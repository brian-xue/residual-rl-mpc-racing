"""Residual RL on top of the kinematic MPC.

The learned policy adds a steering residual to the MPC command at every control instant (20 Hz):

    delta = clip(delta_MPC + r_max * pi(obs), +-s_max)        (then the shared 3.2 rad/s rate limit)

Everything else is the pilot's closed loop (``pilot.closedloop.LapSim``): same reference line, speed plan,
longitudinal controller, MPC, servo, steering limits and per-step validity checks. The policy only sees
generic, track-independent signals (tracking errors, vehicle state, the MPC's own command and a curvature /
lateral-acceleration preview); it never sees the arc length s or the true friction.

Training distribution (domain randomisation): friction mu/mu0 ~ U(mu_range), speed level a_lat drawn from a
0.5 m/s^2 grid over a_lat_range, start anywhere on the lap with a small pose offset and steady-state
steering, episodes of ``episode_steps`` control steps (truncated, so values bootstrap).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .closedloop import LapSim, Scenario, Track
from .sim import make_env

PREVIEW_T = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)  # [s] look-ahead times for the curvature preview
OBS_DIM = 10 + 2 * len(PREVIEW_T)


def residual_obs(sim: LapSim, prev_res: float, r_max: float) -> np.ndarray:
    """Observation at a control instant (after ``sim.plan()``). Roughly unit-scaled, clipped to +-5."""
    x, y, delta, v, yaw, r, beta = sim.st[:7]
    s, ref = sim.s, sim.ref
    s_ahead = s + max(v, 0.5) * np.asarray(PREVIEW_T)
    kap = ref.kappa_at(s_ahead)
    vpl = sim.vplan_at(s_ahead)
    k_now = kap[0]
    o = np.r_[
        sim.e_y / 0.3,
        sim.e_psi / 0.3,
        v / 8.0,
        (v - sim.vp) / 2.0,
        r / 3.0,
        beta / 0.2,
        (r - v * k_now) / 1.0,  # yaw-rate error w.r.t. the path
        delta / sim.s_max,
        sim.d_mpc / sim.s_max,
        prev_res / r_max,
        kap / 0.3,  # curvature preview
        vpl**2 * kap / 8.0,  # planned lateral-acceleration preview
    ]
    return np.clip(o, -5.0, 5.0).astype(np.float32)


class ResidualRacingEnv(gym.Env):
    """Gymnasium env: one step = one MPC control interval (Ts = 50 ms, 5 physics steps)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None):
        super().__init__()
        rc = cfg["residual"]
        self.cfg, self.rc = cfg, rc
        self.r_max = rc["r_max"]
        self.K = rc.get("base_K", 0.0)
        self.track = Track(cfg)
        self._env = make_env(cfg["track"], 1.0, cfg["steer_mode"], cfg["dt"], max_steps=10**9)
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        lo, hi = rc["a_lat_range"]
        self.a_lat_levels = np.arange(lo, hi + 1e-9, 0.5)
        self.sim = None
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng, rc, opt = self._rng, self.rc, options or {}
        scn = Scenario(
            a_lat=float(opt.get("a_lat", rng.choice(self.a_lat_levels))),
            mu_scale=float(opt.get("mu_scale", rng.uniform(*rc["mu_range"]))),
            start_s=float(opt.get("start_s", rng.uniform(0.0, self.track.ref.length))),
        )
        ey0 = rng.uniform(-1, 1) * rc["init_ey_m"]
        epsi0 = rng.uniform(-1, 1) * rc["init_epsi_rad"]
        self.sim = LapSim(self.track, scn, self.K, self.cfg, log=False, env=self._env, ey0=ey0, epsi0=epsi0,
                          steady_init=True)
        self.prev_res = 0.0
        self.n = 0
        self.sim.plan()
        return residual_obs(self.sim, 0.0, self.r_max), {"a_lat": scn.a_lat, "mu_scale": scn.mu_scale}

    def step(self, action):
        sim, rc = self.sim, self.rc
        res = float(np.clip(action[0], -1.0, 1.0)) * self.r_max
        p0 = sim.progress
        sim.apply(res)
        sim.advance_interval()
        self.n += 1
        w = rc["reward"]
        dprog = sim.progress - p0
        rew = (w["progress"] * dprog / max(sim.vp * sim.Ts, 1e-3)
               - w["e_y"] * sim.e_y**2
               - w["res"] * res**2
               - w["dres"] * (res - self.prev_res) ** 2
               - w["off_track"] * (sim.off_in_interval > 0))
        self.prev_res = res
        terminated = truncated = False
        info = {}
        if sim.outcome == "wall_collision":
            rew -= w["crash"]
            terminated = True
        elif abs(sim.e_y) > self.cfg["qualifying"]["max_abs_ey_m"]:
            rew -= w["crash"]
            terminated = True
            sim.outcome = "lateral_error"
        elif sim.done or self.n >= rc["episode_steps"]:
            truncated = True
        if terminated or truncated:
            info["outcome"] = sim.outcome or "running"
            obs = np.zeros(OBS_DIM, np.float32) if sim.outcome == "wall_collision" else residual_obs(sim, res, self.r_max)
            return obs, float(rew), terminated, truncated, info
        sim.plan()
        return residual_obs(sim, res, self.r_max), float(rew), False, False, info


# ---------------------------------------------------------------------------------- evaluation-time policy
_MODELS: dict = {}


class ResidualPolicy:
    """Picklable residual factory for ``run_lap(..., residual=ResidualPolicy(path, r_max))``.
    Calling it returns a per-lap callable ``g(sim) -> delta_res`` (deterministic policy mean)."""

    def __init__(self, model_path: str, r_max: float):
        self.model_path, self.r_max = str(model_path), r_max

    def __call__(self):
        return _LapResidual(self.model_path, self.r_max)


class _LapResidual:
    def __init__(self, path, r_max):
        if path not in _MODELS:
            import torch
            from stable_baselines3 import PPO

            torch.set_num_threads(1)
            _MODELS[path] = PPO.load(path, device="cpu")
        self.model, self.r_max, self.prev = _MODELS[path], r_max, 0.0

    def reset(self):
        self.prev = 0.0

    def __call__(self, sim):
        obs = residual_obs(sim, self.prev, self.r_max)
        a, _ = self.model.predict(obs, deterministic=True)
        self.prev = float(np.clip(a[0], -1, 1)) * self.r_max
        return self.prev
