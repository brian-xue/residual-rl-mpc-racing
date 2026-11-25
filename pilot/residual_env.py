"""Residual RL on top of the kinematic MPC (or, opt-in, the slip-aware MPC: docs/slip_rl_plan.md).

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

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .closedloop import LapSim, Scenario, Track
from .sim import NOMINAL_PARAMS, make_env

PREVIEW_T = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)  # [s] look-ahead times for the curvature preview
OBS_DIM = 10 + 2 * len(PREVIEW_T)  # single channel (steering residual)
OBS_DIM_DUAL = OBS_DIM + 1 + len(PREVIEW_T)  # + previous accel residual, planned-acceleration preview
WALL_PREVIEW_T = (0.0, 0.25, 0.5, 1.0)  # [s] look-ahead times for the wall-clearance observation
WALL_OBS_SCALE, WALL_CAP = 0.3, 1.5  # [m]


def obs_dim(ra_max: float, wall: bool = False) -> int:
    return (OBS_DIM_DUAL if ra_max > 0 else OBS_DIM) + (2 * len(WALL_PREVIEW_T) if wall else 0)


HIST_DIM = 6  # per history frame: r, beta, yaw-rate error, delta, commanded accel, v
HIST_DIMS = {"f0": HIST_DIM, "f1": HIST_DIM + 1, "f2": HIST_DIM + 4}  # residual.obs_history_features
# scales of the extra frame signals (docs/history_mu_plan.md): realized a_y [m/s^2] and the SlipMPC one-step
# innovation in (beta [rad], r [rad/s], v [m/s]); roughly 2-3 standard deviations of each on arm-R rollouts
AY_SCALE, INN_SCALE = 8.0, (0.01, 0.1, 0.01)


def residual_opts(rc: dict) -> dict:
    """The opt-in residual options beyond (r_max, accel_r_max, obs_wall_clearance); defaults reproduce the
    published policies' observation and action exactly.
      kind: "output" (residual added to the base's commands) or "speed" (arm P' of docs/slip_rl_plan.md: one
            action, a speed-reference scale inside the slip-aware MPC; needs longitudinal.mode: slip)
      speed_range: [lo, hi] scale range for kind "speed"; action 0 -> 1.0, -1 -> lo, +1 -> hi
      obs_a_mpc: append the base MPC's acceleration command (slip / coupled base)
      obs_history: append the last N control-rate frames of (r, beta, r - v kappa, delta, a_cmd, v)
      obs_history_features: "f0" (the six above), "f1" (+ realized a_y over the last interval) or "f2" (f1 + the
            SlipMPC one-step innovation in beta, r, v; needs longitudinal.mode: slip). docs/history_mu_plan.md
      obs_history_mode: "past" (the last N frames) or "repeat" (the current frame N times: a control with the
            same input size and no past information)"""
    return dict(kind=rc.get("kind", "output"), speed_range=tuple(rc.get("speed_range", (0.85, 1.05))),
                obs_a_mpc=bool(rc.get("obs_a_mpc", False)), obs_history=int(rc.get("obs_history", 0)),
                obs_history_features=rc.get("obs_history_features", "f0"),
                obs_history_mode=rc.get("obs_history_mode", "past"))


def speed_scale(a: float, rng: tuple) -> float:
    """Map an action in [-1, 1] to a speed scale: 0 -> 1.0 (the base controller), -1 -> lo, +1 -> hi."""
    lo, hi = rng
    a = float(np.clip(a, -1.0, 1.0))
    return 1.0 + a * ((hi - 1.0) if a > 0 else (1.0 - lo))


class ObsExtras:
    """Observation terms appended after ``residual_obs`` (all off by default, then this adds nothing)."""

    def __init__(self, opts: dict):
        self.a_mpc, self.n_hist = opts["obs_a_mpc"], opts["obs_history"]
        self.feat, self.mode = opts["obs_history_features"], opts["obs_history_mode"]
        if self.feat not in HIST_DIMS or self.mode not in ("past", "repeat"):
            raise ValueError(f"bad obs_history_features / obs_history_mode: {self.feat!r} / {self.mode!r}")
        self.dim = int(self.a_mpc) + HIST_DIMS[self.feat] * self.n_hist
        self.hist = None
        self.prev = None  # (x = [e_y, e_psi, beta, r, v], delta, kappa, a_mpc) at the previous control instant

    def reset(self):
        self.hist = None
        self.prev = None

    def frame_signals(self, sim: LapSim) -> np.ndarray:
        """One history frame at the current control instant (after ``sim.plan()``), scaled. Called exactly once
        per control instant: f1 / f2 difference against the state stored by the previous call."""
        _, _, delta, v, _, r, beta = sim.st[:7]
        kap = sim.ref.kappa[sim.i_g]
        a_cmd = (sim.a_mpc or 0.0) + sim.a_res
        f = [r / 3.0, beta / 0.2, (r - v * kap) / 1.0, delta / sim.s_max, a_cmd / 4.0, v / 8.0]
        if self.feat == "f0":
            return np.array(f)
        x = np.array([sim.e_y, sim.e_psi, beta, r, v], float)
        if self.prev is None:  # first instant of an episode / lap: steady estimate, no innovation
            a_y, inn = v * r, np.zeros(3)
        else:
            x0, d0, k0, am0 = self.prev
            a_y = 0.5 * (x0[4] + v) * (0.5 * (x0[3] + r) + (beta - x0[2]) / sim.Ts)
            if self.feat == "f2":
                # command applied over the last interval: the previous solve's acceleration + the residual
                # applied since (sim.a_res, not changed by plan()); steering: mean realized angle
                a_app = float(np.clip(am0 + sim.a_res, -sim.a_max, sim.a_max))
                pred = sim.smpc.predict(x0, (0.5 * (d0 + delta), a_app), k0, sim.Ts)
                inn = (x[2:5] - pred[2:5]) / np.asarray(INN_SCALE)
        self.prev = (x, float(delta), float(kap), float(sim.a_mpc or 0.0))
        f.append(a_y / AY_SCALE)
        if self.feat == "f2":
            f.extend(inn)
        return np.array(f)

    def __call__(self, sim: LapSim) -> np.ndarray:
        parts = []
        if self.a_mpc:
            parts.append([(sim.a_mpc or 0.0) / 4.0])
        if self.n_hist:
            f = self.frame_signals(sim)
            if self.hist is None or self.mode == "repeat":
                self.hist = deque([f] * self.n_hist, maxlen=self.n_hist)
            else:
                self.hist.append(f)
            parts.append(np.concatenate(self.hist))
        return np.clip(np.concatenate(parts), -5.0, 5.0).astype(np.float32) if parts else np.zeros(0, np.float32)


def wall_gaps(sim: LapSim, s_ahead: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gap between the car body's left / right side and the first wall along the reference normal at the
    preview points, assuming the current lateral offset e_y persists: clearance(s) -/+ e_y - width/2."""
    s_ext, left, right = sim.track.wall_side_clearance(WALL_CAP)
    sa = np.mod(s_ahead, sim.ref.length)
    half_w = NOMINAL_PARAMS["width"] / 2
    return np.interp(sa, s_ext, left) - sim.e_y - half_w, np.interp(sa, s_ext, right) + sim.e_y - half_w


def residual_obs(sim: LapSim, prev_res: float, r_max: float, prev_ares: float = 0.0, ra_max: float = 0.0,
                 wall: bool = False) -> np.ndarray:
    """Observation at a control instant (after ``sim.plan()``). Roughly unit-scaled, clipped to +-5.
    With ``ra_max > 0`` (dual channel) the previous acceleration residual and the planned longitudinal
    acceleration v dv/ds along the preview are appended; otherwise the observation is the single-channel one."""
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
    if ra_max > 0:
        o = np.r_[o, prev_ares / ra_max, sim.aplan_at(s_ahead) / 4.0]
    if wall:  # gap to the side walls now and ahead (residual.obs_wall_clearance), clipped to [-1, 4] x 0.3 m
        gl, gr = wall_gaps(sim, s + max(v, 0.5) * np.asarray(WALL_PREVIEW_T))
        o = np.r_[o, np.clip(gl / WALL_OBS_SCALE, -1.0, 4.0), np.clip(gr / WALL_OBS_SCALE, -1.0, 4.0)]
    return np.clip(o, -5.0, 5.0).astype(np.float32)


def build_obs(sim, r_max, ra_max, wall, opts, extras, prev_res=0.0, prev_ares=0.0, prev_s=1.0):
    """Full policy observation: ``residual_obs`` + the opt-in extras. Shared by training and evaluation."""
    if opts["kind"] == "speed":
        lo, hi = opts["speed_range"]
        o = residual_obs(sim, prev_s - 1.0, max(hi - 1.0, 1.0 - lo), 0.0, 1.0, wall)
    else:
        o = residual_obs(sim, prev_res, r_max, prev_ares, ra_max, wall)
    return np.r_[o, extras(sim)] if extras.dim else o


class ResidualRacingEnv(gym.Env):
    """Gymnasium env: one step = one MPC control interval (Ts = 50 ms, 5 physics steps)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None):
        super().__init__()
        rc = cfg["residual"]
        self.cfg, self.rc = cfg, rc
        self.r_max = rc["r_max"]
        self.ra_max = float(rc.get("accel_r_max", 0.0))  # > 0: dual channel (steering + acceleration residual)
        self.wall_obs = bool(rc.get("obs_wall_clearance", False))  # side-wall gaps in the observation
        self.K = rc.get("base_K", 0.0)
        self.opts = residual_opts(rc)
        self.speed_kind = self.opts["kind"] == "speed"
        if self.speed_kind and cfg["longitudinal"].get("mode") != "slip":
            raise ValueError("residual.kind: speed needs longitudinal.mode: slip (it scales SlipMPC's speed reference)")
        if self.speed_kind and self.ra_max > 0:
            raise ValueError("residual.kind: speed has a single action; set accel_r_max: 0")
        if self.opts["obs_history"] and self.opts["obs_history_features"] == "f2" and \
                cfg["longitudinal"].get("mode") != "slip":
            raise ValueError("obs_history_features: f2 needs longitudinal.mode: slip (SlipMPC's model innovation)")
        self.extras = ObsExtras(self.opts)
        # training tracks (residual.tracks, default: the config's track). One Track + simulator env per track,
        # built once; with several tracks each episode draws one uniformly (episodes have a fixed length, so this
        # gives about equal samples per track). A single track draws nothing extra from the rng.
        self.track_names = list(rc.get("tracks") or [cfg["track"]])
        self._tracks = {}
        for name in self.track_names:
            tc = {**cfg, "track": name}
            self._tracks[name] = (tc, Track(tc), make_env(name, 1.0, cfg["steer_mode"], cfg["dt"], max_steps=10**9))
        self.track_name = self.track_names[0]
        self.track_cfg, self.track, self._env = self._tracks[self.track_name]
        # kind "speed" uses the dual-channel layout (planned-acceleration preview) with the scale in the
        # previous-residual slot; its previous-acceleration slot is always 0
        self.obs_dim = obs_dim(1.0 if self.speed_kind else self.ra_max, self.wall_obs) + self.extras.dim
        self.observation_space = spaces.Box(-5.0, 5.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2 if self.ra_max > 0 else 1,), np.float32)
        lo, hi = rc["a_lat_range"]
        self.a_lat_levels = np.arange(lo, hi + 1e-9, 0.5)
        self.sim = None
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng, rc, opt = self._rng, self.rc, options or {}
        if "track" in opt:
            self.track_name = opt["track"]
        elif len(self.track_names) > 1:
            self.track_name = self.track_names[int(rng.integers(len(self.track_names)))]
        self.track_cfg, self.track, self._env = self._tracks[self.track_name]
        scn = Scenario(
            a_lat=float(opt.get("a_lat", rng.choice(self.a_lat_levels))),
            mu_scale=float(opt.get("mu_scale", rng.uniform(*rc["mu_range"]))),
            start_s=float(opt.get("start_s", rng.uniform(0.0, self.track.ref.length))),
        )
        ey0 = rng.uniform(-1, 1) * rc["init_ey_m"]
        epsi0 = rng.uniform(-1, 1) * rc["init_epsi_rad"]
        self.sim = LapSim(self.track, scn, self.K, self.track_cfg, log=False, env=self._env, ey0=ey0, epsi0=epsi0,
                          steady_init=True)
        self.prev_res = self.prev_ares = 0.0
        self.prev_s = 1.0
        self.extras.reset()
        self.n = 0
        self.sim.plan()
        return self._obs(0.0, 0.0), {"a_lat": scn.a_lat, "mu_scale": scn.mu_scale, "track": self.track_name}

    def _obs(self, res, ares):
        return build_obs(self.sim, self.r_max, self.ra_max, self.wall_obs, self.opts, self.extras, res, ares,
                         self.prev_s)

    def step(self, action):
        sim, rc = self.sim, self.rc
        if self.speed_kind:  # arm P': the action scales the speed reference of the NEXT solve (50 ms later)
            sc = speed_scale(action[0], self.opts["speed_range"])
            sim.smpc.set_speed_scale(sc)
            res = ares = 0.0
        else:
            res = float(np.clip(action[0], -1.0, 1.0)) * self.r_max
            ares = float(np.clip(action[1], -1.0, 1.0)) * self.ra_max if self.ra_max > 0 else 0.0
        p0 = sim.progress
        sim.apply(res, ares)
        sim.advance_interval()
        self.n += 1
        w = rc["reward"]
        dprog = sim.progress - p0
        ratio = dprog / max(sim.vp * sim.Ts, 1e-3)  # progress relative to the speed plan this step
        rew = (w["progress"] * ratio
               - w["e_y"] * sim.e_y**2
               - w["res"] * res**2
               - w["dres"] * (res - self.prev_res) ** 2
               - w["off_track"] * (sim.off_in_interval > 0))
        if self.ra_max > 0:  # dual channel: gentle acceleration-residual regularisation (size and chatter)
            rew -= w.get("ares", 0.0) * ares**2 + w.get("dares", 0.0) * (ares - self.prev_ares) ** 2
        if w.get("slow", 0.0) > 0:  # opt-in: hinge on running slower than slow_ratio x plan (the lap-time rule is 1.05)
            rew -= w["slow"] * max(w["slow_ratio"] - ratio, 0.0)
        if w.get("beta", 0.0) > 0:  # high sideslip, before it becomes a spin
            rew -= w["beta"] * max(abs(sim.st[6]) - w["beta_soft_rad"], 0.0) ** 2
        if self.speed_kind:  # opt-in chatter penalty on the speed scale
            rew -= w.get("dspeed", 0.0) * (sc - self.prev_s) ** 2
            self.prev_s = sc
        self.prev_res, self.prev_ares = res, ares
        terminated = truncated = False
        info = {}
        if sim.outcome in ("wall_collision", "spin"):
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
            info["track"] = self.track_name
            obs = np.zeros(self.obs_dim, np.float32) if sim.outcome == "wall_collision" else self._obs(res, ares)
            return obs, float(rew), terminated, truncated, info
        sim.plan()
        return self._obs(res, ares), float(rew), False, False, info


# ---------------------------------------------------------------------------------- evaluation-time policy
_MODELS: dict = {}


class ResidualPolicy:
    """Picklable residual factory for ``run_lap(..., residual=ResidualPolicy(path, r_max[, ra_max]))``.
    Calling it returns a per-lap callable ``g(sim)`` (deterministic policy mean) that returns delta_res, or
    (delta_res, a_res) for a dual-channel policy (``ra_max > 0``)."""

    def __init__(self, model_path: str, r_max: float, ra_max: float = 0.0, wall_obs: bool = False, opts=None):
        self.model_path, self.r_max, self.ra_max, self.wall_obs = str(model_path), r_max, float(ra_max), bool(wall_obs)
        self.opts = opts or residual_opts({})

    @classmethod
    def from_rc(cls, model_path, rc: dict):
        """All residual settings from a config's ``residual`` block."""
        return cls(model_path, rc["r_max"], rc.get("accel_r_max", 0.0), rc.get("obs_wall_clearance", False),
                   residual_opts(rc))

    def __call__(self):
        return _LapResidual(self.model_path, self.r_max, self.ra_max, self.wall_obs, self.opts)


class _LapResidual:
    def __init__(self, path, r_max, ra_max=0.0, wall_obs=False, opts=None):
        if path not in _MODELS:
            import torch
            from stable_baselines3 import PPO

            torch.set_num_threads(1)
            _MODELS[path] = PPO.load(path, device="cpu")
        self.model, self.r_max, self.ra_max, self.wall_obs = _MODELS[path], r_max, ra_max, wall_obs
        self.opts = opts or residual_opts({})
        self.extras = ObsExtras(self.opts)
        self.prev = self.prev_a = 0.0
        self.prev_s = 1.0

    def reset(self):
        self.prev = self.prev_a = 0.0
        self.prev_s = 1.0
        self.extras.reset()

    def __call__(self, sim):
        obs = build_obs(sim, self.r_max, self.ra_max, self.wall_obs, self.opts, self.extras, self.prev, self.prev_a,
                        self.prev_s)
        a, _ = self.model.predict(obs, deterministic=True)
        if self.opts["kind"] == "speed":  # same timing as training: scales the next solve's speed reference
            self.prev_s = speed_scale(a[0], self.opts["speed_range"])
            sim.smpc.set_speed_scale(self.prev_s)
            return 0.0, 0.0
        self.prev = float(np.clip(a[0], -1, 1)) * self.r_max
        if self.ra_max > 0:
            self.prev_a = float(np.clip(a[1], -1, 1)) * self.ra_max
            return self.prev, self.prev_a
        return self.prev
