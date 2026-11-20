"""One closed-loop lap: shared reference, speed plan, longitudinal controller, MPC, limits.

Controllers differ only in the steering correction added after the MPC:
Lap validity is decided by pilot.geometry at every physics step: ``off_track`` (no tire patch on the
legal region) or ``wall_collision`` (body touches a wall) invalidate the lap; a wall collision also ends it.
Optional ``spin`` (``geometry.spin_beta_max_rad``): |sideslip| above the threshold ends the lap. Without it a
spinning car on the ST plant keeps its speed (V' = a) and runs until the timeout.

    nominal  : delta = clip(delta_mpc)
    ff       : delta = clip(delta_mpc + K v^2 kappa_ref(s))
    residual : delta = clip(delta_mpc + pi(obs))          (learned, see pilot.residual_env)
The final clip (|delta| <= s_max) and rate limit (|d delta/dt| <= sv_max, enforced by the
servo every simulator step) are applied after the correction and are identical for both.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import TrackGeometry
from .kernel import KERNELS, make_kernel
from .mpc import CoupledMPC, KinematicMPC, SlipMPC
from .reference import Reference, speed_plan, wall_safe_raceline
from .sim import (NOMINAL_PARAMS, SteeringServo, agent, make_env, plant_model, reset_to_state, set_env_params,
                  set_plant, vehicle_params)


@dataclass
class Scenario:
    a_lat: float  # speed-level parameter of the speed plan [m/s^2]
    mu_scale: float  # true mu / mu_nominal, simulator only
    start_s: float  # start position along the reference [m]
    v_cap: float | None = None  # straight-line speed cap of the speed plan [m/s]; None -> config value


class Track:
    """Loaded once per process: env track, legal/wall geometry, fixed reference line."""

    def __init__(self, cfg):
        self.cfg = cfg
        set_plant(cfg.get("model", "st"), cfg.get("tire"))
        rc = cfg["reference"]
        env = make_env(cfg["track"], 1.0, cfg["steer_mode"], cfg["dt"])
        self.geo = TrackGeometry(env.track, NOMINAL_PARAMS, {**cfg["geometry"], "track": cfg["track"]})
        self.half_width = self.geo.half_width
        c = env.track.centerline
        center = Reference(c.xs, c.ys, ds=0.05, smooth_sigma_m=0.15, kappa_sigma_m=0.3)
        self.info = {}
        if rc["source"] == "raceline":
            r = env.track.raceline_regular
            wall_dist = self.geo.centerline_wall_distance()
            reach = rc["wall_search_m"]
            hw, mg = NOMINAL_PARAMS["width"] / 2, rc["wall_margin_m"]

            def n_max(pts):  # allowed offset: inf where no wall within reach of the centerline point
                d = self.geo.wall_distance(pts)
                return np.where(d > reach, np.inf, d - hw - mg)

            xs, ys, n_raw, n_safe = wall_safe_raceline(np.c_[r.xs, r.ys], center, n_max, rc["offset_smooth_sigma_m"],
                                                       float(np.mean(np.diff(r.ss))), rc["wall_lookaround_m"])
            self.info.update(centerline_wall_distance_min_m=wall_dist, raw_offset_max=float(np.abs(n_raw).max()),
                             frac_points_changed=float(np.mean(np.abs(n_raw - n_safe) > 1e-3)),
                             max_shift_m=float(np.abs(n_raw - n_safe).max()))
        else:
            xs, ys = c.xs, c.ys
        self.ref = Reference(xs, ys, ds=rc["ds"], smooth_sigma_m=rc["smooth_sigma_m"], kappa_sigma_m=rc["kappa_sigma_m"])
        self.info.update(wall_cells_map=int(self.geo.n_wall_cells_map), wall_cells_kept=len(self.geo.wall_boxes),
                         centerline_frac_curve=float(self.geo.curve_mask.mean()))
        # perfect-tracking check of the reference itself (body on the line, heading = line heading)
        L = NOMINAL_PARAMS["lf"] + NOMINAL_PARAMS["lr"]
        clear, n_wall, n_off = [], 0, 0
        for i in range(0, self.ref.n, 2):
            x, y, p = self.ref.x[i], self.ref.y[i], self.ref.psi[i]
            off, wall, _ = self.geo.check(x, y, p, np.arctan(L * self.ref.kappa[i]))
            n_wall += wall
            n_off += off
            clear.append(self.geo.wall_clearance(x, y, p))
        self.info.update(source=rc["source"], length_m=self.ref.length, kappa_abs_max=float(np.abs(self.ref.kappa).max()),
                         perfect_tracking_wall_contacts=int(n_wall), perfect_tracking_off_track=int(n_off),
                         perfect_tracking_min_wall_clearance_m=float(np.min(clear)))
        self._plans = {}
        self._env_pool = {}  # idle simulator envs per (track, steer_mode, dt, plant): see acquire_env
        self.release_env(env, self._env_key(cfg))

    @staticmethod
    def _env_key(cfg):
        return cfg["track"], cfg["steer_mode"], cfg["dt"], plant_model()

    def acquire_env(self, cfg, mu_scale, max_steps):
        """Simulator env for one lap: an idle env from an earlier lap of this process, re-parameterised, or a new one.

        ``make_env`` costs ~0.3 s (map, distance transform, raceline spline), more than stepping a whole lap with
        the Rust kernel. A reused env is reset by ``reset_to_state`` exactly like a new one and gets the lap's
        params and episode limit, so laps are bit-identical either way. Returns (env, pool key)."""
        key = self._env_key(cfg)
        idle = self._env_pool.get(key)
        if idle:
            env = idle.pop()
            set_env_params(env, vehicle_params(mu_scale))
            env.max_episode_steps = max_steps
        else:
            env = make_env(cfg["track"], mu_scale, cfg["steer_mode"], cfg["dt"], max_steps=max_steps)
        return env, key

    def release_env(self, env, key):
        self._env_pool.setdefault(key, []).append(env)

    def wall_side_clearance(self, cap=1.5):
        """(left, right): distance from each reference point to the first wall cell along the left / right normal
        of the reference line (``cap`` where there is none, e.g. on curves without walls). Computed once per
        track from its own walls (residual observation ``residual.obs_wall_clearance``)."""
        if getattr(self, "_wall_side", None) is None:
            ref = self.ref
            xy = np.c_[ref.x, ref.y]
            nrm = np.c_[-np.sin(ref.psi), np.cos(ref.psi)]  # left normal (e_y > 0 is left of the line)
            left = self.geo.ray_wall_distance(xy, nrm, cap)
            right = self.geo.ray_wall_distance(xy, -nrm, cap)
            s_ext = np.r_[ref.s, ref.length]
            self._wall_side = (s_ext, np.r_[left, left[0]], np.r_[right, right[0]])
        return self._wall_side

    def plan(self, a_lat, v_cap=None, friction_ellipse=False):
        sp_ = self.cfg["speed_plan"]
        v_cap = sp_["v_cap"] if v_cap is None else v_cap
        fe = bool(friction_ellipse)
        if (a_lat, v_cap, fe) not in self._plans:
            self._plans[(a_lat, v_cap, fe)] = speed_plan(self.ref, a_lat, v_cap, sp_["a_acc"], sp_["a_dec"],
                                                         friction_ellipse=fe)
        return self._plans[(a_lat, v_cap, fe)]


class LapSim:
    """One closed-loop lap that can be advanced one control interval (Ts) at a time.

    Used by ``run_lap`` (whole lap) and by the residual-RL environment (``pilot.residual_env``), so training
    and evaluation share one implementation of the servo, limits, speed controller and validity checks.

    Per control instant: ``plan()`` solves the MPC (or ``steer_planner``) and the K v^2 kappa term,
    ``apply(d_res)`` adds an optional learned residual, clips and logs the command, and
    ``advance_interval()`` runs the physics steps up to the next control instant (or the end of the lap).

    ``env`` (optional) reuses an existing simulator instance (friction is updated in place); ``ey0``/``epsi0``
    offset the start pose from the reference; ``steady_init`` starts with the kinematic steady-state steering
    and yaw rate of the reference curvature instead of delta = r = 0. Defaults reproduce the pilot exactly.
    """

    def __init__(self, track: Track, scn: Scenario, K: float, cfg, log=True, steer_planner=None, env=None,
                 ey0=0.0, epsi0=0.0, steady_init=False, residual_logged=False):
        ref, dt = track.ref, cfg["dt"]
        self.track, self.scn, self.K, self.cfg, self.log = track, scn, K, cfg, log
        self.ref, self.dt, self.geo = ref, dt, track.geo
        mc, lim = cfg["mpc"], cfg["limits"]
        self.mc = mc
        self.s_max, self.sv_max = lim["s_max"], lim["sv_max"]
        self.Ts = mc["Ts"]
        self.n_sub = int(round(self.Ts / dt))
        vplan = track.plan(scn.a_lat, scn.v_cap, cfg["speed_plan"].get("friction_ellipse", False))
        # periodic lookups on arrays built once (called several times per physics step)
        s_ext, L_ref = np.r_[ref.s, ref.length], ref.length
        v_ext = np.r_[vplan, vplan[0]]
        self._plan_s, self._plan_v = s_ext, v_ext  # also handed to the Rust kernel
        self.vplan_at = lambda s: np.interp(np.mod(s, L_ref), s_ext, v_ext)  # noqa: E731
        aplan = vplan * np.gradient(vplan, ref.ds)  # plan acceleration v dv/ds (feedforward for the coupled MPC)
        a_ext = np.r_[aplan, aplan[0]]
        self.aplan_at = lambda s: np.interp(np.mod(s, L_ref), s_ext, a_ext)  # noqa: E731
        self.plan_lap_time = float(np.sum(ref.ds / vplan))
        self.t_max = cfg["qualifying"]["timeout_factor"] * self.plan_lap_time

        set_plant(cfg.get("model", "st"), cfg.get("tire"))
        self.a_max = vehicle_params(scn.mu_scale)["a_max"]
        self._own_env = env is None  # own env: taken from the Track's pool, truncates at max_episode_steps
        if env is None:
            env, self._env_key = track.acquire_env(cfg, scn.mu_scale, int(self.t_max / dt) + 10)
        else:
            env.sim.update_params(vehicle_params(scn.mu_scale))
        env._check_boundary_frenet = lambda idx: False  # replaced by the four-tire / body checks below
        self.env = env
        self._sim = env.sim  # stepped directly, see _physics
        L = NOMINAL_PARAMS["lf"] + NOMINAL_PARAMS["lr"]
        self.L = L
        x0, y0, psi0 = ref.pose_at(scn.start_s)
        v0 = float(self.vplan_at(scn.start_s))
        x0, y0, psi0 = x0 - ey0 * np.sin(psi0), y0 + ey0 * np.cos(psi0), psi0 + epsi0
        d0 = r0 = 0.0
        if steady_init:
            k0 = float(ref.kappa_at(scn.start_s))
            d0, r0 = float(np.arctan(L * k0)), v0 * k0
        reset_to_state(env, [x0, y0, d0, v0, psi0 % (2 * np.pi), r0, 0.0])
        self.mpc = KinematicMPC(NOMINAL_PARAMS["lf"], NOMINAL_PARAMS["lr"], self.Ts, mc["N"], mc["q_y"], mc["q_psi"],
                                mc["r_du"], mc["q_terminal"], self.s_max, self.sv_max * self.Ts)
        self.planner = steer_planner(cfg, scn.mu_scale) if steer_planner is not None else None
        self.servo = SteeringServo(cfg["steer_mode"], dt, self.sv_max)
        lc = cfg["longitudinal"]
        self.kv = lc["kv"]
        # longitudinal mode: "pfb" = plan feedforward + P (pilot default), "mpc" = coupled steering + speed MPC
        self.cmpc, self.smpc, self.a_mpc, self.a_prev = None, None, None, 0.0
        self.a_res, self.sum_abs_ares = 0.0, 0.0  # longitudinal residual [m/s^2], held over the control interval
        if lc.get("mode", "pfb") == "slip" and self.planner is None:
            sw_ = lc["slip_mpc"]
            ell = None
            if sw_.get("friction_ellipse", False):
                ell = dict(ax_max=sw_.get("ax_max", cfg["speed_plan"]["a_dec"]), ay_max=scn.a_lat,
                           n_facets=sw_["n_facets"])
            # nominal parameters (mu0, linear stiffness): the slip-aware MPC never sees the true mu or tire curve
            self.smpc = SlipMPC(vehicle_params(1.0), self.Ts, mc["N"],
                                {k: mc[k] for k in ("q_y", "q_psi", "r_du", "q_terminal")}, sw_,
                                self.s_max, self.sv_max * self.Ts, ellipse=ell)
        if lc.get("mode", "pfb") == "mpc" and self.planner is None:
            lm = lc["mpc"]
            vp_ = vehicle_params(scn.mu_scale)
            ell = None
            if lm.get("friction_ellipse", False):
                ell = dict(ax_max=lm.get("ax_max", cfg["speed_plan"]["a_dec"]), ay_max=scn.a_lat,
                           n_facets=lm["n_facets"], slack_l1=lm["slack_l1"], slack_l2=lm["slack_l2"])
            self.cmpc = CoupledMPC(NOMINAL_PARAMS["lf"], NOMINAL_PARAMS["lr"], self.Ts, mc["N"],
                                   {k: mc[k] for k in ("q_y", "q_psi", "r_du", "q_terminal")}, lm,
                                   self.s_max, self.sv_max * self.Ts, vp_["a_max"], vp_["v_switch"], ellipse=ell)
        # measured acceleration usage, same ellipse for every controller: a_x,max = speed_plan.a_dec, a_y,max = a_lat
        self.ell_ax, self.ell_ay = cfg["speed_plan"]["a_dec"], scn.a_lat
        self.ell_max, self.n_ell_over, self.max_ay, self.min_ax = 0.0, 0, 0.0, 0.0
        self.tau = mc["delay_comp_s"]
        self.spin_beta_max = cfg["geometry"].get("spin_beta_max_rad")  # None: no spin check
        self.residual_logged = residual_logged

        s0, _, self.i_g = ref.project(x0, y0)
        self.progress, self.s_prev = 0.0, s0
        self.delta_mpc_prev, self.delta_cmd = 0.0, 0.0
        self.d_mpc = self.d_ff = self.d_res = 0.0
        self.t, self.step = 0.0, 0
        self.rows, self.solve_times = [], []
        self.n_fail = self.n_sat_mag = self.n_ctrl = self.n_sat_rate = 0
        self.max_ey, self.sum_ey2, self.n_ey, self.sum_dv, self.sum_abs_res = 0.0, 0.0, 0, 0.0, 0.0
        self.outcome = None
        self.n_off, self.first_off_s, self.first_off_progress = 0, None, None
        self.tires_in = np.ones(4, bool)
        self.off_in_interval = 0  # off-track physics steps since the last control instant
        self.done = self._pre()
        # physics-step loop: "python" (_physics/_pre below) or "rust" (pilot_core.LapKernel, bit-identical)
        kernel = cfg.get("kernel", "python")
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.kernel = make_kernel(self) if kernel == "rust" else None
        # with the Rust kernel the kinematic MPC step (previews, QP, OSQP) also runs in Rust (bit-identical)
        self.rqp = self.mpc.rust_qp() if self.kernel is not None else None

    # ------------------------------------------------------------------ per physics step
    def _pre(self):
        """Read the state, update metrics, check lap completion / timeout. Returns True when the lap ended."""
        ref = self.ref
        self.st = st = agent(self.env).state
        x, y, delta, v, yaw, r, beta = st[:7]
        s, e_y, self.i_g = ref.project(x, y, self.i_g)
        ds_ = (s - self.s_prev + ref.length / 2) % ref.length - ref.length / 2
        self.progress += ds_
        self.s_prev = s
        self.s, self.e_y, self.v = s, e_y, v
        self.e_psi = (yaw - ref.psi[self.i_g] + np.pi) % (2 * np.pi) - np.pi
        self.max_ey = max(self.max_ey, abs(e_y))
        self.sum_ey2 += e_y**2
        self.n_ey += 1
        self.vp = float(self.vplan_at(s))
        self.sum_dv += abs(v - self.vp)
        if self.progress >= ref.length:
            self.outcome = "completed"
            return True
        if self.t >= self.t_max:
            self.outcome = "timeout"
            return True
        return False

    def _physics(self):
        """Servo + longitudinal control + one simulator step + validity checks. Returns True when the lap ended.

        ``_physics``, ``_pre`` and ``advance_interval`` are mirrored operation by operation in
        ``pilot_core/src/kernel.rs`` (``kernel: rust``): change both, then run ``python -m pilot.kernel_check``."""
        ref, st = self.ref, self.st
        u_steer = self.servo.command(self.delta_cmd, st[2])
        if self.cfg["steer_mode"] == "steering_speed":
            self.n_sat_rate += abs(u_steer) >= self.sv_max - 1e-9
        # shared longitudinal controller: speed-plan feed-forward + P feedback
        s, v = self.s, self.v
        if self.a_mpc is not None:
            a_req = self.a_mpc
        else:
            dvds = (self.vplan_at(s + 0.2) - self.vplan_at(s - 0.2)) / 0.4
            a_req = v * dvds + self.kv * (self.vp - v)
        if self.a_res != 0.0:  # longitudinal residual (dual-channel residual RL); mirrored in pilot_core
            a_req = a_req + self.a_res
        a_cmd = float(np.clip(a_req, -self.a_max, self.a_max))
        # Simulator.step is the physics of GKEnv.step; the env's reward (a Frenet nearest-point search), observation,
        # render and done bookkeeping are not used here and cost ~4x the physics. Only its step counter is kept,
        # for the same max_episode_steps truncation.
        self._sim.step(np.array([[u_steer, a_cmd]]))
        env = self.env
        env.current_time, env.current_step = env.current_time + env.timestep, env.current_step + 1
        trunc = env.current_step > env.max_episode_steps
        self.t += self.dt
        self.step += 1
        # track-limit and wall checks at every physics step (same for every controller)
        st = agent(self.env).state
        # measured CG accelerations: longitudinal dv/dt, lateral v (yaw rate + d slip/dt)
        ax, ay = (st[3] - v) / self.dt, st[3] * (st[5] + (st[6] - self.st[6]) / self.dt)
        e = np.hypot(ax / self.ell_ax, ay / self.ell_ay)
        self.ell_max, self.n_ell_over = max(self.ell_max, e), self.n_ell_over + (e > 1.02)
        self.max_ay, self.min_ax = max(self.max_ay, abs(ay)), min(self.min_ax, ax)
        off, wall, self.tires_in = self.geo.check(st[0], st[1], st[4], st[2])
        if off:
            self.n_off += 1
            self.off_in_interval += 1
            if self.first_off_s is None:
                self.first_off_s = float(ref.project(st[0], st[1], self.i_g)[0])
                self.first_off_progress = float(self.progress)
        if wall:
            self.outcome = "wall_collision"
            self.s, self.e_y, self.i_g = ref.project(st[0], st[1], self.i_g)
            return True
        if self.spin_beta_max is not None and abs(st[6]) > self.spin_beta_max:
            self.outcome = "spin"
            self.s, self.e_y, self.i_g = ref.project(st[0], st[1], self.i_g)
            return True
        if trunc and self._own_env:
            self.outcome = "timeout"
            return True
        return False

    # ------------------------------------------------------------------ per control instant (20 Hz)
    def plan(self):
        """MPC (or alternative planner) + K v^2 kappa feedforward at the current control instant."""
        ref, mc, st = self.ref, self.mc, self.st
        delta, v, e_y, e_psi, s = st[2], self.v, self.e_y, self.e_psi, self.s
        tau, L, Ts = self.tau, self.L, self.Ts
        d_ff = None  # K v^2 kappa(s_p), computed below unless the Rust MPC step returned it
        if self.planner is not None:  # alternative steering planner (20 Hz)
            d_mpc, ok, tsol = self.planner.solve(st, e_y, e_psi, s, self.delta_mpc_prev, self.vplan_at, ref)
            s_p = s + tau * v
        elif self.smpc is not None:  # slip-aware coupled MPC (20 Hz): measures beta and r, model-based delay comp.
            d_mpc, a_mpc, ok, tsol, _ = self.smpc.solve(e_y, e_psi, st[6], st[5], v, self.delta_mpc_prev, self.a_prev,
                                                       s, self.vplan_at, self.aplan_at, ref.kappa_at, tau=tau)
            s_p = s + tau * v
            self.a_mpc = self.a_prev = a_mpc
        elif self.cmpc is not None:  # coupled steering + speed MPC (20 Hz), same delay compensation
            b = np.arctan(NOMINAL_PARAMS["lr"] / L * np.tan(delta))
            e_y_p = e_y + tau * v * np.sin(e_psi + b)
            e_psi_p = e_psi + tau * (v * np.cos(b) * np.tan(delta) / L - v * ref.kappa[self.i_g])
            s_p = s + tau * v
            v_p = v + tau * self.a_prev
            d_mpc, a_mpc, ok, tsol, _ = self.cmpc.solve(e_y_p, e_psi_p, self.delta_mpc_prev, v_p, self.a_prev, s_p,
                                                       self.vplan_at, self.aplan_at, ref.kappa_at)
            self.a_mpc = self.a_prev = a_mpc
        elif self.rqp is not None:  # MPC update (20 Hz) in Rust: same steps as the branch below
            d_mpc, ok, tsol, s_p, d_ff = self.kernel.plan_kinematic(self.rqp, self.delta_mpc_prev, tau, self.K)
        else:  # MPC update (20 Hz)
            # delay compensation: propagate the kinematic error state by tau with current steering
            b = np.arctan(NOMINAL_PARAMS["lr"] / L * np.tan(delta))
            e_y_p = e_y + tau * v * np.sin(e_psi + b)
            e_psi_p = e_psi + tau * (v * np.cos(b) * np.tan(delta) / L - v * ref.kappa[self.i_g])
            s_p = s + tau * v
            s_k = s_p + np.cumsum(np.r_[0.0, [Ts * max(self.vplan_at(s_p), 0.5)] * (mc["N"] - 1)])
            s_k = s_p + np.r_[0.0, np.cumsum(Ts * self.vplan_at(s_k[:-1]))]
            v_seq = self.vplan_at(s_k)
            v_seq[0] = v
            kap = ref.kappa_at(s_k)
            d_mpc, ok, tsol, _ = self.mpc.solve(e_y_p, e_psi_p, self.delta_mpc_prev, v_seq, kap)
        self.solve_times.append(tsol)
        if not ok:
            self.n_fail += 1
        self.solver_ok = ok
        self.delta_mpc_prev = d_mpc
        self.d_mpc, self.s_p = d_mpc, s_p
        self.d_ff = d_ff if d_ff is not None else self.K * v**2 * ref.kappa_at(s_p)

    def apply(self, d_res=0.0, a_res=0.0):
        """Add the residual(s), clip to the steering limit, log. The servo enforces the rate limit; ``a_res``
        (dual-channel residual) is added to the speed command at every physics step of the interval."""
        self.d_res = float(d_res)
        self.a_res = float(a_res)
        self.sum_abs_ares += abs(self.a_res)
        self.sum_abs_res += abs(self.d_res)
        req = self.d_mpc + self.d_ff + self.d_res
        self.delta_cmd = float(np.clip(req, -self.s_max, self.s_max))
        self.n_ctrl += 1
        self.n_sat_mag += abs(req) >= self.s_max
        self.off_in_interval = 0
        if self.log:
            x, y, delta, v, yaw, r, beta = self.st[:7]
            row = dict(t=self.t, progress=self.progress, s=self.s, e_y=self.e_y, e_psi=self.e_psi, v=v, v_plan=self.vp,
                       delta_mpc=self.d_mpc, delta_ff=self.d_ff, delta_cmd=self.delta_cmd, delta=delta,
                       yaw_rate=r, slip=beta, kappa=self.ref.kappa[self.i_g], solve_ms=self.solve_times[-1] * 1e3,
                       solver_ok=self.solver_ok, x=x, y=y, yaw=yaw, n_tires_in=int(self.tires_in.sum()))
            if self.cmpc is not None or self.smpc is not None:
                row["a_mpc"] = self.a_mpc
            if self.residual_logged:
                row["delta_res"] = self.d_res
                row["a_res"] = self.a_res
            self.rows.append(row)

    def advance_interval(self):
        """Run physics steps up to the next control instant or the end of the lap. Returns ``self.done``."""
        if self.kernel is not None:
            return self._advance_kernel()
        while not self.done:
            if self._physics() or self._pre():
                self.done = True
            elif self.step % self.n_sub == 0:
                break
        return self.done

    def _advance_kernel(self):
        """``advance_interval`` in Rust; copies back every attribute the 20 Hz code and the summary read."""
        if self.done:
            return True
        (done, outcome, st, x, (t, s, e_y, e_psi, v, vp, progress), (step, i_g, off, env_step),
         tires_in) = self.kernel.advance(self.delta_cmd, self.a_mpc, self.a_res)
        self.st, self.t, self.s, self.e_y, self.e_psi, self.v, self.vp = st, t, s, e_y, e_psi, v, vp
        self.progress, self.step, self.i_g, self.off_in_interval = progress, step, i_g, off
        self.tires_in, self.outcome, self.done = np.array(tires_in), outcome, done
        agent(self.env).state = x  # the simulator's view of the car (e.g. for a reused env)
        self.env.current_step = env_step
        if done:
            self._sync_kernel_metrics()
        return done

    def _sync_kernel_metrics(self):
        for k, val in self.kernel.metrics().items():
            setattr(self, k, val)

    def close(self):
        """Return an own env to the Track's pool (a reused env passed in by the caller stays with the caller)."""
        if self._own_env:
            self.track.release_env(self.env, self._env_key)
            self._own_env = False

    def summary(self):
        if self.kernel is not None:
            self._sync_kernel_metrics()
        scn, cfg, ref, outcome, t = self.scn, self.cfg, self.ref, self.outcome, self.t
        st = np.array(self.solve_times) * 1e3
        summ = dict(
            a_lat=scn.a_lat, v_cap=scn.v_cap if scn.v_cap is not None else cfg["speed_plan"]["v_cap"], mu_scale=scn.mu_scale, start_s=scn.start_s, K=self.K,
            outcome=outcome, off_track_steps=self.n_off, first_off_track_s=self.first_off_s,
            first_off_track_progress=self.first_off_progress, progress_m=self.progress, progress_frac=self.progress / ref.length,
            fail_s=None if outcome == "completed" else float(self.s),
            lap_time=t if outcome == "completed" else None, plan_lap_time=self.plan_lap_time,
            lap_time_ratio=t / self.plan_lap_time if outcome == "completed" else None,
            mean_speed=self.progress / max(t, 1e-9), mean_abs_dv=self.sum_dv / max(self.n_ey, 1),
            rms_ey=np.sqrt(self.sum_ey2 / max(self.n_ey, 1)), max_ey=self.max_ey,
            sat_mag_frac=self.n_sat_mag / max(self.n_ctrl, 1), sat_rate_frac=self.n_sat_rate / max(self.step, 1),
            solver_failures=self.n_fail, solve_ms_median=float(np.median(st)) if len(st) else None,
            solve_ms_p99=float(np.percentile(st, 99)) if len(st) else None,
            solve_ms_max=float(st.max()) if len(st) else None,
            ellipse_max=self.ell_max, ellipse_over_frac=self.n_ell_over / max(self.step, 1),
            max_abs_ay=self.max_ay, max_brake=-self.min_ax,
        )
        if self.residual_logged:
            summ["mean_abs_res"] = self.sum_abs_res / max(self.n_ctrl, 1)
            summ["mean_abs_ares"] = self.sum_abs_ares / max(self.n_ctrl, 1)
        return summ


def run_lap(track: Track, scn: Scenario, K: float, cfg, log=True, steer_planner=None, residual=None):
    """``steer_planner`` (optional) replaces the kinematic MPC: a factory ``f(cfg, mu_scale)`` returning an
    object with ``solve(state, e_y, e_psi, s, delta_prev, vplan_at, ref) -> (delta, ok, solve_s)``.
    ``residual`` (optional) is a factory ``f()`` returning a callable ``g(sim: LapSim) -> delta_res [rad]``,
    called at every control instant after the MPC; the residual is added before the steering clip.
    Everything else (servo, limits, speed controller, validity checks, metrics) is unchanged."""
    res_fn = residual() if residual is not None else None
    sim = LapSim(track, scn, K, cfg, log=log, steer_planner=steer_planner, residual_logged=res_fn is not None)
    if res_fn is not None and hasattr(res_fn, "reset"):
        res_fn.reset()
    while not sim.done:
        sim.plan()
        r = res_fn(sim) if res_fn is not None else 0.0
        sim.apply(*r) if isinstance(r, tuple) else sim.apply(r)  # (d_delta, d_a) for a dual-channel policy
        sim.advance_interval()
    sim.close()
    return sim.summary(), pd.DataFrame(sim.rows)


def qualifies(summ, q):
    """Qualifying-lap rule (thresholds in config, fixed before the reported sweep)."""
    if summ["outcome"] != "completed":
        return False, summ["outcome"]
    if summ["off_track_steps"] > 0:
        return False, "off_track"
    if summ["solver_failures"] > q["max_solver_failures"]:
        return False, "solver_error"
    if summ["max_ey"] > q["max_abs_ey_m"]:
        return False, "lateral_error"
    if summ["lap_time_ratio"] > q["max_lap_time_ratio"]:
        return False, "speed_tolerance"
    return True, "qualifying"


# ---------------------------------------------------------------------------------------------- parallel runs
_W: dict = {}


def _init_worker(cfg):
    _W["cfg"], _W["track"] = cfg, Track(cfg)


def _run_job(job):
    scn, K, log, planner, *rest = job
    return run_lap(_W["track"], scn, K, _W["cfg"], log=log, steer_planner=planner, residual=rest[0] if rest else None)


def speed_chain(track, cfg, K, mu, grid, starts, planner=None, label="", residual=None):
    """Raise the speed level a_lat along ``grid`` until a lap does not qualify from every start pose.
    Returns a list of (a_lat, start_s, summary, trace) for every lap run (the failing level included)."""
    out = []
    for a_lat in grid:
        ok_all = True
        for s0 in starts:
            sm, tr = run_lap(track, Scenario(a_lat, mu, s0), K, cfg, steer_planner=planner, residual=residual)
            ok, reason = qualifies(sm, cfg["qualifying"])
            sm.update(qualifying=ok, status=reason, controller=label)
            out.append((a_lat, s0, sm, tr))
            ok_all &= ok
        if not ok_all:
            break
    return out


def _run_chain(job):
    K, mu, grid, starts, planner, label, *rest = job
    return speed_chain(_W["track"], _W["cfg"], K, mu, grid, starts, planner, label, rest[0] if rest else None)


def _workers(cfg, n_jobs):
    w = cfg.get("workers") or max(1, (os.cpu_count() or 2) - 2)
    return max(1, min(w, n_jobs))


def run_many(cfg, jobs):
    """jobs: list of (Scenario, K, log, steer_planner[, residual]). Returns [(summary, trace)] in order."""
    w = _workers(cfg, len(jobs))
    if w == 1:
        _init_worker(cfg)
        return [_run_job(j) for j in jobs]
    with ProcessPoolExecutor(w, initializer=_init_worker, initargs=(cfg,)) as ex:
        return list(ex.map(_run_job, jobs))


def run_chains(cfg, chains):
    """chains: list of (K, mu, grid, starts, steer_planner, label[, residual]). Returns one speed_chain result per chain."""
    w = _workers(cfg, len(chains))
    if w == 1:
        _init_worker(cfg)
        return [_run_chain(c) for c in chains]
    with ProcessPoolExecutor(w, initializer=_init_worker, initargs=(cfg,)) as ex:
        return list(ex.map(_run_chain, chains))
