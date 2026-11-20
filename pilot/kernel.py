"""Rust lap kernel (``pilot_core``) for ``pilot.closedloop.LapSim``.

With ``kernel: rust`` in the config, every physics step of a lap (servo, longitudinal controller, ST plant with
the configured tires, acceleration metrics, track-limit / wall / spin checks, Frenet projection, progress and
lap metrics) runs in Rust; Python crosses once per control interval (20 Hz) and keeps the MPC, the residual
policy, logging and the summary. The kernel mirrors the Python loop operation by operation and gives
bit-identical laps (checked against ``kernel: python``).

Supported: the ST plant (linear, Pacejka or Fiala tires), ``steer_mode: steering_speed``. The STD plant and the
simulator's own bang-bang servo (``steering_angle``) stay on the Python loop.
"""

from __future__ import annotations

import numpy as np
import shapely

from .sim import ST_TIRE_PLANT, agent, plant_model

KERNELS = ("python", "rust")


def unsupported(sim) -> str | None:
    """Why the Rust kernel cannot run this lap (None if it can)."""
    if plant_model() != "st":
        return f"plant '{plant_model()}' (only the ST plant is ported)"
    if sim.cfg["steer_mode"] != "steering_speed":
        return f"steer_mode '{sim.cfg['steer_mode']}' (only steering_speed is ported)"
    a = agent(sim.env)
    if a.model is not ST_TIRE_PLANT:
        return "the agent does not run the ST tire plant"
    if a.integrator.type != "rk4":
        return f"integrator '{a.integrator.type}' (only rk4 is ported)"
    if sim.servo.d != 2 or len(sim.servo.pending) != 2:
        return "steering servo delay other than 2 steps"
    if abs(a.time_step - sim.dt) > 0.0:
        return "simulator timestep differs from the lap dt"
    return None


def track_objects(track):
    """``pilot_core.Geometry`` and ``pilot_core.RefLine`` for a ``closedloop.Track``, built once per Track."""
    cached = getattr(track, "_pilot_core", None)
    if cached is None:
        import pilot_core

        g, ref = track.geo, track.ref
        rings = [shapely.get_coordinates(r) for p in shapely.get_parts(g.legal) for r in shapely.get_rings(p)]
        geo = pilot_core.Geometry(rings, shapely.bounds(g.wall_boxes), g.tire_offsets, g.patch_len, g.patch_w,
                                  g.length, g.width)
        refl = pilot_core.RefLine(ref.s, ref.x, ref.y, ref.psi, ref.length)
        cached = track._pilot_core = (geo, refl)
    return cached


def make_kernel(sim):
    """Kernel continuing ``sim`` (a LapSim right after its initial ``_pre``) from its exact current state."""
    import pilot_core

    reason = unsupported(sim)
    if reason is not None:
        raise ValueError(f"kernel: rust does not support {reason}; use kernel: python")
    geo, refl = track_objects(sim.track)
    a, env = agent(sim.env), sim.env
    f = float
    init = dict(
        state=np.asarray(a.state, dtype=float), st=np.asarray(sim.st, dtype=float),
        steer_buffer=[f(v) for v in a.steer_buffer], servo_pending=[f(v) for v in sim.servo.pending],
        plan_s=sim._plan_s, plan_v=sim._plan_v, plan_length=f(sim.ref.length),
        dt=f(sim.dt), n_sub=int(sim.n_sub), t_max=f(sim.t_max), kv=f(sim.kv), a_max=f(sim.a_max),
        ell_ax=f(sim.ell_ax), ell_ay=f(sim.ell_ay), sv_max=f(sim.sv_max),
        spin_beta_max=None if sim.spin_beta_max is None else f(sim.spin_beta_max),
        own_env=bool(sim._own_env), max_episode_steps=int(env.max_episode_steps), env_step=int(env.current_step),
        t=f(sim.t), step=int(sim.step), s=f(sim.s), e_y=f(sim.e_y), e_psi=f(sim.e_psi), v=f(sim.v), vp=f(sim.vp),
        i_g=int(sim.i_g), s_prev=f(sim.s_prev), progress=f(sim.progress), off_in_interval=int(sim.off_in_interval),
        tires_in=tuple(bool(t) for t in sim.tires_in), done=bool(sim.done),
        max_ey=f(sim.max_ey), sum_ey2=f(sim.sum_ey2), n_ey=int(sim.n_ey), sum_dv=f(sim.sum_dv),
        ell_max=f(sim.ell_max), n_ell_over=int(sim.n_ell_over), max_ay=f(sim.max_ay), min_ax=f(sim.min_ax),
        n_off=int(sim.n_off), first_off_s=sim.first_off_s, first_off_progress=sim.first_off_progress,
        n_sat_rate=int(sim.n_sat_rate),
    )
    return pilot_core.LapKernel(geo, refl, a.params, init)
