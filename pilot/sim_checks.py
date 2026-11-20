"""Step 1: simulator verification (step tests, mu replay, grip test, ST dynamics regression).

Writes results/sim_checks/{steer_step,long_step,mu_replay,grip_test}.csv, summary.json and
sim_checks.png. Grip test: ``a_y = v r`` (kept for continuity) overstates the lateral acceleration
whenever the sideslip changes (spin transients); ``a_perp = v (r + beta')`` is the acceleration normal to
the velocity, bounded by mu g for saturating tires.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gymkhana.envs.dynamic_models.single_track import vehicle_dynamics_st

from .sim import (NOMINAL_PARAMS, SteeringServo, agent, make_env, plant_label, plant_tire,
                  reset_to_state, set_plant, vehicle_params)
from .tires import st_force_dynamics


def _log(env, t, cmd_steer, cmd_acc):
    s = agent(env).state
    return dict(t=t, cmd_steer=cmd_steer, cmd_acc=cmd_acc, delta=s[2], v=s[3], yaw_rate=s[5], slip=s[6])


def steer_step(track, dt, out):
    rows = []
    x0 = np.array([0, 0, 0, 3.0, 0, 0, 0], float)
    for mode in ["steering_angle", "steering_speed"]:
        env = make_env(track, 1.0, mode, dt)
        start = env.track.centerline
        x0[0], x0[1], x0[4] = start.xs[0], start.ys[0], start.yaws[0]
        reset_to_state(env, x0)
        servo = SteeringServo(mode, dt, NOMINAL_PARAMS["sv_max"])
        for k in range(int(round(0.8 / dt))):
            t = k * dt
            target = 0.2 if t >= 0.1 else 0.0
            u = servo.command(target, agent(env).state[2])
            env.step(np.array([[u, 0.0]]))
            r = _log(env, t + dt, target, 0.0)
            r["mode"] = mode
            rows.append(r)
        env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / "steer_step.csv", index=False)
    return df


def long_step(track, dt, out):
    env = make_env(track, 1.0, "steering_speed", dt)
    c = env.track.centerline
    reset_to_state(env, [c.xs[0], c.ys[0], 0, 2.0, c.yaws[0], 0, 0])
    rows = []
    for k in range(int(round(1.5 / dt))):
        t = k * dt
        a = 3.0 if t < 0.75 else -3.0
        env.step(np.array([[0.0, a]]))
        rows.append(_log(env, t + dt, 0.0, a))
    env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / "long_step.csv", index=False)
    return df


def mu_replay(track, dt, out, mu_scales, v0=6.0, delta_target=0.08, T=1.5):
    """Same initial state, same open-loop steering-velocity sequence, different mu."""
    servo = SteeringServo("steering_speed", dt, NOMINAL_PARAMS["sv_max"])
    # build the open-loop sv sequence once (servo on a pure integrator, no plant feedback)
    seq, d = [], 0.0
    for k in range(int(T / dt)):
        tgt = delta_target if k * dt >= 0.1 else 0.0
        sv = servo.command(tgt, d)
        seq.append(sv)
        if k >= 2:
            d += dt * seq[k - 2]
    rows = []
    for mu in mu_scales:
        env = make_env(track, mu, "steering_speed", dt)
        # open straight-ish area: use a fixed pose far from walls is not needed, env boundary is ignored here
        c = env.track.centerline
        reset_to_state(env, [c.xs[0], c.ys[0], 0, v0, c.yaws[0], 0, 0])
        for k, sv in enumerate(seq):
            env.step(np.array([[sv, 0.0]]))
            r = _log(env, (k + 1) * dt, sv, 0.0)
            r["mu_scale"] = mu
            rows.append(r)
        env.close()
    df = pd.DataFrame(rows)
    L = NOMINAL_PARAMS["lf"] + NOMINAL_PARAMS["lr"]
    df["yaw_rate_kin"] = df.v * np.tan(df.delta) / L
    df.to_csv(out / "mu_replay.csv", index=False)
    return df


def grip_test(track, dt, out, mu_scales, v_hold=5.0, ramp_rate=0.1, delta_end=0.4):
    """Constant speed (P-held), slow steering ramp: peak quasi-steady lateral acceleration a_y = v * r."""
    rows = []
    for mu in mu_scales:
        env = make_env(track, mu, "steering_speed", dt)
        env._check_boundary_frenet = lambda idx: False
        c = env.track.centerline
        reset_to_state(env, [c.xs[0], c.ys[0], 0, v_hold, c.yaws[0], 0, 0])
        servo = SteeringServo("steering_speed", dt, NOMINAL_PARAMS["sv_max"])
        for k in range(int(delta_end / ramp_rate / dt)):
            st = agent(env).state
            beta_prev = float(st[6])
            u = servo.command(ramp_rate * k * dt, st[2])
            a = 3.0 * (v_hold - st[3])
            env.step(np.array([[u, a]]))
            st = agent(env).state
            rows.append(dict(mu_scale=mu, t=(k + 1) * dt, delta=st[2], v=st[3], yaw_rate=st[5], slip=st[6],
                             a_y=st[3] * st[5], a_perp=st[3] * (st[5] + (st[6] - beta_prev) / dt)))
        env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / "grip_test.csv", index=False)
    return df


def st_regression(n=2000, seed=0):
    """Axle-force form with the linear law vs gymkhana's ``vehicle_dynamics_st`` (random states and inputs,
    V >= 0.6 m/s, both mu levels). This is what makes the nonlinear laws a tire-only change."""
    rng = np.random.default_rng(seed)
    err = 0.0
    for mu in (1.0, 0.6):
        p = vehicle_params(mu, model="st")
        for _ in range(n):
            x = np.array([0.0, 0.0, rng.uniform(-0.4, 0.4), rng.uniform(0.6, 15.0), rng.uniform(0, 2 * np.pi),
                          rng.uniform(-5, 5), rng.uniform(-0.5, 0.5)])
            u = np.array([rng.uniform(-3.2, 3.2), rng.uniform(-9.5, 9.5)])
            err = max(err, float(np.abs(st_force_dynamics(x, u, p, "linear") - vehicle_dynamics_st(x, u, p)).max()))
    return dict(max_abs_derivative_diff=err, n_samples=2 * n)


def run(track: str, dt: float, out: pathlib.Path, mu_scales=(1.0, 0.6), model: str = "st", tire=None):
    set_plant(model, tire)
    out.mkdir(parents=True, exist_ok=True)
    ss = steer_step(track, dt, out)
    ls = long_step(track, dt, out)
    mr = mu_replay(track, dt, out, mu_scales)
    gt = grip_test(track, dt, out, mu_scales)

    summ = {}
    for mode, g in ss.groupby("mode"):
        late = g[g.t > 0.4]
        moving = g[(g.delta > 0.01) & (g.delta < 0.19)]
        summ[f"steer_{mode}"] = dict(
            first_motion_t=float(g[g.delta.abs() > 1e-6].t.min()),
            rate_rad_s=float(np.diff(moving.delta).mean() / dt) if len(moving) > 1 else None,
            final_delta=float(late.delta.mean()),
            chatter_pp_rad=float(late.delta.max() - late.delta.min()),
        )
    up = ls[ls.t <= 0.75]
    summ["long_accl"] = dict(
        measured_accel_up=float(np.polyfit(up.t, up.v, 1)[0]),
        measured_accel_down=float(np.polyfit(ls[ls.t > 0.8].t, ls[ls.t > 0.8].v, 1)[0]),
    )
    for mu, g in mr.groupby("mu_scale"):
        e = g[g.t > 1.2]
        summ[f"mu_{mu}"] = dict(
            yaw_rate_ss=float(e.yaw_rate.mean()),
            slip_ss=float(e.slip.mean()),
            kinematic_yaw_rate=float(e.yaw_rate_kin.mean()),
            yaw_rate_ratio_to_kinematic=float(e.yaw_rate.mean() / e.yaw_rate_kin.mean()),
            lat_acc_ss=float((e.yaw_rate * e.v).mean()),
        )
    a, b = mr[mr.mu_scale == mu_scales[0]], mr[mr.mu_scale == mu_scales[1]]
    summ["mu_effect"] = dict(
        max_abs_yaw_rate_diff=float(np.abs(a.yaw_rate.values - b.yaw_rate.values).max()),
        max_abs_slip_diff=float(np.abs(a.slip.values - b.slip.values).max()),
        max_abs_delta_diff=float(np.abs(a.delta.values - b.delta.values).max()),
    )
    summ["plant_model"] = model
    if model == "st":
        summ["tire"] = plant_tire()
        summ["st_linear_regression"] = st_regression()
    for mu, g in gt.groupby("mu_scale"):
        summ[f"grip_mu_{mu}"] = dict(max_abs_a_y=float(g.a_y.abs().max()),
                                     max_abs_a_y_g=float(g.a_y.abs().max() / 9.81),
                                     max_abs_a_perp=float(g.a_perp.abs().max()),
                                     max_abs_a_perp_g=float(g.a_perp.abs().max() / 9.81),
                                     mu_g=float(NOMINAL_PARAMS["mu"] * mu * 9.81),
                                     delta_at_max_a_y=float(g.delta.iloc[int(np.argmax(g.a_y.abs().values))]),
                                     max_abs_slip=float(g.slip.abs().max()))
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(2, 3, figsize=(14, 6.5))
    for mode, g in ss.groupby("mode"):
        ax[0, 0].plot(g.t, g.delta, label=f"actual δ ({mode})")
    ax[0, 0].plot(g.t, g.cmd_steer, "k--", lw=1, label="target δ")
    ax[0, 0].set(title="Steering step (v = 3 m/s)", xlabel="t [s]", ylabel="δ [rad]")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(ls.t, ls.v, label="actual v")
    ax[0, 1].plot(ls.t, 2 + np.cumsum(ls.cmd_acc) * dt, "k--", lw=1, label="∫ commanded a")
    ax[0, 1].set(title="Acceleration step (accl mode)", xlabel="t [s]", ylabel="v [m/s]")
    ax[0, 1].legend(fontsize=7)
    for mu, g in mr.groupby("mu_scale"):
        ax[1, 0].plot(g.t, g.yaw_rate, label=f"yaw rate, μ/μ₀={mu}")
        ax[1, 1].plot(g.t, np.degrees(g.slip), label=f"sideslip, μ/μ₀={mu}")
    ax[1, 0].plot(g.t, g.yaw_rate_kin, "k--", lw=1, label="kinematic v·tanδ/L")
    ax[1, 0].set(title="μ replay: same state & steering, v0 = 6 m/s", xlabel="t [s]", ylabel="r [rad/s]")
    ax[1, 1].set(title="μ replay: sideslip", xlabel="t [s]", ylabel="β [deg]")
    for i, (mu, g) in enumerate(gt.groupby("mu_scale")):
        col = f"C{i}"
        ax[0, 2].plot(g.delta, g.a_perp, color=col, label=f"v(r+β'), μ/μ₀={mu}")
        ax[0, 2].plot(g.delta, g.a_y, color=col, ls="--", lw=0.8, label=f"v·r, μ/μ₀={mu}")
        ax[0, 2].axhline(NOMINAL_PARAMS["mu"] * mu * 9.81, color=col, ls=":", lw=0.8)
        ax[1, 2].plot(g.delta, np.degrees(g.slip), color=col, label=f"μ/μ₀={mu}")
    ax[0, 2].set(title="Grip test: steering ramp at 5 m/s (dotted: μg)", xlabel="δ [rad]",
                 ylabel="lateral acceleration [m/s²]")
    ax[1, 2].set(title="Grip test: sideslip", xlabel="δ [rad]", ylabel="β [deg]")
    for a_ in [*ax[1], ax[0, 2]]:
        a_.legend(fontsize=7)
    fig.suptitle(f"Simulator checks, plant model = {plant_label().upper()}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sim_checks.png", dpi=130)
    plt.close(fig)
    return summ


if __name__ == "__main__":
    from . import cli

    cfg, _, out = cli.parse(__doc__)
    print(json.dumps(run(cfg["track"], cfg["dt"], out / "sim_checks", model=cfg.get("model", "st"), tire=cfg.get("tire")),
                     indent=2))
