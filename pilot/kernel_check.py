"""Check that the Rust lap kernel (``kernel: rust``) reproduces the Python loop (``kernel: python``) exactly.

1. MPC discretisation: ``KinematicMPC``'s closed-form ZOH (shared by both kernels) against ``scipy.linalg.expm``
   of the full augmented matrix M built from every Jacobian column of the model ``KinematicMPC._f``, that
   M^3 = 0 exactly (the closed form's assumption), and that the guard rejects a model breaking it. (The Rust
   side of the guard is covered by ``cargo test`` in ``pilot_core``.)
2. Runs a few laps (linear and Pacejka ST, kinematic and coupled MPC, a completed lap, a crash, a spin) and a
   residual-env rollout with both kernels and compares every logged trace column and summary entry bit for bit
   (solver timings excepted).

Run it after any change to ``LapSim._physics`` / ``_pre``, to the MPC model, or to ``pilot_core``.

Usage:  uv run python -m pilot.kernel_check
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .closedloop import Scenario, Track, run_lap
from .mpc import KinematicMPC
from .residual_env import ResidualRacingEnv
from .sim import NOMINAL_PARAMS

CASES = [  # (config, longitudinal mode override, a_lat, mu_scale, K)
    ("configs/pilot.yaml", None, 3.0, 1.0, 0.0),
    ("configs/pilot.yaml", None, 4.0, 0.6, 0.0),
    ("configs/pilot_st_pacejka.yaml", None, 2.0, 1.0, 0.0),
    ("configs/pilot_st_pacejka.yaml", None, 3.0, 1.0, 0.0),
    ("configs/pilot.yaml", "mpc", 3.0, 1.0, 0.0),
]


def _lap(cfg, kernel, a_lat, mu, K):
    cfg = copy.deepcopy(cfg)
    cfg["kernel"] = kernel
    sm, tr = run_lap(Track(cfg), Scenario(a_lat, mu, 0.0), K, cfg, log=True)
    return sm, pd.DataFrame(tr)


def _rollout(cfg, kernel, n=400, seed=3):
    cfg = copy.deepcopy(cfg)
    cfg["kernel"] = kernel
    env, rng = ResidualRacingEnv(cfg, seed=seed), np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for _ in range(n):
        obs, rew, term, trunc, _ = env.step(rng.uniform(-1, 1, 1).astype(np.float32))
        out.append(np.r_[obs, rew, term, trunc])
        if term or trunc:
            out.append(env.reset()[0])
    return out


def _same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _mpc(cls, cfg):
    mc, lim = cfg["mpc"], cfg["limits"]
    return cls(NOMINAL_PARAMS["lf"], NOMINAL_PARAMS["lr"], mc["Ts"], mc["N"], mc["q_y"], mc["q_psi"], mc["r_du"],
               mc["q_terminal"], lim["s_max"], lim["sv_max"] * mc["Ts"])


def check_discretization(cfg, n=500, seed=0, tol=1e-12):
    """Closed-form ZOH vs expm of the full M, M^3 = 0, and the guard. Returns True if all hold."""
    from scipy.linalg import expm

    m, rng, h = _mpc(KinematicMPC, cfg), np.random.default_rng(seed), 1e-6
    worst, nilpotent = 0.0, True
    for _ in range(n):
        v = rng.uniform(0.5, 15.0, m.N)
        kappa = rng.uniform(-1.5, 1.5, m.N) * rng.choice([1.0, 0.01], m.N)
        Ad, Bd, cd, e1 = m._discretize_many(v, kappa)
        d_r = np.arctan(m.L * kappa)
        f0 = m._f(0.0, e1, d_r, v, kappa)
        cols = (m._f(h, e1, d_r, v, kappa), m._f(0.0, e1 + h, d_r, v, kappa), m._f(0.0, e1, d_r + h, v, kappa))
        M = np.zeros((m.N, 4, 4))  # every column, including the ones the closed form assumes to be zero
        for i in range(2):
            for j, c in enumerate(cols):
                M[:, i, j] = (c[i] - f0[i]) / h
            M[:, i, 3] = f0[i] - M[:, i, 1] * e1 - M[:, i, 2] * d_r
        Mt = M * m.Ts
        nilpotent &= bool(np.all(Mt @ Mt @ Mt == 0.0))
        E = expm(Mt)
        got = np.concatenate([Ad[:, :2, :2].reshape(m.N, -1), Ad[:, :2, 2], Bd[:, :2], cd[:, :2]], axis=1)
        want = np.concatenate([E[:, :2, :2].reshape(m.N, -1), E[:, :2, 2], E[:, :2, 2], E[:, :2, 3]], axis=1)
        worst = max(worst, float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0))))

    class HeadingDamped(KinematicMPC):  # e_psi' depending on e_psi breaks M^3 = 0
        def _f(self, ey, epsi, d, v, kappa):
            a, b = super()._f(ey, epsi, d, v, kappa)
            return a, b - 0.5 * epsi

    try:
        _mpc(HeadingDamped, cfg)._discretize_many(np.full(m.N, 5.0), np.full(m.N, 0.1))
        guard = False
    except ValueError:
        guard = True
    ok = nilpotent and worst <= tol and guard
    print(f"MPC discretisation ({n} horizons): M^3 = 0 {'yes' if nilpotent else 'NO'}, closed form vs expm "
          f"max rel diff {worst:.1e} (tol {tol:g}), guard on a non-nilpotent model {'raises' if guard else 'MISSING'}"
          f" -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ok = check_discretization(yaml.safe_load(open("configs/pilot.yaml")))
    for path, mode, a_lat, mu, K in CASES:
        cfg = yaml.safe_load(open(path))
        if mode:
            cfg["longitudinal"]["mode"] = mode
        t0 = time.perf_counter()
        sp, tp = _lap(cfg, "python", a_lat, mu, K)
        t1 = time.perf_counter()
        sr, trr = _lap(cfg, "rust", a_lat, mu, K)
        t2 = time.perf_counter()
        cols = [c for c in tp.columns if c != "solve_ms" and tp[c].dtype != object]
        trace_ok = list(tp.columns) == list(trr.columns) and _same(tp[cols].values, trr[cols].values)
        keys = [k for k in sp if "solve" not in k]
        summ_ok = all(sp[k] == sr[k] or (isinstance(sp[k], float) and np.isnan(sp[k]) and np.isnan(sr[k]))
                      for k in keys)
        ok &= trace_ok and summ_ok
        print(f"{path.split('/')[-1]:24s} mode={mode or cfg['longitudinal'].get('mode', 'pfb'):4s} a_lat={a_lat} "
              f"mu={mu}: {sp['outcome']:15s} trace {'identical' if trace_ok else 'DIFFERS'}, summary "
              f"{'identical' if summ_ok else 'DIFFERS'} | python {t1 - t0:.2f}s rust {t2 - t1:.2f}s")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    a, b = _rollout(cfg, "python"), _rollout(cfg, "rust")
    roll_ok = len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    ok &= roll_ok
    print(f"residual env rollout ({len(a)} records): {'identical' if roll_ok else 'DIFFERS'}")
    print("OK: kernels identical" if ok else "FAIL: kernels differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
