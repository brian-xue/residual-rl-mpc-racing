"""Check that the Rust lap kernel (``kernel: rust``) reproduces the Python loop (``kernel: python``) exactly.

Runs a few laps (linear and Pacejka ST, kinematic and coupled MPC, a completed lap, a crash, a spin) and a
residual-env rollout with both kernels and compares every logged trace column and summary entry bit for bit
(solver timings excepted). Run it after any change to ``LapSim._physics`` / ``_pre`` or to ``pilot_core``.

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
from .residual_env import ResidualRacingEnv

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


def main():
    ok = True
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
