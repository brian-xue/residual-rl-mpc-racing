"""SlipMPC + RL, Stage 1 (docs/slip_rl_plan.md): 300k-step smoke runs before the 3-seed runs.

    uv run python results/analysis/slip_rl/stage1.py zero      # S-E reference on the TRAINING distribution (zero action)
    uv run python results/analysis/slip_rl/stage1.py smoke     # train every variant for --steps (default 300k)
    uv run python results/analysis/slip_rl/stage1.py summary   # learning curves vs the zero-action reference

Variants (seed 0, Austin, published PPO settings):
    R       configs/pilot_st_pacejka_slip_res.yaml as is (r_max 0.2 rad, accel_r_max 2.0 m/s^2)
    R-low   the same with half the authority (r_max 0.1, accel_r_max 1.0)
    P       configs/pilot_st_pacejka_slip_speed.yaml (arm P': speed-reference scale in [0.85, 1.05])

zero: every variant's policy starts at mean action 0, i.e. S-E plus exploration noise, so the question for a smoke
run is whether training moves crash rate and episode return away from S-E's own values on the same episodes.
The reference runs ResidualRacingEnv with action 0 (identical for all variants: checked in Stage 1).
Episodes in train_log.csv are training episodes WITH exploration noise; the reference has none.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "stage1"
RUNS = ROOT / "results_st_pacejka/slip_smoke"
R_CFG = ROOT / "configs/pilot_st_pacejka_slip_res.yaml"
P_CFG = ROOT / "configs/pilot_st_pacejka_slip_speed.yaml"


def variants():
    r = yaml.safe_load(open(R_CFG))
    lo = copy.deepcopy(r)
    lo["residual"].update(r_max=0.1, accel_r_max=1.0)
    return {"R": r, "R-low": lo, "P": yaml.safe_load(open(P_CFG))}


def _zero_eps(args):
    seed, n_ep = args
    from pilot.residual_env import ResidualRacingEnv

    cfg = yaml.safe_load(open(R_CFG))
    env = ResidualRacingEnv(cfg, seed=seed)
    out = []
    for _ in range(n_ep):
        info0 = env.reset()[1]
        ret, n = 0.0, 0
        while True:
            _, r, te, tu, info = env.step(np.zeros(2, np.float32))
            ret, n = ret + r, n + 1
            if te or tu:
                break
        out.append(dict(seed=seed, a_lat=info0["a_lat"], mu_scale=info0["mu_scale"], ret=ret, len=n,
                        outcome=info["outcome"], crash=info["outcome"] in ("wall_collision", "spin", "lateral_error")))
    return out


def zero(args):
    f = OUT / "zero_episodes.csv"
    if f.exists() and not args.redo:
        df = pd.read_csv(f)
    else:
        jobs = [(10_000 + i, args.episodes // 8) for i in range(8)]
        with ProcessPoolExecutor(8) as ex:
            df = pd.DataFrame([r for rs in ex.map(_zero_eps, jobs) for r in rs])
        df.to_csv(f, index=False, float_format="%.4f")
    summ = dict(episodes=len(df), crash_rate=float(df.crash.mean()), ep_return=float(df.ret.mean()),
                ep_len=float(df.len.mean()), outcomes=df.outcome.value_counts().to_dict(),
                crash_rate_by_mu=df.groupby(pd.cut(df.mu_scale, [0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True),
                                            observed=True).crash.mean().round(3).rename(index=str).to_dict())
    (OUT / "zero_reference.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


def smoke(args):
    for name, cfg in variants().items():
        if args.only and name not in args.only:
            continue
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        cfg["residual"]["out"] = str(out.relative_to(ROOT))
        cfg_path = out / "smoke_config.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        print(f"== {name}", flush=True)
        with open(out / "train_stdout.log", "w") as log:
            subprocess.run([sys.executable, "-m", "pilot.train_residual", "--config", str(cfg_path),
                            "--steps", str(args.steps)], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def summary(args):
    ref = json.loads((OUT / "zero_reference.json").read_text())
    rows = []
    for name in variants():
        f = RUNS / name / "train_log.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        # episode-weighted means over windows of training
        for lab, m in (("first 50k", d.steps <= 50_000), ("100-200k", (d.steps > 100_000) & (d.steps <= 200_000)),
                       ("last 100k", d.steps > d.steps.max() - 100_000)):
            w = d[m]
            rows.append(dict(variant=name, window=lab, episodes=int(w.episodes.sum()),
                             crash_rate=float(np.average(w.crash_rate, weights=w.episodes)),
                             ep_return=float(np.average(w.ep_return, weights=w.episodes)),
                             ep_len=float(np.average(w.ep_len, weights=w.episodes)), std=float(w["std"].iloc[-1]),
                             wall_min=float(d.wall_s.max() / 60)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "smoke_summary.csv", index=False, float_format="%.4f")
    print(f"zero-action S-E reference: crash_rate {ref['crash_rate']:.3f}, ep_return {ref['ep_return']:.1f}, "
          f"ep_len {ref['ep_len']:.1f} ({ref['episodes']} episodes)")
    print(tab.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["zero", "smoke", "summary"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--episodes", type=int, default=240, help="zero: reference episodes (8 workers)")
    ap.add_argument("--only", nargs="*", help="smoke: subset of variants")
    ap.add_argument("--redo", action="store_true", help="zero: rerun the episodes even if the CSV exists")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"zero": zero, "smoke": smoke, "summary": summary}[args.what](args)
