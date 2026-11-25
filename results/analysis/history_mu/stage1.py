"""Stage 1 of docs/history_mu_plan.md: checks before training, then the R-H smoke run.

    uv run python results/analysis/history_mu/stage1.py checks    # passivity, f0 unchanged, history invariants, cost
    uv run python results/analysis/history_mu/stage1.py smoke     # 300k-step R-H run, seed 0 (as arm R's Stage 1)
    uv run python results/analysis/history_mu/stage1.py summary   # R-H learning curve against arm R's smoke run

Checks (all must pass before any training):
  1. passive: R, R-H and R-H-zero envs, same seed and the same action sequence, give identical simulator states, and
     the first 38 observation entries (arm R's observation) are identical. The history only adds observation terms.
  2. f0 unchanged: arm P' (obs_history 10, default f0) gives the same history frames as the pre-change formula.
  3. history invariants, in the env (training) and in run_lap + ResidualPolicy.from_rc (evaluation):
     past:   frames 0..3 of obs_t == frames 1..4 of obs_{t-1} (one frame per control instant, oldest first);
     repeat: all 5 frames == the newest one;
     the first instant of every episode / lap has zero innovation (no previous state).
  4. step cost of the R-H env against the R env (SlipMPC dominates; the innovation adds one model prediction).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "stage1"
CFG = {"R": "configs/pilot_st_pacejka_slip_res.yaml", "RH": "configs/pilot_st_pacejka_slip_res_hist.yaml",
       "RH0": "configs/pilot_st_pacejka_slip_res_hist_zero.yaml", "P": "configs/pilot_st_pacejka_slip_speed.yaml"}
SMOKE = ROOT / "results_st_pacejka/history_mu_smoke/RH"
R_SMOKE = ROOT / "results_st_pacejka/slip_smoke/R/train_log.csv"
N_R_OBS = 38
FD = 10  # f2 frame size
NH = 5


def load(k):
    return yaml.safe_load(open(ROOT / CFG[k]))


def frames(o):
    return np.asarray(o[N_R_OBS:]).reshape(NH, FD)


def rollout(key, seed=7, n_ep=3, act_seed=1):
    from pilot.residual_env import ResidualRacingEnv

    env = ResidualRacingEnv(load(key), seed=seed)
    rng = np.random.default_rng(act_seed)
    eps = []
    for _ in range(n_ep):
        o, info = env.reset()
        obs, sts = [o], [env.sim.st.copy()]
        for _ in range(250):
            a = np.clip(rng.normal(0, 0.3, env.action_space.shape), -1, 1).astype(np.float32)
            o, _, te, tu, _ = env.step(a)
            obs.append(o)
            sts.append(env.sim.st.copy())
            if te or tu:
                break
        eps.append((np.array(obs), np.array(sts), info))
    return eps


def check_history(obs_seq, mode):
    """obs_seq: observations of one episode / lap in order. Returns (ok, message)."""
    f = [frames(o) for o in obs_seq]
    if np.any(f[0][-1][7:] != 0):
        return False, "first frame has a nonzero innovation"
    for t in range(1, len(f)):
        if mode == "past" and not np.array_equal(f[t][:-1], f[t - 1][1:]):
            return False, f"past: frames not shifted at t={t}"
        if mode == "repeat" and not all(np.array_equal(f[t][i], f[t][-1]) for i in range(NH)):
            return False, f"repeat: frames differ at t={t}"
    return True, f"{len(f)} instants"


def checks(_args):
    OUT.mkdir(exist_ok=True)
    res = {}
    # 1. passive
    base = rollout("R")
    for key in ("RH", "RH0"):
        other = rollout(key)
        same_st = all(np.array_equal(a[1], b[1]) for a, b in zip(base, other))
        same_obs = all(np.array_equal(a[0], b[0][:, :N_R_OBS]) for a, b in zip(base, other))
        res[f"1 passive {key}"] = dict(ok=bool(same_st and same_obs), states=same_st, r_obs=same_obs,
                                       steps=int(sum(len(e[1]) for e in other)))
        # 3. history invariants, env
        mode = "past" if key == "RH" else "repeat"
        # a wall-collision terminal observation is all zeros by design (ResidualRacingEnv.step): leave it out
        out = [check_history(e[0] if e[0][-1].any() else e[0][:-1], mode) for e in other]
        res[f"3 history env {key}"] = dict(ok=all(o[0] for o in out), detail=[o[1] for o in out])
    # 2. f0 unchanged (arm P', obs_history 10): recompute the newest frame with the pre-change formula
    from pilot.residual_env import ResidualRacingEnv

    pc = load("P")
    env = ResidualRacingEnv(pc, seed=3)
    rng = np.random.default_rng(2)
    n_bad, n = 0, 0
    for _ in range(2):
        o, _ = env.reset()
        for _ in range(250):
            sim = env.sim
            _, _, delta, v, _, r, beta = sim.st[:7]
            a_cmd = (sim.a_mpc or 0.0) + sim.a_res
            old = np.clip([r / 3.0, beta / 0.2, (r - v * sim.ref.kappa[sim.i_g]) / 1.0, delta / sim.s_max, a_cmd / 4.0,
                           v / 8.0], -5, 5).astype(np.float32)
            n_bad += not np.array_equal(o[-6:], old)
            n += 1
            o, _, te, tu, _ = env.step(np.clip(rng.normal(0, 0.3, 1), -1, 1).astype(np.float32))
            if te or tu:
                break
    res["2 f0 unchanged (P')"] = dict(ok=n_bad == 0, instants=n, mismatches=n_bad)
    # 3. history invariants, evaluation path (untrained policies with the R-H / R-H-zero observation)
    res.update(eval_path_checks())
    # 4. step cost
    res["4 step cost"] = step_cost()
    for k, v in res.items():
        print(("PASS " if v.get("ok", True) else "FAIL ") + k, {kk: vv for kk, vv in v.items() if kk != "ok"})
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    kc = subprocess.run([sys.executable, "-m", "pilot.kernel_check"], cwd=ROOT, capture_output=True, text=True)
    print("kernel_check:", "PASS" if kc.returncode == 0 else "FAIL", kc.stdout.strip().splitlines()[-1:])
    return all(v.get("ok", True) for v in res.values()) and kc.returncode == 0


def eval_path_checks():
    import tempfile

    from stable_baselines3 import PPO

    from pilot.closedloop import Scenario, Track, run_lap
    from pilot.residual_env import ResidualPolicy, ResidualRacingEnv, _MODELS

    out = {}
    for key, mode in (("RH", "past"), ("RH0", "repeat")):
        cfg = load(key)
        env = ResidualRacingEnv(cfg, seed=0)
        model = PPO("MlpPolicy", env, seed=0, device="cpu", policy_kwargs=dict(log_std_init=-1.5))
        path = pathlib.Path(tempfile.mkdtemp()) / f"{key}.zip"
        model.save(path)
        pol = ResidualPolicy.from_rc(str(path), cfg["residual"])
        pol()  # loads into the per-process cache
        m = _MODELS[str(path)]
        seen, predict = [], m.predict
        m.predict = lambda o, deterministic=True: (seen.append(np.array(o)), predict(o, deterministic=deterministic))[1]
        track = Track(cfg)
        laps = []
        for scn in (Scenario(a_lat=6.0, mu_scale=0.7, start_s=0.0), Scenario(a_lat=5.0, mu_scale=1.0, start_s=100.0)):
            seen.clear()
            summ, _ = run_lap(track, scn, 0.0, cfg, log=False, residual=pol)
            ok, msg = check_history(list(seen), mode)
            laps.append(dict(ok=ok, msg=msg, outcome=summ["outcome"]))
        m.predict = predict
        out[f"3 history eval {key}"] = dict(ok=all(l["ok"] for l in laps), laps=laps)
    return out


def step_cost(n=1500):
    from pilot.residual_env import ResidualRacingEnv

    out = {}
    for key in ("R", "RH"):
        env = ResidualRacingEnv(load(key), seed=11)
        rng = np.random.default_rng(0)
        env.reset()
        t0, k = time.perf_counter(), 0
        while k < n:
            _, _, te, tu, _ = env.step(np.clip(rng.normal(0, 0.2, 2), -1, 1).astype(np.float32))
            k += 1
            if te or tu:
                env.reset()
        out[f"{key}_us_per_step"] = round((time.perf_counter() - t0) / n * 1e6)
    return out


def smoke(args):
    SMOKE.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pilot.train_residual", "--config", CFG["RH"], "--steps", str(args.steps),
           "--seed", "0", "--out", str(SMOKE)]
    with open(SMOKE.parent / "RH.train.log", "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def window_stats(df, lo, hi):
    d = df[(df.steps > lo) & (df.steps <= hi)]
    w = d.episodes
    return dict(crash_rate=float((d.crash_rate * w).sum() / w.sum()), ep_return=float((d.ep_return * w).sum() / w.sum()))


def summary(_args):
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, f in (("R (slip_rl Stage 1)", R_SMOKE), ("R-H", SMOKE / "train_log.csv")):
        df = pd.read_csv(f)
        for lab, lo, hi in (("first 50k", 0, 50_000), ("100-200k", 100_000, 200_000), ("last 100k", 200_000, 300_000)):
            rows.append(dict(variant=name, window=lab, **window_stats(df, lo, hi)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "smoke_summary.csv", index=False, float_format="%.3f")
    print(t.pivot(index="variant", columns="window", values=["crash_rate", "ep_return"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "smoke", "summary"])
    ap.add_argument("--steps", type=int, default=300_000)
    a = ap.parse_args()
    if a.cmd == "checks":
        sys.exit(0 if checks(a) else 1)
    {"smoke": smoke, "summary": summary}[a.cmd](a)
