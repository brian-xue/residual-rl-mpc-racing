"""Stage 0 of docs/history_mu_plan.md: offline friction probe on arm R's rollouts, before any training.

    uv run python results/analysis/history_mu/probe.py collect [n_eps_per_job]   # rollouts -> raw/probe_data.csv.gz
    uv run python results/analysis/history_mu/probe.py fit                       # probes -> probe/*.csv, gate verdict

Rollouts: the SELECTED checkpoint of each arm-R run (results_st_pacejka/slip_rl/R/s{0,1,2}, protocol_selected.txt),
stochastic actions (as in training), training-distribution episodes (ResidualRacingEnv, a_lat 3-9, mu/mu0 0.6-1.0)
on all four tracks. R was trained on Austin only, so the other three are unseen.

Per-frame signal sets (all non-privileged; none uses mu):
    F0  r, beta, r - v kappa, delta, a_cmd, v                     (the existing residual.obs_history signals)
    F1  F0 + realized lateral acceleration over the last interval, a_y = v_mean (r_mean + d beta / Ts)
    F2  F1 + SlipMPC one-step innovation: realized (beta, r, v) at instant k minus SlipMPC's nominal-mu0 model
        prediction from the state at k-1 with the command applied over that interval (mean realized steering angle,
        the k-1 solve's acceleration + the applied residual)
    All three are pilot.residual_env.ObsExtras frames (scaled as the policy sees them). v1 of this probe
    (probe_v1/) computed the innovation with the NEXT solve's acceleration instead; see docs/history_mu_plan.md.
Probe inputs: the current arm-R policy observation, plus N frames of a signal set and their first differences.
Models: ridge and a 2x64 MLP (as in results/analysis/followup/feasibility.py). Splits: leave-one-track-out
(train on three tracks, test on the fourth), Austin-only -> each other track, and Austin episode-grouped CV.

Gate (fixed in the plan): the smallest set and window whose leave-one-track-out high-excitation (|v r| > 4) R^2 is
>= 0.3 on at least 3 of 4 held-out tracks, preferring F0 < F1 < F2 and N 5 < 10 when within 0.05 (mean over tracks).
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "followup"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

RUNS = ROOT / "results_st_pacejka/slip_rl/R"
TRACKS = ["Austin", "Catalunya", "Silverstone", "Spielberg"]
RAW = HERE / "raw"
OUT = HERE / "probe"
SEEDS = (0, 1, 2)
F0 = ["r", "beta", "r_err", "delta", "a_cmd", "v"]
F1 = F0 + ["a_y"]
F2 = F1 + ["inn_beta", "inn_r", "inn_v"]
SETS = {"F0": F0, "F1": F1, "F2": F2}


def selected_ckpt(seed: int) -> pathlib.Path:
    rd = RUNS / f"s{seed}"
    return rd / (rd / "protocol_selected.txt").read_text().strip()


def run_cfg(seed: int, track: str) -> dict:
    cfg = copy.deepcopy(yaml.safe_load(open(RUNS / f"s{seed}" / "train_config.yaml")))
    cfg["track"] = track
    cfg["residual"].pop("tracks", None)
    return cfg


# ------------------------------------------------------------------------------------------- rollouts
def _collect(job):
    """Rollouts with arm R's policy. The F0-F2 signals are the environment's own history frames
    (pilot.residual_env.ObsExtras with obs_history_features f2, one frame), i.e. exactly what an R-H policy sees at
    the same control instant: the frame at row t is computed at the instant of the observation the action uses."""
    seed, track, chunk, n_ep = job
    import torch
    from stable_baselines3 import PPO

    from pilot.residual_env import ObsExtras, residual_opts, ResidualRacingEnv

    torch.set_num_threads(1)
    cfg = run_cfg(seed, track)
    env = ResidualRacingEnv(cfg, seed=10_000 * (seed + 1) + 100 * TRACKS.index(track) + chunk)
    model = PPO.load(str(selected_ckpt(seed)), device="cpu")
    frames = ObsExtras(residual_opts(dict(obs_history=1, obs_history_features="f2")))
    rows = []
    for e in range(n_ep):
        obs, info = env.reset()
        frames.reset()
        sim = env.sim
        mu, a_lat = info["mu_scale"], info["a_lat"]
        ep = f"{seed}-{track}-{chunk}-{e}"
        f = frames(sim)
        for t in range(cfg["residual"]["episode_steps"]):
            v, r = float(sim.st[3]), float(sim.st[5])
            act, _ = model.predict(obs, deterministic=False)
            rows.append((seed, track, ep, t, mu, a_lat, v, r, *f, *obs))
            obs, _, te, tu, _ = env.step(act)
            if te or tu:
                break
            f = frames(sim)
    return rows


def collect(n_ep=12):
    RAW.mkdir(exist_ok=True)
    p = RAW / "probe_data.csv.gz"
    jobs = [(s, tr, c, n_ep) for s in SEEDS for tr in TRACKS for c in range(6)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(8) as ex:
        rows = [r for part in ex.map(_collect, jobs) for r in part]
    n_obs = len(rows[0]) - 8 - len(F2)
    cols = ["seed", "track", "episode", "t", "mu", "a_lat", "v_raw", "r_raw"] + F2 + [f"o{i}" for i in range(n_obs)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(p, index=False, float_format="%.6g")
    print(f"{len(df)} steps, {df.episode.nunique()} episodes in {time.perf_counter() - t0:.0f} s -> {p}")
    return df


# ------------------------------------------------------------------------------------------- probes
def features(df, sig, k):
    """Current policy observation + k frames (lags 0..k-1) of the signal set and their first differences.
    Rows with fewer than k frames in their episode are dropped."""
    obs = [c for c in df.columns if c.startswith("o") and c[1:].isdigit()]
    out = [df[obs].values]
    keep = np.ones(len(df), bool)
    if k > 0:
        g = df.groupby("episode", sort=False)
        cur = df[sig].values
        out.append(cur)
        prev = cur
        for lag in range(1, k):
            sh = g[sig].shift(lag).values
            out += [sh, prev - sh]
            prev = sh
        keep = (df.t >= k - 1).values
    X = np.concatenate(out, axis=1)
    return X[keep], keep


def fit_all(torch_threads=8):
    import torch

    from feasibility import mlp_fit, ridge_fit

    torch.set_num_threads(torch_threads)
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(RAW / "probe_data.csv.gz")
    vr = (df.v_raw * df.r_raw).abs()
    df["exc"] = np.where(vr < 2.0, "low", np.where(vr > 4.0, "high", "mid"))
    rng = np.random.default_rng(0)
    variants = [("obs", 0)] + [(s, k) for s in SETS for k in (5, 10)]
    rows = []
    for name, k in variants:
        X, keep = features(df, SETS.get(name, []), k)
        d = df[keep].reset_index(drop=True)
        y = d.mu.values
        trk = d.track.values
        splits = []  # (split, test track, train mask, test mask)
        for tr in TRACKS:
            splits.append(("loto", tr, trk != tr, trk == tr))
        for tr in TRACKS[1:]:
            splits.append(("austin_only", tr, trk == "Austin", trk == tr))
        aus = trk == "Austin"
        eps = np.asarray(d.episode[aus].unique(), dtype=object)
        rng.shuffle(eps)
        for i, f in enumerate(np.array_split(eps, 5)):
            te = aus & d.episode.isin(f).values
            splits.append(("austin_cv", f"fold{i}", aus & ~te, te))
        for model_name, fit in (("ridge", ridge_fit), ("mlp", mlp_fit)):
            t0 = time.perf_counter()
            cv_pred = np.full(len(d), np.nan)
            for split, test, trm, tem in splits:
                pr = fit(X[trm], y[trm])(X[tem])
                if split == "austin_cv":
                    cv_pred[tem] = pr
                    continue
                full = np.full(len(d), np.nan)
                full[tem] = pr
                rows += _score(name, k, model_name, split, test, full, tem, y, d, y[trm].mean())
            rows += _score(name, k, model_name, "austin_cv", "Austin", cv_pred, aus, y, d, y[aus].mean())
            print(f"{name} N={k} {model_name}: {time.perf_counter() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "probe_scores.csv", index=False, float_format="%.4f")
    return t


def _score(name, k, model, split, test, pred, mask, y, d, const):
    out = []
    for exc in ("all", "low", "mid", "high"):
        m = mask & ((d.exc == exc).values if exc != "all" else True)
        if m.sum() < 50:
            continue
        e = pred[m] - y[m]
        out.append(dict(set=name, N=k, model=model, split=split, test=test, excitation=exc, n=int(m.sum()),
                        rmse=float(np.sqrt(np.mean(e**2))), rmse_const=float(np.sqrt(np.mean((const - y[m]) ** 2))),
                        r2=float(1 - np.mean(e**2) / np.var(y[m]))))
    return out


def gate(t: pd.DataFrame) -> dict:
    """Apply the plan's gate to the leave-one-track-out, high-excitation R^2 (better of ridge / MLP per variant)."""
    h = t[(t.split == "loto") & (t.excitation == "high") & (t.set != "obs")]
    best = h.groupby(["set", "N", "test"]).r2.max().unstack("test")
    best["n_pass"] = (best[TRACKS] >= 0.3).sum(axis=1)
    best["mean_r2"] = best[TRACKS].mean(axis=1)
    order = [(s, k) for s in SETS for k in (5, 10)]  # preference: smallest set, then shortest window
    passing = [v for v in order if best.loc[v, "n_pass"] >= 3]
    verdict = dict(table=best.round(3).reset_index().to_dict("records"))
    if not passing:
        verdict["choice"] = None
        verdict["verdict"] = "STOP: no signal set reaches R^2 >= 0.3 on 3 of 4 held-out tracks"
        return verdict
    top = max(best.loc[v, "mean_r2"] for v in passing)
    choice = next(v for v in passing if best.loc[v, "mean_r2"] >= top - 0.05)
    verdict["choice"] = dict(set=choice[0], N=int(choice[1]))
    verdict["verdict"] = f"PASS: {choice[0]}, N = {choice[1]}"
    return verdict


def fit():
    t = fit_all()
    v = gate(t)
    (OUT / "gate.json").write_text(json.dumps(v, indent=1))
    pd.set_option("display.width", 250)
    for split in ("loto", "austin_only", "austin_cv"):
        s = t[t.split == split]
        print(f"\n== {split}: R^2 (rows: model, excitation; cols: set, N, test track)")
        print(s.pivot_table(index=["model", "excitation"], columns=["set", "N", "test"], values="r2").round(2)
              .to_string())
    print("\n", pd.DataFrame(v["table"]).to_string())
    print(v["verdict"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(*(int(a) for a in sys.argv[2:]))
    else:
        fit()
