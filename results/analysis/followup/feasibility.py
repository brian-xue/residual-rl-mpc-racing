"""Part 3: two feasibility checks (no training).

    uv run python results/analysis/followup/feasibility.py slipbase   # SlipMPC as an RL base: recoverable failures, step cost
    uv run python results/analysis/followup/feasibility.py probe      # offline friction probe from observation history

slipbase: S-E vs MPC + RL (δ, a) + wall (ckpt 1.25M) on the same friction-ellipse plan and the same 72 sweep cells
(results/analysis/data/r1_summary.csv, results/analysis/behavior/failure_classification.csv); env.step cost of
ResidualRacingEnv with the kinematic MPC vs with SlipMPC as the base controller (config changed in memory only).

probe: frozen 1.25M wall policy, stochastic actions, training-distribution episodes (ResidualRacingEnv) on Austin
(train) and Spielberg (held-out track). Targets: simulator mu/mu0. Inputs: the policy observation (+ action), no
mu-revealing quantity (the observation contains none). Ridge regression and a small MLP, episode-grouped CV.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

RAW = HERE / "raw"
OUT = HERE / "feasibility"
OUT.mkdir(exist_ok=True)
pc = time.perf_counter


# ------------------------------------------------------------------------------------------- SlipMPC base
def env_step_cost(mode, n=3000, seed=3):
    from pilot.residual_env import ResidualRacingEnv

    cfg = R.load_cfg("wall")
    cfg["longitudinal"]["mode"] = mode  # "pfb" (as trained) or "slip" (SlipMPC base); in memory only
    env = ResidualRacingEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    for _ in range(300):
        _, _, te, tu, _ = env.step(rng.normal(0, 0.075, 2))
        if te or tu:
            env.reset()
    st, rs, cpu0 = [], [], time.process_time()
    t_all = pc()
    for _ in range(n):
        t0 = pc()
        _, _, te, tu, info = env.step(rng.normal(0, 0.075, 2))  # small exploration-like actions
        st.append(pc() - t0)
        if te or tu:
            t0 = pc()
            env.reset()
            rs.append(pc() - t0)
    wall = pc() - t_all
    return dict(mode=mode, n=n, step_us_median=float(np.median(st) * 1e6), step_us_p95=float(np.percentile(st, 95) * 1e6),
                step_us_p99=float(np.percentile(st, 99) * 1e6), step_us_mean=float(np.mean(st) * 1e6),
                reset_us_median=float(np.median(rs) * 1e6) if rs else None, n_resets=len(rs),
                wall_per_step_us=wall / n * 1e6, cpu_per_step_us=(time.process_time() - cpu0) / n * 1e6)


def slipbase():
    sm = pd.read_csv(R.OUT / "data/r1_summary.csv")
    fc = pd.read_csv(R.OUT / "behavior/failure_classification.csv")
    a = sm[sm.ctrl == "wall"].set_index(["track", "a_lat", "mu_scale", "start_s"])
    b = sm[sm.ctrl == "SE"].set_index(["track", "a_lat", "mu_scale", "start_s"])
    j = a[["status", "lap_id", "fail_s", "lap_time_ratio"]].join(b[["status", "fail_s", "lap_time_ratio"]], rsuffix="_SE")
    j = j.reset_index()
    j = j.merge(fc[["lap_id", "cls"]], on="lap_id", how="left")
    rec = j[(j.status != "qualifying") & (j.status_SE == "qualifying")]
    lost = j[(j.status == "qualifying") & (j.status_SE != "qualifying")]
    rec.to_csv(OUT / "slipmpc_recovers.csv", index=False)
    lost.to_csv(OUT / "slipmpc_loses.csv", index=False)
    summary = dict(cells=len(j), both=int(((j.status == "qualifying") & (j.status_SE == "qualifying")).sum()),
                   se_recovers=len(rec), se_loses=len(lost),
                   recovers_by_class=rec.cls.fillna(rec.status).value_counts().to_dict(),
                   recovers_by_track=rec.track.value_counts().to_dict(),
                   recovers_by_mu=rec.mu_scale.value_counts().to_dict())
    costs = [env_step_cost("pfb"), env_step_cost("slip")]
    for c in costs:
        print(c, flush=True)
    # implied cost per million env steps
    roll = pd.read_csv(HERE / "rollout/throughput.csv") if (HERE / "rollout/throughput.csv").exists() else None
    kin, slip = costs
    summary["env_step_cost"] = costs
    summary["cpu_s_per_1M_steps"] = {c["mode"]: c["cpu_per_step_us"] for c in costs}  # µs/step == s per 1M steps
    summary["ratio_slip_over_kin"] = slip["wall_per_step_us"] / kin["wall_per_step_us"]
    if roll is not None:
        r8 = roll[(roll.ckpt == "ckpt_1250000.zip") & (roll.n_envs == 8) & (~roll.traced)]["median"].iloc[0]
        r4 = roll[(roll.ckpt == "ckpt_1250000.zip") & (roll.n_envs == 4) & (~roll.traced)]["median"].iloc[0]
        summary["measured_rollout_steps_per_s"] = {"4": r4, "8": r8}
    (OUT / "slipbase.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "env_step_cost"}, indent=1, default=float))


# ------------------------------------------------------------------------------------------- friction probe
def _collect(args):
    track, seed, n_ep = args
    import torch

    from pilot.residual_env import ResidualRacingEnv

    torch.set_num_threads(1)
    cfg = R.load_cfg("wall", track)
    env = ResidualRacingEnv(cfg, seed=seed)
    model = R._model(str(R.EXPS["wall"][1] / R.SELECTED))
    rows = []
    for e in range(n_ep):
        obs, info = env.reset()
        mu, a_lat = info["mu_scale"], info["a_lat"]
        for t in range(600):
            act, _ = model.predict(obs, deterministic=False)  # stochastic, as in training
            rows.append((track, seed * 1000 + e, t, mu, a_lat, *obs, *act))
            obs, _, te, tu, _ = env.step(act)
            if te or tu:
                break
    return rows


def collect():
    p = RAW / "probe_data.csv.gz"
    if p.exists():
        return pd.read_csv(p)
    jobs = [("Austin", 100 + i, 25) for i in range(8)] + [("Spielberg", 200 + i, 10) for i in range(8)]
    with ProcessPoolExecutor(8) as ex:
        rows = [r for part in ex.map(_collect, jobs) for r in part]
    d = 37
    cols = ["track", "episode", "t", "mu", "a_lat"] + [f"o{i}" for i in range(d)] + ["act_d", "act_a"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(p, index=False, float_format="%.5g")
    return df


def features(df, k):
    """Current obs + action, plus k frames of history (obs, action, and first differences); rows with < k history
    frames in their episode are dropped."""
    base = [c for c in df.columns if c.startswith("o")] + ["act_d", "act_a"]
    out, keep = [df[base].values], np.ones(len(df), bool)
    if k > 0:
        g = df.groupby("episode", sort=False)
        for lag in range(1, k + 1):
            sh = g[base].shift(lag)
            out.append(sh.values)
            out.append(df[base].values - sh.values)
        keep = (df.t >= k).values
    X = np.concatenate(out, axis=1)
    return X[keep], keep


def ridge_fit(X, y, lam=1.0):
    mu_, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu_) / sd]
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    A[0, 0] -= lam
    w = np.linalg.solve(A, Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu_) / sd] @ w


def mlp_fit(X, y, epochs=30, seed=0):
    import torch

    torch.manual_seed(seed)
    mu_, sd = X.mean(0), X.std(0) + 1e-9
    Xt = torch.tensor((X - mu_) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()

    def pred(Xn):
        with torch.no_grad():
            return net(torch.tensor((Xn - mu_) / sd, dtype=torch.float32)).numpy()[:, 0]

    return pred


def probe():
    import torch

    torch.set_num_threads(4)
    df = collect()
    v = df.o2 * 8.0  # obs[2] = v / 8
    r = df.o4 * 3.0  # obs[4] = r / 3
    df["ay_abs"] = (v * r).abs()  # measured lateral acceleration proxy (excitation; does not use mu)
    df["excitation"] = np.where(df.ay_abs < 2.0, "low (|v·r| < 2)", np.where(df.ay_abs > 4.0, "high (|v·r| > 4)", "mid"))
    rows = []
    rng = np.random.default_rng(0)
    for k in (0, 5, 10):
        X, keep = features(df, k)
        d = df[keep].reset_index(drop=True)
        y = d.mu.values
        aus = (d.track == "Austin").values
        eps = d.episode[aus].unique()
        rng.shuffle(eps)
        folds = np.array_split(eps, 5)
        for model_name, fit in (("ridge", ridge_fit), ("mlp", mlp_fit)):
            preds = np.full(len(d), np.nan)
            for f in folds:  # episode-grouped 5-fold CV on Austin
                te = aus & d.episode.isin(f).values
                tr_ = aus & ~d.episode.isin(f).values
                preds[te] = fit(X[tr_], y[tr_])(X[te])
            hold = fit(X[aus], y[aus])(X[~aus])  # train on all Austin, test on Spielberg
            preds_h = np.full(len(d), np.nan)
            preds_h[~aus] = hold
            for split, p_, m in (("Austin CV (episode-grouped)", preds, aus), ("held-out track (Spielberg)", preds_h, ~aus)):
                for exc in ("all", "low (|v·r| < 2)", "mid", "high (|v·r| > 4)"):
                    mm = m & ((d.excitation == exc).values if exc != "all" else True)
                    if mm.sum() < 50:
                        continue
                    e = p_[mm] - y[mm]
                    base = y[aus].mean() - y[mm]
                    rows.append(dict(history_k=k, model=model_name, split=split, excitation=exc, n=int(mm.sum()),
                                     n_episodes=int(d.episode[mm].nunique()), rmse=float(np.sqrt(np.mean(e**2))),
                                     rmse_const=float(np.sqrt(np.mean(base**2))),
                                     r2=float(1 - np.mean(e**2) / np.var(y[mm]))))
            print(k, model_name, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "friction_probe.csv", index=False, float_format="%.4f")
    meta = dict(steps=len(df), episodes_austin=int(df[df.track == "Austin"].episode.nunique()),
                episodes_spielberg=int(df[df.track == "Spielberg"].episode.nunique()),
                excitation_share=df.excitation.value_counts(normalize=True).to_dict(),
                mu_range=[float(df.mu.min()), float(df.mu.max())])
    (OUT / "friction_probe_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 250)
    print(meta)
    print(t.pivot_table(index=["split", "excitation"], columns=["model", "history_k"], values="rmse").round(4).to_string())


if __name__ == "__main__":
    {"slipbase": slipbase, "probe": probe}[sys.argv[1]]()
