"""Stage 4 of docs/history_mu_plan.md: diagnostics, lapping only (no training, no selection).

    uv run python results/analysis/history_mu/stage4.py lap        # R-H, R-H-zero on the 144-lap grid with traces
    uv run python results/analysis/history_mu/stage4.py rollout    # training-distribution episodes + hidden layers
    uv run python results/analysis/history_mu/stage4.py analyse    # tables -> stage4/analysis.txt

Controllers: the SELECTED checkpoint of every R-H and R-H-zero run. Arm R's per-lap rows are reused from
results/analysis/slip_rl/stage5/laps.csv (same grid, same aggregates, same code).

Questions (plan, Stage 4):
  1. residual against mu: does the acceleration residual in corners (|kappa| > 0.05) depend on mu/mu0? This is
     decision-rule criterion 4. Measured on the 144-lap grid (mu/mu0 0.6 vs 1.0, same cells otherwise) and on
     training-distribution episodes (continuous mu/mu0 in [0.6, 1.0], slope of a_res on mu).
  2. in-policy mu probe: is mu decodable from the policy's (actor) and value function's (critic) last hidden layer?
     Ridge and the 2x64 MLP, leave-one-track-out, by excitation (as Stage 0).
  3. spin and wall locations;
  4. Stage-5 soft-limit columns (slack activity, plant slip angles, residual size).
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "slip_rl"))
sys.path.insert(0, str(HERE.parent / "followup"))

D = ROOT / "results_st_pacejka"
OUT = HERE / "stage4"
RAW = HERE / "raw"
TRACES = D / "history_mu/stage4/traces"
R5 = HERE.parent / "slip_rl/stage5/laps.csv"
RUNS = {"R": D / "slip_rl/R", "RH": D / "history_mu/RH", "RH0": D / "history_mu/RH0"}
TRACKS = ["Austin", "Catalunya", "Silverstone", "Spielberg"]
SEEDS = (0, 1, 2)
CORNER = 0.05


def run_dir(arm, s):
    return RUNS[arm] / f"s{s}"


def selected(arm, s):
    rd = run_dir(arm, s)
    return rd / (rd / "protocol_selected.txt").read_text().strip()


# ------------------------------------------------------------------------------------------- 144-lap grid
def lap(args):
    from pilot.closedloop import Scenario, qualifies, run_many
    from pilot.protocol import _hotspot, report_cells
    from pilot.residual_env import ResidualPolicy
    from stage5 import lap_aggregates

    OUT.mkdir(exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    cells, rows = report_cells(), []
    for arm in ("RH", "RH0"):
        for s in SEEDS:
            name, rd = f"{arm}/s{s}", run_dir(arm, s)
            cfg = yaml.safe_load(open(rd / "train_config.yaml"))
            cfg["residual"].pop("tracks", None)
            pol = ResidualPolicy.from_rc(str(selected(arm, s)), cfg["residual"])
            sw, q = cfg["longitudinal"]["slip_mpc"], cfg["qualifying"]
            ref = pd.read_csv(rd / "protocol_report.csv")
            traces = []
            for track in TRACKS:
                tc = copy.deepcopy(cfg)
                tc["track"] = track
                cs = [c for c in cells if c[0] == track]
                K = cfg["residual"].get("base_K", 0.0)
                for c, (sm, tr) in zip(cs, run_many(tc, [(Scenario(c[1], c[2], c[3]), K, True, None, pol) for c in cs])):
                    ok, why = qualifies(sm, q)
                    r = ref[(ref.track == track) & (ref.a_lat == c[1]) & (ref.mu_scale == c[2]) & (ref.start_s == c[3])]
                    assert len(r) == 1 and r.status.iloc[0] == why, (name, c, why, r.status.tolist())
                    fs = sm.get("fail_s")
                    rows.append(dict(run=name, variant=arm, track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3],
                                     grid=c[4], status=why, qualifying=bool(ok), fail_s=fs,
                                     hotspot=_hotspot(track, fs), lap_time_ratio=sm["lap_time_ratio"],
                                     **lap_aggregates(tr, sw["alpha_f_max"], sw["alpha_r_max"])))
                    keep = [k for k in ("t", "s", "v", "v_plan", "kappa", "slip", "yaw_rate", "delta", "delta_res",
                                        "a_res", "a_mpc", "speed_scale", "slack0", "slack_max", "e_y") if k in tr]
                    traces.append(tr[keep].assign(track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3]))
            pd.concat(traces).to_csv(TRACES / f"{arm}_s{s}.csv.gz", index=False, float_format="%.5g")
            print(name, "outcomes match the protocol report", time.strftime("%T"), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "laps.csv", index=False, float_format="%.5g")


# ------------------------------------------------------------------------------------------- rollouts
def _rollout(job):
    """Deterministic (evaluation) actions, training-distribution episodes. Per control instant: mu, excitation,
    curvature, the applied residuals and the actor / critic last hidden layer."""
    arm, s, track, chunk, n_ep = job
    import torch
    from stable_baselines3 import PPO

    from pilot.residual_env import ResidualRacingEnv

    torch.set_num_threads(1)
    cfg = copy.deepcopy(yaml.safe_load(open(run_dir(arm, s) / "train_config.yaml")))
    cfg["track"] = track
    cfg["residual"].pop("tracks", None)
    env = ResidualRacingEnv(cfg, seed=20_000 * (s + 1) + 100 * TRACKS.index(track) + chunk)  # same episodes per arm
    model = PPO.load(str(selected(arm, s)), device="cpu")
    pol = model.policy
    rmax, ramax = cfg["residual"]["r_max"], cfg["residual"]["accel_r_max"]
    meta, hid = [], []
    for e in range(n_ep):
        obs, info = env.reset()
        sim = env.sim
        for t in range(cfg["residual"]["episode_steps"]):
            v, r = float(sim.st[3]), float(sim.st[5])
            kap = float(sim.ref.kappa[sim.i_g])
            with torch.no_grad():
                x = torch.as_tensor(obs[None], dtype=torch.float32)
                f = pol.extract_features(x, pol.pi_features_extractor) if hasattr(pol, "pi_features_extractor") \
                    else pol.extract_features(x)
                hp, hv = pol.mlp_extractor.forward_actor(f), pol.mlp_extractor.forward_critic(f)
                act = pol.action_net(hp).numpy()[0]
            a = np.clip(act, -1, 1)
            meta.append((arm, s, track, f"{arm}{s}-{track}-{chunk}-{e}", t, info["mu_scale"], info["a_lat"], v, r, kap,
                         a[0] * rmax, a[1] * ramax))
            hid.append(np.concatenate([hp.numpy()[0], hv.numpy()[0]]).astype(np.float32))
            obs, _, te, tu, _ = env.step(act.astype(np.float32))
            if te or tu:
                break
    return meta, np.array(hid)


def rollout(args):
    RAW.mkdir(exist_ok=True)
    jobs = [(arm, s, tr, c, args.n_ep) for arm in RUNS for s in SEEDS for tr in TRACKS for c in range(args.chunks)]
    t0 = time.perf_counter()
    meta, hid = [], []
    with ProcessPoolExecutor(8) as ex:
        for m, h in ex.map(_rollout, jobs):
            meta += m
            hid.append(h)
    cols = ["arm", "seed", "track", "episode", "t", "mu", "a_lat", "v", "r", "kappa", "dres", "ares"]
    df = pd.DataFrame(meta, columns=cols)
    df.to_csv(RAW / "stage4_rollouts.csv.gz", index=False, float_format="%.6g")
    np.save(RAW / "stage4_hidden.npy", np.concatenate(hid).astype(np.float16))
    print(f"{len(df)} steps, {df.episode.nunique()} episodes in {time.perf_counter() - t0:.0f} s")


# ------------------------------------------------------------------------------------------- analysis
def mu_effect_grid(d):
    """Per run: corner a_res (lap mean over |kappa| > 0.05) at mu/mu0 0.6 and 1.0, and their difference, over the
    cells (track, a_lat, start) where both laps exist. Failed laps are included (they are the cells that matter);
    their mean is over the part driven."""
    rows = []
    for run, g in d.groupby("run"):
        p = g.pivot_table(index=["track", "a_lat", "start_s"], columns="mu_scale", values="ares_corner")
        rows.append(dict(run=run, variant=g.variant.iloc[0], ares_c_06=p[0.6].mean(), ares_c_10=p[1.0].mean(),
                         diff=(p[0.6] - p[1.0]).mean(), diff_q=(p[0.6] - p[1.0]).abs().mean(),
                         dres_06=g[g.mu_scale == 0.6].dres_mean.mean(), dres_10=g[g.mu_scale == 1.0].dres_mean.mean()))
    return pd.DataFrame(rows).set_index("run")


def mu_effect_rollouts(df):
    """Per run: slope of the corner a_res on mu/mu0 (training-distribution episodes, deterministic actions), with
    the corner state controlled for: a_res ~ mu + v + |v r| + kappa sign-folded, least squares. Also the plain
    difference of means, mu/mu0 < 0.7 against > 0.9."""
    rows = []
    c = df[df.kappa.abs() > CORNER]
    for (arm, s), g in c.groupby(["arm", "seed"]):
        X = np.column_stack([np.ones(len(g)), g.mu, g.v, (g.v * g.r).abs(), g.kappa.abs()])
        b = np.linalg.lstsq(X, g.ares.values, rcond=None)[0]
        rows.append(dict(run=f"{arm}/s{s}", variant=arm, slope_mu=b[1], slope_mu_raw=np.polyfit(g.mu, g.ares, 1)[0],
                         lo_minus_hi=g[g.mu < 0.7].ares.mean() - g[g.mu > 0.9].ares.mean(), n=len(g)))
    return pd.DataFrame(rows).set_index("run")


def hidden_probe(df, H):
    import torch

    from feasibility import mlp_fit, ridge_fit

    torch.set_num_threads(8)
    vr = (df.v * df.r).abs().values
    exc = np.where(vr < 2, "low", np.where(vr > 4, "high", "mid"))
    rows = []
    for (arm, s), g in df.groupby(["arm", "seed"]):
        idx = g.index.values
        for layer, sl in (("actor", slice(0, 128)), ("critic", slice(128, 256))):
            X, y, trk = H[idx, sl].astype(np.float32), g.mu.values, g.track.values
            for mname, fit in (("ridge", ridge_fit), ("mlp", mlp_fit)):
                for tr in TRACKS:
                    te = trk == tr
                    pr = fit(X[~te], y[~te])(X[te])
                    for ex in ("all", "high"):
                        m = np.ones(te.sum(), bool) if ex == "all" else exc[idx][te] == "high"
                        e = pr[m] - y[te][m]
                        rows.append(dict(arm=arm, seed=s, layer=layer, model=mname, test=tr, excitation=ex,
                                         n=int(m.sum()), r2=1 - np.mean(e**2) / np.var(y[te][m])))
            print(arm, s, layer, flush=True)
    return pd.DataFrame(rows)


def analyse(args):
    OUT.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(R5).query("variant == 'R'"), pd.read_csv(OUT / "laps.csv")], ignore_index=True)
    pd.set_option("display.width", 220)
    out = []

    def show(title, tab):
        s = f"\n## {title}\n{tab.to_string(float_format=lambda x: f'{x:.3f}')}"
        print(s)
        out.append(s)

    # 1. residual against mu (criterion 4)
    g = mu_effect_grid(d)
    show("1a. 144-lap grid: corner a_res [m/s^2] at mu/mu0 0.6 and 1.0 (same track/a_lat/start cells), difference, "
         "mean |difference|; |delta_res| [rad]", g)
    ro = pd.read_csv(RAW / "stage4_rollouts.csv.gz")
    r = mu_effect_rollouts(ro)
    show("1b. training-distribution episodes (deterministic): corner a_res slope on mu/mu0 [m/s^2 per unit mu/mu0], "
         "controlled for v, |v r|, |kappa|; raw slope; mean(mu<0.7) - mean(mu>0.9)", r)
    crit = []
    for lab, t, col in (("grid diff", g, "diff"), ("rollout slope", r, "slope_mu")):
        R = t[t.variant == "R"][col]
        for v in ("RH", "RH0"):
            x = t[t.variant == v][col]
            crit.append(dict(measure=lab, variant=v, R_range=f"{R.min():.3f} .. {R.max():.3f}",
                             R_spread=R.max() - R.min(), median=x.median(),
                             median_minus_R_median=x.median() - R.median(),
                             outside_R_range=int(((x < R.min()) | (x > R.max())).sum())))
    show("1c. criterion 4: the mu effect of R-H / R-H-zero against arm R's seed range of the same quantity",
         pd.DataFrame(crit).set_index(["measure", "variant"]))
    # 2. in-policy probe
    H = np.load(RAW / "stage4_hidden.npy")
    pf = OUT / "hidden_probe.csv"
    if args.refit or not pf.exists():
        hidden_probe(ro, H).to_csv(pf, index=False, float_format="%.4f")
    hp = pd.read_csv(pf)
    best = hp.groupby(["arm", "seed", "layer", "excitation", "test"]).r2.max().reset_index()
    show("2. mu probe on the last hidden layer (better of ridge / MLP), leave-one-track-out R^2, mean over the 4 "
         "held-out tracks; per seed",
         best.groupby(["layer", "excitation", "arm", "seed"]).r2.mean().unstack(["arm", "seed"]))
    # 3. failure locations
    fl = d[d.status.isin(["spin", "wall_collision"])].copy()
    fl["where"] = np.where(fl.hotspot.fillna("").isin(["", "other"]), fl.track.str[:3] + "-" +
                           (fl.fail_s // 25 * 25).astype(int).astype(str), fl.hotspot)
    show("3. spins and walls by place (summed over the 3 seeds of each arm)",
         fl.pivot_table(index="where", columns="variant", values="status", aggfunc="size", fill_value=0)
         .assign(tot=lambda t: t.sum(axis=1)).sort_values("tot", ascending=False).head(15))
    show("3b. spins and walls by mu/mu0, a_lat (summed over seeds)",
         fl.pivot_table(index=["mu_scale", "a_lat"], columns=["variant", "status"], values="track", aggfunc="size",
                        fill_value=0))
    # 4. soft-limit columns
    cols = ["slack_active", "slack_any_active", "ar_over", "af_over", "ar_max", "dres_mean", "ares_abs",
            "ares_corner", "ares_straight"]
    show("4. soft-limit activity and residual size per run (mean over the 144 laps)", d.groupby("run")[cols].mean())
    key = ["track", "a_lat", "mu_scale", "start_s"]
    allq = d.groupby(key).qualifying.all().reset_index().query("qualifying")[key]
    show(f"4b. the same on the {len(allq)} cells every run qualifies", d.merge(allq, on=key).groupby("run")[cols].mean())
    show("4c. last 1 s before a spin or wall", fl.groupby(["variant", "status"])[
        ["last_slack_active", "last_ar_max", "last_dres", "last_ares"]].mean())
    (OUT / "analysis.txt").write_text("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lap", "rollout", "analyse"])
    ap.add_argument("--n-ep", type=int, default=6)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    {"lap": lap, "rollout": rollout, "analyse": analyse}[a.what](a)
