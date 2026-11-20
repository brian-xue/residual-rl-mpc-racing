"""Stage 1 of MPC speed control: coupled steering + speed MPC vs the pilot's speed controller.

Longitudinal controllers (each with steering K = 0 and K = K_frozen):
    P     plan feedforward + P speed control (pilot), standard speed plan
    P-E   same controller on the friction-ellipse speed plan
    C     coupled MPC (pilot.mpc.CoupledMPC), standard plan, no ellipse constraint
    C-E   coupled MPC with the friction-ellipse constraint, on the friction-ellipse plan
    S     slip-aware coupled MPC (pilot.mpc.SlipMPC: dynamic model, nominal mu0, slip-angle limits), standard plan
    S-E   slip-aware MPC on the friction-ellipse plan (slip limits only, no ellipse constraint)
The steering MPC weights, horizon, rate, limits, servo, validity checks and qualifying rule are the pilot's.
The lap-time ratio in the qualifying rule is taken against each controller's own speed plan.

Outputs (<config out>/longitudinal/<track>/, e.g. results/ for linear ST, results_st_pacejka/ for Pacejka): sweep.csv, lap_time_grid.csv, lap_time_comparison.csv/.png, report.json.
Usage:  uv run python -m pilot.longitudinal [--config configs/pilot.yaml] [--track Austin]
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .closedloop import Scenario, Track, qualifies, run_lap, speed_chain

LONG = {  # name: (longitudinal mode, friction-ellipse plan, friction-ellipse MPC constraint)
    "P": ("pfb", False, False),
    "P-E": ("pfb", True, False),
    "C": ("mpc", False, False),
    "C-E": ("mpc", True, True),
    "S": ("slip", False, False),
    "S-E": ("slip", True, False),
}
COLORS = {"P": "#c0392b", "P-E": "#e67e22", "C": "#1f6fb4", "C-E": "#27ae60", "S": "#8e44ad", "S-E": "#d35ec5"}


def variant_cfg(cfg, name):
    mode, plan_e, mpc_e = LONG[name]
    c = copy.deepcopy(cfg)
    c["longitudinal"]["mode"] = mode
    c["speed_plan"]["friction_ellipse"] = plan_e
    c["longitudinal"]["mpc"]["friction_ellipse"] = mpc_e
    if "slip_mpc" in c["longitudinal"]:
        c["longitudinal"]["slip_mpc"]["friction_ellipse"] = mpc_e
    return c


_W: dict = {}


def _init(cfg):
    _W["track"] = Track(cfg)


def _lap(job):
    c, scn, K, tag = job
    sm, _ = run_lap(_W["track"], scn, K, c, log=False)
    ok, why = qualifies(sm, c["qualifying"])
    return {**tag, **sm, "qualifying": ok, "status": why}


def _chain(job):
    c, K, mu, grid, starts, tag = job
    out = []
    for a, s0, sm, _ in speed_chain(_W["track"], c, K, mu, grid, starts):
        out.append({**sm, **tag})  # tag last: speed_chain sets its own (empty) controller label
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--track", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-grid", action="store_true",
                    help="skip the laps; rebuild lap_time_comparison/report from the saved sweep.csv and lap_time_grid.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.track:
        cfg["track"] = args.track
    out = pathlib.Path(args.out or f"{cfg.get('out', 'results')}/longitudinal/{cfg['track']}")  # per plant config
    out.mkdir(parents=True, exist_ok=True)
    rep_path = pathlib.Path(cfg.get("out", "results")) / "report.json"
    K_ff = json.loads(rep_path.read_text()).get("K_frozen", 0.03) if rep_path.exists() else 0.03
    t0 = time.time()
    ctrls = [(n, K) for K in (0.0, K_ff) for n in LONG]
    label = lambda n, K: f"{n}{'+ff' if K else ''}"  # noqa: E731
    sw, lc = cfg["sweep"], cfg["lap_time"]
    starts = (sw["start_s"], sw["recheck_start_s"])

    grid = sorted(set(lc["a_lat_grid"]) | {9.0, 10.0})
    if args.from_grid:
        sdf, gdf = pd.read_csv(out / "sweep.csv"), pd.read_csv(out / "lap_time_grid.csv")
        gdf["controller"] = [label(n, K) for n, K in zip(gdf["long"], gdf["K"])]
    else:
        sdf, gdf = run_laps(cfg, ctrls, label, starts, grid, out)
    summarize(cfg, ctrls, label, starts, grid, sdf, gdf, out, K_ff, t0)


def run_laps(cfg, ctrls, label, starts, grid, out):
    sw, lc = cfg["sweep"], cfg["lap_time"]
    with ProcessPoolExecutor(cfg.get("workers") or 8, initializer=_init, initargs=(cfg,)) as ex:
        print("[1/2] sweep", flush=True)
        jobs = [(variant_cfg(cfg, n), Scenario(a, mu, s0), K, dict(controller=label(n, K), long=n, K=K))
                for n, K in ctrls for a in sw["a_lat_levels"] for mu in sw["mu_scales"] for s0 in starts]
        sdf = pd.DataFrame(list(ex.map(_lap, jobs)))
        sdf.to_csv(out / "sweep.csv", index=False, float_format="%.4f")
        piv = sdf.pivot_table(index=["a_lat", "mu_scale", "start_s"], columns="controller", values="status", aggfunc="first")
        print(piv[[label(n, K) for n, K in ctrls]].to_string(), flush=True)

        print("[2/2] fastest qualifying lap per controller", flush=True)
        chains = [(variant_cfg(cfg, n), K, mu, grid, starts, dict(controller=label(n, K), long=n, K=K))
                  for n, K in ctrls for mu in lc["mu_scales"]]
        rows = [r for ch in ex.map(_chain, chains) for r in ch]
    gdf = pd.DataFrame(rows)
    gdf.to_csv(out / "lap_time_grid.csv", index=False, float_format="%.4f")
    return sdf, gdf


def summarize(cfg, ctrls, label, starts, grid, sdf, gdf, out, K_ff, t0):
    lc = cfg["lap_time"]

    best = []
    for n, K in ctrls:
        c = label(n, K)
        for mu in lc["mu_scales"]:
            g = gdf[(gdf.controller == c) & (gdf.mu_scale == mu)]
            ok_levels = []
            for a in grid:  # contiguous from the slowest level, qualifying from both start poses
                q = g[g.a_lat == a]
                if len(q) < len(starts) or not all(qualifies(r, cfg["qualifying"])[0] for r in q.to_dict("records")):
                    break
                ok_levels.append(a)
            a = max(ok_levels) if ok_levels else None
            r0 = g[(g.a_lat == a) & (g.start_s == starts[0])].iloc[0] if a else None
            best.append(dict(controller=c, long=n, K=K, mu_scale=mu, best_a_lat=a,
                             lap_time=None if r0 is None else r0.lap_time,
                             max_ey=None if r0 is None else g[g.a_lat == a].max_ey.max(),
                             ellipse_max=None if r0 is None else g[g.a_lat == a].ellipse_max.max(),
                             max_abs_ay=None if r0 is None else g[g.a_lat == a].max_abs_ay.max()))
    bdf = pd.DataFrame(best)
    bdf.to_csv(out / "lap_time_comparison.csv", index=False, float_format="%.3f")
    print(bdf.to_string(index=False), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    w = 0.8 / len(LONG)
    for ax, K in zip(axes, (0.0, K_ff)):
        for j, n in enumerate(LONG):
            for i, mu in enumerate(lc["mu_scales"]):
                r = bdf[(bdf.controller == label(n, K)) & (bdf.mu_scale == mu)].iloc[0]
                x = i + (j - (len(LONG) - 1) / 2) * w
                if r.lap_time is None or pd.isna(r.lap_time):
                    ax.text(x, 1, "none", ha="center", va="bottom", fontsize=6, rotation=90)
                    continue
                ax.bar(x, r.lap_time, w, color=COLORS[n], label=n if i == 0 else None)
                ax.text(x, r.lap_time + 0.3, f"{r.lap_time:.1f}\n{r.best_a_lat:g}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(lc["mu_scales"])), [f"μ/μ₀ = {m}" for m in lc["mu_scales"]])
        ax.set_title(f"steering: MPC{' + K v² κ' if K else ''}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    lo = pd.to_numeric(bdf.lap_time, errors="coerce").min()
    axes[0].set_ylim(lo - 6 if pd.notna(lo) else 0, None)
    axes[0].set_ylabel("best qualifying lap time [s] (start s = 0), label: time / a_lat")
    axes[0].legend(fontsize=8, title="longitudinal", title_fontsize=8, loc="upper left")
    fig.suptitle(f"{cfg['track']}: P = plan + P speed control, C = coupled kinematic MPC, S = slip-aware MPC, "
                 "-E = friction-ellipse plan (C-E: + ellipse constraint)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "lap_time_comparison.png", dpi=130)
    plt.close(fig)

    report = dict(track=cfg["track"], K_ff=K_ff, controllers={k: dict(zip(("mode", "plan_ellipse", "mpc_ellipse"), v))
                                                              for k, v in LONG.items()},
                  longitudinal_mpc=cfg["longitudinal"]["mpc"],
                  sweep_qualifying=sdf.groupby("controller").qualifying.sum().astype(int).to_dict(),
                  lap_time_comparison=bdf.to_dict(orient="records"), runtime_s=time.time() - t0)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['runtime_s']:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
