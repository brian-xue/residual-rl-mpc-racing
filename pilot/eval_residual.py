"""Evaluate the trained residual policy against nominal MPC and MPC + K v^2 kappa.

Steps (all laps through ``closedloop.run_lap``: identical servo, limits, validity checks and qualifying rule):
  1. checkpoint selection on validation laps (residual.select: speed levels and start pose NOT used below)
  2. sweep: sweep.a_lat_levels x sweep.mu_scales x {start_s, recheck_start_s} for all three controllers
  3. fastest qualifying lap per controller and friction (lap_time.a_lat_grid + residual.extra_a_lat)
  4. outputs: sweep.csv, lap_time_grid.csv, lap_time_comparison.csv/.png, residual_trace.png, lap_time.gif, report.json

Usage:  uv run python -m pilot.eval_residual [--config configs/pilot.yaml] [--model PATH] [--no-gif]
        uv run python -m pilot.eval_residual --track Spielberg --model results/residual/ckpt_1750000.zip
            (held-out track: same model, same K, same settings; outputs in results/heldout/<track>/)
K for the ff controller is read from <config out>/report.json (K_frozen, from pilot.run_pilot), else 0.03.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from . import animate
from .closedloop import Scenario, Track, qualifies, run_chains, run_many
from .residual_env import ResidualPolicy

NAMES = {"nominal": "Nominal MPC", "ff": "MPC + K v² κ", "residual": "MPC + RL residual"}
COLORS = {"nominal": "#c0392b", "ff": "#1f6fb4", "residual": "#27ae60"}
animate.NAMES.setdefault("residual", NAMES["residual"])
animate.COLORS.setdefault("residual", "#2ecc71")


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for k, v in d.items()}


def select_checkpoint(cfg, out, r_max):
    """Validation laps per checkpoint; rule: most qualifying, then mean progress, then lowest mean RMS e_y."""
    sel = cfg["residual"]["select"]
    ckpts = sorted(out.glob("ckpt_*.zip"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if (out / "final.zip").exists():
        ckpts.append(out / "final.zip")
    if not ckpts:
        raise SystemExit(f"no checkpoints in {out}; run pilot.train_residual first")
    scns = [Scenario(a, mu, sel["start_s"]) for a in sel["a_lat"] for mu in sel["mu_scales"]]
    jobs = [(scn, cfg["residual"].get("base_K", 0.0), False, None, ResidualPolicy(c, r_max, cfg["residual"].get("accel_r_max", 0.0),
                                                                        cfg["residual"].get("obs_wall_clearance", False)))
            for c in ckpts for scn in scns]
    res = run_many(cfg, jobs)
    rows = []
    for i, c in enumerate(ckpts):
        rr = [r[0] for r in res[i * len(scns):(i + 1) * len(scns)]]
        rows.append(dict(checkpoint=c.name, n_qualifying=sum(qualifies(r, cfg["qualifying"])[0] for r in rr),
                         mean_progress=float(np.mean([r["progress_frac"] for r in rr])),
                         mean_rms_ey=float(np.mean([r["rms_ey"] for r in rr]))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "checkpoint_selection.csv", index=False, float_format="%.4f")
    best = df.sort_values(["n_qualifying", "mean_progress", "mean_rms_ey"], ascending=[False, False, True]).iloc[0]
    print(df.to_string(index=False), flush=True)
    return out / best.checkpoint, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model", default=None, help="skip checkpoint selection and use this .zip")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--track", default=None, help="evaluate on another map (held-out track); needs --model")
    ap.add_argument("--out", default=None, help="output folder (default residual.out, or results/heldout/<track>)")
    ap.add_argument("--grid-min", type=float, default=None,
                    help="supplementary: start the fastest-lap search at this a_lat (default: the full lap_time grid)")
    ap.add_argument("--K", type=float, default=None, help="K of the ff controller (default: frozen K of the pilot)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rc = cfg["residual"]
    rep_path = pathlib.Path(cfg.get("out", "results")) / "report.json"  # pilot's frozen K (training track)
    if args.track:
        if not args.model:
            raise SystemExit("--track needs --model: the checkpoint is selected on the training track only")
        cfg["track"] = args.track
    out = pathlib.Path(args.out or (f"results/heldout/{args.track}" if args.track else rc["out"]))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    K = args.K if args.K is not None else (
        json.loads(rep_path.read_text()).get("K_frozen", 0.03) if rep_path.exists() else 0.03)
    report = {"track": cfg["track"], "K_ff": K, "r_max": rc["r_max"], "base_K": rc.get("base_K", 0.0)}

    print("[1/3] checkpoint selection", flush=True)
    if args.model:
        model = pathlib.Path(args.model)
    else:
        model, _ = select_checkpoint(cfg, out, rc["r_max"])
    report["model"] = str(model)
    print(f"   using {model}", flush=True)
    pol = ResidualPolicy(model, rc["r_max"], rc.get("accel_r_max", 0.0), rc.get("obs_wall_clearance", False))
    ctrls = {"nominal": (0.0, None), "ff": (K, None), "residual": (rc.get("base_K", 0.0), pol)}
    if rc.get("base_K", 0.0) > 0:  # residual on top of MPC + K v^2 kappa
        NAMES["residual"] = animate.NAMES["residual"] = "MPC + K v² κ + RL residual"
    if rc.get("accel_r_max", 0.0) > 0:  # dual channel: steering + acceleration residual
        NAMES["residual"] = animate.NAMES["residual"] = "MPC + RL residual (δ, a)" + (
            " + wall obs" if rc.get("obs_wall_clearance", False) else "")
    report["controllers"] = dict(NAMES)
    report["accel_r_max"] = rc.get("accel_r_max", 0.0)
    report["speed_plan_friction_ellipse"] = bool(cfg["speed_plan"].get("friction_ellipse", False))

    # 2. sweep
    print("[2/3] sweep", flush=True)
    sw = cfg["sweep"]
    starts = (sw["start_s"], sw["recheck_start_s"])
    scns = [Scenario(a, mu, s0) for a in sw["a_lat_levels"] for mu in sw["mu_scales"] for s0 in starts]
    keys = [(c, scn) for scn in scns for c in ctrls]
    res = run_many(cfg, [(scn, ctrls[c][0], True, None, ctrls[c][1]) for c, scn in keys])
    rows, traces = [], {}
    for (c, scn), (sm, tr) in zip(keys, res):
        ok, why = qualifies(sm, cfg["qualifying"])
        sm = _clean(sm)
        sm.update(controller=c, qualifying=ok, status=why)
        rows.append(sm)
        traces[(c, scn.a_lat, scn.mu_scale, scn.start_s)] = (sm, tr)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out / "sweep.csv", index=False, float_format="%.4f")
    piv = sdf.pivot_table(index=["a_lat", "mu_scale", "start_s"], columns="controller", values="status", aggfunc="first")
    print(piv[list(ctrls)].to_string(), flush=True)
    report["sweep_qualifying"] = sdf.groupby("controller").qualifying.sum().astype(int).to_dict()

    # residual trace plot: hardest cell where the residual qualifies and ff does not, else a_lat max / mu min
    cand = [(a, mu) for a in sw["a_lat_levels"] for mu in sw["mu_scales"]
            if traces[("residual", a, mu, starts[0])][0]["qualifying"] and not traces[("ff", a, mu, starts[0])][0]["qualifying"]]
    a_p, mu_p = max(cand, key=lambda k: (k[0], -k[1])) if cand else (max(sw["a_lat_levels"]), min(sw["mu_scales"]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for c in ctrls:
        sm, tr = traces[(c, a_p, mu_p, starts[0])]
        if tr.empty:
            continue
        axes[0].plot(tr.progress, tr.e_y, color=COLORS[c], lw=0.9, label=f"{NAMES[c]} ({sm['status']})")
        axes[1].plot(tr.progress, tr.delta, color=COLORS[c], lw=0.9, label=NAMES[c])
        if c == "residual":
            axes[2].plot(tr.progress, tr.delta_res, color=COLORS[c], lw=0.9, label="learned residual")
        if c == "ff":
            axes[2].plot(tr.progress, tr.delta_ff, color=COLORS[c], lw=0.9, ls=":", label="K v² κ term")
    for ax, lab in zip(axes, ("e_y [m]", "δ actual [rad]", "correction [rad]")):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].axhline(cfg["qualifying"]["max_abs_ey_m"], color="k", lw=0.6, ls="--")
    axes[0].axhline(-cfg["qualifying"]["max_abs_ey_m"], color="k", lw=0.6, ls="--")
    axes[2].set_xlabel("progress [m]")
    fig.suptitle(f"{cfg['track']}: a_lat {a_p}, μ/μ₀ {mu_p}, start s = {starts[0]:.0f} m", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "residual_trace.png", dpi=130)
    plt.close(fig)
    report["trace_cell"] = [a_p, mu_p]

    # 3. fastest qualifying lap per controller
    print("[3/3] fastest qualifying lap per controller", flush=True)
    lc = cfg["lap_time"]
    grid = sorted(set(lc["a_lat_grid"]) | set(rc.get("extra_a_lat", [])))
    if args.grid_min is not None:
        grid = [a for a in grid if a >= args.grid_min]
        report["lap_grid_min"] = args.grid_min
    combos = [(c, mu) for mu in lc["mu_scales"] for c in ctrls]
    chains = run_chains(cfg, [(ctrls[c][0], mu, grid, starts, None, c, ctrls[c][1]) for c, mu in combos])
    grows, summs, ltr = [], {}, {}
    for (c, mu), chain in zip(combos, chains):
        for a, s0, sm, tr in chain:
            grows.append({k: sm[k] for k in ("controller", "mu_scale", "a_lat", "start_s", "status", "qualifying",
                                              "progress_frac", "fail_s", "lap_time", "plan_lap_time", "max_ey", "rms_ey")})
            summs[(c, mu, a, s0)], ltr[(c, mu, a, s0)] = sm, tr
    pd.DataFrame(grows).to_csv(out / "lap_time_grid.csv", index=False, float_format="%.4f")
    best = []
    for mu in lc["mu_scales"]:
        for c in ctrls:
            ok = []
            for a in grid:
                if not all((c, mu, a, s) in summs and summs[(c, mu, a, s)]["qualifying"] for s in starts):
                    break
                ok.append(a)
            a = max(ok) if ok else None
            best.append(dict(mu_scale=mu, controller=c, best_a_lat=a,
                             lap_time=summs[(c, mu, a, starts[0])]["lap_time"] if a else None,
                             max_ey=max(summs[(c, mu, a, s)]["max_ey"] for s in starts) if a else None))
    bdf = pd.DataFrame(best)
    nom = bdf[bdf.controller == "nominal"].set_index("mu_scale").lap_time
    bdf["gap_s_vs_nominal"] = [nom[r.mu_scale] - r.lap_time if r.lap_time and pd.notna(nom[r.mu_scale]) else None
                               for r in bdf.itertuples()]
    bdf.to_csv(out / "lap_time_comparison.csv", index=False, float_format="%.3f")
    print(bdf.to_string(index=False), flush=True)
    report["lap_time_comparison"] = bdf.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.27
    for j, c in enumerate(ctrls):
        for i, mu in enumerate(lc["mu_scales"]):
            r = bdf[(bdf.mu_scale == mu) & (bdf.controller == c)].iloc[0]
            x = i + (j - 1) * w
            if r.lap_time is None or pd.isna(r.lap_time):
                ax.text(x, 1, "no\nqualifying\nlap", ha="center", va="bottom", fontsize=7)
                continue
            ax.bar(x, r.lap_time, w, color=COLORS[c], label=NAMES[c] if i == 0 else None)
            ax.text(x, r.lap_time + 0.3, f"{r.lap_time:.2f} s\na_lat {r.best_a_lat:g}", ha="center", va="bottom", fontsize=7)
    lo = pd.to_numeric(bdf.lap_time, errors="coerce").min()
    ax.set_ylim(lo - 6 if pd.notna(lo) else 0, None)
    ax.set_xticks(range(len(lc["mu_scales"])), [f"μ/μ₀ = {m}" for m in lc["mu_scales"]])
    ax.set_ylabel("best qualifying lap time [s] (start s = 0)")
    ax.set_title(f"{cfg['track']}: fastest qualifying lap per controller", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lap_time_comparison.png", dpi=130)
    plt.close(fig)

    if not args.no_gif:
        mu_g = min(lc["mu_scales"])
        pick = {c: bdf[(bdf.mu_scale == mu_g) & (bdf.controller == c)].iloc[0].best_a_lat for c in ctrls}
        pick = {c: a for c, a in pick.items() if a is not None and pd.notna(a)}
        if pick:
            print("[gif] rendering", flush=True)
            tr_g = {c: ltr[(c, mu_g, a, starts[0])] for c, a in pick.items()}
            sm_g = {c: _clean({**summs[(c, mu_g, a, starts[0])]}) for c, a in pick.items()}
            report["lap_time_gif"] = animate.render(Track(cfg), cfg, tr_g, sm_g, out / "lap_time.gif")
    report["runtime_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['runtime_s']:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
