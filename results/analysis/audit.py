"""Part 1: experiment audit from the SAVED results (no laps are run).

    uv run python results/analysis/audit.py
Writes results/analysis/audit/: inventory.csv, consistency.txt, status_counts.csv, sweep_by_mu.csv,
fastest_laps.csv, search_audit.csv, tolerance_sensitivity.csv, plan_comparison.csv, linear_context.csv.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import rlana as R  # noqa: E402

A = R.OUT / "audit"
A.mkdir(parents=True, exist_ok=True)
LOG = []


def note(msg):
    LOG.append(msg)
    print(msg)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:12]


def tracked(p):
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=R.ROOT, capture_output=True)
    return r.returncode == 0


def inventory():
    rows = []
    for exp, (cfgp, run, label) in R.EXPS.items():
        cfg = yaml.safe_load(open(R.ROOT / cfgp))
        tc = yaml.safe_load(open(run / "train_config.yaml"))
        same_cfg = cfg == tc
        sel = pd.read_csv(run / "checkpoint_selection.csv")
        best = sel.sort_values(["n_qualifying", "mean_progress", "mean_rms_ey"], ascending=[False, False, True]).iloc[0]
        # ties at the top (same n_qualifying) and the margin that broke them
        top = sel[sel.n_qualifying == best.n_qualifying].sort_values("mean_progress", ascending=False)
        margin = float(top.mean_progress.iloc[0] - top.mean_progress.iloc[1]) if len(top) > 1 else None
        for track in R.TRACKS:
            rp = run / "report.json" if track == "Austin" else run / "heldout" / track / "report.json"
            rep = json.loads(rp.read_text())
            rows.append(dict(exp=exp, label=label, config=cfgp, train_config_identical=same_cfg, track=track,
                             report_model=rep.get("model"), selected_by_rule=best.checkpoint,
                             model_matches_rule=pathlib.Path(rep.get("model", "")).name == best.checkpoint,
                             ckpt_sha=sha(run / best.checkpoint), ckpt_in_git=tracked(run / best.checkpoint),
                             n_ckpts_on_disk=len(list(run.glob("ckpt_*.zip"))) + (run / "final.zip").exists(),
                             top_ties=len(top), tie_margin_progress=margin,
                             base_K=rep.get("base_K"), accel_r_max=rep.get("accel_r_max"),
                             ellipse_plan=rep.get("speed_plan_friction_ellipse"), K_ff=rep.get("K_ff"),
                             lap_grid_min=rep.get("lap_grid_min")))
        for track in R.TRACKS:
            gp = run / "grid_from3" / track / "report.json"
            if gp.exists():
                rep = json.loads(gp.read_text())
                rows.append(dict(exp=exp + "_from3", label=label, config=cfgp, track=track, report_model=rep.get("model"),
                                 lap_grid_min=rep.get("lap_grid_min"), ellipse_plan=rep.get("speed_plan_friction_ellipse")))
    inv = pd.DataFrame(rows)
    inv.to_csv(A / "inventory.csv", index=False)
    for r in inv[inv.exp.isin(R.EXPS)].itertuples():
        if not r.model_matches_rule:
            note(f"MISMATCH {r.exp}/{r.track}: report model {r.report_model} vs rule {r.selected_by_rule}")
        if not r.train_config_identical:
            note(f"MISMATCH {r.exp}: train_config.yaml differs from {r.config}")
    note("inventory: report models vs selection rule and config vs train_config checked "
         f"({int(inv[inv.exp.isin(R.EXPS)].model_matches_rule.sum())}/{int(inv.exp.isin(R.EXPS).sum())} match)")
    return inv


def all_sweeps():
    """Every saved Pacejka sweep row with a uniform controller key."""
    out = []
    for track in R.TRACKS:
        d = R.saved_sweep("delta", track)
        for c, k in (("nominal", "P (std)"), ("ff", "P+ff (std)"), ("residual", "RL δ")):
            out.append(d[d.controller == c].assign(ctrl=k, track=track))
        for exp, k in (("dual", "RL δa"), ("wall", "RL δa+wall")):
            d = R.saved_sweep(exp, track)
            out.append(d[d.controller == "residual"].assign(ctrl=k, track=track))
            if exp == "dual":
                for c, kk in (("nominal", "P-E (ell)"), ("ff", "P-E+ff (ell)")):
                    out.append(d[d.controller == c].assign(ctrl=kk, track=track))
        out.append(R.saved_sweep("SE", track).assign(ctrl="S-E", track=track))
    lg = pd.read_csv(R.P / "longitudinal/Austin/sweep.csv")
    for c in sorted(set(lg.controller) - {"S-E"}):
        out.append(lg[lg.controller == c].assign(ctrl=f"[Austin only] {c}", track="Austin"))
    return pd.concat(out, ignore_index=True)


def consistency(sw):
    # baselines inside the dual and wall runs must be identical laps (same config except residual settings)
    for track in R.TRACKS:
        a, b = R.saved_sweep("dual", track), R.saved_sweep("wall", track)
        for c in ("nominal", "ff"):
            x, y = a[a.controller == c].reset_index(drop=True), b[b.controller == c].reset_index(drop=True)
            same = (x.status.values == y.status.values).all() and np.allclose(x.rms_ey, y.rms_ey)
            if not same:
                note(f"MISMATCH baseline {c} dual vs wall on {track}")
    note("baselines (nominal/ff) in the dual and wall runs: identical on all tracks" if not any(
        "baseline" in m for m in LOG) else "baseline mismatch found")
    # S-E standalone Austin sweep (all 12 controllers) vs held-out S-E only runs: same schema
    # P-E in the 12-controller Austin run vs nominal in the dual run (same controller, same plan)
    lg = pd.read_csv(R.P / "longitudinal/Austin/sweep.csv")
    pe = lg[lg.controller == "P-E"].sort_values(["a_lat", "mu_scale", "start_s"]).reset_index(drop=True)
    dn = R.saved_sweep("dual", "Austin")
    dn = dn[dn.controller == "nominal"].sort_values(["a_lat", "mu_scale", "start_s"]).reset_index(drop=True)
    note(f"P-E (longitudinal/Austin) vs nominal in the dual run: statuses identical = "
         f"{(pe.status.values == dn.status.values).all()}, max |Δ rms_ey| = {np.abs(pe.rms_ey - dn.rms_ey).max():.2e}")
    pp = lg[lg.controller == "P"].sort_values(["a_lat", "mu_scale", "start_s"]).reset_index(drop=True)
    dd = R.saved_sweep("delta", "Austin")
    dd = dd[dd.controller == "nominal"].sort_values(["a_lat", "mu_scale", "start_s"]).reset_index(drop=True)
    note(f"P (longitudinal/Austin) vs nominal in the δ run: statuses identical = {(pp.status.values == dd.status.values).all()}")
    # comparison_all_tracks.csv against status counts
    cmp_ = pd.read_csv(R.EXPS["wall"][1] / "comparison_all_tracks.csv")
    key = {"S-E (slip-aware MPC + ellipse)": "S-E", "MPC + RL (δ)": "RL δ", "MPC + RL (δ, a)": "RL δa",
           "MPC + RL (δ, a) + wall obs": "RL δa+wall"}
    bad = 0
    for r in cmp_.itertuples():
        k = key.get(r.controller)
        if k is None:
            note(f"comparison_all_tracks: unknown controller label {r.controller!r}")
            continue
        g = sw[(sw.ctrl == k) & (sw.track == r.track)]
        q, sp, wl = int(g.qualifying.sum()), int((g.status == "spin").sum()), int((g.status == "wall_collision").sum())
        if (q, sp, wl) != (r.qual, r.spin, r.wall):
            bad += 1
            note(f"MISMATCH comparison_all_tracks {r.track}/{k}: csv {r.qual}/{r.spin}/{r.wall} vs sweep {q}/{sp}/{wl}")
    note(f"comparison_all_tracks.csv (wall run): {len(cmp_) - bad}/{len(cmp_)} rows match the sweep counts")


def status_counts(sw):
    sc = sw.groupby(["track", "ctrl"]).status.value_counts().unstack(fill_value=0)
    for s in R.STATUSES:
        if s not in sc:
            sc[s] = 0
    sc = sc[R.STATUSES]
    sc["n"] = sc.sum(axis=1)
    bad = sc[sc.n != 18]
    if len(bad):
        note(f"STATUS COUNTS do not sum to 18: {bad.index.tolist()}")
    else:
        note("status counts: every track x controller sums to 18 sweep laps")
    sc = sc.reset_index()
    tot = sc.groupby("ctrl")[R.STATUSES + ["n"]].sum().reset_index().assign(track="ALL")
    sc = pd.concat([sc, tot], ignore_index=True)
    # qualifying-lap quality
    q = sw[sw.qualifying].groupby(["track", "ctrl"]).agg(q_ratio_mean=("lap_time_ratio", "mean"),
                                                       q_rms_ey=("rms_ey", "mean"), q_max_ey=("max_ey", "max"))
    sc = sc.merge(q.reset_index(), on=["track", "ctrl"], how="left")
    sc.to_csv(A / "status_counts.csv", index=False, float_format="%.4f")
    by_mu = sw.groupby(["track", "ctrl", "mu_scale"]).agg(qual=("qualifying", "sum"), n=("qualifying", "size"),
                                                         spin=("status", lambda s: (s == "spin").sum()),
                                                         wall=("status", lambda s: (s == "wall_collision").sum()),
                                                         slow=("status", lambda s: (s == "speed_tolerance").sum()))
    by_mu.reset_index().to_csv(A / "sweep_by_mu.csv", index=False)
    return sc


def best_from_grid(g, starts=(0.0, 200.0)):
    """Fastest qualifying level under the published chain rule (all lower levels qualify from both starts)."""
    ok = []
    for a in sorted(g.a_lat.unique()):
        gg = g[np.isclose(g.a_lat, a)]
        if len(gg) < len(starts) or not gg.qualifying.astype(bool).all():
            stop = gg[~gg.qualifying.astype(bool)]
            return (max(ok) if ok else None), a, (stop.status.iloc[0] if len(stop) else "missing start"), ok
        ok.append(a)
    return (max(ok) if ok else None), None, "grid exhausted", ok


def fastest_and_search():
    rows = []
    srcs = []
    for track in R.TRACKS:
        for exp, k in (("delta", None), ("dual", None), ("wall", None)):
            g = R.saved_grid(exp, track)
            for c in g.controller.unique():
                name = {"delta": {"nominal": "P (std)", "ff": "P+ff (std)", "residual": "RL δ"},
                        "dual": {"nominal": "P-E (ell)", "ff": "P-E+ff (ell)", "residual": "RL δa"},
                        "wall": {"nominal": "P-E (ell)", "ff": "P-E+ff (ell)", "residual": "RL δa+wall"}}[exp][c]
                if exp == "wall" and c != "residual":
                    continue
                srcs.append((track, name, "full grid", g[g.controller == c]))
            if exp in ("dual", "wall"):
                g3 = R.saved_grid(exp, track, from3=True)
                srcs.append((track, "RL δa" if exp == "dual" else "RL δa+wall", "from a_lat 3", g3[g3.controller == "residual"]))
        srcs.append((track, "S-E", "full grid", R.saved_grid("SE", track)))
    for track, name, search, g in srcs:
        for mu in sorted(g.mu_scale.unique(), reverse=True):
            gg = g[np.isclose(g.mu_scale, mu)]
            best, stop_a, why, ok = best_from_grid(gg)
            lt = gg[np.isclose(gg.a_lat, best) & np.isclose(gg.start_s, 0.0)].lap_time.iloc[0] if best else None
            stop_rows = gg[np.isclose(gg.a_lat, stop_a)] if stop_a is not None else gg.iloc[0:0]
            ratio = (stop_rows.lap_time / stop_rows.plan_lap_time).max() if len(stop_rows) else None
            rows.append(dict(track=track, ctrl=name, search=search, mu=mu, best_a_lat=best, lap_time=lt,
                             first_level=min(ok) if ok else gg.a_lat.min(), stop_a_lat=stop_a, stop_reason=why,
                             stop_lap_time_ratio=ratio, levels_passed=len(ok)))
    fl = pd.DataFrame(rows)
    fl.to_csv(A / "fastest_laps.csv", index=False, float_format="%.3f")
    # check against published lap_time_comparison.csv
    bad = 0
    for track in R.TRACKS:
        for exp, nm in (("delta", "RL δ"), ("dual", "RL δa"), ("wall", "RL δa+wall")):
            run = R.EXPS[exp][1]
            for search, p in (("full grid", run / ("lap_time_comparison.csv" if track == "Austin" else f"heldout/{track}/lap_time_comparison.csv")),
                              ("from a_lat 3", run / f"grid_from3/{track}/lap_time_comparison.csv")):
                if not p.exists():
                    continue
                lc = pd.read_csv(p)
                for r in lc[lc.controller == "residual"].itertuples():
                    m = fl[(fl.track == track) & (fl.ctrl == nm) & (fl.search == search) & np.isclose(fl.mu, r.mu_scale)].iloc[0]
                    a, b = r.best_a_lat, m.best_a_lat
                    if not ((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and np.isclose(a, b))):
                        bad += 1
                        note(f"MISMATCH fastest lap {track}/{nm}/{search}/mu {r.mu_scale}: csv {a} vs recomputed {b}")
    note(f"fastest-lap levels recomputed from lap_time_grid.csv: {bad} mismatches with lap_time_comparison.csv")
    return fl


def tolerance_sensitivity(sw):
    """Re-score saved sweep laps at other lap-time tolerances (sensitivity only; the published rule is 5%)."""
    rows = []
    for tol in (1.03, 1.05, 1.07, 1.10, np.inf):
        ok_other = (sw.outcome == "completed") & (sw.off_track_steps == 0) & (sw.solver_failures == 0) & (sw.max_ey <= 0.5)
        q = ok_other & (sw.lap_time_ratio.fillna(np.inf) <= tol)
        for (ctrl), g in sw.assign(q=q).groupby("ctrl"):
            rows.append(dict(tol=tol, ctrl=ctrl, qual_of_72=int(g.q.sum()), n=len(g)))
    t = pd.DataFrame(rows).pivot(index="ctrl", columns="tol", values="qual_of_72")
    t.to_csv(A / "tolerance_sensitivity.csv")
    return t


def plan_comparison():
    """Standard vs friction-ellipse plan: plan lap time and peak combined demand per a_lat (Austin + held-out)."""
    from pilot.closedloop import Track

    rows = []
    for track in R.TRACKS:
        cfg = R.load_cfg("dual", track)
        tr = Track(cfg)
        ref = tr.ref
        for a in (1.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            for fe in (False, True):
                v = tr.plan(a, None, fe)
                ax = v * np.gradient(v, ref.ds)
                ay = v**2 * np.abs(ref.kappa)
                ell = np.hypot(ax / 4.0, ay / a)
                rows.append(dict(track=track, a_lat=a, plan="ellipse" if fe else "standard",
                                 plan_lap_time=float(np.sum(ref.ds / v)), ellipse_max=float(ell.max()),
                                 frac_over_ellipse=float((ell > 1.02).mean()),
                                 frac_brake_in_turn=float(((ax < -0.5) & (ay > 0.5 * a)).mean())))
    pc = pd.DataFrame(rows)
    pc.to_csv(A / "plan_comparison.csv", index=False, float_format="%.4f")
    return pc


def linear_context():
    rows = []
    for name, p in (("linear RL δ (results/residual)", R.ROOT / "results/residual"),
                    ("linear MPC+Kv²κ+RL (results/ff_residual)", R.ROOT / "results/ff_residual")):
        for track in R.TRACKS:
            sp = p / "sweep.csv" if track == "Austin" else p / "heldout" / track / "sweep.csv"
            if not sp.exists():
                sp = R.ROOT / "results/heldout" / track / "sweep.csv" if "results/residual" in str(p) else sp
            if not sp.exists():
                continue
            d = pd.read_csv(sp)
            d = d[d.controller == "residual"]
            rows.append(dict(run=name, track=track, qual=int(d.qualifying.sum()), n=len(d),
                             **d.status.value_counts().to_dict()))
    lc = pd.DataFrame(rows).fillna(0)
    lc.to_csv(A / "linear_context.csv", index=False)
    return lc


if __name__ == "__main__":
    inventory()
    sw = all_sweeps()
    sw.to_csv(A / "all_sweeps.csv", index=False)
    consistency(sw)
    sc = status_counts(sw)
    fl = fastest_and_search()
    ts = tolerance_sensitivity(sw)
    pc = plan_comparison()
    lc = linear_context()
    (A / "consistency.txt").write_text("\n".join(LOG) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sc[sc.track == "ALL"].to_string(index=False))
        print(ts.to_string())
        print(fl.to_string(index=False))
        print(pc.to_string(index=False))
        print(lc.to_string(index=False))


def fig_status():
    """Stacked outcome counts of the 18 sweep laps per track for the main Pacejka controllers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = pd.read_csv(A / "status_counts.csv")
    ctrls = ["P+ff (std)", "P-E+ff (ell)", "S-E", "RL δ", "RL δa", "RL δa+wall"]
    cols = {"qualifying": "#2e7d32", "speed_tolerance": "#fdd835", "lateral_error": "#8d6e63", "wall_collision": "#1565c0",
            "spin": "#c62828", "off_track": "#9e9e9e", "timeout": "#000000", "solver_error": "#7b1fa2"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    seen = set()
    for ax, trk in zip(axes, R.TRACKS):
        g = sc[(sc.track == trk)].set_index("ctrl").reindex(ctrls)
        bottom = np.zeros(len(ctrls))
        for s in R.STATUSES:
            v = g[s].fillna(0).values
            if v.sum() == 0:
                continue
            ax.bar(range(len(ctrls)), v, bottom=bottom, color=cols[s], label=None if s in seen else s)
            seen.add(s)
            bottom += v
        ax.set_xticks(range(len(ctrls)), ctrls, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{trk}{' (training)' if trk == 'Austin' else ''}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("sweep laps (of 18)")
    fig.legend(fontsize=7, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(A / "fig_sweep_outcomes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_status()
