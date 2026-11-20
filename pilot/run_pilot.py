"""Feasibility pilot: sim checks -> conservative lap -> K tuning -> paired sweep -> recheck -> outputs.

Usage:  uv run python -m pilot.run_pilot [--config configs/pilot.yaml] [--out DIR]   (DIR defaults to the config's `out`)
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import sim_checks
from . import animate, cli, lap_time
from .closedloop import Scenario, Track, qualifies, run_lap, run_many
from .drawing import draw_track

CTRL = {"nominal": "Nominal MPC", "ff": "MPC + K v² κ"}


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for k, v in d.items()}


def common_interval(tr_a, tr_b):
    """RMS / max |e_y| of both traces over their common progress interval."""
    p = min(tr_a.progress.max(), tr_b.progress.max())
    out = {"common_progress_m": float(p)}
    for name, tr in (("nominal", tr_a), ("ff", tr_b)):
        seg = tr[tr.progress <= p]
        out[f"{name}_rms_ey_common"] = float(np.sqrt((seg.e_y**2).mean()))
        out[f"{name}_mean_v_common"] = float(seg.v.mean())
    return out


def run_pairs(cfg, scns, K, tag, rows, traces):
    """Run nominal and feedforward on every scenario (in parallel); same scenario for both controllers."""
    jobs = [(scn, k, True, None) for scn in scns for k in (0.0, K)]
    results = run_many(cfg, jobs)
    out = {}
    for i, scn in enumerate(scns):
        res = {}
        for j, (ctrl, k) in enumerate((("nominal", 0.0), ("ff", K))):
            summ, tr = results[2 * i + j]
            ok, reason = qualifies(summ, cfg["qualifying"])
            summ = _clean(summ)
            summ.update(controller=ctrl, qualifying=ok, status=reason, phase=tag)
            rows.append(summ)
            traces[(tag, scn.a_lat, scn.mu_scale, scn.start_s, ctrl)] = tr
            res[ctrl] = (summ, tr)
        ci = common_interval(res["nominal"][1], res["ff"][1])
        for ctrl in ("nominal", "ff"):
            res[ctrl][0].update(ci)
        out[(scn.a_lat, scn.mu_scale)] = res
    return out


def plot_cell(track, cfg, key_nom, key_ff, traces, summ_by_ctrl, path):
    ref = track.ref
    q = cfg["qualifying"]
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
    ax_e, ax_v, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
    ax_m = fig.add_subplot(gs[:, 1])
    colors = {"nominal": "#c0392b", "ff": "#1f6fb4"}
    for ctrl, key in (("nominal", key_nom), ("ff", key_ff)):
        tr, sm = traces[key], summ_by_ctrl[ctrl]
        lab = f"{CTRL[ctrl]} ({sm['status']}, max|e_y|={sm['max_ey']:.2f} m)"
        ax_e.plot(tr.progress, tr.e_y, color=colors[ctrl], lw=0.9, label=lab)
        ax_v.plot(tr.progress, tr.v, color=colors[ctrl], lw=0.9, label=f"{CTRL[ctrl]} v")
        ax_d.plot(tr.progress, tr.delta, color=colors[ctrl], lw=0.8, label=f"{CTRL[ctrl]} δ actual")
        if ctrl == "ff":
            ax_d.plot(tr.progress, tr.delta_ff, color=colors[ctrl], lw=0.8, ls=":", label="K v² κ term")
        if sm["outcome"] != "completed":
            for a in (ax_e, ax_v, ax_d):
                a.axvline(sm["progress_m"], color=colors[ctrl], ls="--", lw=1)
        x, y = tr.x.values, tr.y.values
        ax_m.plot(x, y, color=colors[ctrl], lw=0.9, zorder=4)
        if sm["outcome"] != "completed":
            ax_m.plot(x[-1], y[-1], "X", color=colors[ctrl], ms=11, mec="w", zorder=5)
        if sm.get("first_off_track_progress") is not None:
            j = int(np.searchsorted(tr.progress.values, sm["first_off_track_progress"]))
            ax_m.plot(x[min(j, len(x) - 1)], y[min(j, len(x) - 1)], "o", mfc="none", color=colors[ctrl], ms=12, mew=2, zorder=5)
            ax_e.axvline(sm["first_off_track_progress"], color=colors[ctrl], ls=":", lw=1)
    tr = max((traces[key_nom], traces[key_ff]), key=len)
    ax_v.plot(tr.progress, tr.v_plan, "k--", lw=0.8, label="speed plan")
    for a in (ax_e,):
        for sgn in (1, -1):
            a.axhline(sgn * q["max_abs_ey_m"], color="grey", ls=":", lw=1)
    ax_d.axhline(cfg["limits"]["s_max"], color="grey", ls=":", lw=1)
    ax_d.axhline(-cfg["limits"]["s_max"], color="grey", ls=":", lw=1)
    ax_e.set(ylabel="lateral error e_y [m]")
    ax_v.set(ylabel="speed [m/s]")
    ax_d.set(ylabel="steering δ [rad]", xlabel="track progress [m]")
    for a in (ax_e, ax_v, ax_d):
        a.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)
    draw_track(ax_m, track, lw_scale=0.6, show_ref=False)
    ax_m.set_title("path (X = wall collision, o = first off-track)", fontsize=9)
    sm = summ_by_ctrl["nominal"]
    fig.suptitle(f"{cfg['track']} | a_lat plan = {sm['a_lat']} m/s² | μ/μ₀ = {sm['mu_scale']} | start s = {sm['start_s']} m "
                 f"| K = {summ_by_ctrl['ff']['K']}  (grey dotted: |e_y| tolerance; dashed: wall collision; dotted: first off-track)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg, _, out = cli.parse(__doc__)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {"config": cfg}

    # 1. simulator verification
    print("[1/6] simulator checks", flush=True)
    report["sim_checks"] = sim_checks.run(cfg["track"], cfg["dt"], out / "sim_checks", model=cfg.get("model", "st"),
                                           tire=cfg.get("tire"))

    track = Track(cfg)
    report["reference_and_geometry"] = track.info
    rows, traces = [], {}

    # 2. conservative nominal lap
    print("[2/6] conservative nominal lap", flush=True)
    sw = cfg["sweep"]
    s, tr = run_lap(track, Scenario(sw["conservative_a_lat"], 1.0, sw["start_s"]), 0.0, cfg)
    ok, reason = qualifies(s, cfg["qualifying"])
    s = _clean(s)
    s.update(controller="nominal", qualifying=ok, status=reason, phase="conservative")
    rows.append(s)
    report["conservative"] = s

    # 3. tune K on development rollouts, then freeze
    print("[3/6] tuning K", flush=True)
    ff = cfg["feedforward"]
    tune = []
    jobs = [(Scenario(ff["dev_a_lat"], mu, ff["dev_start_s"]), K, False, None)
            for K in ff["K_grid"] for mu in ff["dev_mu_scales"]]
    res = run_many(cfg, jobs)
    nm = len(ff["dev_mu_scales"])
    for i, K in enumerate(ff["K_grid"]):
        rr = [r[0] for r in res[i * nm:(i + 1) * nm]]
        tune.append(dict(K=K, all_completed=all(r["outcome"] == "completed" for r in rr),
                         all_valid=all(r["outcome"] == "completed" and r["off_track_steps"] == 0 for r in rr),
                         mean_progress_frac=float(np.mean([r["progress_frac"] for r in rr])),
                         mean_rms_ey=float(np.mean([r["rms_ey"] for r in rr])),
                         mean_max_ey=float(np.mean([r["max_ey"] for r in rr]))))
        print(f"   K={K:.3f}  {tune[-1]}", flush=True)
    tdf = pd.DataFrame(tune)
    tdf.to_csv(out / "k_tuning.csv", index=False)
    # rule: most mean progress (all valid laps first), then lowest mean RMS lateral error
    best_k = tdf.sort_values(["all_valid", "mean_progress_frac", "mean_rms_ey"], ascending=[False, False, True]).iloc[0]
    K = float(best_k["K"])
    report["K_frozen"] = K
    print(f"   frozen K = {K}", flush=True)

    # 4. paired sweep (same scenario for both controllers in every cell)
    print("[4/6] paired sweep", flush=True)
    cells = run_pairs(cfg, [Scenario(a, mu, sw["start_s"]) for a in sw["a_lat_levels"] for mu in sw["mu_scales"]],
                      K, "sweep", rows, traces)
    for (a_lat, mu), res in cells.items():
        n, f = res["nominal"][0], res["ff"][0]
        print(f"   a_lat={a_lat} mu={mu}: nominal {n['status']} prog={n['progress_frac']:.2f} max|ey|={n['max_ey']:.3f} | "
              f"ff {f['status']} prog={f['progress_frac']:.2f} max|ey|={f['max_ey']:.3f}", flush=True)

    # 5. recheck promising cells from the second start pose
    print("[5/6] recheck promising cells", flush=True)
    promising = [k for k, r in cells.items() if (not r["nominal"][0]["qualifying"]) and r["ff"][0]["qualifying"]]
    report["promising_cells"] = [list(k) for k in promising]
    rechecks = run_pairs(cfg, [Scenario(a, mu, sw["recheck_start_s"]) for a, mu in promising], K, "recheck",
                         rows, traces) if promising else {}
    for (a_lat, mu), res in rechecks.items():
        n, f = res["nominal"][0], res["ff"][0]
        print(f"   a_lat={a_lat} mu={mu} start={sw['recheck_start_s']}: nominal {n['status']} | ff {f['status']}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["phase", "controller", "a_lat", "mu_scale", "start_s", "K", "status", "qualifying", "outcome",
            "progress_frac", "fail_s", "off_track_steps", "first_off_track_s", "lap_time", "plan_lap_time", "lap_time_ratio", "mean_speed", "mean_abs_dv",
            "rms_ey", "max_ey", "common_progress_m", "nominal_rms_ey_common", "ff_rms_ey_common",
            "sat_mag_frac", "sat_rate_frac", "solver_failures", "solve_ms_median", "solve_ms_p99", "solve_ms_max"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    # most informative cell: confirmed promising (both starts) > promising > largest progress gap (either
    # direction) > largest RMS gap
    def score(k):
        r = cells[k]
        conf = k in rechecks and (not rechecks[k]["nominal"][0]["qualifying"]) and rechecks[k]["ff"][0]["qualifying"]
        n, f = r["nominal"][0], r["ff"][0]
        return (conf, k in promising, abs(f["progress_frac"] - n["progress_frac"]), abs(n["rms_ey"] - f["rms_ey"]))

    best = max(cells, key=score)
    r = cells[best]
    key = lambda c: ("sweep", best[0], best[1], sw["start_s"], c)  # noqa: E731
    plot_cell(track, cfg, key("nominal"), key("ff"), traces, {c: r[c][0] for c in r}, out / "comparison.png")
    report["plotted_cell"] = list(best)
    for (tag, a, m, s0, c), tr in traces.items():
        if (a, m) == best:
            tr.to_csv(out / f"trace_{tag}_alat{a}_mu{m}_s{int(s0)}_{c}.csv", index=False, float_format="%.5f")

    print("[gif] rendering", flush=True)
    report["gif"] = animate.render(track, cfg, {c: traces[key(c)] for c in ("nominal", "ff")},
                                   {c: r[c][0] for c in r}, out / "lap.gif")

    # 6. lap-time comparison: each controller at its own fastest qualifying speed level
    print("[6/6] fastest qualifying lap per controller", flush=True)
    bdf, pick = lap_time.search(track, cfg, K, out)
    report["lap_time_comparison"] = bdf.to_dict(orient="records")
    if pick is not None:
        report["lap_time_gif"] = animate.render(track, cfg, pick["traces"], {c: _clean(v) for c, v in pick["summs"].items()},
                                                out / "lap_time.gif")
        report["lap_time_gif"].update(mu_scale=pick["mu_scale"], a_lat_nominal=pick["a_lat_nominal"], a_lat_ff=pick["a_lat_ff"])
    report["runtime_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['runtime_s']:.0f} s -> {out}/summary.csv, {out}/comparison.png", flush=True)


if __name__ == "__main__":
    main()
