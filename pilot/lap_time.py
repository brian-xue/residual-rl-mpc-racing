"""Best qualifying lap per controller: the speed level (a_lat) of the shared speed plan is raised
step by step until the controller no longer produces a qualifying lap. Each controller then runs at
its own fastest qualifying level, and the lap times are compared at equal friction.

A level counts only if the lap qualifies from BOTH predetermined start poses, and the best level is
the highest one with every slower level also qualifying. Each (controller, friction) search stops at its
first failing level; searches run in parallel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .closedloop import run_chains

NAMES = {"nominal": "Nominal MPC", "ff": "MPC + K v² κ"}
COLORS = {"nominal": "#c0392b", "ff": "#1f6fb4"}


def search(track, cfg, K, out):
    lc = cfg["lap_time"]
    starts = (cfg["sweep"]["start_s"], cfg["sweep"]["recheck_start_s"])
    rows, traces, summs = [], {}, {}
    combos = [(ctrl, mu, k) for mu in lc["mu_scales"] for ctrl, k in (("nominal", 0.0), ("ff", K))]
    chains = run_chains(cfg, [(k, mu, list(lc["a_lat_grid"]), starts, None, ctrl) for ctrl, mu, k in combos])
    for (ctrl, mu, _), chain in zip(combos, chains):
        for a_lat, s0, sm, tr in chain:
            rows.append({c: sm[c] for c in ("controller", "mu_scale", "a_lat", "start_s", "status", "qualifying",
                                            "progress_frac", "fail_s", "lap_time", "plan_lap_time",
                                            "lap_time_ratio", "max_ey", "rms_ey")})
            traces[(ctrl, mu, a_lat, s0)] = tr
            summs[(ctrl, mu, a_lat, s0)] = sm
        print(f"   mu={mu} {ctrl}: levels run = {sorted({c[0] for c in chain})}, last = "
              f"{chain[-1][2]['status']}", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "lap_time_grid.csv", index=False, float_format="%.4f")

    best = []
    for mu in lc["mu_scales"]:
        for ctrl in ("nominal", "ff"):
            ok_levels = []
            for a in lc["a_lat_grid"]:  # contiguous from the slowest level
                if not all((ctrl, mu, a, s) in summs and summs[(ctrl, mu, a, s)]["qualifying"] for s in starts):
                    break
                ok_levels.append(a)
            if ok_levels:
                a = max(ok_levels)
                best.append(dict(mu_scale=mu, controller=ctrl, best_a_lat=a,
                                 lap_time_s0=summs[(ctrl, mu, a, starts[0])]["lap_time"],
                                 lap_time_s1=summs[(ctrl, mu, a, starts[1])]["lap_time"],
                                 plan_lap_time=summs[(ctrl, mu, a, starts[0])]["plan_lap_time"],
                                 max_ey=max(summs[(ctrl, mu, a, s)]["max_ey"] for s in starts)))
            else:
                best.append(dict(mu_scale=mu, controller=ctrl, best_a_lat=None, lap_time_s0=None, lap_time_s1=None,
                                 plan_lap_time=None, max_ey=None))
    bdf = pd.DataFrame(best)
    piv = bdf.pivot(index="mu_scale", columns="controller", values="lap_time_s0")
    bdf["gap_s_vs_nominal"] = [
        (piv.loc[r.mu_scale, "nominal"] - r.lap_time_s0) if (r.controller == "ff" and r.lap_time_s0 is not None
                                                             and pd.notna(piv.loc[r.mu_scale, "nominal"])) else None
        for r in bdf.itertuples()]
    bdf.to_csv(out / "lap_time_comparison.csv", index=False, float_format="%.3f")

    # bar chart
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mus = list(lc["mu_scales"])
    w = 0.38
    for j, ctrl in enumerate(("nominal", "ff")):
        for i, mu in enumerate(mus):
            r = bdf[(bdf.mu_scale == mu) & (bdf.controller == ctrl)].iloc[0]
            x = i + (j - 0.5) * w
            if r.lap_time_s0 is None or pd.isna(r.lap_time_s0):
                ax.text(x, 1, "no\nqualifying\nlap", ha="center", va="bottom", fontsize=7)
                continue
            ax.bar(x, r.lap_time_s0, w, color=COLORS[ctrl], label=NAMES[ctrl] if i == 0 else None)
            ax.text(x, r.lap_time_s0 + 0.3, f"{r.lap_time_s0:.2f} s\na_lat {r.best_a_lat:g}", ha="center",
                    va="bottom", fontsize=7)
    lo = np.nanmin(pd.to_numeric(bdf.lap_time_s0, errors="coerce"))
    ax.set_ylim(lo - 6 if np.isfinite(lo) else 0, None)
    ax.set_xticks(range(len(mus)), [f"μ/μ₀ = {m}" for m in mus])
    ax.set_ylabel("best qualifying lap time [s] (start s = 0)")
    ax.set_title(f"{cfg['track']}: fastest qualifying lap per controller\n"
                 "(speed level raised until the lap no longer qualifies from both start poses)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lap_time_comparison.png", dpi=130)
    plt.close(fig)

    # GIF cell: friction with the largest gain where both controllers have a qualifying lap (ties -> lower mu)
    cand = []
    for mu in mus:
        n = bdf[(bdf.mu_scale == mu) & (bdf.controller == "nominal")].iloc[0]
        f = bdf[(bdf.mu_scale == mu) & (bdf.controller == "ff")].iloc[0]
        if pd.notna(n.best_a_lat) and pd.notna(f.best_a_lat):
            cand.append((n.lap_time_s0 - f.lap_time_s0, -mu, mu, n.best_a_lat, f.best_a_lat))
    pick = None
    if cand:
        _, _, mu, a_n, a_f = max(cand)
        pick = dict(mu_scale=mu, a_lat_nominal=a_n, a_lat_ff=a_f,
                    traces={"nominal": traces[("nominal", mu, a_n, starts[0])], "ff": traces[("ff", mu, a_f, starts[0])]},
                    summs={"nominal": summs[("nominal", mu, a_n, starts[0])], "ff": summs[("ff", mu, a_f, starts[0])]})
    return bdf, pick
