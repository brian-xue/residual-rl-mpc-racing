"""Closed-loop sweep of the acceleration-residual strength of the wall-aware dual-channel policy.

    uv run python results/analysis/ares_sweep/ares_sweep.py smoke     # 18 laps; checks alpha 0 / 1 vs saved a0 / orig
    uv run python results/analysis/ares_sweep/ares_sweep.py sweep     # 720 laps (~2-3 min on 8 workers)
    uv run python results/analysis/ares_sweep/ares_sweep.py analyze

Uses the lap loop of results/analysis/followup/ablate.py (variant "scale"): the policy's acceleration residual is
replaced by alpha * a_res at the same point where the a0 ablation sets it to 0, i.e. before it is added to the base
speed command (plan feedforward + P) and before the usual actuator limit. The steering residual, observation
(previous-action input = the applied residual, as in a0), speed plan and simulator settings are unchanged.
alpha = 1 is bit-identical to the policy as evaluated; alpha = 0 to the a0 ablation.
"""

from __future__ import annotations

import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "followup"))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablate as A  # noqa: E402
import rlana as R  # noqa: E402

CKPTS = A.CKPTS  # {"1.25M": ckpt_1250000.zip, "2.5M": ckpt_2500000.zip}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SAVED = HERE.parent / "followup/ablation"


def job(ck, track, a_lat, mu, s0, alpha):
    return dict(track=track, ckpt=CKPTS[ck], a_lat=a_lat, mu=mu, start_s=s0, variant="scale", alpha=alpha,
                window=None, bias=0.0, seg="full", ck=ck)


def run_one(j):
    sm, tr = A.run_case(j)
    gap, _ = A._gaps(tr, j["track"])
    keep = ("outcome", "status", "qualifying", "lap_time", "plan_lap_time", "lap_time_ratio", "fail_s", "progress_frac",
            "rms_ey", "max_ey", "mean_abs_ares")
    return dict(ck=j["ck"], track=j["track"], a_lat=j["a_lat"], mu_scale=j["mu"], start_s=j["start_s"], alpha=j["alpha"],
                **{k: sm.get(k) for k in keep}, min_wall_gap=float(np.min(gap)),
                min_wall_gap_tight=float(np.min(gap[A._gaps(tr, j["track"])[1] < 0.4])) if (A._gaps(tr, j["track"])[1] < 0.4).any() else np.nan,
                mean_a_res_applied=float(tr.a_res_app.mean()), max_abs_slip=float(tr.slip.abs().max()))


def run(jobs, name):
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df["completed"] = df.outcome == "completed"
    df.to_csv(HERE / f"{name}.csv", index=False, float_format="%.6g")
    return df


def saved():
    """Saved orig / a0 full-lap results of the earlier ablation (paired_full_a0.csv: orig columns, *_v = a0)."""
    a = pd.read_csv(SAVED / "paired_full_a0.csv")
    key = ["ck", "track", "a_lat", "mu_scale", "start_s"]
    o = a[key + ["status", "lap_time", "fail_s"]].assign(alpha=1.0)
    z = a[key + ["status_v", "lap_time_v", "fail_s_v"]].rename(columns={"status_v": "status", "lap_time_v": "lap_time",
                                                                        "fail_s_v": "fail_s"}).assign(alpha=0.0)
    return pd.concat([o, z], ignore_index=True)


def check(df):
    """alpha = 0 / 1 against the saved a0 / orig: status, lap time and failure location must match exactly."""
    key = ["ck", "track", "a_lat", "mu_scale", "start_s", "alpha"]
    m = df[df.alpha.isin([0.0, 1.0])].merge(saved(), on=key, suffixes=("", "_saved"))
    m["match"] = (m.status == m.status_saved) & (
        (m.lap_time.isna() & m.lap_time_saved.isna()) | np.isclose(m.lap_time, m.lap_time_saved, atol=1e-9)) & (
        (m.fail_s.isna() & m.fail_s_saved.isna()) | np.isclose(m.fail_s, m.fail_s_saved, atol=1e-9))
    return m


def smoke():
    cells = [("Austin", 4.0, 1.0, 0.0),    # routine finish (both checkpoints finish with and without a_res)
             ("Austin", 8.0, 0.8, 0.0),    # 1.25M finishes only with a_res (a0: wall collision)
             ("Austin", 4.0, 1.0, 200.0)]  # 2.5M finishes too slowly with a_res, qualifies at a0
    jobs = [job(ck, *c, a) for ck in CKPTS for c in cells for a in (0.0, 0.5, 1.0)]
    df = run(jobs, "smoke")
    m = check(df)
    pd.set_option("display.width", 250)
    print(df[["ck", "track", "a_lat", "mu_scale", "start_s", "alpha", "status", "lap_time", "lap_time_ratio", "fail_s",
              "min_wall_gap"]].to_string(index=False))
    print(m[["ck", "track", "a_lat", "mu_scale", "start_s", "alpha", "status", "status_saved", "lap_time", "lap_time_saved",
             "match"]].to_string(index=False))
    print("all alpha 0/1 cases match the saved a0/orig laps:", bool(m.match.all()), f"({int(m.match.sum())}/{len(m)})")


def sweep():
    jobs = [job(ck, track, a, mu, s0, al) for ck in CKPTS for track in R.TRACKS
            for a, mu, s0 in R.sweep_cells(R.load_cfg("wall", track)) for al in ALPHAS]
    df = run(jobs, "per_cell")
    m = check(df)
    m.to_csv(HERE / "reproduction_check.csv", index=False)
    print(f"{len(df)} laps; alpha 0/1 reproduce saved a0/orig: {int(m.match.sum())}/{len(m)}")


if __name__ == "__main__":
    {"smoke": smoke, "sweep": sweep}[sys.argv[1]]() if sys.argv[1] != "analyze" else None


# ------------------------------------------------------------------------------------------------ analysis
def analyze():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(HERE / "per_cell.csv")
    key = ["ck", "track", "a_lat", "mu_scale", "start_s"]
    ref = df[df.alpha == 1.0].set_index(key)
    zero = df[df.alpha == 0.0].set_index(key)
    rescued = ref.completed & ~zero.completed  # finishes only with the full residual
    rows, cell_rows = [], []
    for (ck, al), g in df.groupby(["ck", "alpha"]):
        g = g.set_index(key)
        r = ref.loc[g.index]
        both = g.completed & r.completed
        dl = (g.lap_time - r.lap_time)[both]
        resc = rescued.loc[g.index]
        rows.append(dict(
            ck=ck, alpha=al, completed=int(g.completed.sum()), qualifying=int(g.qualifying.sum()),
            too_slow=int((g.status == "speed_tolerance").sum()), wall=int((g.status == "wall_collision").sum()),
            spin=int((g.status == "spin").sum()), lateral=int((g.status == "lateral_error").sum()),
            q_gained_vs_1=int((g.qualifying & ~r.qualifying).sum()), q_lost_vs_1=int((~g.qualifying & r.qualifying).sum()),
            finishes_lost_vs_1=int((r.completed & ~g.completed).sum()), finishes_gained_vs_1=int((~r.completed & g.completed).sum()),
            rescued_subset=int(resc.sum()), rescued_preserved=int((resc & g.completed).sum()),
            rescued_qualifying=int((resc & g.qualifying).sum()),
            paired_n=int(both.sum()), dlap_mean=float(dl.mean()) if len(dl) else np.nan,
            dlap_median=float(dl.median()) if len(dl) else np.nan, dlap_min=float(dl.min()) if len(dl) else np.nan,
            dlap_max=float(dl.max()) if len(dl) else np.nan,
            ratio_mean_completed=float(g.lap_time_ratio[g.completed].mean()),
            dgap_tight_mean=float((g.min_wall_gap_tight - r.min_wall_gap_tight)[both].mean()),
            dslip_mean=float((g.max_abs_slip - r.max_abs_slip)[both].mean())))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "tradeoff_summary.csv", index=False, float_format="%.4f")

    # breakdowns: qualifying / completion per alpha by track, a_lat, mu
    for by in ("track", "a_lat", "mu_scale"):
        b = df.groupby(["ck", by, "alpha"]).agg(qual=("qualifying", "sum"), done=("completed", "sum"), n=("completed", "size"))
        b.unstack("alpha").to_csv(HERE / f"by_{by}.csv")
    hd = df.assign(cls=np.where((df.a_lat >= 6) & (df.mu_scale <= 0.8), "high demand & low grip (a_lat ≥ 6, μ ≤ 0.8)",
                                np.where(df.a_lat >= 6, "high demand (a_lat ≥ 6, μ 1.0)", "a_lat 4")))
    hb = hd.groupby(["ck", "cls", "alpha"]).agg(qual=("qualifying", "sum"), done=("completed", "sum"), n=("completed", "size"))
    hb.unstack("alpha").to_csv(HERE / "by_demand_class.csv")

    # per-cell best alpha (oracle) -> which cells need an intermediate or cell-specific scale
    for (ck, *k), g in df.groupby(["ck", "track", "a_lat", "mu_scale", "start_s"]):
        g = g.sort_values("alpha")
        q = g[g.qualifying]
        c = g[g.completed]
        cell_rows.append(dict(ck=ck, track=k[0], a_lat=k[1], mu_scale=k[2], start_s=k[3],
                              status_by_alpha=" / ".join(g.status.str.replace("wall_collision", "wall").str.replace(
                                  "speed_tolerance", "slow").str.replace("qualifying", "Q").str.replace("lateral_error", "lat")),
                              qual_alphas=",".join(f"{a:g}" for a in q.alpha), done_alphas=",".join(f"{a:g}" for a in c.alpha),
                              min_alpha_finish=c.alpha.min() if len(c) else np.nan,
                              max_alpha_qualify=q.alpha.max() if len(q) else np.nan,
                              min_alpha_qualify=q.alpha.min() if len(q) else np.nan))
    cells = pd.DataFrame(cell_rows)
    def cat(r):
        qa = [float(x) for x in r.qual_alphas.split(",")] if r.qual_alphas else []
        if not qa:
            return "never qualifies"
        if len(qa) == len(ALPHAS):
            return "qualifies at every alpha"
        lo, hi = min(qa), max(qa)
        contiguous = [a for a in ALPHAS if lo <= a <= hi] == qa
        if not contiguous:
            return "non-monotone"
        if hi == 1.0:
            return f"qualifies only for alpha >= {lo:g}"
        if lo == 0.0:
            return f"qualifies only for alpha <= {hi:g}"
        return f"qualifies only for {lo:g} <= alpha <= {hi:g}"

    cells["category"] = cells.apply(cat, axis=1)
    cells.to_csv(HERE / "per_cell_by_alpha.csv", index=False)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ck in zip(axes, CKPTS):
        g = t[t.ck == ck]
        ax.plot(g.alpha, g.qualifying, "o-", color="#2e7d32", label="qualifying (of 72)")
        ax.plot(g.alpha, g.completed, "s-", color="#1565c0", label="completed (of 72)")
        ax.plot(g.alpha, g.too_slow, "^:", color="#f9a825", label="completed but too slow")
        rs = int(g.rescued_subset.iloc[0])
        ax.plot(g.alpha, g.rescued_preserved, "d--", color="#c62828", label=f"finishes kept of the {rs} that need a_res")
        ax.set_xlabel("α (a_res scale)")
        ax.set_ylabel("laps")
        ax.set_title(f"ckpt {ck}", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(g.alpha, g.dlap_mean, "k-", lw=0.8, alpha=0.6)
        for x, y, n in zip(g.alpha, g.dlap_mean, g.paired_n):
            ax2.annotate(f"{y:+.2f} s\n(n={n})", (x, y), fontsize=6, ha="center", va="bottom", alpha=0.7)
        ax2.set_ylabel("paired lap-time change vs α=1 [s] (grey)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_tradeoff.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).to_string(index=False))
    print(hb.unstack("alpha").to_string())
    print(pd.read_csv(HERE / "by_track.csv", header=[0, 1], index_col=[0, 1]).to_string())
    print(cells.groupby(["ck", "category"]).size().unstack(fill_value=0).to_string())
    print(cells[~cells.category.isin(["never qualifies", "qualifies at every alpha"])].to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "analyze":
    analyze()
