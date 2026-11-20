"""Validation of the acceleration-residual scale alpha (wall-aware dual-channel policy, frozen checkpoints).

    uv run python results/analysis/ares_validation/validate.py compare   # 1. 1.25M/a=1 vs 2.5M/a=0.75, saved laps only
    uv run python results/analysis/ares_validation/validate.py fine      # 2. finer alpha grid, 504 laps (~1 min)
    uv run python results/analysis/ares_validation/validate.py freeze    # 3a. pick candidates from the 72 sweep cells only
    uv run python results/analysis/ares_validation/validate.py fresh     # 3b. frozen candidates on a fresh grid (~2 min)
    uv run python results/analysis/ares_validation/validate.py window    # 4. windowed-alpha diagnostic, 5 cells
    uv run python results/analysis/ares_validation/validate.py figure

Same lap loop as results/analysis/ares_sweep (ablate.run_case, variant "scale"): the policy's acceleration residual
is replaced by alpha * a_res before it enters the speed command. Window runs use alpha inside a window of arc length
and alpha_out elsewhere; before the window opens the lap is identical to the alpha_out lap (checked).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
AN = HERE.parent
for p in (AN / "ares_sweep", AN / "followup", AN):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablate as A  # noqa: E402
import ares_sweep as S  # noqa: E402
import rlana as R  # noqa: E402

KEY = ["track", "a_lat", "mu_scale", "start_s"]
FINE = {"1.25M": [0.80, 0.85, 0.90, 0.95], "2.5M": [0.70, 0.80, 0.85]}
EY_MAX, RATIO_MAX = 0.5, 1.05
# fresh grid: sweep a_lat levels, start poses and friction values not used by the sweep (s 0/200, mu 1.0/0.8/0.6),
# by checkpoint selection (Austin a_lat 5/7, mu 1.0/0.7, s 100: avoided by keeping a_lat 4/6/8) or by alpha choice
FRESH = dict(a_lat=[4.0, 6.0, 8.0], mu_scale=[1.0, 0.9, 0.7], start_s=[100.0, 300.0])


def reason(status):
    return {"qualifying": "qualifies", "wall_collision": "crash (wall)", "spin": "crash (spin)",
            "speed_tolerance": "5% lap-time rule", "lateral_error": "0.5 m tracking limit"}.get(status, status)


def margins(d, sfx=""):
    """Margin to each qualifying threshold (positive = inside the limit)."""
    return dict(**{f"ratio_margin{sfx}": RATIO_MAX - d[f"lap_time_ratio{sfx}"], f"ey_margin{sfx}": EY_MAX - d[f"max_ey{sfx}"]})


def run(jobs, name=None):
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(S.run_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df["completed"] = df.outcome == "completed"
    if name:
        df.to_csv(HERE / f"{name}.csv", index=False, float_format="%.6g")
    return df


def coarse():
    return pd.read_csv(AN / "ares_sweep/per_cell.csv")


# ------------------------------------------------------------------------------------------------ 1. compare
def compare():
    df = coarse()
    a = df[(df.ck == "1.25M") & (df.alpha == 1.0)].set_index(KEY)
    b = df[(df.ck == "2.5M") & (df.alpha == 0.75)].set_index(KEY)
    cols = ["status", "qualifying", "completed", "lap_time", "lap_time_ratio", "max_ey", "fail_s", "min_wall_gap"]
    m = a[cols].join(b[cols], suffixes=("_A", "_B")).reset_index()
    m["group"] = np.select([m.qualifying_A & m.qualifying_B, m.qualifying_A, m.qualifying_B],
                           ["both qualify", "1.25M/α=1 only", "2.5M/α=0.75 only"], "neither")
    m["why_A"], m["why_B"] = m.status_A.map(reason), m.status_B.map(reason)
    for s in ("_A", "_B"):
        m = m.assign(**margins(m, s))
    m["dlap_B_minus_A"] = np.where(m.completed_A & m.completed_B, m.lap_time_B - m.lap_time_A, np.nan)
    m.to_csv(HERE / "compare_cells.csv", index=False, float_format="%.4f")

    rows = []
    for name, g in [("all", m)] + list(m.groupby("group")):
        d = g.dlap_B_minus_A.dropna()
        rows.append(dict(group=name, cells=len(g), jointly_completed=len(d), dlap_mean=d.mean(), dlap_median=d.median(),
                         dlap_min=d.min(), dlap_max=d.max()))
    s = pd.DataFrame(rows)
    s.to_csv(HERE / "compare_summary.csv", index=False, float_format="%.3f")
    x = m[m.group.isin(["1.25M/α=1 only", "2.5M/α=0.75 only"])]
    pd.set_option("display.width", 250)
    print(m.group.value_counts().to_string())
    print(pd.crosstab(m.why_A, m.why_B).to_string())
    print(x[KEY + ["group", "why_A", "why_B", "lap_time_ratio_A", "lap_time_ratio_B", "max_ey_A", "max_ey_B", "fail_s_A",
                   "fail_s_B", "dlap_B_minus_A"]].round(3).to_string(index=False))
    print(s.round(3).to_string(index=False))
    print(m.groupby("track").group.value_counts().unstack(fill_value=0).to_string())
    print(m.groupby(["a_lat", "mu_scale"]).group.value_counts().unstack(fill_value=0).to_string())


# ------------------------------------------------------------------------------------------------ 2. fine sweep
def fine():
    jobs = [S.job(ck, t, a, mu, s0, al) for ck, als in FINE.items() for t in R.TRACKS
            for a, mu, s0 in R.sweep_cells(R.load_cfg("wall", t)) for al in als]
    run(jobs, "fine_per_cell")
    fine_analyze()


def all_alpha():
    df = pd.concat([coarse(), pd.read_csv(HERE / "fine_per_cell.csv")], ignore_index=True)
    return df.sort_values(["ck"] + KEY + ["alpha"]).reset_index(drop=True)


def tradeoff(df, ref_alpha=1.0, ref=None):
    """Totals per (ck, alpha) relative to the checkpoint's alpha = 1 laps on the same cells."""
    k = ["ck"] + KEY
    ref = (df if ref is None else ref)
    r1 = ref[ref.alpha == ref_alpha].set_index(k)
    r0 = ref[ref.alpha == 0.0].set_index(k) if (ref.alpha == 0.0).any() else None
    rows = []
    for (ck, al), g in df.groupby(["ck", "alpha"]):
        g = g.set_index(k)
        r = r1.loc[g.index]
        both = g.completed & r.completed
        dl = (g.lap_time - r.lap_time)[both]
        q = g[g.qualifying]
        row = dict(ck=ck, alpha=al, cells=len(g), qualifying=int(g.qualifying.sum()), completed=int(g.completed.sum()),
                   crash=int(g.status.isin(["wall_collision", "spin"]).sum()),
                   too_slow=int((g.status == "speed_tolerance").sum()), ey_limit=int((g.status == "lateral_error").sum()),
                   q_gained_vs_1=int((g.qualifying & ~r.qualifying).sum()), q_lost_vs_1=int((~g.qualifying & r.qualifying).sum()),
                   finishes_retained=int((r.completed & g.completed).sum()), finishes_at_1=int(r.completed.sum()),
                   paired_n=int(both.sum()), dlap_mean=dl.mean(), dlap_min=dl.min(), dlap_max=dl.max(),
                   ratio_mean_completed=g.lap_time_ratio[g.completed].mean(),
                   q_within_0p5pt_of_5pct=int((q.lap_time_ratio > RATIO_MAX - 0.005).sum()),
                   q_within_2cm_of_ey=int((q.max_ey > EY_MAX - 0.02).sum()))
        if r0 is not None:
            resc = r.completed & ~r0.loc[g.index].completed
            row.update(rescued_subset=int(resc.sum()), rescued_kept=int((resc & g.completed).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def fine_analyze():
    df = all_alpha()
    t = tradeoff(df)
    t.to_csv(HERE / "fine_tradeoff.csv", index=False, float_format="%.4f")
    # cells whose qualifying result differs from the checkpoint's alpha = 1 at any alpha >= 0.5
    rows = []
    for (ck, *k), g in df[df.alpha >= 0.5].groupby(["ck"] + KEY):
        g = g.sort_values("alpha")
        if g.qualifying.nunique() == 1 and g.status.nunique() == 1:
            continue
        for _, r in g.iterrows():
            rows.append(dict(ck=ck, **dict(zip(KEY, k)), alpha=r.alpha, status=r.status, why=reason(r.status),
                             lap_time=r.lap_time, lap_time_ratio=r.lap_time_ratio, ratio_margin=RATIO_MAX - r.lap_time_ratio,
                             max_ey=r.max_ey, ey_margin=EY_MAX - r.max_ey, fail_s=r.fail_s, min_wall_gap=r.min_wall_gap))
    ch = pd.DataFrame(rows)
    ch.to_csv(HERE / "fine_changed_cells.csv", index=False, float_format="%.4f")
    wide = ch.assign(st=ch.status.str.replace("wall_collision", "wall").str.replace("speed_tolerance", "slow")
                     .str.replace("qualifying", "Q").str.replace("lateral_error", "ey")) \
        .pivot_table(index=["ck"] + KEY, columns="alpha", values="st", aggfunc="first")
    wide.to_csv(HERE / "fine_changed_cells_wide.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t[t.alpha >= 0.5].round(3).to_string(index=False))
    print(wide.fillna("").to_string())


# ------------------------------------------------------------------------------------------------ 3. freeze + fresh
RULE = ("Per checkpoint, over all tested alpha (coarse + fine) on the 72 sweep cells only: maximise qualifying laps; "
        "ties -> more completed laps; remaining ties -> the alpha closest to the middle of the contiguous alpha range "
        "that is within 1 qualifying lap of the maximum (robustness). For 1.25M the best alpha < 1 under the same "
        "rule is also kept as a challenger. Reference settings evaluated alongside: 1.25M/alpha=1 (published baseline) "
        "and 2.5M/alpha=1.")


def pick(t):
    t = t.sort_values("alpha").reset_index(drop=True)
    best = t.qualifying.max()
    near = t[t.qualifying >= best - 1].alpha.values
    # contiguous run (in the tested grid) around the best alpha
    top = t[t.qualifying == best]
    top = top[top.completed == top.completed.max()]
    idx = t.index[t.alpha.isin(top.alpha)]
    lo = hi = idx[0]
    while lo - 1 >= 0 and t.alpha[lo - 1] in near:
        lo -= 1
    while hi + 1 < len(t) and t.alpha[hi + 1] in near:
        hi += 1
    mid = 0.5 * (t.alpha[lo] + t.alpha[hi])
    a = float(top.alpha.iloc[np.argmin(np.abs(top.alpha.values - mid))])
    return a, dict(max_qualifying=int(best), plateau=[float(t.alpha[lo]), float(t.alpha[hi])])


def freeze():
    t = tradeoff(all_alpha())
    out = dict(frozen_at=datetime.datetime.now().isoformat(timespec="seconds"), rule=RULE, data="72 sweep cells",
               candidates={})
    for ck in ("1.25M", "2.5M"):
        a, info = pick(t[t.ck == ck])
        out["candidates"][f"{ck} best"] = dict(ck=ck, alpha=a, **info)
    a, info = pick(t[(t.ck == "1.25M") & (t.alpha < 1.0)])
    out["candidates"]["1.25M best alpha<1"] = dict(ck="1.25M", alpha=a, **info)
    out["reference"] = [dict(ck="1.25M", alpha=1.0), dict(ck="2.5M", alpha=1.0)]
    (HERE / "freeze.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def fresh_cells():
    return [(t, a, mu, s0) for t in R.TRACKS for a in FRESH["a_lat"] for mu in FRESH["mu_scale"] for s0 in FRESH["start_s"]]


def fresh():
    fz = json.loads((HERE / "freeze.json").read_text())
    settings = {(c["ck"], c["alpha"]) for c in fz["candidates"].values()} | {(r["ck"], r["alpha"]) for r in fz["reference"]}
    settings |= {("2.5M", 0.0), ("1.25M", 0.0)}  # needed to define "finishes that need a_res" on the fresh grid
    settings |= {("2.5M", 0.75)}  # the coarse-sweep pick, kept as a reference
    jobs = [S.job(ck, *c, al) for ck, al in sorted(settings) for c in fresh_cells()]
    print(f"{len(jobs)} laps, settings {sorted(settings)}")
    run(jobs, "fresh_per_cell")
    fresh_analyze()


def fresh_analyze():
    fz = json.loads((HERE / "freeze.json").read_text())
    df = pd.read_csv(HERE / "fresh_per_cell.csv")
    t = tradeoff(df)
    t.to_csv(HERE / "fresh_tradeoff.csv", index=False, float_format="%.4f")
    base = df[(df.ck == "1.25M") & (df.alpha == 1.0)].set_index(KEY)
    cols = ["status", "qualifying", "completed", "lap_time", "lap_time_ratio", "max_ey", "fail_s", "min_wall_gap"]
    rows, cells = [], []
    for name, c in list(fz["candidates"].items()) + [("2.5M alpha=0.75 (coarse pick)", dict(ck="2.5M", alpha=0.75)),
                                                     ("2.5M alpha=1", dict(ck="2.5M", alpha=1.0))]:
        x = df[(df.ck == c["ck"]) & (df.alpha == c["alpha"])].set_index(KEY)
        m = base[cols].join(x[cols], suffixes=("_A", "_B")).reset_index()
        m["setting"] = f"{c['ck']}/α={c['alpha']:g}"
        m["group"] = np.select([m.qualifying_A & m.qualifying_B, m.qualifying_A, m.qualifying_B],
                               ["both qualify", "baseline only", "candidate only"], "neither")
        m["why_A"], m["why_B"] = m.status_A.map(reason), m.status_B.map(reason)
        for s in ("_A", "_B"):
            m = m.assign(**margins(m, s))
        both = m.completed_A & m.completed_B
        d = (m.lap_time_B - m.lap_time_A)[both]
        m["dlap_B_minus_A"] = np.where(both, m.lap_time_B - m.lap_time_A, np.nan)
        cells.append(m)
        hd = (m.a_lat >= 6) & (m.mu_scale <= 0.9)
        rows.append(dict(setting=m.setting[0], frozen_as=name, qualifying=int(m.qualifying_B.sum()),
                         baseline_qualifying=int(m.qualifying_A.sum()),
                         both=int((m.group == "both qualify").sum()), candidate_only=int((m.group == "candidate only").sum()),
                         baseline_only=int((m.group == "baseline only").sum()), neither=int((m.group == "neither").sum()),
                         completed=int(m.completed_B.sum()), baseline_completed=int(m.completed_A.sum()),
                         crash=int(m.status_B.isin(["wall_collision", "spin"]).sum()),
                         too_slow=int((m.status_B == "speed_tolerance").sum()), ey_limit=int((m.status_B == "lateral_error").sum()),
                         completed_but_miss=int((m.completed_B & ~m.qualifying_B).sum()),
                         paired_n=int(both.sum()), dlap_mean=d.mean(), dlap_median=d.median(), dlap_min=d.min(), dlap_max=d.max(),
                         q_high_demand=int(m.qualifying_B[hd].sum()), base_q_high_demand=int(m.qualifying_A[hd].sum()),
                         q_routine=int(m.qualifying_B[m.a_lat == 4].sum()), base_q_routine=int(m.qualifying_A[m.a_lat == 4].sum()),
                         q_within_0p5pt_of_5pct=int((m.qualifying_B & (m.lap_time_ratio_B > RATIO_MAX - 0.005)).sum()),
                         q_within_2cm_of_ey=int((m.qualifying_B & (m.max_ey_B > EY_MAX - 0.02)).sum())))
    s = pd.DataFrame(rows)
    s.to_csv(HERE / "fresh_vs_baseline.csv", index=False, float_format="%.3f")
    cc = pd.concat(cells, ignore_index=True)
    cc.to_csv(HERE / "fresh_cells_vs_baseline.csv", index=False, float_format="%.4f")
    by = df.groupby(["ck", "alpha", "track"]).qualifying.sum().unstack("track")
    by.to_csv(HERE / "fresh_by_track.csv")
    bd = df.assign(cls=np.where(df.a_lat == 4, "a_lat 4", np.where(df.mu_scale == 1.0, "a_lat ≥ 6, μ 1.0", "a_lat ≥ 6, μ ≤ 0.9"))) \
        .groupby(["ck", "alpha", "cls"]).qualifying.sum().unstack("cls")
    bd.to_csv(HERE / "fresh_by_demand.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).to_string(index=False))
    print(s.round(3).to_string(index=False))
    print(by.to_string())
    print(bd.to_string())
    x = cc[cc.group.isin(["baseline only", "candidate only"]) & (cc.setting != "2.5M/α=1")]
    print(x[["setting"] + KEY + ["group", "why_A", "why_B", "lap_time_ratio_A", "lap_time_ratio_B", "max_ey_A", "max_ey_B",
                                 "fail_s_A", "fail_s_B"]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ 4. windows
# The five 2.5M cells that qualify only at alpha = 0.75 in the coarse sweep. At alpha = 0.5 each hits the wall at
# a corner exit (fail_s from ares_sweep/per_cell.csv, alpha 0.5). Windows are fixed relative to that point before
# any window run: entry [f-35, f-15), exit [f-15, f+5), corner = entry + exit, far control [f-95, f-55).
FIVE = [("Austin", 8.0, 0.8, 0.0), ("Austin", 8.0, 0.8, 200.0), ("Spielberg", 4.0, 0.6, 0.0),
        ("Spielberg", 6.0, 1.0, 0.0), ("Spielberg", 6.0, 1.0, 200.0)]
WINDOWS = {"entry": (-35, -15), "exit": (-15, 5), "corner": (-35, 5), "far": (-95, -55)}
# (name, alpha_out, alpha_in, window)
WRUNS = [("0.5 → 0.75 in " + w, 0.5, 0.75, w) for w in WINDOWS] + \
        [("0.5 → 1.0 in " + w, 0.5, 1.0, w) for w in ("entry", "exit", "corner")] + \
        [("0.75 → 0.5 in " + w, 0.75, 0.5, w) for w in ("corner", "far")]


def _wjob(cell, alpha_out, alpha_in, win, f, L):
    t, a, mu, s0 = cell
    j = S.job("2.5M", t, a, mu, s0, alpha_in)
    if win is not None:
        lo, hi = WINDOWS[win]
        j.update(window=((f + lo) % L, (f + hi) % L), alpha_out=alpha_out)
    return j


def _wrun(j):
    sm, tr = A.run_case(j)
    gap, _ = A._gaps(tr, j["track"])
    tr = tr.assign(gap=gap)
    return sm, tr[["t", "s", "x", "y", "v", "e_y", "active", "a_res_app", "gap"]]


def window():
    from pilot.closedloop import Track

    pc = coarse()
    jobs, meta = [], []
    for cell in FIVE:
        t, a, mu, s0 = cell
        L = Track(R.load_cfg("wall", t)).ref.length
        r = pc[(pc.ck == "2.5M") & (pc.track == t) & (pc.a_lat == a) & (pc.mu_scale == mu) & (pc.start_s == s0)
               & (pc.alpha == 0.5)].iloc[0]
        f = float(r.fail_s) % L
        for al in (0.5, 0.75, 1.0):  # uniform references (= coarse sweep laps)
            jobs.append(_wjob(cell, al, al, None, f, L))
            meta.append(dict(cell=cell, run=f"uniform {al:g}", alpha_out=al, alpha_in=al, window=None, f=f, L=L))
        for name, ao, ai, w in WRUNS:
            jobs.append(_wjob(cell, ao, ai, w, f, L))
            meta.append(dict(cell=cell, run=name, alpha_out=ao, alpha_in=ai, window=w, f=f, L=L))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_wrun, jobs))
    ref = {(m["cell"], m["alpha_out"]): tr for m, (sm, tr) in zip(meta, res) if m["window"] is None}
    rows = []
    for m, (sm, tr) in zip(meta, res):
        t, a, mu, s0 = m["cell"]
        f, L = m["f"], m["L"]
        # prefix identity: until the window first opens the lap must equal the alpha_out lap
        prefix_rows, prefix_maxdiff = 0, 0.0
        if m["window"] is not None:
            r0 = ref[(m["cell"], m["alpha_out"])]
            k = int(np.argmax(tr.active.values)) if tr.active.any() else len(tr)
            k = min(k, len(r0))
            prefix_rows = k
            prefix_maxdiff = float(np.max(np.abs(tr[["x", "y", "v"]].values[:k] - r0[["x", "y", "v"]].values[:k]))) if k else 0.0
        # the failure zone [f-15, f+5) on the first pass
        ds = (np.mod(tr.s.values, L) - f + L / 2) % L - L / 2
        z = (ds >= -15) & (ds < 5)
        zi = np.flatnonzero(z)
        e = np.flatnonzero((ds >= -35) & (ds < -34))
        rows.append(dict(track=t, a_lat=a, mu_scale=mu, start_s=s0, f=round(f, 2), run=m["run"], alpha_out=m["alpha_out"],
                         alpha_in=m["alpha_in"], window=m["window"], status=sm["status"], qualifying=sm["qualifying"],
                         lap_time=sm.get("lap_time"), lap_time_ratio=sm.get("lap_time_ratio"), max_ey=sm.get("max_ey"),
                         fail_s=sm.get("fail_s"), v_corner_start=float(tr.v.values[e[0]]) if len(e) else np.nan,
                         v_zone_mean=float(tr.v.values[zi].mean()) if len(zi) else np.nan,
                         min_gap_zone=float(tr.gap.values[zi].min()) if len(zi) else np.nan,
                         mean_a_res_corner=float(tr.a_res_app.values[(ds >= -35) & (ds < 5)].mean()),
                         prefix_rows=prefix_rows, prefix_maxdiff=prefix_maxdiff))
    w = pd.DataFrame(rows)
    w.to_csv(HERE / "window_diagnostic.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(w.drop(columns=["f"]).round(3).to_string(index=False))
    print("prefix rows checked:", int(w.prefix_rows.sum()), " max |diff| in x, y, v:", w.prefix_maxdiff.max())


# ------------------------------------------------------------------------------------------------ figure
def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.read_csv(HERE / "fine_tradeoff.csv")
    fz = json.loads((HERE / "freeze.json").read_text())
    fr = pd.read_csv(HERE / "fresh_tradeoff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ck in zip(axes, ("1.25M", "2.5M")):
        g = t[(t.ck == ck) & (t.alpha >= 0.5)]
        ax.plot(g.alpha, g.qualifying, "o-", color="#2e7d32", label="qualifying, 72 sweep cells")
        ax.plot(g.alpha, g.completed, "s-", color="#1565c0", label="completed, 72 sweep cells")
        h = fr[(fr.ck == ck) & (fr.alpha >= 0.5)]
        ax.plot(h.alpha, h.qualifying, "o", mfc="none", mec="#2e7d32", ms=9, mew=1.5,
                label="qualifying, 72 fresh cells")
        ax.plot(h.alpha, h.completed, "s", mfc="none", mec="#1565c0", ms=9, mew=1.5, label="completed, 72 fresh cells")
        for c in fz["candidates"].values():
            if c["ck"] == ck:
                ax.axvline(c["alpha"], color="0.5", ls=":", lw=1)
        ax.set_xlabel("α (a_res scale)")
        ax.set_ylabel("laps")
        ax.set_title(f"ckpt {ck} (dotted: frozen candidate)", fontsize=10)
        ax.set_xlim(0.47, 1.03)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(g.alpha, g.dlap_mean, "k-", lw=0.8, alpha=0.6)
        ax2.set_ylabel("paired lap-time change vs α=1 [s] (grey)", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    {"compare": compare, "fine": fine, "fine_analyze": fine_analyze, "freeze": freeze, "fresh": fresh,
     "fresh_analyze": fresh_analyze, "window": window, "figure": figure}[sys.argv[1]]()
