"""Part 2: paired closed-loop action ablations of the wall-aware policy (ckpt 1.25M vs a late checkpoint).

    uv run python results/analysis/followup/ablate.py run       # ~3 min on 8 workers
    uv run python results/analysis/followup/ablate.py analyze

Variants (analysis-only interventions on a frozen policy):
    orig     the policy as evaluated (bit-identical to pilot.closedloop.run_lap)
    a0       acceleration residual forced to 0
    d0       steering residual forced to 0
    debias   the checkpoint's mean Austin straight-line steering residual subtracted, only inside the
             pre-defined straight segments (|kappa_ref| < 0.01 over >= 15 m, fixed from geometry before any run)
Full laps: the variant is active over the whole lap. Segment tests: the variant is active only for
s in [branch, exit) of a pre-defined segment; before the branch point the lap is identical to `orig`
(checked), so simulator state, MPC warm start and policy history match at the branch.
The lap loop below is run_lap's loop (plan -> policy -> apply -> advance_interval), with logging of the
requested (policy mean, unclipped and clipped) and applied actions.
"""

from __future__ import annotations

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
OUT = HERE / "ablation"
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
CKPTS = {"1.25M": "ckpt_1250000.zip", "2.5M": "ckpt_2500000.zip"}
# pre-defined segments: (track, name, branch s, hotspot s, exit s) -- hotspots from the earlier failure analysis
SEGMENTS = [
    ("Austin", "A-T1 (control: corner RL passes)", 20.0, 50.0, 75.0),
    ("Spielberg", "S-283 tight straight", 245.0, 283.0, 295.0),
    ("Spielberg", "S-303 corner exit into wall", 285.0, 303.0, 318.0),
    ("Silverstone", "V-176 corner exit into wall", 140.0, 176.0, 190.0),
    ("Silverstone", "V-218 tight curve", 190.0, 218.0, 232.0),
]


def straight_segments(track, min_len=15.0, kmax=0.01):
    """Pre-defined straights: contiguous reference stretches with |kappa| < kmax of at least min_len."""
    ref = track.ref
    m = np.abs(ref.kappa) < kmax
    segs, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) * ref.ds >= min_len:
                segs.append((ref.s[i], ref.s[j - 1]))
            i = j
        else:
            i += 1
    return segs


_W: dict = {}


def _track(name):
    from pilot.closedloop import Track

    if name not in _W:
        _W[name] = Track(R.load_cfg("wall", name))
    return _W[name]


def run_case(job):
    """job: dict(track, ckpt, a_lat, mu, start_s, variant, window=(lo, hi) or None, bias)."""
    import torch

    from pilot.closedloop import Scenario
    from pilot.closedloop import LapSim
    from pilot.residual_env import residual_obs

    cfg = R.load_cfg("wall", job["track"])
    tr = _track(job["track"])
    L = tr.ref.length
    rc = cfg["residual"]
    model = R._model(str(R.EXPS["wall"][1] / job["ckpt"]))
    straights = straight_segments(tr) if job["variant"] == "debias" else []
    win = job.get("window")

    def active(s):
        if win is None:
            return True
        lo, hi = win
        return (s - lo) % L < (hi - lo) % L

    def in_straight(s):
        return any(a <= s <= b for a, b in straights)

    sim = LapSim(tr, Scenario(job["a_lat"], job["mu"], job["start_s"]), 0.0, cfg, log=True, residual_logged=True)
    prev = prev_a = 0.0
    rows = []
    while not sim.done:
        sim.plan()
        obs = residual_obs(sim, prev, rc["r_max"], prev_a, rc["accel_r_max"], True)
        with torch.no_grad():
            raw = model.policy.get_distribution(torch.as_tensor(obs)[None]).distribution.mean.numpy()[0]
        a, _ = model.predict(obs, deterministic=True)
        d_req, a_req_res = float(np.clip(a[0], -1, 1)) * rc["r_max"], float(np.clip(a[1], -1, 1)) * rc["accel_r_max"]
        d, da = d_req, a_req_res
        s_now = float(sim.s % L)
        on = active(s_now) and job["variant"] != "orig"
        if on and job["variant"] == "a0":
            da = 0.0
        elif on and job["variant"] == "scale":  # results/analysis/ares_sweep: a_res scaled by alpha
            da = job["alpha"] * da
        elif on and job["variant"] == "d0":
            d = 0.0
        elif on and job["variant"] == "debias" and in_straight(s_now):
            d = d - job["bias"]
        prev, prev_a = d, da
        sim.apply(d, da)
        v = sim.v
        dvds = (sim.vplan_at(sim.s + 0.2) - sim.vplan_at(sim.s - 0.2)) / 0.4
        a_long_req = float(v * dvds + sim.kv * (sim.vp - v) + da)
        rows.append(dict(t=sim.t, s=s_now, progress=sim.progress, active=on, raw_d=raw[0], raw_a=raw[1],
                         d_res_req=d_req, a_res_req=a_req_res, d_res_app=d, a_res_app=da,
                         steer_req=sim.d_mpc + sim.d_ff + d, steer_cmd=sim.delta_cmd,
                         steer_clipped=abs(sim.d_mpc + sim.d_ff + d) >= sim.s_max,
                         a_long_req=a_long_req, a_long_app=float(np.clip(a_long_req, -sim.a_max, sim.a_max)),
                         a_clipped=abs(a_long_req) > sim.a_max))
        sim.advance_interval()
    sim.close()
    sm = sim.summary()
    from pilot.closedloop import qualifies

    ok, why = qualifies(sm, cfg["qualifying"])
    trace = pd.DataFrame(sim.rows)
    extra = pd.DataFrame(rows)
    trace = pd.concat([trace.reset_index(drop=True), extra.drop(columns=["t", "s", "progress"]).reset_index(drop=True)], axis=1)
    sm = {k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for k, v in sm.items()}
    sm.update(qualifying=ok, status=why, **{k: job[k] for k in ("track", "ckpt", "variant", "seg", "bias")})
    return sm, trace


def austin_bias():
    """Mean steering residual on Austin's pre-defined straights for each checkpoint (orig sweep laps)."""
    from pilot.closedloop import Track

    tr = Track(R.load_cfg("wall", "Austin"))
    segs = straight_segments(tr)
    cfg = R.load_cfg("wall")
    jobs = [dict(track="Austin", ckpt=c, a_lat=a, mu=mu, start_s=s0, variant="orig", window=None, bias=0.0, seg="full")
            for c in CKPTS.values() for a, mu, s0 in R.sweep_cells(cfg)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(run_case, jobs))
    out = {}
    for c in CKPTS.values():
        vals = [tr_.d_res_req[[any(a <= s <= b for a, b in segs) for s in tr_.s]].values
                for sm, tr_ in res if sm["ckpt"] == c]
        out[c] = float(np.mean(np.concatenate(vals)))
    return out, segs


def run():
    t0 = time.time()
    bias, segs = austin_bias()
    print("Austin straight-line bias:", bias, "straights:", [(round(a), round(b)) for a, b in segs], flush=True)
    jobs = []
    for track in R.TRACKS:
        cells = R.sweep_cells(R.load_cfg("wall", track))
        for c in CKPTS.values():
            for a, mu, s0 in cells:
                for v in ("orig", "a0", "d0", "debias"):
                    jobs.append(dict(track=track, ckpt=c, a_lat=a, mu=mu, start_s=s0, variant=v, window=None,
                                     bias=bias[c], seg="full"))
    for track, name, lo, hot, hi in SEGMENTS:
        cells = R.sweep_cells(R.load_cfg("wall", track))
        for c in CKPTS.values():
            for a, mu, s0 in cells:
                for v in ("a0", "d0", "debias"):
                    jobs.append(dict(track=track, ckpt=c, a_lat=a, mu=mu, start_s=s0, variant=v, window=(lo, hi),
                                     bias=bias[c], seg=name))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(run_case, jobs, chunksize=2))
    sms, trs = [], []
    for i, ((sm, tr_), j) in enumerate(zip(res, jobs)):
        sm["case_id"] = i
        sm["a_lat"], sm["mu_scale"], sm["start_s"] = j["a_lat"], j["mu"], j["start_s"]
        sms.append(sm)
        tr_["case_id"] = i
        trs.append(tr_)
    pd.DataFrame(sms).to_csv(RAW / "ablation_summary.csv", index=False)
    pd.concat(trs).to_csv(RAW / "ablation_traces.csv.gz", index=False, float_format="%.6g")
    pd.Series(bias).to_csv(RAW / "ablation_bias.csv")
    pd.DataFrame(segs, columns=["s0", "s1"]).to_csv(RAW / "austin_straights.csv", index=False)
    print(f"{len(jobs)} laps in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()


# ------------------------------------------------------------------------------------------------ analysis
FAIL = ("wall_collision", "spin", "lateral_error", "timeout")


def _gaps(tr, track):
    from pilot.sim import NOMINAL_PARAMS

    t = _track(track)
    s_ext, left, right = t.wall_side_clearance(1.5)
    hw = NOMINAL_PARAMS["width"] / 2
    s = np.mod(tr.s.values, t.ref.length)
    gl = np.interp(s, s_ext, left) - tr.e_y.values - hw
    gr = np.interp(s, s_ext, right) + tr.e_y.values - hw
    return np.minimum(gl, gr), np.minimum(np.interp(s, s_ext, left), np.interp(s, s_ext, right))


def _seg_class(track, a_lat, s):
    t = _track(track)
    ref = t.ref
    v = t.plan(a_lat, None, True)
    aplan = v * np.gradient(v, ref.ds)
    se = np.r_[ref.s, ref.length]
    k = np.abs(np.interp(s, se, np.r_[ref.kappa, ref.kappa[0]]))
    a = np.interp(s, se, np.r_[aplan, aplan[0]])
    return np.where(k > 0.05, "corner", np.where(a < -0.5, "braking", np.where(k < 0.01, "straight", "transition")))


def analyze():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = pd.read_csv(RAW / "ablation_summary.csv")
    tr = pd.read_csv(RAW / "ablation_traces.csv.gz")
    sm["completed"] = sm.outcome == "completed"
    sm["ck"] = sm.ckpt.map({v: k for k, v in CKPTS.items()})
    key = ["track", "a_lat", "mu_scale", "start_s"]
    full = sm[sm.seg == "full"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # A1. outcome table per checkpoint x variant
    tab = full.groupby(["ck", "variant"]).agg(laps=("completed", "size"), completed=("completed", "sum"),
                                             qualifying=("qualifying", "sum"),
                                             slow=("status", lambda s: (s == "speed_tolerance").sum()),
                                             wall=("status", lambda s: (s == "wall_collision").sum()),
                                             spin=("status", lambda s: (s == "spin").sum()),
                                             lateral=("status", lambda s: (s == "lateral_error").sum()),
                                             ratio_completed=("lap_time_ratio", "mean"))
    tab.to_csv(OUT / "full_lap_outcomes.csv", float_format="%.4f")
    tabt = full.groupby(["ck", "variant", "track"]).agg(completed=("completed", "sum"), qualifying=("qualifying", "sum"))
    tabt.unstack("track").to_csv(OUT / "full_lap_outcomes_by_track.csv")

    # A2. paired deltas vs orig (same checkpoint, same cell)
    o = full[full.variant == "orig"].set_index(["ck"] + key)
    pair_rows = []
    for v in ("a0", "d0", "debias"):
        x = full[full.variant == v].set_index(["ck"] + key)
        j = o[["completed", "qualifying", "lap_time", "lap_time_ratio", "fail_s", "status"]].join(
            x[["completed", "qualifying", "lap_time", "lap_time_ratio", "fail_s", "status"]], rsuffix="_v")
        for ck, g in j.groupby(level=0):
            both = g[g.completed & g.completed_v]
            pair_rows.append(dict(ck=ck, variant=v, cells=len(g), both_completed=len(both),
                                  dlap_mean=(both.lap_time_v - both.lap_time).mean(),
                                  dlap_median=(both.lap_time_v - both.lap_time).median(),
                                  dratio_mean=(both.lap_time_ratio_v - both.lap_time_ratio).mean(),
                                  orig_only_completed=int((g.completed & ~g.completed_v).sum()),
                                  variant_only_completed=int((~g.completed & g.completed_v).sum()),
                                  dqual=int(g.qualifying_v.sum() - g.qualifying.sum()),
                                  q_gained=int((~g.qualifying & g.qualifying_v).sum()),
                                  q_lost=int((g.qualifying & ~g.qualifying_v).sum())))
        j.reset_index().to_csv(OUT / f"paired_full_{v}.csv", index=False)
    pt = pd.DataFrame(pair_rows)
    pt.to_csv(OUT / "paired_full_summary.csv", index=False, float_format="%.4f")

    # A3. late vs early checkpoint, orig: why does 2.5M lose qualifying?
    e = o.xs("1.25M").join(o.xs("2.5M"), rsuffix="_late")
    lost = e[e.qualifying & ~e.qualifying_late]
    late = dict(cells=len(e), q_early=int(e.qualifying.sum()), q_late=int(e.qualifying_late.sum()),
                lost_total=len(lost), lost_to_speed=int((lost.status_late == "speed_tolerance").sum()),
                lost_to_failure=int(lost.status_late.isin(FAIL).sum()),
                gained=int((~e.qualifying & e.qualifying_late).sum()),
                completed_early=int(e.completed.sum()), completed_late=int(e.completed_late.sum()),
                both_completed=int((e.completed & e.completed_late).sum()),
                dlap_both_mean=float((e.lap_time_late - e.lap_time)[e.completed & e.completed_late].mean()),
                ratio_early=float(e.lap_time_ratio[e.completed].mean()), ratio_late=float(e.lap_time_ratio_late[e.completed_late].mean()),
                late_status=e.status_late.value_counts().to_dict(), early_status=e.status.value_counts().to_dict())
    (OUT / "late_vs_early.json").write_text(pd.Series(late).to_json(indent=1))

    # A4. where the time goes: excess time vs plan per segment class (completed orig laps), and time added by a_res
    tr = tr.merge(sm[["case_id", "ck", "variant", "seg"] + key + ["completed"]], on="case_id")
    rows = []
    for (cid, ck, var, trk, a), g in tr[(tr.seg == "full") & tr.completed].groupby(["case_id", "ck", "variant", "track", "a_lat"]):
        g = g.sort_values("t")
        dt = np.diff(g.t.values, append=g.t.values[-1] + 0.05)
        ds = np.diff(g.progress.values, append=g.progress.values[-1])
        plan_dt = ds / np.maximum(g.v_plan.values, 0.3)
        seg = _seg_class(trk, a, np.mod(g.s.values, _track(trk).ref.length))
        gap, clr = _gaps(g, trk)
        risky = clr < 0.4  # within a tight-wall stretch
        for sc in ("straight", "braking", "corner", "transition"):
            for rk in (False, True):
                m = (seg == sc) & (risky == rk)
                rows.append(dict(case_id=cid, ck=ck, variant=var, track=trk, a_lat=a, seg=sc, tight_wall=rk,
                                 time=dt[m].sum(), plan_time=plan_dt[m].sum(), dist=ds[m].sum()))
    ex = pd.DataFrame(rows)
    agg = ex.groupby(["ck", "variant", "seg", "tight_wall"])[["time", "plan_time", "dist"]].sum().reset_index()
    agg["excess"] = agg.time - agg.plan_time
    tot = agg.groupby(["ck", "variant"]).agg(ex_tot=("excess", "sum"), d_tot=("dist", "sum"))
    agg = agg.merge(tot.reset_index(), on=["ck", "variant"])
    agg["share_of_excess"] = agg.excess / agg.ex_tot
    agg["share_of_distance"] = agg.dist / agg.d_tot
    agg["excess_pct_of_plan"] = agg.excess / agg.plan_time * 100
    agg.to_csv(OUT / "excess_time_by_segment.csv", index=False, float_format="%.4f")

    # A5. a_res: survival vs time without clearance benefit (paired orig vs a0, full laps)
    fa = pd.read_csv(OUT / "paired_full_a0.csv")
    surv = fa[fa.completed & ~fa.completed_v]
    hurt = fa[~fa.completed & fa.completed_v]
    clr_rows = []
    for ck in CKPTS:
        for _, r in fa[(fa.ck == ck) & fa.completed & fa.completed_v].iterrows():
            for v in ("orig", "a0"):
                cid = sm[(sm.seg == "full") & (sm.ck == ck) & (sm.variant == v) & (sm.track == r.track) & (sm.a_lat == r.a_lat)
                         & (sm.mu_scale == r.mu_scale) & (sm.start_s == r.start_s)].case_id.iloc[0]
                g = tr[tr.case_id == cid]
                gap, clr = _gaps(g, r.track)
                m = clr < 0.4
                clr_rows.append(dict(ck=ck, track=r.track, a_lat=r.a_lat, mu=r.mu_scale, start_s=r.start_s, variant=v,
                                     lap_time=r.lap_time if v == "orig" else r.lap_time_v,
                                     min_gap_tight=gap[m].min() if m.any() else np.nan,
                                     p05_gap_tight=np.quantile(gap[m], 0.05) if m.any() else np.nan,
                                     max_abs_slip=g.slip.abs().max()))
    cg = pd.DataFrame(clr_rows).pivot_table(index=["ck", "track", "a_lat", "mu", "start_s"], columns="variant",
                                            values=["lap_time", "min_gap_tight", "max_abs_slip"])
    cg.columns = [f"{a}_{b}" for a, b in cg.columns]
    cg["dlap_from_ares"] = cg.lap_time_orig - cg.lap_time_a0
    cg["dgap_from_ares"] = cg.min_gap_tight_orig - cg.min_gap_tight_a0
    cg["dslip_from_ares"] = cg.max_abs_slip_orig - cg.max_abs_slip_a0
    cg.reset_index().to_csv(OUT / "ares_time_vs_clearance.csv", index=False, float_format="%.4f")
    surv.to_csv(OUT / "ares_survival_cases.csv", index=False)

    # B. segment tests
    seg_rows, prefix_bad = [], 0
    for (track, name, lo, hot, hi) in SEGMENTS:
        L = _track(track).ref.length
        for (ck, a, mu, s0), _ in o.reset_index().query("track == @track").groupby(["ck", "a_lat", "mu_scale", "start_s"]):
            cid_o = sm[(sm.seg == "full") & (sm.variant == "orig") & (sm.ck == ck) & (sm.track == track) & (sm.a_lat == a)
                       & (sm.mu_scale == mu) & (sm.start_s == s0)].case_id.iloc[0]
            to = tr[tr.case_id == cid_o].sort_values("t")
            rel = lambda s: np.mod(s - lo, L)  # noqa: E731
            w = (hi - lo) % L
            reached = (rel(to.s.values) < w).any()
            if not reached:
                continue  # orig failed before the branch point
            for v in ("orig", "a0", "d0", "debias"):
                if v == "orig":
                    g = to
                else:
                    cid = sm[(sm.seg == name) & (sm.variant == v) & (sm.ck == ck) & (sm.a_lat == a) & (sm.mu_scale == mu)
                             & (sm.start_s == s0)].case_id.iloc[0]
                    g = tr[tr.case_id == cid].sort_values("t")
                    first = np.argmax(g.active.values) if g.active.any() else len(g)
                    n = min(first, len(to))
                    if not np.array_equal(g.e_y.values[:n], to.e_y.values[:n]) or not np.array_equal(g.v.values[:n], to.v.values[:n]):
                        prefix_bad += 1
                inw = rel(g.s.values) < w
                idx = np.where(inw)[0]
                if not len(idx):
                    continue
                i0 = idx[0]
                gw = g.iloc[i0:]
                in_seg = rel(gw.s.values) < w
                exit_i = np.argmax(~in_seg) if (~in_seg).any() else None
                gs = gw.iloc[:exit_i] if exit_i is not None else gw
                st = sm[sm.case_id == g.case_id.iloc[0]].iloc[0]
                failed_in = exit_i is None and st.status in FAIL
                gap, clr = _gaps(gs, track)
                hot_i = np.argmin(np.abs(rel(gs.s.values) - (hot - lo) % L))
                seg_rows.append(dict(seg=name, track=track, ck=ck, a_lat=a, mu=mu, start_s=s0, variant=v,
                                     entry_v=gs.v.iloc[0], v_at_hot_minus10=gs.v.iloc[max(hot_i - int(10 / max(gs.v.iloc[0] * 0.05, 0.1)), 0)],
                                     seg_time=(gw.t.iloc[exit_i] - gw.t.iloc[0]) if exit_i is not None else np.nan,
                                     min_gap=gap.min(), exit_abs_ey=abs(gw.e_y.iloc[exit_i]) if exit_i is not None else np.nan,
                                     failed_in_segment=failed_in, fail_s=st.fail_s if failed_in else np.nan,
                                     lap_status=st.status, mean_d_res_app=gs.d_res_app.mean(), mean_a_res_app=gs.a_res_app.mean(),
                                     steer_clip_frac=gs.steer_clipped.mean(), a_clip_frac=gs.a_clipped.mean()))
    sg = pd.DataFrame(seg_rows)
    sg.to_csv(OUT / "segment_cases.csv", index=False, float_format="%.4f")
    base = sg[sg.variant == "orig"].set_index(["seg", "ck", "a_lat", "mu", "start_s"])
    prs = []
    for v in ("a0", "d0", "debias"):
        x = sg[sg.variant == v].set_index(["seg", "ck", "a_lat", "mu", "start_s"])
        j = base.join(x, rsuffix="_v", how="inner")
        for (sname, ck), g in j.groupby(level=[0, 1]):
            both = g[~g.failed_in_segment & ~g.failed_in_segment_v]
            prs.append(dict(seg=sname, ck=ck, variant=v, cases=len(g), fail_orig=int(g.failed_in_segment.sum()),
                            fail_v=int(g.failed_in_segment_v.sum()),
                            d_seg_time=(both.seg_time_v - both.seg_time).mean(),
                            d_min_gap=(g.min_gap_v - g.min_gap).mean(),
                            d_min_gap_both=(both.min_gap_v - both.min_gap).mean(),
                            d_exit_ey=(both.exit_abs_ey_v - both.exit_abs_ey).mean(),
                            d_v_hot=(g.v_at_hot_minus10_v - g.v_at_hot_minus10).mean(),
                            orig_closer_by_2cm=int(((g.min_gap - g.min_gap_v) < -0.02).sum()) if v == "d0" else None,
                            orig_steer_clip=g.steer_clip_frac.mean(), orig_a_clip=g.a_clip_frac.mean()))
    ps = pd.DataFrame(prs)
    ps.to_csv(OUT / "segment_paired_summary.csv", index=False, float_format="%.4f")

    # C. overlays: two segments, one cell, ckpt 1.25M, all variants
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex="col")
    cols = {"orig": "#000000", "a0": "#1565c0", "d0": "#c62828", "debias": "#2e7d32"}
    for j, (track, name, lo, hot, hi) in enumerate([SEGMENTS[2], SEGMENTS[3]]):
        L = _track(track).ref.length
        for v in ("orig", "a0", "d0", "debias"):
            q = sm[(sm.seg == ("full" if v == "orig" else name)) & (sm.variant == v) & (sm.ck == "1.25M") & (sm.track == track)
                   & (sm.a_lat == 6.0) & (sm.mu_scale == 1.0) & (sm.start_s == 0.0)]
            g = tr[tr.case_id == q.case_id.iloc[0]].sort_values("t")
            m = np.mod(g.s - lo + 20, L) < (hi - lo) + 35
            g = g[m]
            x = np.mod(g.s.values, L)
            gap, _ = _gaps(g, track)
            lab = f"{v} ({q.status.iloc[0]})"
            axes[0, j].plot(x, g.e_y, color=cols[v], lw=1, label=lab)
            axes[1, j].plot(x, gap, color=cols[v], lw=1)
            axes[2, j].plot(x, g.d_res_app, color=cols[v], lw=1)
            axes[2, j].plot(x, g.d_res_req, color=cols[v], lw=0.6, ls=":")
            axes[3, j].plot(x, g.v / g.v_plan, color=cols[v], lw=1)
        for a_ in axes[:, j]:
            a_.axvspan(lo, hi, color="#fff3e0", zorder=0)
            a_.axvline(hot, color="r", lw=0.6)
            a_.grid(alpha=0.3)
        axes[0, j].set_title(f"{name} — a_lat 6, μ/μ₀ 1.0, start 0, ckpt 1.25M\n(shaded: intervention window; red line: hotspot)", fontsize=8)
        axes[0, j].legend(fontsize=7)
        axes[-1, j].set_xlabel("s [m]")
    for a_, lab in zip(axes[:, 0], ("e_y [m]", "min body–wall gap [m]", "δ_res applied (solid) / requested (dotted)", "v / v_plan")):
        a_.set_ylabel(lab, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_segment_overlays.png", dpi=125)
    plt.close(fig)

    print(tab.to_string())
    print(tabt.unstack("track").to_string())
    print(pt.to_string(index=False))
    print(late)
    print(agg[agg.variant == "orig"].round(3).to_string(index=False))
    print(agg[agg.variant == "a0"].groupby("ck")[["excess", "plan_time"]].sum())
    print("a_res survival cases:", len(surv), "a_res hurt cases:", len(hurt))
    print(cg.groupby(level=0)[["dlap_from_ares", "dgap_from_ares", "dslip_from_ares"]].describe().round(3).T.to_string())
    print("prefix mismatches:", prefix_bad)
    print(ps.round(3).to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "analyze":
    analyze()
