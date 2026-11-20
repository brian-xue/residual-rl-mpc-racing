"""Replay saved evaluation laps with traces, checkpoint diagnostics and counterfactual ablations.

    uv run python results/analysis/replay.py r1      # sweep cells, 4 tracks: RL δ / δa / δa+wall, S-E, P/P+ff, P-E/P-E+ff
    uv run python results/analysis/replay.py r1low   # dual policies at a_lat 1-3 on Austin (below the training range)
    uv run python results/analysis/replay.py r2      # all 13 checkpoints x 3 runs: validation laps + sweep on 4 tracks
    uv run python results/analysis/replay.py r3      # ablations of the selected policies on the sweep cells, 4 tracks
Outputs in results/analysis/data/ (summaries .csv, traces .csv.gz, observations .npz); a batch is skipped if its
summary file exists. r1 also writes reproduction.csv: every replayed lap against its saved sweep.csv row.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

D = R.OUT / "data"
D.mkdir(parents=True, exist_ok=True)


def _save(name, results, jobs):
    rows, traces, obs = [], [], {}
    for i, ((sm, tr, o, a), j) in enumerate(zip(results, jobs)):
        sm["lap_id"] = i
        rows.append(sm)
        if tr is not None and len(tr):
            tr = tr.copy()
            tr["lap_id"] = i
            traces.append(tr)
        if o is not None:
            obs[f"obs_{i}"], obs[f"act_{i}"] = o, a
    df = pd.DataFrame(rows)
    df.to_csv(D / f"{name}_summary.csv", index=False)
    if traces:
        pd.concat(traces).to_csv(D / f"{name}_traces.csv.gz", index=False, float_format="%.6g")
    if obs:
        np.savez_compressed(D / f"{name}_obs.npz", **obs)
    return df


def _run(name, jobs, force=False):
    if (D / f"{name}_summary.csv").exists() and not force:
        print(f"[skip] {name}: exists")
        return pd.read_csv(D / f"{name}_summary.csv")
    t0 = time.time()
    res = R.run_jobs(jobs)
    df = _save(name, res, jobs)
    print(f"[done] {name}: {len(jobs)} laps in {time.time() - t0:.0f} s")
    return df


def r1():
    jobs = []
    for track in R.TRACKS:
        cells = R.sweep_cells(R.load_cfg("delta", track))
        for a, mu, s0 in cells:
            base = dict(track=track, a_lat=a, mu=mu, start_s=s0, log=True)
            for exp in R.EXPS:
                jobs.append({**base, "exp": exp, "K": 0.0, "residual": R.residual_for(exp, record=True),
                             "tags": dict(ctrl=exp, track=track)})
            jobs.append({**base, "exp": "S-E", "K": 0.0, "tags": dict(ctrl="SE", track=track)})
            for exp, plan in (("delta", "std"), ("dual", "ell")):  # baselines: kinematic MPC on each plan
                jobs.append({**base, "exp": exp, "K": 0.0, "tags": dict(ctrl=f"nominal_{plan}", track=track)})
                jobs.append({**base, "exp": exp, "K": R.K_FF, "tags": dict(ctrl=f"ff_{plan}", track=track)})
    df = _run("r1", jobs)
    reproduction(df)


def reproduction(df):
    """Compare every replayed lap with its saved sweep row (outcome, status, lap_time, rms_ey, fail_s)."""
    rows = []
    src = {"delta": ("delta", "residual"), "dual": ("dual", "residual"), "wall": ("wall", "residual"),
           "SE": ("SE", "S-E"), "nominal_std": ("delta", "nominal"), "ff_std": ("delta", "ff"),
           "nominal_ell": ("dual", "nominal"), "ff_ell": ("dual", "ff")}
    for (ctrl, track), g in df.groupby(["ctrl", "track"]):
        exp, c = src[ctrl]
        sv = R.saved_sweep(exp, track)
        sv = sv[sv.controller == c]
        for r in g.itertuples():
            m = sv[(np.isclose(sv.a_lat, r.a_lat)) & (np.isclose(sv.mu_scale, r.mu_scale)) & (np.isclose(sv.start_s, r.start_s))]
            if len(m) != 1:
                rows.append(dict(ctrl=ctrl, track=track, a_lat=r.a_lat, mu=r.mu_scale, start_s=r.start_s, match=False,
                                 note=f"{len(m)} saved rows"))
                continue
            m = m.iloc[0]
            same = (m.outcome == r.outcome and m.status == r.status
                    and (pd.isna(m.lap_time) and pd.isna(r.lap_time) or abs(m.lap_time - r.lap_time) < 5e-3)
                    and abs(m.rms_ey - r.rms_ey) < 5e-4 and abs(m.progress_m - r.progress_m) < 5e-3)
            rows.append(dict(ctrl=ctrl, track=track, a_lat=r.a_lat, mu=r.mu_scale, start_s=r.start_s, match=bool(same),
                             saved_status=m.status, new_status=r.status, saved_lap=m.lap_time, new_lap=r.lap_time,
                             saved_rms=m.rms_ey, new_rms=r.rms_ey, d_progress=r.progress_m - m.progress_m))
    rep = pd.DataFrame(rows)
    rep.to_csv(D / "reproduction.csv", index=False)
    print(rep.groupby("ctrl").match.agg(["sum", "count"]).to_string())


def r1low():
    jobs = []
    for exp in ("dual", "wall"):
        for a in (1.0, 1.5, 2.0, 2.5, 3.0):
            for mu in (1.0, 0.6):
                base = dict(track="Austin", a_lat=a, mu=mu, start_s=0.0, log=True)
                jobs.append({**base, "exp": exp, "K": 0.0, "residual": R.residual_for(exp, record=True),
                             "tags": dict(ctrl=exp, track="Austin")})
                jobs.append({**base, "exp": exp, "K": 0.0, "residual": R.residual_for(exp, ablation="zero_a"),
                             "tags": dict(ctrl=f"{exp}_zero_a", track="Austin")})
    _run("r1low", jobs)


def r2():
    jobs = []
    ckpts = [f"ckpt_{k}.zip" for k in range(250000, 3000001, 250000)] + ["final.zip"]
    for exp in R.EXPS:
        cfg = R.load_cfg(exp)
        sel = cfg["residual"]["select"]
        for ck in ckpts:
            res = R.residual_for(exp, ck)
            step = 3006464 if ck == "final.zip" else int(ck[5:-4])
            for a in sel["a_lat"]:
                for mu in sel["mu_scales"]:
                    jobs.append(dict(exp=exp, track="Austin", a_lat=a, mu=mu, start_s=sel["start_s"], K=0.0, residual=res,
                                     tags=dict(ctrl=exp, ckpt=ck, step=step, track="Austin", set="validation")))
            for track in R.TRACKS:
                for a, mu, s0 in R.sweep_cells(cfg):
                    jobs.append(dict(exp=exp, track=track, a_lat=a, mu=mu, start_s=s0, K=0.0, residual=res,
                                     tags=dict(ctrl=exp, ckpt=ck, step=step, track=track, set="sweep")))
    _run("r2", jobs)


def straight_bias():
    """Mean steering residual on straights (|kappa| < 0.01) of each selected policy over all r1 laps."""
    tr = pd.read_csv(D / "r1_traces.csv.gz", usecols=["lap_id", "kappa", "delta_res"])
    sm = pd.read_csv(D / "r1_summary.csv")
    tr = tr.merge(sm[["lap_id", "ctrl"]], on="lap_id")
    st = tr[(tr.kappa.abs() < 0.01) & tr.ctrl.isin(list(R.EXPS))]
    return st.groupby("ctrl").delta_res.mean().to_dict()


def r3():
    bias = straight_bias()
    print("straight-line mean residual:", bias)
    jobs = []
    plan = [("delta", "zero_d"), ("delta", "debias"), ("dual", "zero_d"), ("dual", "zero_a"), ("dual", "debias"),
            ("wall", "zero_d"), ("wall", "zero_a"), ("wall", "debias"), ("wall", "wall_cap")]
    for track in R.TRACKS:
        for a, mu, s0 in R.sweep_cells(R.load_cfg("delta", track)):
            for exp, abl in plan:
                jobs.append(dict(exp=exp, track=track, a_lat=a, mu=mu, start_s=s0, K=0.0, log=True,
                                 residual=R.residual_for(exp, ablation=abl, bias=bias.get(exp, 0.0)),
                                 tags=dict(ctrl=exp, ablation=abl, bias=bias.get(exp, 0.0), track=track)))
    _run("r3", jobs)


if __name__ == "__main__":
    {"r1": r1, "r1low": r1low, "r2": r2, "r3": r3, "repro": lambda: reproduction(pd.read_csv(D / "r1_summary.csv"))}[
        sys.argv[1]]()
