"""Checkpoint-selection and reporting protocol for residual policies (multi-seed).

Usage:
    uv run python -m pilot.protocol run --config CFG --run-dir DIR [--all-ckpts]
        select a checkpoint of one training run on the selection grid, then lap the reporting grid with it
        (--all-ckpts: also lap the reporting grid with every checkpoint, as a diagnostic that selection never sees)
    uv run python -m pilot.protocol run ... --select-tracks Austin Catalunya Silverstone --tag noSpielberg
        same, with the selection grid laid on each listed track (24 laps per track); outputs get the suffix _<tag>.
        If the run already has protocol_report_all_ckpts.csv, the selected checkpoint's report is taken from it
        (laps are deterministic) instead of being lapped again.
    uv run python -m pilot.protocol compare --out DIR LABEL=RUN_DIR[@TAG] [LABEL=RUN_DIR[@TAG] ...]
        per-run and per-variant (label prefix before '/') summaries; e.g. base/s0=... base/s1=... slow/s0=...

Fixed before any run of the multi-seed study, and shared by every variant:
  * Selection grid (24 laps / checkpoint): training track only, cells used by no reporting lap:
    Austin x a_lat {3.5, 5, 7, 8.5} x mu/mu0 {1.0, 0.85, 0.65} x start s {50, 250}.
  * Selection rule: most qualifying laps -> most completed laps -> lowest mean lap-time ratio over completed laps
    -> earliest checkpoint. (No mean-progress tie-break: it separated checkpoints by 1e-4.)
  * Reporting grid (144 laps): 4 tracks x a_lat {4, 6, 8} x [the sweep cells mu/mu0 {1.0, 0.8, 0.6} x s {0, 200}
    + fresh cells mu/mu0 {1.0, 0.9, 0.7} x s {100, 300}]. Austin is the training track; the others are held out.
  * Reported per lap: status (qualifying or the failure reason), lap-time ratio and its margin to 1.05, max |e_y|
    and its margin to 0.5 m, failure location and known crash hotspot. Per variant: median and range over seeds.
The evaluation rule itself (pilot.closedloop.qualifies, cfg['qualifying']) is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import re

import numpy as np
import pandas as pd
import yaml

from .closedloop import Scenario, qualifies, run_many
from .residual_env import ResidualPolicy

TRACKS = ["Austin", "Catalunya", "Silverstone", "Spielberg"]
SELECT = dict(track="Austin", a_lat=[3.5, 5.0, 7.0, 8.5], mu_scale=[1.0, 0.85, 0.65], start_s=[50.0, 250.0])
REPORT = [("sweep", [1.0, 0.8, 0.6], [0.0, 200.0]), ("fresh", [1.0, 0.9, 0.7], [100.0, 300.0])]
REPORT_A_LAT = [4.0, 6.0, 8.0]
# corner-exit wall pinches found in results/analysis (ares_sweep / ares_validation); +-10 m
HOTSPOTS = {"Austin": [152.0], "Spielberg": [42.0, 304.0], "Silverstone": [176.0]}
CRASH = ("wall_collision", "spin")


def _ckpts(run_dir: pathlib.Path):
    c = sorted(run_dir.glob("ckpt_*.zip"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if (run_dir / "final.zip").exists():
        c.append(run_dir / "final.zip")
    return c


def _hotspot(track, fail_s):
    if fail_s is None or not np.isfinite(fail_s):
        return ""
    for h in HOTSPOTS.get(track, []):
        if abs(fail_s - h) <= 10.0:
            return f"{track[:3]}-{h:.0f}"
    return "other"


def lap(cfg, ckpts, cells):
    """cells: list of (track, a_lat, mu_scale, start_s[, grid]). Returns one row per (checkpoint, cell)."""
    rc, q = cfg["residual"], cfg["qualifying"]
    rows = []
    for track in sorted({c[0] for c in cells}):
        tc = copy.deepcopy(cfg)
        tc["track"] = track
        cs = [c for c in cells if c[0] == track]
        jobs, keys = [], []
        for ck in ckpts:
            pol = ResidualPolicy.from_rc(str(ck), rc)
            for c in cs:
                jobs.append((Scenario(c[1], c[2], c[3]), rc.get("base_K", 0.0), False, None, pol))
                keys.append((ck.name, c))
        for (name, c), (sm, _) in zip(keys, run_many(tc, jobs)):
            ok, why = qualifies(sm, q)
            fs = sm.get("fail_s")
            rows.append(dict(checkpoint=name, track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3],
                             grid=c[4] if len(c) > 4 else "select", status=why, qualifying=bool(ok),
                             completed=sm["outcome"] == "completed", lap_time=sm["lap_time"],
                             lap_time_ratio=sm["lap_time_ratio"], max_ey=sm["max_ey"], rms_ey=sm["rms_ey"],
                             fail_s=fs, progress_frac=sm["progress_frac"], mean_abs_ares=sm.get("mean_abs_ares"),
                             ratio_margin=(q["max_lap_time_ratio"] - sm["lap_time_ratio"]) if sm["lap_time_ratio"] else np.nan,
                             ey_margin=q["max_abs_ey_m"] - sm["max_ey"], hotspot=_hotspot(track, fs)))
    return pd.DataFrame(rows)


def _steps(name):
    return int(re.findall(r"\d+", name)[0]) if name.startswith("ckpt_") else 10**12


def select(sel: pd.DataFrame):
    g = sel.groupby("checkpoint")
    t = pd.DataFrame(dict(
        qualifying=g.qualifying.sum(), completed=g.completed.sum(),
        crash=g.status.apply(lambda s: s.isin(CRASH).sum()), too_slow=g.status.apply(lambda s: (s == "speed_tolerance").sum()),
        ey_limit=g.status.apply(lambda s: (s == "lateral_error").sum()),
        mean_ratio_completed=g.apply(lambda x: x.lap_time_ratio[x.completed].mean(), include_groups=False))).reset_index()
    t["steps"] = t.checkpoint.map(_steps)
    t = t.sort_values("steps").reset_index(drop=True)
    best = t.sort_values(["qualifying", "completed", "mean_ratio_completed", "steps"],
                         ascending=[False, False, True, True], na_position="last").iloc[0]
    return t, best.checkpoint


def train_tracks(run_dir: pathlib.Path):
    """Tracks a run was trained on (residual.tracks of its train_config.yaml; default its track)."""
    tc = yaml.safe_load(open(run_dir / "train_config.yaml"))
    return list(tc["residual"].get("tracks") or [tc["track"]])


def _suffix(tag):
    return f"_{tag}" if tag else ""


def report_cells():
    return [(t, a, mu, s0, grid) for t in TRACKS for a in REPORT_A_LAT for grid, mus, ss in REPORT for mu in mus for s0 in ss]


def cmd_run(args):
    cfg = yaml.safe_load(open(args.config))
    run = pathlib.Path(args.run_dir)
    ck = _ckpts(run)
    if not ck:
        raise SystemExit(f"no checkpoints in {run}")
    sel_tracks = args.select_tracks or [SELECT["track"]]
    sx = _suffix(args.tag)
    cells = [(tr, a, mu, s0) for tr in sel_tracks for a in SELECT["a_lat"] for mu in SELECT["mu_scale"]
             for s0 in SELECT["start_s"]]
    sel = lap(cfg, ck, cells)
    sel.to_csv(run / f"protocol_selection_laps{sx}.csv", index=False, float_format="%.5g")
    t, best = select(sel)
    t.to_csv(run / f"protocol_selection{sx}.csv", index=False, float_format="%.4f")
    (run / f"protocol_selected{sx}.txt").write_text(best + "\n")
    print(t.to_string(index=False))
    print("selected:", best, flush=True)
    allp = run / "protocol_report_all_ckpts.csv"
    if args.all_ckpts and not allp.exists():
        pd.DataFrame(lap(cfg, ck, report_cells())).to_csv(allp, index=False, float_format="%.5g")
    cached = pd.read_csv(allp) if allp.exists() else None
    if cached is not None and (cached.checkpoint == best).sum() == len(report_cells()):
        rep = cached[cached.checkpoint == best].reset_index(drop=True)
        print(f"report of {best} taken from {allp.name}")
    else:
        rep = lap(cfg, [ck_ for ck_ in ck if ck_.name == best], report_cells())
    rep.to_csv(run / f"protocol_report{sx}.csv", index=False, float_format="%.5g")
    print(summarize(rep, train_tracks(run)).to_string(), flush=True)


def summarize(rep: pd.DataFrame, trained_on=("Austin",)) -> pd.Series:
    """Held-out = reporting tracks the policy was not trained on (``trained_on``)."""
    c = rep[rep.completed]
    q = rep[rep.qualifying]
    aus = rep.track == "Austin"
    tr = rep.track.isin(list(trained_on))
    return pd.Series(dict(
        qualifying=int(rep.qualifying.sum()), qualifying_austin=int(rep.qualifying[aus].sum()),
        qualifying_heldout=int(rep.qualifying[~tr].sum()), qualifying_train=int(rep.qualifying[tr].sum()),
        **{f"qualifying_{t}": int(rep.qualifying[rep.track == t].sum()) for t in TRACKS},
        **{f"crash_{t}": int(rep.status[rep.track == t].isin(CRASH).sum()) for t in TRACKS},
        qualifying_sweep=int(rep.qualifying[rep.grid == "sweep"].sum()), qualifying_fresh=int(rep.qualifying[rep.grid == "fresh"].sum()),
        completed=int(rep.completed.sum()), crash=int(rep.status.isin(CRASH).sum()),
        too_slow=int((rep.status == "speed_tolerance").sum()), ey_limit=int((rep.status == "lateral_error").sum()),
        mean_ratio_completed=float(c.lap_time_ratio.mean()), max_ratio_qualifying=float(q.lap_time_ratio.max()) if len(q) else np.nan,
        q_within_0p5pt=int((q.ratio_margin < 0.005).sum()), q_within_2cm=int((q.ey_margin < 0.02).sum()),
        **{f"crash_{h}": int(((rep.hotspot == h) & rep.status.isin(CRASH)).sum())
           for h in [f"{t[:3]}-{x:.0f}" for t, xs in HOTSPOTS.items() for x in xs] + ["other"]}))


def cmd_compare(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = dict(a.split("=", 1) for a in args.runs)
    reps, rows, sels = [], [], []
    for label, d in runs.items():
        d, _, tag = d.partition("@")
        d, sx = pathlib.Path(d), _suffix(tag)
        rep = pd.read_csv(d / f"protocol_report{sx}.csv").assign(run=label, variant=label.split("/")[0])
        reps.append(rep)
        s = summarize(rep, train_tracks(d))
        rows.append(dict(run=label, variant=label.split("/")[0], selected=rep.checkpoint.iloc[0], **s))
        sels.append(pd.read_csv(d / f"protocol_selection{sx}.csv").assign(run=label, variant=label.split("/")[0]))
        if (d / "protocol_report_all_ckpts.csv").exists():
            a = pd.read_csv(d / "protocol_report_all_ckpts.csv")
            for ck, g in a.groupby("checkpoint"):
                sels[-1].loc[sels[-1].checkpoint == ck, ["report_qualifying", "report_completed", "report_too_slow",
                                                         "report_mean_ratio"]] = [
                    g.qualifying.sum(), g.completed.sum(), (g.status == "speed_tolerance").sum(),
                    g.lap_time_ratio[g.completed].mean()]
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False, float_format="%.4f")
    num = per_run.drop(columns=["run", "selected"]).groupby("variant")
    agg = pd.concat({"median": num.median(), "min": num.min(), "max": num.max()}, axis=1)
    agg.to_csv(out / "per_variant.csv", float_format="%.4f")
    allrep = pd.concat(reps, ignore_index=True)
    allrep.to_csv(out / "report_laps.csv", index=False, float_format="%.5g")
    pd.concat(sels, ignore_index=True).to_csv(out / "selection_curves.csv", index=False, float_format="%.4f")
    # per-cell qualifying share by variant (how many seeds qualify each cell)
    key = ["track", "a_lat", "mu_scale", "start_s", "grid"]
    share = allrep.pivot_table(index=key, columns="variant", values="qualifying", aggfunc="mean")
    share.to_csv(out / "cell_qualifying_share.csv", float_format="%.3f")
    # lap time on cells completed by every run
    done = allrep.pivot_table(index=key, columns="run", values="completed", aggfunc="first")
    common = done.index[done.all(axis=1)]
    lt = allrep.set_index(key).loc[common].groupby("run").lap_time_ratio.mean()
    per_run["mean_ratio_common_cells"] = per_run.run.map(lt)
    per_run["n_common_cells"] = len(common)
    per_run.to_csv(out / "per_run.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(per_run.to_string(index=False))
    print(agg.T.to_string())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--all-ckpts", action="store_true")
    r.add_argument("--select-tracks", nargs="+", default=None, help="selection-grid tracks (default: Austin)")
    r.add_argument("--tag", default="", help="suffix for the output files (keeps earlier selections)")
    c = sp.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("runs", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
