"""SlipMPC + RL, Stages 2-4 (docs/slip_rl_plan.md): no-learning baselines on the protocol's reporting grid.

    uv run python results/analysis/slip_rl/stage2.py baselines

Laps the 144-lap reporting grid of pilot.protocol (4 tracks x a_lat {4, 6, 8} x 12 cells) with
    SE      S-E: configs/pilot_st_pacejka_slip_res.yaml with no residual (the published S-E, Stage 0 check)
    SE097   S-E tracking 0.97 x the planned speed (slip_mpc.speed_scale: 0.97), the Stage 0 oracle's free baseline
and writes each as a protocol "run directory" (protocol_report.csv, protocol_selection.csv, train_config.yaml), so
``pilot.protocol compare`` treats them like the trained runs. Both are deterministic: one "seed" each.
The training and protocol runs themselves are in run_stage2.sh.
"""

from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pilot.closedloop import Scenario, qualifies, run_many  # noqa: E402
from pilot.protocol import _hotspot, report_cells  # noqa: E402

CFG = ROOT / "configs/pilot_st_pacejka_slip_res.yaml"
OUT = ROOT / "results_st_pacejka/slip_rl/baselines"


def lap_baseline(cfg):
    q, rows = cfg["qualifying"], []
    cells = report_cells()
    for track in sorted({c[0] for c in cells}):
        tc = copy.deepcopy(cfg)
        tc["track"] = track
        cs = [c for c in cells if c[0] == track]
        jobs = [(Scenario(c[1], c[2], c[3]), 0.0, False, None) for c in cs]
        for c, (sm, _) in zip(cs, run_many(tc, jobs)):
            ok, why = qualifies(sm, q)
            fs = sm.get("fail_s")
            rows.append(dict(checkpoint="none", track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3], grid=c[4],
                             status=why, qualifying=bool(ok), completed=sm["outcome"] == "completed",
                             lap_time=sm["lap_time"], lap_time_ratio=sm["lap_time_ratio"], max_ey=sm["max_ey"],
                             rms_ey=sm["rms_ey"], fail_s=fs, progress_frac=sm["progress_frac"], mean_abs_ares=None,
                             ratio_margin=(q["max_lap_time_ratio"] - sm["lap_time_ratio"]) if sm["lap_time_ratio"] else np.nan,
                             ey_margin=q["max_abs_ey_m"] - sm["max_ey"], hotspot=_hotspot(track, fs)))
    return pd.DataFrame(rows)


def baselines():
    base = yaml.safe_load(open(CFG))
    for name, speed in (("SE", None), ("SE097", 0.97)):
        cfg = copy.deepcopy(base)
        if speed is not None:
            cfg["longitudinal"]["slip_mpc"]["speed_scale"] = speed
        d = OUT / name / "s0"
        d.mkdir(parents=True, exist_ok=True)
        rep = lap_baseline(cfg)
        rep.to_csv(d / "protocol_report.csv", index=False, float_format="%.5g")
        pd.DataFrame([dict(checkpoint="none", qualifying=np.nan, completed=np.nan, steps=0)]).to_csv(
            d / "protocol_selection.csv", index=False)
        yaml.safe_dump(cfg, open(d / "train_config.yaml", "w"), sort_keys=False)
        print(name, "qualifying", int(rep.qualifying.sum()), "of", len(rep),
              rep.groupby("track").qualifying.sum().to_dict(), flush=True)


if __name__ == "__main__":
    {"baselines": baselines}[sys.argv[1]]()
