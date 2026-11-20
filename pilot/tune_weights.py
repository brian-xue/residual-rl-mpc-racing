"""Select the shared MPC weights for the configured plant (nominal MPC, mu_scale = 1.0 only).

Rule: grid over (q_y, q_psi, r_du); every development lap (a_lat in DEV_A_LAT, both predetermined start
poses, mu_scale = 1.0) must be valid, then lowest mean RMS lateral error, then lowest mean max |e_y|.
DEV_A_LAT stays inside the STD grip limit (~10 m/s^2 steady, less under braking): at a_lat 8 every weight
set spins on corner entry (results of that first attempt are noted in the README). The result is written to
results/mpc_weight_grid.csv and printed; copy the winner into configs/pilot.yaml (mpc.q_y / q_psi / r_du).

Usage:  uv run python -m pilot.tune_weights [--config configs/pilot.yaml] [--out DIR]
"""

from __future__ import annotations

import itertools
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from . import cli
from .closedloop import Scenario, Track, qualifies, run_lap

GRID = dict(q_y=(3.0, 10.0), q_psi=(1.0, 3.0, 10.0), r_du=(30.0, 100.0, 300.0, 1000.0))
DEV_A_LAT = (4.0, 6.0)


def _job(args):
    config, w, a_lat, s0 = args
    cfg = yaml.safe_load(open(config))
    cfg["mpc"].update(w)
    s, _ = run_lap(Track(cfg), Scenario(a_lat, 1.0, s0), 0.0, cfg, log=False)
    ok, reason = qualifies(s, cfg["qualifying"])
    return dict(**w, a_lat=a_lat, start_s=s0, status=reason, valid=ok, progress_frac=float(s["progress_frac"]),
                rms_ey=float(s["rms_ey"]), max_ey=float(s["max_ey"]))


def main(config="configs/pilot.yaml", out="results"):
    cfg = yaml.safe_load(open(config))
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    starts = (cfg["sweep"]["start_s"], cfg["sweep"]["recheck_start_s"])
    jobs = [(config, w, a, s0) for w in combos for a in DEV_A_LAT for s0 in starts]
    with ProcessPoolExecutor(cfg.get("workers", 8)) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    pathlib.Path(out).mkdir(exist_ok=True)
    df.to_csv(pathlib.Path(out) / "mpc_weight_grid.csv", index=False, float_format="%.4f")
    agg = df.assign(progress_frac=df.progress_frac.round(3)).groupby(list(GRID)).agg(all_valid=("valid", "all"), mean_progress=("progress_frac", "mean"),
                                     mean_rms_ey=("rms_ey", "mean"), mean_max_ey=("max_ey", "mean")).reset_index()
    agg = agg.sort_values(["all_valid", "mean_progress", "mean_rms_ey", "mean_max_ey"],
                          ascending=[False, False, True, True])
    print(agg.to_string(index=False))
    best = agg.iloc[0]
    print("selected:", {k: float(best[k]) for k in GRID}, "plant:", cfg.get("model", "st"))
    return {k: float(best[k]) for k in GRID}


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    _, config, out = cli.parse(__doc__)
    main(config, out)
