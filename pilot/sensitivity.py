"""Sensitivity of the sweep outcome to the wall margin of the reference and to K (start s = 0).

Usage:  uv run python -m pilot.sensitivity [--config configs/pilot.yaml] [--out DIR]  -> DIR/sensitivity.csv
"""

from __future__ import annotations

import copy
import itertools
import pathlib

import pandas as pd
import yaml

from . import cli
from .closedloop import Scenario, qualifies, run_many


def main(config="configs/pilot.yaml", out="results"):
    base = yaml.safe_load(open(config))
    rows = []
    for margin in (0.10, 0.15, 0.20):
        cfg = copy.deepcopy(base)
        cfg["reference"]["wall_margin_m"] = margin
        combos = list(itertools.product((0.0, 0.02, 0.03), (3.0, 4.0, 6.0, 8.0), (1.0, 0.6)))
        res = run_many(cfg, [(Scenario(a_lat, mu, base["sweep"]["start_s"]), K, False, None) for K, a_lat, mu in combos])
        for (K, a_lat, mu), (s, _) in zip(combos, res):
            ok, reason = qualifies(s, cfg["qualifying"])
            rows.append(dict(wall_margin_m=margin, K=K, a_lat=a_lat, mu_scale=mu, status=reason, qualifying=ok,
                             progress_frac=round(float(s["progress_frac"]), 3), fail_s=s["fail_s"],
                             max_ey=round(float(s["max_ey"]), 3), rms_ey=round(float(s["rms_ey"]), 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    pathlib.Path(out).mkdir(exist_ok=True)
    df.to_csv(pathlib.Path(out) / "sensitivity.csv", index=False)
    print(df.pivot_table(index=["wall_margin_m", "a_lat", "mu_scale"], columns="K", values="status", aggfunc="first"))


if __name__ == "__main__":
    _, config, out = cli.parse(__doc__)
    main(config, out)
