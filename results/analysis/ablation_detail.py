"""Part 2: what the ablations change inside the laps (r3 traces vs r1): straight-line residual / offset, sideslip,
rear-tire utilisation in corners and speed ratio. Writes results/analysis/behavior/ablation_detail.csv.
    uv run python results/analysis/ablation_detail.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import behavior as Bh  # noqa: E402
import rlana as R  # noqa: E402

D = R.OUT / "data"
base = pd.read_csv(D / "r1_traces_enriched.csv.gz")
base = base[base.ctrl.isin(list(R.EXPS))].assign(ablation="none")
sm3 = pd.read_csv(D / "r3_summary.csv")
tr3 = pd.read_csv(D / "r3_traces.csv.gz")
sm3["ctrl_key"] = sm3.ctrl
tr3 = Bh.enrich(sm3, tr3)
tr3 = tr3.merge(sm3[["lap_id", "ablation"]], on="lap_id")
allt = pd.concat([base, tr3], ignore_index=True)
rows = []
for (ctrl, abl, trk), g in allt.groupby(["ctrl", "ablation", "track"]):
    st, co = g[g.seg == "straight"], g[g.seg == "corner"]
    rows.append(dict(ctrl=ctrl, ablation=abl, track=trk, straight_res=st.delta_res.mean(), straight_ey=st.e_y.mean(),
                     corner_abs_slip=co.slip.abs().mean(), corner_util_r_p95=co.util_r.quantile(0.95),
                     frac_util_r_gt_0_95=(g.util_r > 0.95).mean(), v_ratio=(g.v / g.v_plan).mean()))
t = pd.DataFrame(rows)
t.to_csv(R.OUT / "behavior/ablation_detail.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
