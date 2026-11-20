"""Part 3: per-term reward magnitudes along the replayed sweep laps, and acceleration residual vs mu / a_lat.

    uv run python results/analysis/reward_decomp.py
Re-evaluates the training reward (pilot.residual_env.ResidualRacingEnv.step) on the 20 Hz traces of r1 (sweep laps,
which start on the reference line, so they approximate but do not equal training episodes).
Writes results/analysis/training/reward_terms.csv and ares_by_condition.csv.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

T = R.OUT / "training"


def main():
    tr = pd.read_csv(R.OUT / "data/r1_traces_enriched.csv.gz")
    tr = tr[tr.ctrl.isin(list(R.EXPS))].sort_values(["lap_id", "t"])
    rows = []
    for ctrl, g in tr.groupby("ctrl"):
        w = R.load_cfg(ctrl)["residual"]["reward"]
        dprog = g.groupby("lap_id").progress.diff().shift(-1)  # progress over the next interval
        terms = pd.DataFrame({
            "progress": w["progress"] * dprog / np.maximum(g.v_plan * 0.05, 1e-3),
            "e_y": -w["e_y"] * g.e_y**2,
            "res": -w["res"] * g.delta_res**2,
            "dres": -w["dres"] * g.groupby("lap_id").delta_res.diff().fillna(0) ** 2,
            "ares": -w.get("ares", 0) * g.a_res.fillna(0) ** 2,
            "dares": -w.get("dares", 0) * g.groupby("lap_id").a_res.diff().fillna(0) ** 2,
            "beta": -w.get("beta", 0) * np.maximum(g.slip.abs() - w.get("beta_soft_rad", 0.15), 0) ** 2,
        }).dropna()
        rows.append(dict(ctrl=ctrl, **{k: v for k, v in terms.mean().items()}, n_steps=len(terms)))
        # what a constant straight-line bias b costs per step: res*b^2 (+ e_y term of the offset it causes)
        st = g[g.seg == "straight"]
        b, ey = st.delta_res.mean(), st.e_y.mean()
        rows[-1].update(straight_bias=b, straight_ey=ey, bias_cost_per_step=w["res"] * b**2 + w["e_y"] * ey**2,
                        slowdown_cost_per_step=1 - (g.v / g.v_plan).mean())
    rt = pd.DataFrame(rows)
    rt.to_csv(T / "reward_terms.csv", index=False, float_format="%.5f")
    du = tr[tr.ctrl.isin(["dual", "wall"])]
    ab = du.groupby(["ctrl", "a_lat", "mu_scale"]).agg(a_res=("a_res", "mean"), v_ratio=("v", "sum"))
    ab["v_ratio"] = du.groupby(["ctrl", "a_lat", "mu_scale"]).apply(lambda x: (x.v / x.v_plan).mean())
    ab.to_csv(T / "ares_by_condition.csv", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(rt.to_string(index=False))
    print(ab.unstack("mu_scale").round(3).to_string())


if __name__ == "__main__":
    main()
