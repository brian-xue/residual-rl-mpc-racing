"""Part 3: training dynamics and checkpoint selection (train_log.csv + replay.py r2).

    uv run python results/analysis/training.py
Writes results/analysis/training/: checkpoint_diagnostics.csv, selection_check.csv, selection_alternatives.csv,
train_log_summary.csv, fig_training_checkpoints.png.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

T = R.OUT / "training"
T.mkdir(parents=True, exist_ok=True)
COL = {"delta": "#2e7d32", "dual": "#1565c0", "wall": "#e65100"}


def main():
    r2 = pd.read_csv(R.OUT / "data/r2_summary.csv")
    r2["fail"] = r2.status.isin(["wall_collision", "spin", "lateral_error", "timeout"])
    rows = []
    for (ctrl, ck, step), g in r2.groupby(["ctrl", "ckpt", "step"]):
        v = g[g.set == "validation"]
        row = dict(ctrl=ctrl, ckpt=ck, step=step, val_qual=int(v.qualifying.sum()),
                   val_progress=v.progress_frac.mean(), val_rms=v.rms_ey.mean(),
                   val_slow=int((v.status == "speed_tolerance").sum()), val_crash=int(v.fail.sum()),
                   val_ratio_max=v.lap_time_ratio.max())
        for trk in R.TRACKS:
            s = g[(g.set == "sweep") & (g.track == trk)]
            row[f"q_{trk}"] = int(s.qualifying.sum())
            row[f"slow_{trk}"] = int((s.status == "speed_tolerance").sum())
            row[f"wall_{trk}"] = int((s.status == "wall_collision").sum())
            row[f"spin_{trk}"] = int((s.status == "spin").sum())
        s = g[g.set == "sweep"]
        row.update(q_all=int(s.qualifying.sum()), slow_all=int((s.status == "speed_tolerance").sum()),
                   crash_all=int(s.fail.sum()), ratio_completed=s[s.outcome == "completed"].lap_time_ratio.mean(),
                   q_heldout=int(s[s.track != "Austin"].qualifying.sum()))
        rows.append(row)
    cd = pd.DataFrame(rows).sort_values(["ctrl", "step"])
    cd.to_csv(T / "checkpoint_diagnostics.csv", index=False, float_format="%.5f")

    # selection rule with unrounded values vs the published checkpoint; agreement with the saved csv
    chk, alt = [], []
    for ctrl, g in cd.groupby("ctrl"):
        saved = pd.read_csv(R.EXPS[ctrl][1] / "checkpoint_selection.csv").set_index("checkpoint")
        g = g.set_index("ckpt")
        same = all(saved.loc[c, "n_qualifying"] == g.loc[c, "val_qual"] for c in saved.index)
        ranked = g.sort_values(["val_qual", "val_progress", "val_rms"], ascending=[False, False, True])
        top = ranked[ranked.val_qual == ranked.val_qual.iloc[0]]
        chk.append(dict(ctrl=ctrl, saved_csv_reproduced=same, selected=ranked.index[0],
                        runner_up=ranked.index[1], tie_on_qual=len(top),
                        progress_margin=float(top.val_progress.iloc[0] - top.val_progress.iloc[1]) if len(top) > 1 else None,
                        sel_q_all=int(ranked.q_all.iloc[0]), best_q_all=int(g.q_all.max()),
                        best_q_all_ckpt=g.q_all.idxmax(), best_q_heldout=int(g.q_heldout.max()),
                        best_q_heldout_ckpt=g.q_heldout.idxmax(), sel_q_heldout=int(ranked.q_heldout.iloc[0])))
        # alternative rules (reported only): validation + Austin sweep; ignoring the progress tie-break (rms only)
        g2 = g.assign(val_plus=g.val_qual + g["q_Austin"])
        for name, key in (("published: val qual > progress > rms", ["val_qual", "val_progress", "val_rms"]),
                          ("val qual > rms (no progress tie-break)", ["val_qual", "val_rms"]),
                          ("val + Austin sweep qual > rms", ["val_plus", "val_rms"])):
            asc = [False] + [k == "val_rms" for k in key[1:]]
            pick = g2.sort_values(key, ascending=asc).index[0]
            alt.append(dict(ctrl=ctrl, rule=name, pick=pick, q_Austin=int(g.loc[pick, "q_Austin"]),
                            q_heldout=int(g.loc[pick, "q_heldout"]), q_all=int(g.loc[pick, "q_all"])))
        # rank correlation of validation qual with held-out qual over checkpoints (>= 500k)
        gg = g[g.step >= 500000]
        chk[-1]["spearman_val_vs_heldout"] = float(gg.val_qual.rank().corr(gg.q_heldout.rank()))
        chk[-1]["spearman_austin_vs_heldout"] = float(gg["q_Austin"].rank().corr(gg.q_heldout.rank()))
    pd.DataFrame(chk).to_csv(T / "selection_check.csv", index=False)
    pd.DataFrame(alt).to_csv(T / "selection_alternatives.csv", index=False)

    # train log summary: crash-rate noise, episodes per rollout, ep length
    tl_rows = []
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ctrl in R.EXPS:
        tl = pd.read_csv(R.EXPS[ctrl][1] / "train_log.csv")
        tl["crash_roll"] = tl.crash_rate.rolling(15, center=True).mean()
        for lo, hi in ((0, 0.5e6), (1.0e6, 1.5e6), (2.5e6, 3.1e6)):
            w = tl[(tl.steps >= lo) & (tl.steps < hi)]
            tl_rows.append(dict(ctrl=ctrl, window=f"{lo / 1e6:.1f}-{hi / 1e6:.1f}M", episodes_per_rollout=w.episodes.mean(),
                                crash_rate=w.crash_rate.mean(), crash_rate_sd=w.crash_rate.std(), ep_len=w.ep_len.mean(),
                                ep_return=w.ep_return.mean(), std=w["std"].mean(),
                                steps_per_s=(w.steps.diff() / w.wall_s.diff()).median()))
        ax[0].plot(tl.steps / 1e6, tl.ep_return.rolling(15, center=True).mean(), color=COL[ctrl], label=R.LABEL[ctrl])
        ax[1].plot(tl.steps / 1e6, tl.crash_roll, color=COL[ctrl])
        ax[1].plot(tl.steps / 1e6, tl.crash_rate, color=COL[ctrl], alpha=0.15, lw=0.5)
        g = cd[cd.ctrl == ctrl]
        ax[2].plot(g.step / 1e6, g.val_qual, "o-", color=COL[ctrl], ms=4)
        ax[2].plot(g.step / 1e6, g.val_slow, "x:", color=COL[ctrl], ms=5)
        ax[3].plot(g.step / 1e6, g.q_Austin, "o-", color=COL[ctrl], ms=4)
        ax[3].plot(g.step / 1e6, g.q_heldout, "s--", color=COL[ctrl], ms=4)
    ax[0].set_ylabel("episode return (rolling)")
    ax[1].set_ylabel("training crash rate")
    ax[2].set_ylabel("validation laps (of 4)\n● qualifying  × too slow (5% rule)")
    ax[3].set_ylabel("sweep qualifying\n● Austin (/18)  ■ 3 held-out (/54)")
    ax[3].set_xlabel("training steps [M]")
    for a in ax:
        a.grid(alpha=0.3)
        a.axvline(1.25, color="k", lw=0.6, ls=":")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(T / "fig_training_checkpoints.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(tl_rows).to_csv(T / "train_log_summary.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(cd[["ctrl", "step", "val_qual", "val_slow", "val_crash", "val_ratio_max", "q_Austin", "slow_Austin", "q_Spielberg",
              "q_Catalunya", "q_Silverstone", "q_all", "slow_all", "crash_all", "ratio_completed"]].to_string(index=False))
    print(pd.DataFrame(chk).to_string(index=False))
    print(pd.DataFrame(alt).to_string(index=False))
    print(pd.DataFrame(tl_rows).to_string(index=False))


if __name__ == "__main__":
    main()
