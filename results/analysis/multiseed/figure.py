"""Reporting-grid results of every checkpoint per run (diagnostic; selection never sees it), selected marked.

    uv run python results/analysis/multiseed/figure.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {"base/s0": ROOT / "results_st_pacejka/residual_dual_wall"} | {
    f"{v}/s{k}": ROOT / f"results_st_pacejka/seeds/{v}/s{k}" for v in ("base", "slow") for k in (0, 1, 2)
    if not (v == "base" and k == 0)}
COL = {"base": "#1565c0", "slow": "#c62828"}


def steps(name):
    return 3_000_000 + 6_464 if name == "final.zip" else int(name.split("_")[1].split(".")[0])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, d in RUNS.items():
        f = d / "protocol_report_all_ckpts.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        g = a.groupby("checkpoint")
        t = pd.DataFrame(dict(q=g.qualifying.sum(), done=g.completed.sum(),
                              ratio=g.apply(lambda x: x.lap_time_ratio[x.completed].mean(), include_groups=False)))
        t["steps"] = [steps(c) for c in t.index]
        t = t.sort_values("steps")
        v = label.split("/")[0]
        sel = (d / "protocol_selected.txt").read_text().strip()
        ls = {"s0": "-", "s1": "--", "s2": ":"}[label.split("/")[1]]
        for ax, col in zip(axes, ("q", "done", "ratio")):
            ax.plot(t.steps / 1e6, t[col], ls, color=COL[v], lw=1.3, label=label)
            ax.plot(steps(sel) / 1e6, t.loc[sel, col], "o", color=COL[v], ms=7, mfc="white", mew=1.5)
    for ax, title in zip(axes, ("qualifying laps (of 144)", "completed laps (of 144)", "mean lap-time ratio, completed")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training steps [M]")
        ax.grid(alpha=0.3)
    axes[2].axhline(1.05, color="k", lw=0.8, ls="--")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Reporting grid (4 tracks, 144 cells) for every checkpoint; circle = checkpoint picked by the selection rule",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig_checkpoints.png", dpi=130)


if __name__ == "__main__":
    main()
