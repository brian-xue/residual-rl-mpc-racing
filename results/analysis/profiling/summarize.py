"""Figures and tables for docs/performance_profile.md from results/analysis/profiling/out/.

    uv run python results/analysis/profiling/summarize.py
"""

from __future__ import annotations

import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

O = pathlib.Path(__file__).resolve().parent / "out"


def top_of_stack(path, n=15):
    """Parse the 'Sort by top of stack' section of a /usr/bin/sample report: (frame, samples) pairs."""
    txt = path.read_text(errors="ignore")
    i = txt.find("Sort by top of stack")
    if i < 0:
        return []
    rows = []
    for ln in txt[i:].splitlines()[1:]:
        m = re.match(r"\s+(.+?)\s+(\d+)\s*$", ln)
        if m:
            rows.append((m.group(1)[:110], int(m.group(2))))
        elif rows and not ln.strip():
            break
    tot = sum(c for _, c in rows)
    return [(f, c, c / tot) for f, c in rows[:n]], tot


def main():
    out = {}
    # 1. per-step composition (Rust RL step and S-E)
    st = json.loads((O / "steps.json").read_text())["result"]
    comp = []
    for name, v in st.items():
        if name.startswith("env.step"):
            continue
        c = v["components_us"]
        pol = c.get("policy_total", {}).get("total", 0) / max(v["ctrl_steps"], 1)
        obs = c.get("residual_obs", {}).get("total", 0) / v["ctrl_steps"]
        pred = c.get("sb3_predict", {}).get("total", 0) / v["ctrl_steps"]
        plan = c["plan"]["total"] / v["ctrl_steps"]
        adv = c["advance_interval"]["total"] / v["ctrl_steps"]
        app = c["apply"]["total"] / v["ctrl_steps"]
        tot = v["us_per_step_total"]
        comp.append(dict(workload=name, total_us=tot, mpc_plan_us=plan, physics_us=adv, apply_us=app, obs_us=obs,
                         sb3_predict_us=pred, policy_other_us=pol - obs - pred if pol else 0.0,
                         loop_other_us=tot - plan - adv - app - pol))
    cdf = pd.DataFrame(comp)
    cdf.to_csv(O / "step_composition.csv", index=False, float_format="%.2f")
    out["step_composition"] = cdf
    fig, ax = plt.subplots(figsize=(9, 3.8))
    parts = [("mpc_plan_us", "MPC plan", "#1565c0"), ("physics_us", "physics (5 steps)", "#2e7d32"),
             ("obs_us", "observation", "#fb8c00"), ("sb3_predict_us", "SB3 predict", "#c62828"),
             ("policy_other_us", "policy wrapper", "#ef9a9a"), ("apply_us", "apply", "#757575"),
             ("loop_other_us", "loop / other", "#bdbdbd")]
    left = np.zeros(len(cdf))
    for k, lab, col in parts:
        v = cdf[k].clip(lower=0).values
        ax.barh(range(len(cdf)), v, left=left, color=col, label=lab)
        left += v
    ax.set_yticks(range(len(cdf)), cdf.workload, fontsize=8)
    ax.set_xlabel("µs per 50 ms control step (whole lap / steps), single process")
    ax.legend(fontsize=7, ncol=2, loc="center right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(O / "fig_step_composition.png", dpi=130)
    plt.close(fig)

    # 2. PPO scaling
    pp = pd.read_csv(O / "ppo_scaling.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for exp, col in (("delta", "#2e7d32"), ("dual", "#1565c0"), ("wall", "#e65100")):
        g = pp[(pp.exp == exp) & (pp.threads == pp.threads.mode()[0])].drop_duplicates("n_envs")
        ax[0].plot(g.n_envs, g.steady_steps_per_s, "o-", color=col, label=f"{exp}: rollout")
        ax[0].plot(g.n_envs, g.end_to_end_steps_per_s, "s--", color=col, alpha=0.6, label=f"{exp}: rollout + update")
        ax[1].plot(g.n_envs, g.worker_busy_frac, "o-", color=col, label=exp)
    ax[0].axvline(4, color="k", lw=0.6, ls=":")
    ax[0].text(4.1, 9500, "4 P-cores", fontsize=7)
    ax[0].set_xlabel("n_envs (SubprocVecEnv workers)")
    ax[0].set_ylabel("env steps / s (single run each)")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("n_envs")
    ax[1].set_ylabel("worker busy fraction of rollout")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(O / "fig_ppo_scaling.png", dpi=130)
    plt.close(fig)

    # 3. MPC latency distributions
    ms = pd.read_csv(O / "mpc_solves.csv.gz")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for kind, col, lab in (("kin_rust", "#2e7d32", "kinematic MPC, Rust (plan call)"),
                           ("kin_py", "#1565c0", "kinematic MPC, Python (solve)"),
                           ("slip", "#8e24aa", "SlipMPC, Python (solve, total)")):
        x = np.sort(ms[(ms.kind == kind) & (~ms["first"].astype(bool))].total.values * 1e3)
        ax.plot(x, 1 - np.arange(len(x)) / len(x), color=col, label=lab)
    if (O / "pool_scaling.csv").exists():
        pl = pd.read_csv(O / "pool_scaling.csv")
        out["pool"] = pl
    x = np.sort(ms[ms.kind == "slip"].osqp_solve.values * 1e3)
    ax.plot(x, 1 - np.arange(len(x)) / len(x), color="#8e24aa", ls=":", label="SlipMPC: OSQP solve only (the logged solve_ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("latency per control step [ms] (single process)")
    ax.set_ylabel("P(latency > x)")
    ax.axvline(50, color="r", lw=0.6)
    ax.text(40, 0.5, "50 ms period", rotation=90, fontsize=7, color="r")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(O / "fig_mpc_latency.png", dpi=130)
    plt.close(fig)

    # 4. native top-of-stack
    nat = {}
    for nm in ("rl", "se"):
        p = O / "native" / f"sample_{nm}.txt"
        if p.exists():
            nat[nm] = top_of_stack(p)
    out["native"] = nat
    with open(O / "native_top_of_stack.txt", "w") as f:
        for nm, (rows, tot) in nat.items():
            f.write(f"== {nm}: {tot} samples (top-of-stack)\n")
            for fr, c, sh in rows:
                f.write(f"{sh * 100:5.1f}%  {c:6d}  {fr}\n")
    pd.set_option("display.width", 250)
    print(cdf.round(1).to_string(index=False))
    print((O / "native_top_of_stack.txt").read_text())


if __name__ == "__main__":
    main()
