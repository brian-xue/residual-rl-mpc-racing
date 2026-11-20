"""Part 1: frozen-policy PPO rollouts, traced vs untraced, 4 vs 8 envs, two wall-aware checkpoints.

    uv run python results/analysis/followup/trace_rollouts.py run      # ~6 min, writes raw/rollout_*.csv.gz
    uv run python results/analysis/followup/trace_rollouts.py analyze  # tables + timeline figure

Rollouts are SB3 PPO ``learn`` with ``train`` replaced by a no-op: stochastic action sampling, rollout buffer,
VecMonitor + VecNormalize exactly as in pilot.train_residual, but the policy stays frozen. Each run: 3 rollouts of
n_steps (1024) per env; the first rollout is warm-up. Worker seeds 1000+i, so the 4-env runs use the first four
initial-condition streams of the 8-env runs.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

RAW = HERE / "raw"
OUT = HERE / "rollout"
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
CKPTS = ["ckpt_1250000.zip", "ckpt_2500000.zip"]


def load_state():
    load = os.getloadavg()
    ps = subprocess.run(["ps", "-Ao", "%cpu,comm"], capture_output=True, text=True).stdout.splitlines()[1:]
    other = sum(float(ln.split(None, 1)[0]) for ln in ps if "python" not in ln.lower() and ln.strip())
    return dict(load1=load[0], other_cpu_pct=round(other, 1))


def one_run(ckpt, n_envs, traced, rollouts=3):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    from rollout_lib import EnvFactory, TracedSubprocVecEnv

    cfg = R.load_cfg("wall")
    pc = cfg["residual"]["ppo"]
    fns = [EnvFactory(cfg, 1000 + i) for i in range(n_envs)]
    base = TracedSubprocVecEnv(fns) if traced else SubprocVecEnv(fns, start_method="spawn")
    venv = VecNormalize(VecMonitor(base, info_keywords=("outcome",)), norm_obs=False, norm_reward=True, gamma=pc["gamma"])
    model = PPO.load(str(R.EXPS["wall"][1] / ckpt), env=venv, device="cpu")
    model.train = lambda: None  # frozen policy: rollouts only
    marks = []

    class CB(BaseCallback):
        def _on_rollout_start(self):
            marks.append(("rs", time.perf_counter_ns()))

        def _on_rollout_end(self):
            marks.append(("re", time.perf_counter_ns()))

        def _on_step(self):
            return True

    st0 = load_state()
    model.learn(pc["n_steps"] * n_envs * rollouts, callback=CB(), reset_num_timesteps=True)
    st1 = load_state()
    roll = [(b[1] - a[1]) / 1e9 for a, b in zip(marks, marks[1:]) if a[0] == "rs" and b[0] == "re"]
    win = [(a[1], b[1]) for a, b in zip(marks, marks[1:]) if a[0] == "rs" and b[0] == "re"]
    recs = base.records if traced else []
    venv.close()
    return dict(ckpt=ckpt, n_envs=n_envs, traced=traced, rollout_s=roll, steady_rollout_s=float(np.median(roll[1:])),
                steps_per_s=pc["n_steps"] * n_envs / float(np.median(roll[1:])), torch_threads=torch.get_num_threads(),
                load_before=st0, load_after=st1), recs, win


def flatten(recs, win, tag):
    """One row per (vectorised step, worker); steps in the warm-up rollout are flagged."""
    rows = []
    w0 = win[1][0] if len(win) > 1 else 0
    for k, r in enumerate(recs):
        for j, w in enumerate(r["workers"]):
            rows.append(dict(tag=tag, vstep=k, worker=j, warm=r["async0"] < w0, async0=r["async0"],
                             wait1=r["wait1"], main_cpu=r["wait_cpu1"] - r["async_cpu0"],
                             dispatch=r["dispatch"][j], got=r["got"][j], **{k2: v for k2, v in w.items()}))
    return rows


def run():
    meta, rows = [], []
    order = [(c, n, t) for rep in range(3) for c in CKPTS for n in (4, 8) for t in (False, True)]
    for i, (c, n, t) in enumerate(order):
        m, recs, win = one_run(c, n, t)
        m["rep"] = i // 8
        meta.append(m)
        print({k: v for k, v in m.items() if k != "rollout_s"}, flush=True)
        if t:
            rows += flatten(recs, win, f"{c}|{n}|{m['rep']}")
    (RAW / "rollout_runs.json").write_text(json.dumps(meta, indent=1))
    pd.DataFrame(rows).to_csv(RAW / "rollout_trace.csv.gz", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()


# ------------------------------------------------------------------------------------------------ analysis
def per_step(df):
    """One row per vectorised step: latency, cycle, straggler and the critical-path decomposition (µs)."""
    out = []
    for (tag, k), g in df.groupby(["tag", "vstep"], sort=False):
        g = g.sort_values("worker")
        s = g.loc[g.send.idxmax()]  # straggler: last worker to send its result
        a0, w1 = g.async0.iloc[0], g.wait1.iloc[0]
        reset = (s.reset1 - s.reset0) if s.reset0 else 0
        exec_ = s.step1 - s.step0
        out.append(dict(
            tag=tag, vstep=k, warm=bool(g.warm.iloc[0]), n=len(g), async0=a0, wait1=w1,
            latency=(w1 - a0) / 1e3, main_cpu_in_step=g.main_cpu.iloc[0] / 1e3,
            straggler=int(s.worker), straggler_reset=bool(s.reset0), straggler_solve_ms=s.solve_ms,
            straggler_plan_us=s.plan_ns / 1e3, straggler_exec=exec_ / 1e3, straggler_cpu=s.cpu_step / 1e3,
            straggler_preempt=(exec_ - s.cpu_step) / 1e3,
            # critical path pieces (sum exactly to latency)
            cp_dispatch=(s.dispatch - a0) / 1e3,  # main sends commands to workers 0..straggler
            cp_ipc_in=(s.recv - s.dispatch) / 1e3,  # pipe transfer + worker wake-up (not separable)
            cp_pre=(s.step0 - s.recv) / 1e3, cp_exec=exec_ / 1e3, cp_reset=reset / 1e3,
            cp_post=(s.send - s.step1 - reset) / 1e3,  # info dict, terminal obs, pickling in worker
            cp_tail=(w1 - s.send) / 1e3,  # transfer back + main receipt/unpickle of remaining results
            mean_exec=((g.step1 - g.step0) / 1e3).mean(), max_exec=((g.step1 - g.step0) / 1e3).max(),
            any_reset=bool((g.reset0 > 0).any()), exec_spread=((g.step1 - g.step0).max() - (g.step1 - g.step0).min()) / 1e3,
            max_ipc_in=((g.recv - g.dispatch) / 1e3).max(), busy_sum=((g.send - g.recv) / 1e3).sum()))
    st = pd.DataFrame(out).sort_values(["tag", "vstep"])
    st["cycle"] = st.groupby("tag").async0.diff().shift(-1) / 1e3  # to the next step_async
    st["main_gap"] = (st.groupby("tag").async0.shift(-1) - st.wait1) / 1e3  # policy, buffer, wrappers
    return st


def analyze():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = pd.DataFrame(json.loads((RAW / "rollout_runs.json").read_text()))
    meta["load1_before"] = [m["load1"] for m in meta.load_before]
    meta["other_cpu"] = [m["other_cpu_pct"] for m in meta.load_before]
    thr = meta.groupby(["ckpt", "n_envs", "traced"]).steps_per_s.agg(["median", "min", "max", "count"]).reset_index()
    thr.to_csv(OUT / "throughput.csv", index=False, float_format="%.1f")
    df = pd.read_csv(RAW / "rollout_trace.csv.gz")
    st = per_step(df)
    st = st[~st.warm & st.cycle.notna()]
    st[["ckpt", "n_envs", "rep"]] = st.tag.str.split("|", expand=True)
    st["n_envs"] = st.n_envs.astype(int)
    st.to_csv(RAW / "rollout_steps.csv.gz", index=False, float_format="%.2f")
    cp = ["cp_dispatch", "cp_ipc_in", "cp_pre", "cp_exec", "cp_reset", "cp_post", "cp_tail"]
    rows = []
    for (ck, n), g in st.groupby(["ckpt", "n_envs"]):
        p95 = g.latency.quantile(0.95)
        slow, norm = g[g.latency >= p95], g[g.latency < g.latency.quantile(0.5)]
        cyc = g.cycle
        r = dict(ckpt=ck, n_envs=n, vsteps=len(g),
                 steps_per_s_traced=n / (cyc.mean() / 1e6),
                 lat_p50=g.latency.median(), lat_p95=p95, lat_p99=g.latency.quantile(0.99),
                 cycle_p50=cyc.median(), cycle_mean=cyc.mean(), main_gap_mean=g.main_gap.mean(),
                 main_share=g.main_gap.mean() / cyc.mean(), wait_share=g.latency.mean() / cyc.mean(),
                 worker_util=g.busy_sum.mean() / (n * cyc.mean()),
                 straggler_exec_share=g.cp_exec.mean() / g.latency.mean(),
                 **{f"mean_{c}": g[c].mean() for c in cp},
                 **{f"slow_{c}": slow[c].mean() for c in cp},
                 slow_reset_frac=slow.straggler_reset.mean(), all_reset_frac=g.straggler_reset.mean(),
                 slow_any_reset_frac=slow.any_reset.mean(), base_any_reset_frac=g.any_reset.mean(),
                 slow_solve_ms=slow.straggler_solve_ms.mean(), norm_solve_ms=norm.straggler_solve_ms.mean(),
                 slow_preempt=slow.straggler_preempt.mean(), norm_preempt=norm.straggler_preempt.mean(),
                 slow_exec=slow.cp_exec.mean(), norm_exec=norm.cp_exec.mean(),
                 slow_outside=(slow.latency - slow.cp_exec - slow.cp_reset).mean(),
                 norm_outside=(norm.latency - norm.cp_exec - norm.cp_reset).mean(),
                 straggler_counts=g.straggler.value_counts().sort_index().tolist(),
                 slow_straggler_counts=slow.straggler.value_counts().reindex(range(n), fill_value=0).tolist())
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "critical_path_summary.csv", index=False, float_format="%.2f")
    # classify each slow step (>= p95) by its dominant cause
    cls = []
    for (ck, n), g in st.groupby(["ckpt", "n_envs"]):
        p95 = g.latency.quantile(0.95)
        med = g[["cp_exec", "cp_ipc_in", "cp_tail", "straggler_solve_ms", "straggler_preempt"]].median()
        for r in g[g.latency >= p95].itertuples():
            if r.cp_reset > 0.3 * r.latency:
                c = "reset on the critical path"
            elif r.straggler_preempt > 0.3 * r.cp_exec and r.straggler_preempt > 100:
                c = "straggler descheduled during env.step (wall >> CPU)"
            elif r.straggler_solve_ms > 3 * med.straggler_solve_ms:
                c = "slow OSQP solve (>3x median)"
            elif r.cp_exec > 2 * med.cp_exec:
                c = "slow env.step execution (CPU-bound)"
            elif (r.cp_ipc_in + r.cp_tail + r.cp_dispatch) > 0.5 * r.latency:
                c = "outside env.step (dispatch/pipe/wake-up/receipt)"
            else:
                c = "mixed / not attributable"
            cls.append(dict(ckpt=ck, n_envs=n, cause=c))
    cdf = pd.DataFrame(cls).groupby(["ckpt", "n_envs", "cause"]).size().unstack(fill_value=0)
    cdf.to_csv(OUT / "slow_step_causes.csv")
    # timeline figure: a median and a p99 vectorised step, 8 envs, ckpt 1.25M, last repeat
    g = st[(st.ckpt == "ckpt_1250000.zip") & (st.n_envs == 8)]
    tag = g.tag.iloc[-1]
    gg = g[g.tag == tag]
    picks = [("median step", gg.iloc[(gg.latency - gg.latency.median()).abs().argmin()]),
             ("slow steady-state step (p99, no cold reset)",
              (lambda h: h.iloc[(h.latency - h.latency.quantile(0.99)).abs().argmin()])(gg[gg.cp_reset < 5000]))]
    raw = df[df.tag == tag]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, (title, sr) in zip(axes, picks):
        w = raw[raw.vstep == sr.vstep].sort_values("worker")
        nxt = st[(st.tag == tag) & (st.vstep == sr.vstep)].iloc[0]
        t0 = sr.async0
        rel = lambda x: (x - t0) / 1e3  # noqa: E731
        y = len(w)
        ax.barh(y, rel(sr.wait1), left=0, color="#90a4ae", height=0.6)
        ax.barh(y, nxt.main_gap, left=rel(sr.wait1), color="#ffb74d", height=0.6)
        for r in w.itertuples():
            j = r.worker
            ax.barh(j, rel(r.recv) - rel(r.dispatch), left=rel(r.dispatch), color="#e0e0e0", height=0.6)
            ax.barh(j, (r.plan_ns) / 1e3, left=rel(r.step1) - r.plan_ns / 1e3, color="#1565c0", height=0.6)
            ax.barh(j, (r.step1 - r.step0) / 1e3 - r.plan_ns / 1e3, left=rel(r.step0), color="#64b5f6", height=0.6)
            if r.reset0:
                ax.barh(j, (r.reset1 - r.reset0) / 1e3, left=rel(r.reset0), color="#c62828", height=0.6)
            ax.barh(j, rel(r.got) - rel(r.send), left=rel(r.send), color="#cfd8dc", height=0.6)
            ax.plot([rel(r.send)], [j], "k|", ms=10)
            ax.plot([rel(r.got)], [j], "g|", ms=10)
        ax.set_yticks(range(len(w) + 1), [f"worker {j}" for j in range(len(w))] + ["main"], fontsize=7)
        ax.set_title(f"{title}: latency {sr.latency:.0f} µs; straggler = worker {sr.straggler}"
                     f"{' (reset)' if sr.straggler_reset else ''}; main-process gap after: {nxt.main_gap:.0f} µs", fontsize=9)
        ax.set_xlabel("µs since step_async (ckpt 1.25M, 8 envs)")
        ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch

    fig.legend([Patch(color=c) for c in ("#90a4ae", "#ffb74d", "#e0e0e0", "#64b5f6", "#1565c0", "#c62828", "#cfd8dc")],
               ["main: step_async..step_wait", "main: policy/buffer/wrappers before next step", "pipe + wake-up (in)",
                "env.step: obs/physics/reward", "env.step: MPC plan (incl. OSQP)", "auto-reset", "send → main receipt"],
               fontsize=7, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_rollout_timeline.png", dpi=130)
    plt.close(fig)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(thr.to_string(index=False))
    print(summ.T.to_string())
    print(cdf.to_string())


if __name__ == "__main__" and sys.argv[1] == "analyze":
    analyze()
