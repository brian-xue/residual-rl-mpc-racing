"""Steady-state critical path of the traced rollouts: excludes vectorised steps whose critical path contains a
cold reset (> 5 ms: first use of an a_lat level computes and caches the speed plan; see report).
    uv run python results/analysis/followup/steady_path.py
"""
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
st = pd.read_csv(HERE / "raw/rollout_steps.csv.gz")
df = pd.read_csv(HERE / "raw/rollout_trace.csv.gz")
df = df[~df.warm]
rows = []
for (ck, n), g in st.groupby(["ckpt", "n_envs"]):
    cold = g.cp_reset > 5000
    h = g[~cold]
    w = df[(df.tag.isin(g.tag.unique())) & (df.reset0 == 0)]
    ex = (w.step1 - w.step0) / 1e3
    cpu = w.cpu_step / 1e3
    per_worker = ex.groupby(w.worker).mean()
    lat = h.latency
    rows.append(dict(
        ckpt=ck, n_envs=n, vsteps=len(g), cold_reset_steps=int(cold.sum()),
        cold_reset_time_share=float(g[cold].cycle.sum() / g.cycle.sum()),
        steps_per_s_all=n / g.cycle.mean() * 1e6, steps_per_s_steady=n / h.cycle.mean() * 1e6,
        cycle_mean=h.cycle.mean(), lat_p50=lat.median(), lat_p95=lat.quantile(0.95), lat_p99=lat.quantile(0.99),
        main_gap=h.main_gap.mean(), dispatch=h.cp_dispatch.mean(), ipc_in=h.cp_ipc_in.mean(), pre=h.cp_pre.mean(),
        straggler_exec=h.cp_exec.mean(), warm_reset=h.cp_reset.mean(), post=h.cp_post.mean(), tail=h.cp_tail.mean(),
        mean_worker_exec=ex.mean(), mean_worker_cpu=cpu.mean(), exec_p95=ex.quantile(0.95),
        straggler_excess=(h.cp_exec - h.mean_exec).mean(),
        worker_exec_min=per_worker.min(), worker_exec_max=per_worker.max(),
        util=float(h.busy_sum.mean() / (n * h.cycle.mean())),
        straggler_is_last3=float((h.straggler >= n - 3).mean()),
        main_cpu_per_vstep=float(h.main_cpu_in_step.mean())))
t = pd.DataFrame(rows)
t["main_share"] = t.main_gap / t.cycle_mean
t["exec_share"] = t.straggler_exec / t.cycle_mean
t["comm_share"] = (t["dispatch"] + t["ipc_in"] + t["tail"] + t["pre"] + t["post"]) / t.cycle_mean
t["reset_share"] = t.warm_reset / t.cycle_mean
t["unattributed"] = t.cycle_mean - (t.main_gap + t.dispatch + t["ipc_in"] + t["pre"] + t["straggler_exec"] + t["warm_reset"] + t["post"] + t["tail"])
t.to_csv(HERE / "rollout/steady_critical_path.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 250)
print(t.round(2).T.to_string())
