# Follow-up analysis: commands and files

The findings are in [REPORT.md](REPORT.md). Run everything from the repository root; the times are for an Apple M4 with 8 workers.

| Step | Command | Output | Time |
| --- | --- | --- | --- |
| Rollout tracing (24 runs) | `uv run python results/analysis/followup/trace_rollouts.py run` | `raw/rollout_runs.json`, `raw/rollout_trace.csv.gz` | ~6 min |
| Rollout analysis | `uv run python results/analysis/followup/trace_rollouts.py analyze` then `uv run python results/analysis/followup/steady_path.py` | `rollout/*.csv`, `rollout/fig_rollout_timeline.png`, `raw/rollout_steps.csv.gz` | ~1 min |
| Paired ablations (1,116 laps) | `uv run python results/analysis/followup/ablate.py run` | `raw/ablation_summary.csv`, `raw/ablation_traces.csv.gz`, `raw/ablation_bias.csv`, `raw/austin_straights.csv` | ~3 min |
| Ablation analysis | `uv run python results/analysis/followup/ablate.py analyze` | `ablation/*.csv`, `ablation/late_vs_early.json`, `ablation/fig_segment_overlays.png` | ~1 min |
| SlipMPC base | `uv run python results/analysis/followup/feasibility.py slipbase` | `feasibility/slipbase.json`, `feasibility/slipmpc_recovers.csv`, `feasibility/slipmpc_loses.csv` | ~10 s |
| Friction probe | `uv run python results/analysis/followup/feasibility.py probe` | `raw/probe_data.csv.gz`, `feasibility/friction_probe*.{csv,json}` | ~2 min |

**Shared code:**
- `rollout_lib.py`: the traced `SubprocVecEnv`. It is SB3 2.9.2's worker plus timestamps, with timers patched into `LapSim.plan`.
- `../rlana.py`: experiment registry and policy loading.

**Inputs:**
- **Tracked in git:** `results_st_pacejka/residual_dual_wall/` checkpoints `ckpt_1250000.zip` and `ckpt_2500000.zip`. Only the 1.25M checkpoint is tracked; the 2.5M checkpoint is a local, git-ignored file.
- **Not tracked:** `feasibility.py slipbase` reads `results/analysis/data/r1_summary.csv` (regenerate with `results/analysis/replay.py r1`).

**Timing caveat:** timing results depend on background load, which is recorded per run in `raw/rollout_runs.json`. Laps and replays are deterministic.
