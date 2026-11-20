# Analysis of the Pacejka residual-RL experiments

These scripts back up [`docs/rl_analysis.md`](../../docs/rl_analysis.md) and [`docs/performance_profile.md`](../../docs/performance_profile.md).
- They only import the repository code; no file under `pilot/`, `pilot_core/`, `configs/` or the existing `results*/` folders is written.
- Timing wrappers are monkeypatched inside the scripts.

Run everything from the repository root with `uv run python …`.

| Step | Command | What it does | Time (M4, 8 workers) |
| --- | --- | --- | --- |
| R1 | `results/analysis/replay.py r1` | 576 sweep laps with traces (RL δ / δa / δa+wall, S-E, P/P+ff, P-E/P-E+ff on 4 tracks); writes `data/reproduction.csv` against the saved `sweep.csv` rows | 64 s |
| R1-low | `results/analysis/replay.py r1low` | dual policies at a_lat 1–3, with and without the acceleration residual | 21 s |
| R2 | `results/analysis/replay.py r2` | all 13 checkpoints × 3 runs: validation laps and the 72-lap sweep | 145 s |
| R3 | `results/analysis/replay.py r3` | ablations of the selected policies (`zero_d`, `zero_a`, `debias`, `wall_cap`; see `rlana.py`) on the sweep cells | 58 s |
| Part 1 | `results/analysis/audit.py` | inventory, consistency checks, status counts, fastest laps, search and tolerance audit, plan comparison → `audit/` | 20 s |
| Part 2 | `results/analysis/behavior.py`, `results/analysis/ablation_detail.py` | steering / longitudinal / failure / ablation / observation-sensitivity analysis → `behavior/` | ~1.5 min |
| Part 3 | `results/analysis/training.py`, `results/analysis/reward_decomp.py` | checkpoint diagnostics, selection rule, train-log, reward terms → `training/` | 10 s |
| Part 4 | `results/analysis/profiling/bench.py <cmd>` | benchmarks → `profiling/out/` (see the script docstring for the commands) | ~25 min in total |

**Folders:**
- `data/`: replay outputs.
  - `*_summary.csv`: one row per lap.
  - `*_traces.csv.gz`: 20 Hz traces.
  - `r1_obs.npz`: policy observations and actions.
  - `r1_traces_enriched.csv.gz`: traces with post-hoc wall gaps, slip angles and tire utilisation.
- `audit/`, `behavior/`, `training/`: derived tables (`.csv`) and figures (`fig_*.png`).
- `profiling/out/`:
  - benchmark results (`*.json`, `*.csv`), each with system-load snapshots;
  - `native/`: cProfile output and `/usr/bin/sample` call trees.

**Notes:**
- Laps are deterministic: R1 reproduces all 576 saved sweep laps exactly.
- Replays write a summary file per batch and skip a batch whose summary already exists; delete the file to re-run it.
