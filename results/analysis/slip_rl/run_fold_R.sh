#!/usr/bin/env bash
# SlipMPC + RL, arm R in a leave-one-track-out fold (docs/slip_rl_plan.md, "Next: arm R in a leave-one-track-out
# fold"). Run from the repository root:   results/analysis/slip_rl/run_fold_R.sh Spielberg
# The multi-track fold script with arm R's configs and runs: base = the Stage 2 runs R/s0-s2 (Austin), A3 = the same
# re-selected on the three training tracks, MT = trained and selected on them, MT-A = MT selected on Austin.
set -euo pipefail
H=${1:?usage: run_fold_R.sh HOLDOUT_TRACK}
h=$(echo "$H" | tr '[:upper:]' '[:lower:]')
SEEDS=${SEEDS:-0 1 2} BASE_CFG=configs/pilot_st_pacejka_slip_res.yaml MT_CFG=configs/pilot_st_pacejka_slip_res_mt.yaml \
  BASE_RUN_FMT=results_st_pacejka/slip_rl/R/s%s MT_DIR=results_st_pacejka/slip_rl/heldout_$h \
  CMP_OUT=results/analysis/slip_rl/heldout_$h exec results/analysis/multitrack/run_fold.sh "$H"
