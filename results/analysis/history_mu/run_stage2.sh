#!/usr/bin/env bash
# History window for mu, Stage 2 (docs/history_mu_plan.md). Run from the repository root.
#   R-H and R-H-zero, seeds 0-2, 3M steps each (Austin training, arm R's PPO settings), interleaved by seed so a
#   partial run stays balanced. Each run is followed by the protocol (checkpoint selection on Austin, then the
#   144-lap reporting grid with the selected checkpoint), the first part of Stage 3.
# Skips finished steps, so it can be re-run after an interruption.
set -euo pipefail
H=configs/pilot_st_pacejka_slip_res_hist.yaml
Z=configs/pilot_st_pacejka_slip_res_hist_zero.yaml
D=results_st_pacejka/history_mu
mkdir -p $D/RH $D/RH0

run() {  # cfg out seed
  if [ ! -f "$2/final.zip" ]; then
    echo "== train $2 (seed $3) $(date +%T)"
    uv run python -m pilot.train_residual --config "$1" --seed "$3" --out "$2" > "$2.train.log" 2>&1
  fi
  if [ ! -f "$2/protocol_report.csv" ]; then
    echo "== protocol $2 $(date +%T)"
    uv run python -m pilot.protocol run --config "$1" --run-dir "$2" > "$2.protocol.log" 2>&1
  fi
  echo "== done $2 $(date +%T)"
}
for s in 0 1 2; do
  run $H $D/RH/s$s $s
  run $Z $D/RH0/s$s $s
done
echo "== all done $(date +%T)"
