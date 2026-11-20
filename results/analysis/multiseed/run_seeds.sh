#!/usr/bin/env bash
# Multi-seed study: baseline (wall-aware dual policy) vs + speed-shortfall hinge. Run from the repository root.
# Baseline seed 0 is the existing run results_st_pacejka/residual_dual_wall (protocol run on it separately).
set -euo pipefail
B=configs/pilot_st_pacejka_residual_dual_wall.yaml
S=configs/pilot_st_pacejka_residual_dual_wall_slow.yaml
D=results_st_pacejka/seeds
run() {  # cfg out seed
  if [ ! -f "$2/final.zip" ]; then
    echo "== train $2 (seed $3) $(date +%T)"
    uv run python -m pilot.train_residual --config "$1" --seed "$3" --out "$2" > "$2.train.log" 2>&1
  fi
  echo "== protocol $2 $(date +%T)"
  uv run python -m pilot.protocol run --config "$1" --run-dir "$2" --all-ckpts > "$2.protocol.log" 2>&1
  echo "== done $2 $(date +%T)"
}
mkdir -p $D/base $D/slow
run $S $D/slow/s0 0
run $B $D/base/s1 1
run $S $D/slow/s1 1
run $B $D/base/s2 2
run $S $D/slow/s2 2
