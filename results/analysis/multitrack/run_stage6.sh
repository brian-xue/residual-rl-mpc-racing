#!/usr/bin/env bash
# Stage 6 of docs/multitrack_plan.md (stage 1-4 result was in the "ambiguous" band): seeds 3 and 4 for both the
# Austin-only baseline and the multi-track policy, same protocol, then the 5-seed comparison. Run from the repo root.
set -euo pipefail
M=configs/pilot_st_pacejka_residual_dual_wall_mt.yaml
B=configs/pilot_st_pacejka_residual_dual_wall.yaml
SEL3="--select-tracks Austin Catalunya Silverstone --tag sel3"
train() {  # cfg out seed
  if [ ! -f "$2/final.zip" ]; then
    echo "== train $2 (seed $3) $(date +%T)"
    uv run python -m pilot.train_residual --config "$1" --seed "$3" --out "$2" > "$2.train.log" 2>&1
  fi
  echo "== protocol $2 $(date +%T)"
  uv run python -m pilot.protocol run --config "$1" --run-dir "$2" --all-ckpts > "$2.protocol.log" 2>&1
  uv run python -m pilot.protocol run --config "$1" --run-dir "$2" $SEL3 > "$2.protocol_sel3.log" 2>&1
}
for s in 3 4; do
  train $B results_st_pacejka/seeds/base/s$s $s
  train $M results_st_pacejka/multitrack/s$s $s
done
B0=results_st_pacejka/residual_dual_wall; BS=results_st_pacejka/seeds/base; D=results_st_pacejka/multitrack
args=(base/s0=$B0)
for s in 1 2 3 4; do args+=(base/s$s=$BS/s$s); done
args+=(A3/s0=$B0@sel3); for s in 1 2 3 4; do args+=(A3/s$s=$BS/s$s@sel3); done
for s in 0 1 2 3 4; do args+=(MT/s$s=$D/s$s@sel3); done
for s in 0 1 2 3 4; do args+=(MT-A/s$s=$D/s$s); done
uv run python -m pilot.protocol compare --out results/analysis/multitrack/five_seeds "${args[@]}"
echo "== done $(date +%T)"
