#!/usr/bin/env bash
# Multi-track (leave-one-track-out) study, docs/multitrack_plan.md. Run from the repository root.
#   MT:   trained on Austin + Catalunya + Silverstone, selected on the same three tracks (tag sel3)
#   MT-A: the same runs selected on Austin only (default protocol, no tag)
#   A3:   the Austin-only baseline seeds re-selected on the three tracks (tag sel3, lapping only)
set -euo pipefail
M=configs/pilot_st_pacejka_residual_dual_wall_mt.yaml
B=configs/pilot_st_pacejka_residual_dual_wall.yaml
D=results_st_pacejka/multitrack
SEL3="--select-tracks Austin Catalunya Silverstone --tag sel3"
mkdir -p $D
for s in ${SEEDS:-0 1 2}; do
  if [ ! -f "$D/s$s/final.zip" ]; then
    echo "== train $D/s$s (seed $s) $(date +%T)"
    uv run python -m pilot.train_residual --config $M --seed $s --out $D/s$s > $D/s$s.train.log 2>&1
  fi
  echo "== protocol $D/s$s $(date +%T)"
  uv run python -m pilot.protocol run --config $M --run-dir $D/s$s --all-ckpts $SEL3 > $D/s$s.protocol_sel3.log 2>&1
  uv run python -m pilot.protocol run --config $M --run-dir $D/s$s > $D/s$s.protocol.log 2>&1
done
for r in results_st_pacejka/residual_dual_wall results_st_pacejka/seeds/base/s1 results_st_pacejka/seeds/base/s2; do
  echo "== re-select $r $(date +%T)"
  uv run python -m pilot.protocol run --config $B --run-dir $r $SEL3 > $r.protocol_sel3.log 2>&1
done
uv run python -m pilot.protocol compare --out results/analysis/multitrack \
  base/s0=results_st_pacejka/residual_dual_wall base/s1=results_st_pacejka/seeds/base/s1 base/s2=results_st_pacejka/seeds/base/s2 \
  A3/s0=results_st_pacejka/residual_dual_wall@sel3 A3/s1=results_st_pacejka/seeds/base/s1@sel3 A3/s2=results_st_pacejka/seeds/base/s2@sel3 \
  MT/s0=$D/s0@sel3 MT/s1=$D/s1@sel3 MT/s2=$D/s2@sel3 \
  MT-A/s0=$D/s0 MT-A/s1=$D/s1 MT-A/s2=$D/s2
echo "== done $(date +%T)"
