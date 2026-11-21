#!/usr/bin/env bash
# One leave-one-track-out fold of the multi-track study (docs/multitrack_plan.md). Run from the repository root:
#   results/analysis/multitrack/run_fold.sh Spielberg      # or Silverstone, Catalunya, Austin
#   SEEDS="0 1 2" results/analysis/multitrack/run_fold.sh Silverstone
# Variants (5 seeds by default):
#   base: Austin-only baseline runs, checkpoint selected on Austin (published protocol)
#   A3:   the same runs re-selected on the three training tracks of this fold (tag no<Holdout>, lapping only)
#   MT:   trained on the three training tracks of this fold, selected on them (tag no<Holdout>)
#   MT-A: the MT runs selected on Austin only (no tag)
# Idempotent: training is skipped if final.zip exists and a protocol step if its report exists, so re-running
# only redoes the comparison. Comparison -> results/analysis/multitrack/heldout_<holdout>/.
set -euo pipefail
H=${1:?usage: run_fold.sh HOLDOUT_TRACK}
ALL=(Austin Catalunya Silverstone Spielberg)
TRAIN=(); for t in "${ALL[@]}"; do [ "$t" != "$H" ] && TRAIN+=("$t"); done
[ ${#TRAIN[@]} -eq 3 ] || { echo "unknown track $H (one of ${ALL[*]})"; exit 1; }
SEEDS=${SEEDS:-0 1 2 3 4}
h=$(echo "$H" | tr '[:upper:]' '[:lower:]')
M=configs/pilot_st_pacejka_residual_dual_wall_mt.yaml
B=configs/pilot_st_pacejka_residual_dual_wall.yaml
D=results_st_pacejka/multitrack_heldout_$h
TAG=no$H
base_dir() { [ "$1" = 0 ] && echo results_st_pacejka/residual_dual_wall || echo results_st_pacejka/seeds/base/s$1; }
protocol() {  # cfg run_dir tag [extra args]: selection + report, skipped if the report exists
  local cfg=$1 run=$2 tag=$3; shift 3
  local sx=${tag:+_$tag} log=$run.protocol${tag:+_$tag}.log
  if [ -f "$run/protocol_report$sx.csv" ]; then echo "   (protocol$sx of $run exists, skipped)"; return; fi
  echo "== protocol$sx $run $(date +%T)"
  if [ -n "$tag" ]; then set -- "$@" --select-tracks "${TRAIN[@]}" --tag "$tag"; fi
  uv run python -m pilot.protocol run --config "$cfg" --run-dir "$run" "$@" > "$log" 2>&1
}
train() {  # cfg out seed [extra args]: skipped if the run exists
  local cfg=$1 out=$2 seed=$3; shift 3
  if [ -f "$out/train_config.yaml" ]; then return; fi
  echo "== train $out (seed $seed) $(date +%T)"
  uv run python -m pilot.train_residual --config "$cfg" --seed "$seed" --out "$out" "$@" > "$out.train.log" 2>&1
}
echo "== fold: hold out $H, train on ${TRAIN[*]}, seeds $SEEDS $(date +%T)"
mkdir -p "$D" results_st_pacejka/seeds/base
for s in $SEEDS; do
  b=$(base_dir "$s")
  train $B "$b" "$s"                      # Austin-only baseline (seed 0 is the published run)
  protocol $B "$b" "" --all-ckpts         # base: Austin selection
  protocol $B "$b" "$TAG"                 # A3: re-selection on this fold's training tracks
  train $M "$D/s$s" "$s" --holdout "$H"
  protocol $M "$D/s$s" "$TAG" --all-ckpts # MT
  protocol $M "$D/s$s" ""                 # MT-A
done
args=()
for s in $SEEDS; do args+=("base/s$s=$(base_dir "$s")"); done
for s in $SEEDS; do args+=("A3/s$s=$(base_dir "$s")@$TAG"); done
for s in $SEEDS; do args+=("MT/s$s=$D/s$s@$TAG"); done
for s in $SEEDS; do args+=("MT-A/s$s=$D/s$s"); done
uv run python -m pilot.protocol compare --out "results/analysis/multitrack/heldout_$h" "${args[@]}"
echo "== done $(date +%T)"
