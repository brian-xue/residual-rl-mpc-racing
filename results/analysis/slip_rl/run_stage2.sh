#!/usr/bin/env bash
# SlipMPC + RL, Stages 2-4 (docs/slip_rl_plan.md). Run from the repository root.
#   1. no-learning baselines on the 144-lap reporting grid (S-E, S-E at 0.97 x speed)
#   2. arms R and P', seeds 0-2, 3M steps each (Austin training, published PPO settings), each followed by the
#      protocol (checkpoint selection on Austin, then the reporting grid with the selected checkpoint)
#   3. diagnostic only, last: the reporting grid with every checkpoint (--all-ckpts; selection never sees it)
# Skips finished steps, so it can be re-run after an interruption.
set -euo pipefail
R=configs/pilot_st_pacejka_slip_res.yaml
P=configs/pilot_st_pacejka_slip_speed.yaml
D=results_st_pacejka/slip_rl
mkdir -p $D/R $D/P

if [ ! -f $D/baselines/SE097/s0/protocol_report.csv ]; then
  echo "== baselines $(date +%T)"
  uv run python results/analysis/slip_rl/stage2.py baselines > $D/baselines.log 2>&1
fi

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
  run $R $D/R/s$s $s
  run $P $D/P/s$s $s
done

uv run python -m pilot.protocol compare --out $D/compare \
  SE/s0=$D/baselines/SE/s0 SE097/s0=$D/baselines/SE097/s0 \
  R/s0=$D/R/s0 R/s1=$D/R/s1 R/s2=$D/R/s2 P/s0=$D/P/s0 P/s1=$D/P/s1 P/s2=$D/P/s2 > $D/compare.log 2>&1
echo "== primary results in $D/compare $(date +%T)"

for s in 0 1 2; do
  for arm in R P; do
    cfg=$R; [ $arm = P ] && cfg=$P
    if [ ! -f $D/$arm/s$s/protocol_report_all_ckpts.csv ]; then
      echo "== all-ckpts $D/$arm/s$s $(date +%T)"
      uv run python -m pilot.protocol run --config $cfg --run-dir $D/$arm/s$s --all-ckpts > $D/$arm/s$s.allckpts.log 2>&1
    fi
  done
done
uv run python -m pilot.protocol compare --out $D/compare \
  SE/s0=$D/baselines/SE/s0 SE097/s0=$D/baselines/SE097/s0 \
  R/s0=$D/R/s0 R/s1=$D/R/s1 R/s2=$D/R/s2 P/s0=$D/P/s0 P/s1=$D/P/s1 P/s2=$D/P/s2 > $D/compare.log 2>&1
echo "== all done $(date +%T)"
