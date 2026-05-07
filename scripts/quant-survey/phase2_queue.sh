#!/usr/bin/env bash
# Phase 2 sequencer: 5 variants × 3 trials × 2 A3B models = 30 runs.
#
# Each run: separate phase2_runner.py invocation. Model is reloaded
# every time (cleanest: no risk of cross-variant residue from prior
# in-place mutation). Worst case ~5 min × 30 = 2.5 hours.
#
# Output: /tmp/hiptrx-survey/runs/phase2-<model>/phase2_results.jsonl
# accumulates one line per run (15 lines per model after completion).
#
# Usage on hiptrx:
#   cd ~/hipfire && source ~/venv-torch/bin/activate
#   setsid nohup bash scripts/quant-survey/phase2_queue.sh \
#     > /tmp/hiptrx-survey/logs/phase2-queue.log 2>&1 < /dev/null &

set -u
cd ~/hipfire
source ~/venv-torch/bin/activate

LOG_DIR=/tmp/hiptrx-survey/logs
RUNS_DIR=/tmp/hiptrx-survey/runs
mkdir -p $LOG_DIR

run_one() {
  local model=$1
  local variant=$2
  local trial=$3
  local outdir=$RUNS_DIR/phase2-$model
  local log=$LOG_DIR/phase2-${model}-${variant}-t${trial}.log
  local results=$outdir/phase2_results.jsonl
  # Skip if this (variant, trial) already has a result line.
  if [ -f "$results" ] && grep -q "\"variant\":\"$variant\",\"trial\":$trial," "$results"; then
    echo "[phase2-queue] $(date) SKIP $model $variant trial=$trial (already in results)"
    return 0
  fi
  echo "[phase2-queue] $(date) starting $model $variant trial=$trial"
  HIP_VISIBLE_DEVICES=0,1,2,3 python3 scripts/quant-survey/phase2_runner.py \
    --model "$model" \
    --variant "$variant" \
    --trial "$trial" \
    --output-dir "$outdir" \
    --max-seq-len 1024 \
    > "$log" 2>&1
  local rc=$?
  echo "[phase2-queue] $(date) $model $variant trial=$trial exit=$rc"
}

for model in qwen3.5-a3b qwen3.6-a3b; do
  for variant in V1 V2 V3a V3b V3c; do
    for trial in 1 2 3; do
      run_one "$model" "$variant" "$trial"
    done
  done
done
echo "[phase2-queue] $(date) ALL DONE"
