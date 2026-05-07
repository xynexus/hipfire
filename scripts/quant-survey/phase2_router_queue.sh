#!/usr/bin/env bash
# V_prod + V4 router-Q8 ablation queue.
#
# 2 models × 2 variants × 2 corpora × 1 trial = 8 runs ~12 min.
# Single trial since prior Phase 2 confirmed bit-identical determinism.
#
# Outputs land in:
#   /tmp/hiptrx-survey/runs/phase2-routerq8-<model>/phase2_results.jsonl
# (combined wikitext+agentic, both eval_corpus_md5 fields preserved
#  per-line so synthesis can split them back out).

set -u
cd ~/hipfire
source ~/venv-torch/bin/activate
LOG_DIR=/tmp/hiptrx-survey/logs
RUNS_DIR=/tmp/hiptrx-survey/runs
mkdir -p $LOG_DIR

run_one() {
  local model=$1
  local variant=$2
  local corpus=$3      # "wikitext" or "agentic"
  local outdir=$RUNS_DIR/phase2-routerq8-$model
  local log=$LOG_DIR/phase2-routerq8-${model}-${variant}-${corpus}.log
  local results=$outdir/phase2_results.jsonl
  local eval_path
  if [ "$corpus" = "wikitext" ]; then
    eval_path=scripts/quant-survey/phase2_eval_corpus.jsonl
  else
    eval_path=scripts/quant-survey/phase2_eval_corpus_agentic.jsonl
  fi
  mkdir -p $outdir
  # Skip if already done. Match on (variant, trial=1, eval_corpus_md5).
  local md5=$(md5sum $eval_path | awk '{print $1}')
  if [ -f "$results" ] && jq -e --arg v "$variant" --arg m "$md5" \
        'select(.variant==$v and .trial==1 and .eval_corpus_md5==$m)' "$results" >/dev/null 2>&1; then
    echo "[router-queue] $(date) SKIP $model $variant on $corpus (already done)"
    return 0
  fi
  echo "[router-queue] $(date) starting $model $variant corpus=$corpus"
  HIP_VISIBLE_DEVICES=0,1,2,3 python3 scripts/quant-survey/phase2_runner.py \
    --model "$model" --variant "$variant" --trial 1 \
    --output-dir "$outdir" --max-seq-len 1024 \
    --eval-corpus "$eval_path" \
    > "$log" 2>&1
  echo "[router-queue] $(date) $model $variant corpus=$corpus exit=$?"
}

for model in qwen3.5-a3b qwen3.6-a3b; do
  for variant in V_prod V4; do
    for corpus in wikitext agentic; do
      run_one "$model" "$variant" "$corpus"
    done
  done
done
echo "[router-queue] $(date) ALL DONE"
