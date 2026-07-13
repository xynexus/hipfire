#!/usr/bin/env bash
# Phase 2 pre-decision: hot-budget x fold knee at 2-bit cold (0.8B).
# Answers "does a bigger hot ring at 2-bit reach the target before we build the
# RoPE-dephased merge kernel?" (design-lever #3 vs #1). bf16 ref PPL 24.05.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
REFDUMP=/tmp/ref08_o0.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/hotfold_knee_08b_o0.md
: >"$OUT"
echo "# hot x fold knee at 2-bit cold, 0.8B (mq4+, bf16 ref 24.05, offset 0 ctx 2048)" >>"$OUT"
echo "" >>"$OUT"
echo "| hot | fold | PPL | KLD/tok |" >>"$OUT"
echo "|---|---|---|---|" >>"$OUT"

./target/release/hipfire lock acquire hier-kv-knee --timeout-secs 300 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT

run() {
  local hot="$1" fold="$2"
  local log=/tmp/knee_h${hot}_f${fold}.log
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$hot HIPFIRE_KV_FOLD_M=$fold HIPFIRE_KV_COLD_BITS=2 \
    $PX "$CAND" "$CORPUS" --ctx 2048 --offset 0 --kv-mode kvarn --kld-ref "$REFDUMP" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $hot | $fold | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: hot=$hot fold=$fold ppl=$ppl kld=$kld"
}

for hot in 64 128 256 512; do
  for fold in 4 8; do
    run $hot $fold
  done
done
echo "ALL DONE"
