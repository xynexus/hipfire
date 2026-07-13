#!/usr/bin/env bash
# Phase 0 re-baseline (0.8B): perplexity PPL + KLD vs bf16, per docs/plans/2026-07-12.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
REFDUMP=/tmp/ref08_o0.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/phase0_08b_o0.md
: >"$OUT"
echo "# Phase 0 re-baseline 0.8B (mq4+ candidate, bf16 ref PPL 24.05), offset 0 ctx 2048" >>"$OUT"
echo "" >>"$OUT"
echo "| config | PPL | KLD/tok |" >>"$OUT"
echo "|---|---|---|" >>"$OUT"

./target/release/hipfire lock acquire hier-kv-rebaseline-phase0 --timeout-secs 300 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT

run() { # label ; env-prefix passed via caller
  local label="$1"; shift
  local log=/tmp/phase0_${label//[^a-zA-Z0-9]/_}.log
  env "$@" $PX "$CAND" "$CORPUS" --ctx 2048 --offset 0 --kv-mode kvarn --kld-ref "$REFDUMP" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $label | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: $label ppl=$ppl kld=$kld"
}

run "plain-kvarn (hier off)"
run "hier fold=1 hot=64"                 HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=1
run "hier fold=4 hot=64 (default)"       HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=4
run "hier fold=4 hot=64 2-bit"           HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2
run "hier fold=4 hot=256 (new default)"  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=256 HIPFIRE_KV_FOLD_M=4
echo "ALL DONE"
