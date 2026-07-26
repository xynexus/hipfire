#!/usr/bin/env bash
# Head-to-head: hierarchical (f16 hot + KVarN cold) vs single-tier kvarn vs asym4/asym3.
# Validates the ratified north-star claim "KVarN + hierarchical should outperform asym".
# PPL + KLD vs bf16 at short (2k) and long (16k) ctx. NOT strictly iso-memory — hier
# trades a bigger exact hot window for heavier cold compression; flag in synthesis.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
OUT=benchmarks/results/hier-kv-rebaseline-20260712/asym_headtohead.md
: >"$OUT"
echo "# KV head-to-head: hierarchical vs single-tier kvarn vs asym (0.8B mq4+)" >>"$OUT"
echo "" >>"$OUT"
echo "| ctx | kv system | PPL | KLD/tok |" >>"$OUT"
echo "|---|---|---|---|" >>"$OUT"

./target/release/hipfire lock acquire hier-kv-headtohead --timeout-secs 900 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT

run() { # ctx ref label kvmode [env assignments...]
  local ctx="$1" ref="$2" label="$3" kvmode="$4"; shift 4
  local log=/tmp/h2h_${ctx}_${label//[^a-zA-Z0-9]/_}.log
  env "$@" $PX "$CAND" "$CORPUS" --ctx "$ctx" --offset 0 --kv-mode "$kvmode" --kld-ref "$ref" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $ctx | $label | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: ctx=$ctx $label ppl=$ppl kld=$kld"
}

# short ctx 2048 (bf16 ref PPL 24.05)
R2=/tmp/ref08_o0.pkld
run 2048 "$R2" "asym4"               asym4 HIPFIRE_KV_HIERARCHICAL=0
run 2048 "$R2" "asym3"               asym3 HIPFIRE_KV_HIERARCHICAL=0
run 2048 "$R2" "kvarn (single-tier)" kvarn HIPFIRE_KV_HIERARCHICAL=0
run 2048 "$R2" "hier hot=512 2-bit"  kvarn HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2

# long ctx 16384 (bf16 ref PPL 16.10)
R16=/tmp/ref08_lc16384.pkld
run 16384 "$R16" "asym4"               asym4 HIPFIRE_KV_HIERARCHICAL=0
run 16384 "$R16" "asym3"               asym3 HIPFIRE_KV_HIERARCHICAL=0
run 16384 "$R16" "kvarn (single-tier)" kvarn HIPFIRE_KV_HIERARCHICAL=0
run 16384 "$R16" "hier hot=2048 2-bit" kvarn HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=2048 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2
echo "ALL DONE"
