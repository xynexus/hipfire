#!/usr/bin/env bash
# Phase 1: 1-bit cold probe (0.8B) + parity_kv_hier oracle gate.
# Isolate quant at fold=1 (4/2/1-bit ladder), then fold=4 operating point.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
REFDUMP=/tmp/ref08_o0.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/phase1_1bit_08b_o0.md
: >"$OUT"

./target/release/hipfire lock acquire hier-kv-phase1 --timeout-secs 300 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT

echo "=== parity_kv_hier oracle gate ===" | tee -a "$OUT"
if ./target/release/examples/parity_kv_hier >/tmp/parity_kv_hier.log 2>&1; then
  echo "parity_kv_hier: PASS" | tee -a "$OUT"
else
  echo "parity_kv_hier: FAIL (see /tmp/parity_kv_hier.log)" | tee -a "$OUT"
  tail -20 /tmp/parity_kv_hier.log | tee -a "$OUT"
fi
echo "" >>"$OUT"

echo "# Phase 1 1-bit cold probe 0.8B (mq4+, bf16 ref PPL 24.05, offset 0 ctx 2048)" >>"$OUT"
echo "" >>"$OUT"
echo "| config | PPL | KLD/tok |" >>"$OUT"
echo "|---|---|---|" >>"$OUT"

run() {
  local label="$1"; shift
  local log=/tmp/phase1_${label//[^a-zA-Z0-9]/_}.log
  env "$@" $PX "$CAND" "$CORPUS" --ctx 2048 --offset 0 --kv-mode kvarn --kld-ref "$REFDUMP" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $label | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: $label ppl=$ppl kld=$kld"
}

# isolate quant (fold=1): fill 2-bit and 1-bit rungs (4-bit fold=1 = 26.52 from phase0)
run "fold=1 hot=64 2-bit"       HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=1 HIPFIRE_KV_COLD_BITS=2
run "fold=1 hot=64 1-bit"       HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=1 HIPFIRE_KV_COLD_BITS=1
# operating point (fold=4): 1-bit K, then 1-bit K+V (4-bit=30.75, 2-bit=31.05 from phase0)
run "fold=4 hot=64 1-bit K"     HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=1
run "fold=4 hot=64 1-bit K+V"   HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=1 HIPFIRE_KV_COLD_V_BITS=1
echo "ALL DONE"
