#!/usr/bin/env bash
# Decision 1 (A): long-context merge-penalty measurement. The 2048-ctx knee could
# not settle Phase 2 because hot=512 there = 25% exact (barely compressing). At
# ctx=16384/hot=512 the cache is ~97% cold/merged — the regime the feature exists
# for. Question: does the merge penalty (fold=1 vs fold=4) balloon at long ctx?
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
REF=/home/sadara/.hipfire/w8probe/qwen3.5-0.8b-bf16.hfq
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
CTX=${CTX:-16384}
REFDUMP=/tmp/ref08_lc${CTX}.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/longctx_08b_ctx${CTX}.md
: >"$OUT"
echo "# Long-ctx merge penalty 0.8B (mq4+), ctx=$CTX hot=512 2-bit, offset 0" >>"$OUT"
echo "" >>"$OUT"

./target/release/hipfire lock acquire hier-kv-longctx --timeout-secs 600 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT

echo "=== ref dump (bf16, ctx=$CTX) ==="
$PX "$REF" "$CORPUS" --ctx "$CTX" --offset 0 --dump-ref "$REFDUMP" >/tmp/lc_ref.log 2>&1
refppl=$(grep -oE 'PPL: *[0-9.]+' /tmp/lc_ref.log | tail -1 | grep -oE '[0-9.]+')
echo "bf16 ref PPL=$refppl (ctx=$CTX)"; echo "bf16 ref PPL=$refppl" >>"$OUT"
echo "" >>"$OUT"; echo "| config | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|" >>"$OUT"

run() {
  local label="$1"; shift
  local log=/tmp/lc_${label//[^a-zA-Z0-9]/_}.log
  env "$@" $PX "$CAND" "$CORPUS" --ctx "$CTX" --offset 0 --kv-mode kvarn --kld-ref "$REFDUMP" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $label | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: $label ppl=$ppl kld=$kld"
}

# hot=512, 2-bit cold. fold=1 = no-merge floor; fold=4/8 = the merge penalty at long ctx.
run "hier fold=1 hot=512"  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=1 HIPFIRE_KV_COLD_BITS=2
run "hier fold=4 hot=512"  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2
run "hier fold=8 hot=512"  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=8 HIPFIRE_KV_COLD_BITS=2
echo "ALL DONE"
