#!/usr/bin/env bash
# T1 re-test on 9B (0.8B may be a prototype-rung artifact). hier+TriAttn vs hier+vnorm,
# PPL only (relative comparison; no bf16 ref needed). Matched mq4 model+sidecar pair.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/home/sadara/.hipfire/models/qwen3.5-9b-mq4.hfq
TRIA=/home/sadara/.hipfire/triattn/qwen3.5-9b.mq4.triattn.bin
OUT=benchmarks/results/hier-kv-rebaseline-20260712/t1_9b.md
: >"$OUT"
echo "# T1 on 9B: hier cold-merge TriAttn vs vnorm (qwen3.5-9b-mq4, PPL only)" >>"$OUT"
echo "" >>"$OUT"; echo "| ctx | hier importance | PPL |" >>"$OUT"; echo "|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire hier-t1-9b --timeout-secs 1200 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
run() { # ctx hot imp
  local ctx="$1" hot="$2" imp="$3"
  local log=/tmp/t1_9b_${ctx}_${imp}.log
  local extra=""
  [ "$imp" = "triattn" ] && extra="HIPFIRE_KV_TRIATTN_SIDECAR=$TRIA"
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$hot HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2 \
      HIPFIRE_KV_IMPORTANCE=$imp $extra \
    $PX "$CAND" "$CORPUS" --ctx "$ctx" --offset 0 --kv-mode kvarn >"$log" 2>&1
  local ppl; ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  local loaded; loaded=$(grep -c "TriAttn importance: centers" "$log")
  echo "| $ctx | $imp | ${ppl:-ERR} |" >>"$OUT"
  echo "done: ctx=$ctx imp=$imp ppl=$ppl (centers_loaded=$loaded)"
}
run 2048 512 vnorm
run 2048 512 triattn
echo "ALL DONE"
