#!/usr/bin/env bash
# Lever 2: PyramidKV per-layer fold/core schedule vs uniform (approx iso-memory —
# the schedule redistributes fold/core across layers around the base).
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
OUT=benchmarks/results/hier-kv-rebaseline-20260712/lever2_pyramid.md
: >"$OUT"
echo "# Lever 2: per-layer pyramid budgets vs uniform (0.8B mq4+, base fold=4 core=0.125 2-bit)" >>"$OUT"
echo "" >>"$OUT"; echo "| ctx | schedule | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire lever2 --timeout-secs 900 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
run() { # ctx ref hot sched
  local ctx="$1" ref="$2" hot="$3" sched="$4"
  local log=/tmp/lever2_${ctx}_${sched}.log
  local extra=""; [ "$sched" = "pyramid" ] && extra="HIPFIRE_KV_PYRAMID=1"
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$hot HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2 $extra \
    $PX "$CAND" "$CORPUS" --ctx "$ctx" --offset 0 --kv-mode kvarn --kld-ref "$ref" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $ctx | $sched | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: ctx=$ctx sched=$sched ppl=$ppl kld=$kld"
}
run 2048  /tmp/ref08_o0.pkld     512  uniform
run 2048  /tmp/ref08_o0.pkld     512  pyramid
run 16384 /tmp/ref08_lc16384.pkld 2048 uniform
run 16384 /tmp/ref08_lc16384.pkld 2048 pyramid
echo "ALL DONE"
