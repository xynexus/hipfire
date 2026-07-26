#!/usr/bin/env bash
# T1 gating experiment: does CASK/TriAttn calibrated importance beat vnorm for the
# hier cold-tier merge? hier+triattn vs hier+vnorm, same operating point.
# Baselines from head-to-head: 2k asym4 27.04 kvarn 26.38 hier-vnorm 27.54;
# 16k asym4 18.15 kvarn 17.71 hier-vnorm 18.41.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
TRIA=/srv/hipfire/triattn/qwen3.5-0.8b.mq4.triattn.bin
OUT=benchmarks/results/hier-kv-rebaseline-20260712/t1_triattn.md
: >"$OUT"
echo "# T1: hier cold-merge ranked by TriAttn vs vnorm (0.8B mq4+, PPL/KLD vs bf16)" >>"$OUT"
echo "" >>"$OUT"; echo "| ctx | hier importance | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire hier-t1 --timeout-secs 900 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
run() { # ctx ref hot imp
  local ctx="$1" ref="$2" hot="$3" imp="$4"
  local log=/tmp/t1_${ctx}_${imp}.log
  local extra=""
  [ "$imp" = "triattn" ] && extra="HIPFIRE_KV_TRIATTN_SIDECAR=$TRIA"
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$hot HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2 \
      HIPFIRE_KV_IMPORTANCE=$imp $extra \
    $PX "$CAND" "$CORPUS" --ctx "$ctx" --offset 0 --kv-mode kvarn --kld-ref "$ref" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $ctx | $imp | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: ctx=$ctx imp=$imp ppl=$ppl kld=$kld"
}
run 2048  /tmp/ref08_o0.pkld     512  vnorm
run 2048  /tmp/ref08_o0.pkld     512  triattn
run 16384 /tmp/ref08_lc16384.pkld 2048 vnorm
run 16384 /tmp/ref08_lc16384.pkld 2048 triattn
echo "ALL DONE"
