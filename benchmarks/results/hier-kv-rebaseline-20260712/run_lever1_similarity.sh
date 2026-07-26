#!/usr/bin/env bash
# Lever 1 gate: CASK content-similarity merge grouping vs position merge.
# Baselines: 2k position 27.54/0.153, KVarN 26.38/0.083; 16k position 18.38/0.145, KVarN 17.71/0.085.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
OUT=benchmarks/results/hier-kv-rebaseline-20260712/lever1_similarity.md
: >"$OUT"
echo "# Lever 1: hier merge grouping — position vs similarity (0.8B mq4+, PPL/KLD vs bf16)" >>"$OUT"
echo "" >>"$OUT"; echo "| ctx | grouping | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire lever1 --timeout-secs 900 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
run() { # ctx ref hot grouping
  local ctx="$1" ref="$2" hot="$3" grp="$4"
  local log=/tmp/lever1_${ctx}_${grp}.log
  local extra=""; [ "$grp" = "similarity" ] && extra="HIPFIRE_KV_MERGE=similarity"
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$hot HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2 $extra \
    $PX "$CAND" "$CORPUS" --ctx "$ctx" --offset 0 --kv-mode kvarn --kld-ref "$ref" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $ctx | $grp | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: ctx=$ctx grp=$grp ppl=$ppl kld=$kld"
}
run 2048  /tmp/ref08_o0.pkld     512  position
run 2048  /tmp/ref08_o0.pkld     512  similarity
run 16384 /tmp/ref08_lc16384.pkld 2048 position
run 16384 /tmp/ref08_lc16384.pkld 2048 similarity
echo "ALL DONE"
