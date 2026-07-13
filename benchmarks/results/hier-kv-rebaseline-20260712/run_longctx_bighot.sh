#!/usr/bin/env bash
# Decision 2 payoff: with the f16 hot ring, a larger exact window is affordable at
# long ctx. Does hot=1024/2048 beat hot=512 (20.07) at ctx=16384? This is the
# long-context quality lever now that dephasing is dead.
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
REFDUMP=/tmp/ref08_lc16384.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/longctx_bighot_ctx16384.md
: >"$OUT"
echo "# f16 big-hot payoff, 0.8B ctx=16384 fold=4 2-bit (bf16 ref 16.10; hot=512 f32 was 20.07/0.247)" >>"$OUT"
echo "" >>"$OUT"; echo "| hot (f16) | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire hier-kv-bighot --timeout-secs 600 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
for HOT in 1024 2048; do
  log=/tmp/bighot_${HOT}.log
  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=$HOT HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=2 \
    $PX "$CAND" "$CORPUS" --ctx 16384 --offset 0 --kv-mode kvarn --kld-ref "$REFDUMP" >"$log" 2>&1
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $HOT | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: hot=$HOT ppl=$ppl kld=$kld"
done
echo "ALL DONE"
