#!/usr/bin/env bash
# Levers 3-4 headroom probe: does cold V have compression room below its floor?
# Asymmetric K/V bits (K held, V pushed down). If V=1 holds with a good K, V has
# headroom → the V.Wo/OVC low-rank machinery is worth building; if V=1 tanks, V is
# already near its floor and the fancy V-compression buys little. Testable on wikitext
# (V-quant quality is measurable, unlike the retrieval-regime merge levers).
set -u
cd /home/sadara/hipfire
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PX=./target/release/examples/perplexity
CAND=/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq
REF=/tmp/ref08_o0.pkld
OUT=benchmarks/results/hier-kv-rebaseline-20260712/lever34_vprobe.md
: >"$OUT"
echo "# Levers 3-4 V headroom probe (0.8B mq4+, hier hot=512 fold=4, ctx=2048, vs bf16)" >>"$OUT"
echo "" >>"$OUT"; echo "| K bits | V bits | PPL | KLD/tok |" >>"$OUT"; echo "|---|---|---|---|" >>"$OUT"
./target/release/hipfire lock acquire lever34-vprobe --timeout-secs 600 || { echo LOCKFAIL; exit 1; }
trap './target/release/hipfire lock release' EXIT
run() { # kbits vbits
  local kb="$1" vb="$2"
  local log=/tmp/vprobe_k${kb}_v${vb}.log
  env HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=4 \
      HIPFIRE_KV_COLD_BITS=$kb HIPFIRE_KV_COLD_V_BITS=$vb \
    $PX "$CAND" "$CORPUS" --ctx 2048 --offset 0 --kv-mode kvarn --kld-ref "$REF" >"$log" 2>&1
  local ppl kld
  ppl=$(grep -oE 'PPL: *[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  kld=$(grep -oiE 'KLD[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  echo "| $kb | $vb | ${ppl:-ERR} | ${kld:-NA} |" >>"$OUT"
  echo "done: K=$kb V=$vb ppl=$ppl kld=$kld"
}
run 2 2   # baseline operating point
run 2 1   # push V to 1-bit, K held at 2
run 4 2   # K=4 V=2
run 4 1   # K=4 V=1 (best K, aggressive V)
echo "ALL DONE"
