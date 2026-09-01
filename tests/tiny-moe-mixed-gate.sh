#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Cover for MIXED-PRECISION MoE layers: routed experts where some are
# OqPlusCompact and some are Oq8G256 in the SAME layer.
#
# Nothing else in the tree exercises this. Every tiny fixture is uniform, and
# the only production model that mixes is the 122B, which takes 65s to load at
# 69 GB. The mixed path (per-expert stride table, `block_stride == 0` selecting
# an Oq8 arm inside a compact GEMV) therefore shipped with no small-scale
# coverage at all.
#
# Two things are asserted:
#   1. the artifact really IS mixed (both dtypes present among routed experts)
#   2. it loads and scores, and its KLD tracks the uniform build
#
# `--mixed-bpw` is the per-tensor Oq4->Oq8 promoter and it only runs when the
# INPUT is an .hfq, so this goes source -> anchor -> calib -> HFQ-requantize.
set -uo pipefail
cd "$(dirname "$0")/.."

W="${TMPDIR:-/tmp}/tiny-moe-mixed.$$"
trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

Q=./target/release/hipfire-quantize
P=./target/release/examples/tiny_quant_probe
FAM=qwen3_5_moe_indexed

echo "tiny-moe-mixed-gate: building..."
cargo build --release -p hipfire-quantize --bin hipfire-quantize >/dev/null 2>&1 || { echo "  BUILD FAILED"; exit 2; }
cargo build --release --example tiny_quant_probe >/dev/null 2>&1 || { echo "  BUILD FAILED"; exit 2; }

set -e
$Q --emit-fixture "$FAM" --out "$W/src" --seed 42 >/dev/null 2>&1
$Q --input "$W/src" --output "$W/anchor.fp16.hfq" --format fp16 --arch-id 6 >/dev/null 2>&1
set +e
# The promoter only runs on the .hfq -> .hfq path, so the source path must REFUSE
# --mixed-bpw rather than quietly emitting a uniform artifact (docs/bugs/
# 2026-08-27-mixed-bpw-ignored-off-hfq.md). CPU-only, so it runs before the GPU steps.
# `oq4`, not `oq4.25++`: the `++` formats demand --hessian first, so that arm
# would refuse for an unrelated reason and assert nothing about the promoter.
$Q --input "$W/src" --output "$W/should-not-exist.hfq" --format oq4 --arch-id 6 \
   --mixed-bpw 4.5 >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "tiny-moe-mixed-gate: FAIL — --mixed-bpw was accepted on a SOURCE input"
    echo "  it is threaded only into run_hfq_source_pipeline; accepting it silently"
    echo "  produces a uniform artifact that looks mixed-precision"
    exit 1
fi
set -e

$P collect --arch "$FAM" --model "$W/anchor.fp16.hfq" --out "$W/calib.hfq" --len 128 >/dev/null 2>&1
# uniform reference (source path) and the MIXED build (HFQ path + promotion)
$Q --input "$W/src" --output "$W/uniform.hfq" --format oq4.25++ --arch-id 6 \
   --hessian "$W/calib.hfq" >/dev/null 2>&1
$Q --input "$W/anchor.fp16.hfq" --output "$W/mixed.hfq" --format oq4.25++ --arch-id 6 \
   --hessian "$W/calib.hfq" --mixed-bpw 4.5 >/dev/null 2>&1
set +e

# ── 1. is it actually mixed? ────────────────────────────────────────────────
dts=$(./target/release/hipfire inspect "$W/mixed.hfq" --tensors 2>/dev/null \
      | awk '$1 ~ /\.experts\./ && $1 !~ /awq_scale/ {print $2}' | sort -u | tr '\n' ' ')
echo "  routed-expert dtypes: $dts"
case "$dts" in
    *Oq8G256*) ;;
    *) echo "tiny-moe-mixed-gate: FAIL — no Oq8G256 experts; the fixture is not mixed"
       echo "  (the promoter is silent on non-.hfq input; check --mixed-bpw reached run_hfq_source_pipeline)"
       exit 1 ;;
esac
case "$dts" in
    *OqPlusCompact*) ;;
    *) echo "tiny-moe-mixed-gate: FAIL — no OqPlusCompact experts; nothing to mix against"; exit 1 ;;
esac

# ── 2. does it load and score? ──────────────────────────────────────────────
kld_of() {
    HIPFIRE_QWEN35_MOE_OQ_INDEXED=1 $P kld --arch "$FAM" --ref "$W/anchor.fp16.hfq" \
        --cand "$1" --len 128 --warmup 8 2>&1
}
uni_out=$(kld_of "$W/uniform.hfq"); mix_out=$(kld_of "$W/mixed.hfq")
uni=$(printf '%s' "$uni_out" | grep -oP 'mean_kld:\s*\K[0-9.eE+-]+')
mix=$(printf '%s' "$mix_out" | grep -oP 'mean_kld:\s*\K[0-9.eE+-]+')
if [ -z "$mix" ]; then
    echo "tiny-moe-mixed-gate: FAIL — mixed artifact did not score"
    printf '%s\n' "$mix_out" | tail -3 | sed 's/^/        /'
    exit 1
fi
echo "  mean_kld: uniform=$uni  mixed=$mix"
# The mixed build promotes tensors to HIGHER precision, so it must not be
# dramatically WORSE. A wide bound: this is a tripwire for a broken path, not a
# quality baseline (the two builds take different pipelines, so they are not
# expected to match exactly).
bad=$(python3 -c "u=float('$uni'); m=float('$mix'); print(1 if (m != m or m > max(4*u, u+0.5)) else 0)")
if [ "$bad" != "0" ]; then
    echo "tiny-moe-mixed-gate: FAIL — mixed KLD $mix is far worse than uniform $uni"
    exit 1
fi
echo "tiny-moe-mixed-gate: PASS"
