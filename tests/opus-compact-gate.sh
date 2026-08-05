#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — Opus compact-residency (OqPlusCompact / qt 36) drift gate (GPU).
#
# Quantizes a seeded tiny fixture to oq4.25 (N_out=3, the block shape the
# premier oq4.25++ family uses) and free-runs greedy decode twice: once on the
# expanded path and once with HIPFIRE_OQ_COMPACT_RESIDENT=1. Both hashes are
# checked against committed per-mode baselines.
#
#   ./tests/opus-compact-gate.sh            # check vs committed baselines
#   ./tests/opus-compact-gate.sh --record   # rewrite baselines for THIS gpu
#
# WHY THE TWO MODES ARE NOT ASSERTED EQUAL
#
# They are not equal, and that is understood rather than unexplained:
# OqCompactG256 has no fused kernel arms, so where Oq8G256 dispatches
# fused_qkvza_oq8_gemv / fused_gate_up_oq8_gemv / gemv_oq8_grouped, compact
# falls back to unfused gemm_oq_compact_grouped_wmma. Same math, different
# accumulation. See docs/plans/2026-08-05-opus-across-model-families.md.
#
# The size of that difference is NOT cosmetic. On Qwen3.5-0.8B every one of the
# 128 free-run tokens still matched, because its logits are confident. On this
# random-weight fixture, where logits are near-tied, the SAME difference flips
# argmax and the token hashes diverge too. So token equality is a property of
# the model, not an invariant of the path — which is exactly why this gate
# pins per-mode baselines instead of asserting the two modes agree.
#
# When the fused compact arms land the two modes should converge; the gate says
# so loudly rather than failing, since convergence is the goal, not a
# regression.
#
# Exit: 0 = baselines matched, 1 = drift, 2 = infra, 3 = no baseline recorded.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

BASELINES="tests/opus-compact-baselines.txt"
FAMILY="${HIPFIRE_OPUSCOMPACT_FAMILY:-qwen3_5}"
FMT="${HIPFIRE_OPUSCOMPACT_FORMAT:-oq4.25}"
LEN="${HIPFIRE_OPUSCOMPACT_LEN:-128}"
PROMPT_LEN="${HIPFIRE_OPUSCOMPACT_PROMPT_LEN:-4}"
SEED="${HIPFIRE_OPUSCOMPACT_SEED:-42}"
LOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo "opus-compact-gate: building..."
cargo build --release \
    -p hipfire-quantize --bin hipfire-quantize \
    -p hipfire-serving-core --example tiny_quant_probe >/dev/null || {
    echo "build failed" >&2
    exit 2
}

"$LOCK_BIN" gpu-lock acquire "opus-compact-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
TMP="$(mktemp -d)"
trap '"$LOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$TMP"' EXIT

Q="$ROOT/target/release/hipfire-quantize"
P="$ROOT/target/release/examples/tiny_quant_probe"

hf_dir="$TMP/$FAMILY-hf"
hfq="$TMP/$FAMILY-$FMT.hfq"
if ! "$Q" --emit-fixture "$FAMILY" --out "$hf_dir" --seed "$SEED" >/dev/null 2>&1; then
    echo "FAIL: emit fixture $FAMILY" >&2
    exit 2
fi
if ! "$Q" --input "$hf_dir" --output "$hfq" --format "$FMT" >"$TMP/q.log" 2>&1; then
    echo "SKIP: $FAMILY/$FMT quantize unsupported ($(tail -1 "$TMP/q.log"))"
    exit 3
fi

# The gate is meaningless if nothing in the artifact is actually compact.
n_compact="$("$LOCK_BIN" inspect "$hfq" 2>/dev/null | awk '/OqPlusCompact/ {print $2}')"
if [ -z "$n_compact" ] || [ "$n_compact" -eq 0 ] 2>/dev/null; then
    echo "SKIP: $FAMILY/$FMT produced no OqPlusCompact tensors"
    exit 3
fi
echo "opus-compact-gate: $FAMILY/$FMT, $n_compact OqPlusCompact tensor(s)"

run_mode() { # $1 = 0|1 -> "<logit> <token>"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
        HIPFIRE_OQ_COMPACT_RESIDENT="$1" \
        "$P" ar-hash --arch "$FAMILY" --model "$hfq" \
        --len "$LEN" --prompt-len "$PROMPT_LEN" --seed "$SEED" 2>"$TMP/e.$1" >"$TMP/o.$1" || return 1
    awk '/^logit_hash:/ {l=$2} /^token_hash:/ {t=$2} END {print l" "t}' "$TMP/o.$1"
}

declare -A GOT
for mode in 0 1; do
    label=$([ "$mode" = 0 ] && echo expanded || echo compact)
    if ! out="$(run_mode "$mode")"; then
        echo "  FAIL $label: ar-hash failed: $(tail -2 "$TMP/e.$mode" | tr '\n' ' ')"
        exit 1
    fi
    GOT[$label]="$out"
    echo "  $label: $out"
done

gpu_arch="$(grep -oE 'gfx[0-9a-f]+' "$TMP/e.0" | head -1)"
[ -n "$gpu_arch" ] || gpu_arch="unknown"

if [ "$RECORD" = 1 ]; then
    {
        echo "# gpu family format mode logit_hash token_hash"
        for label in expanded compact; do
            echo "$gpu_arch $FAMILY $FMT $label ${GOT[$label]}"
        done
    } >"$BASELINES"
    echo "opus-compact-gate: recorded $BASELINES for $gpu_arch"
    exit 0
fi

if [ ! -f "$BASELINES" ]; then
    echo "opus-compact-gate: no baselines; run --record on this gpu"
    exit 3
fi

fail=0
have=0
for label in expanded compact; do
    want="$(awk -v g="$gpu_arch" -v f="$FAMILY" -v q="$FMT" -v m="$label" \
        '$1==g && $2==f && $3==q && $4==m {print $5" "$6}' "$BASELINES" | head -1)"
    if [ -z "$want" ]; then
        echo "  SKIP $label: no baseline for $gpu_arch"
        continue
    fi
    have=1
    if [ "$want" = "${GOT[$label]}" ]; then
        echo "  ok   $label"
    else
        echo "  DRIFT $label: want [$want] got [${GOT[$label]}]"
        fail=1
    fi
done
[ "$have" = 1 ] || { echo "opus-compact-gate: no baseline for $gpu_arch"; exit 3; }

# Convergence notice: the goal, not a regression, so never a failure.
if [ "${GOT[expanded]}" = "${GOT[compact]}" ]; then
    echo "opus-compact-gate: NOTE expanded and compact now AGREE exactly."
    echo "  The fused compact arms have most likely landed. Re-record baselines"
    echo "  and update docs/plans/2026-08-05-opus-across-model-families.md."
fi

if [ "$fail" = 0 ]; then
    echo "opus-compact-gate: PASS"
    exit 0
fi
echo "opus-compact-gate: FAIL (baseline drift)"
exit 1
