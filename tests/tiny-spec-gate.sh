#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny structural spec-decode gate.
#
# First-tier coverage for DFlash/DDTree/spec-decode support code. Covers
# deterministic tree construction/linearization/following, qwen35 spec policy
# unit tests, and a tiny trained DFlash draft sidecar through convert + runtime
# smoke.
#
# Exit: 0 pass, 1 test failure, 2 infrastructure/build error.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "tiny-spec-gate: runtime DDTree structural tests..."
if ! cargo test -p hipfire-runtime ddtree --lib; then
    echo "tiny-spec-gate: FAIL hipfire-runtime ddtree tests"
    exit 1
fi

echo "tiny-spec-gate: qwen35 spec policy tests..."
if ! cargo test -p hipfire-arch-qwen35 ddtree --lib; then
    echo "tiny-spec-gate: FAIL hipfire-arch-qwen35 ddtree/spec tests"
    exit 1
fi

if [[ "${HIPFIRE_TINY_SPEC_SKIP_DFLASH:-0}" != "1" ]]; then
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    OUT_DIR="${HIPFIRE_TINY_DFLASH_OUT:-target/tiny-gates/dflash-trained}"
    OUT_HFQ="${HIPFIRE_TINY_DFLASH_HFQ:-target/tiny-gates/dflash-trained.dflash.hfq}"
    STEPS="${HIPFIRE_TINY_DFLASH_STEPS:-20}"
    ROWS="${HIPFIRE_TINY_DFLASH_ROWS:-32}"
    LR="${HIPFIRE_TINY_DFLASH_LR:-0.003}"
    BLOCK="${HIPFIRE_TINY_DFLASH_BLOCK:-16}"
    CTX="${HIPFIRE_TINY_DFLASH_CTX:-32}"

    echo "tiny-spec-gate: tiny DFlash train/export..."
    if ! cargo run -q -p hipfire-train --example tiny_dflash_train -- \
        --out "$OUT_DIR" --steps "$STEPS" --rows "$ROWS" --lr "$LR"; then
        echo "tiny-spec-gate: FAIL tiny DFlash trainer"
        exit 1
    fi

    echo "tiny-spec-gate: tiny DFlash convert..."
    if ! cargo run -q -p hipfire-quantize --bin dflash_convert -- \
        --input "$OUT_DIR" --output "$OUT_HFQ"; then
        echo "tiny-spec-gate: FAIL tiny DFlash convert"
        exit 1
    fi

    echo "tiny-spec-gate: tiny DFlash runtime smoke..."
    if ! cargo run -q -p hipfire-runtime --features deltanet --example dflash_smoke -- \
        "$OUT_HFQ" --block "$BLOCK" --ctx "$CTX"; then
        echo "tiny-spec-gate: FAIL tiny DFlash runtime smoke"
        exit 1
    fi
    # ── Real spec-decode loop with REJECTIONS ────────────────────────────
    # The smoke above is a drafter FORWARD only: it never verifies a block, so
    # it cannot reach rollback. Instrumenting proved it — with
    # HIPFIRE_DFLASH_CHECKPOINT_ROLLBACK=1 the checkpoint path fired ZERO times
    # while the gate still passed. A gate that cannot fail when the feature
    # breaks is not coverage.
    #
    # This stage pairs a tiny qwen3_5 TARGET with the drafter trained above and
    # drives an actual spec loop. Two assertions, and both are load-bearing:
    #   1. `replay_checkpoint > 0` — the rollback path actually ran (positive
    #      probe; without it a skipped path reads as a pass).
    #   2. tokens == --ar-baseline — spec-decode is lossless, which is the
    #      property the rollback exists to preserve.
    #
    # The drafter is random-init trained for a handful of steps, so tau is ~0
    # and EVERY block is rejected. That is deliberate: rejection is what forces
    # rollback, so the weakest possible drafter is the strongest possible test.
    SPEC_DIR="${HIPFIRE_TINY_SPEC_DIR:-target/tiny-gates/spec-loop}"
    rm -rf "$SPEC_DIR"; mkdir -p "$SPEC_DIR"
    echo "tiny-spec-gate: tiny spec-decode loop (checkpoint rollback coverage)..."
    if ! cargo run -q -p hipfire-quantize --bin hipfire-quantize -- \
        --emit-fixture qwen3_5 --out "$SPEC_DIR/hf" --seed 1234 >/dev/null; then
        echo "tiny-spec-gate: FAIL emit qwen3_5 fixture"
        exit 2
    fi
    if ! cargo run -q -p hipfire-quantize --bin hipfire-quantize -- \
        --input "$SPEC_DIR/hf" --output "$SPEC_DIR/target.hfq" --format bf16 >/dev/null; then
        echo "tiny-spec-gate: FAIL quantize qwen3_5 fixture"
        exit 2
    fi
    printf 'hello world this is a test of speculative decoding\n' > "$SPEC_DIR/p.txt"
    # `--no-chatml` is REQUIRED: the fixture tokenizer has no <|im_start|>.
    SPEC_ARGS=(--target "$SPEC_DIR/target.hfq" --draft "$OUT_HFQ"
               --prompt-file "$SPEC_DIR/p.txt" --max 32 --no-chatml)
    if ! cargo run -q -p hipfire-runtime --features deltanet --example dflash_spec_demo -- \
        "${SPEC_ARGS[@]}" --ar-baseline > "$SPEC_DIR/ar.log" 2>&1; then
        echo "tiny-spec-gate: FAIL spec-loop AR baseline"; tail -5 "$SPEC_DIR/ar.log"
        exit 1
    fi
    # HIPFIRE_DN_STATE_FP16=0 is REQUIRED, not belt-and-braces. The checkpoint
    # path needs FP32 DeltaNet state; with the session's default FP16 the
    # runtime prints
    #   "DFlash checkpoint rollback: DISABLED - it needs FP32 DeltaNet state
    #    and this session resolved FP16"
    # and silently skips the snapshot allocation, so replay_checkpoint stays 0
    # and the assertion below fails no matter what the code does. That is what
    # it did from the day this stage landed: the gate demanded coverage it had
    # itself made unreachable.
    if ! HIPFIRE_DFLASH_CHECKPOINT_ROLLBACK=1 HIPFIRE_DN_STATE_FP16=0 \
        cargo run -q -p hipfire-runtime --features deltanet --example dflash_spec_demo -- \
        "${SPEC_ARGS[@]}" > "$SPEC_DIR/ckpt.log" 2>&1; then
        echo "tiny-spec-gate: FAIL spec-loop checkpoint run"; tail -5 "$SPEC_DIR/ckpt.log"
        exit 1
    fi
    CKPT_N=$(grep -oE 'replay_checkpoint=[0-9]+' "$SPEC_DIR/ckpt.log" | tail -1 | cut -d= -f2)
    CKPT_N="${CKPT_N:-0}"
    if [ "$CKPT_N" -lt 1 ]; then
        echo "tiny-spec-gate: FAIL checkpoint rollback never engaged (replay_checkpoint=$CKPT_N)"
        echo "  the loop ran but never rejected a block, so this asserts nothing —"
        echo "  see 825d3ccfc for why a passing gate here would be vacuous."
        exit 1
    fi
    grep -oE 'AR tokens: \[[0-9, ]*\]' "$SPEC_DIR/ar.log" | sed 's/^AR tokens: //' > "$SPEC_DIR/ar.tok"
    grep -oE 'DFlash tokens: \[[0-9, ]*\]' "$SPEC_DIR/ckpt.log" | sed 's/^DFlash tokens: //' > "$SPEC_DIR/ckpt.tok"
    if [ ! -s "$SPEC_DIR/ar.tok" ] || [ ! -s "$SPEC_DIR/ckpt.tok" ]; then
        echo "tiny-spec-gate: FAIL could not capture token streams"
        exit 1
    fi
    if ! cmp -s "$SPEC_DIR/ar.tok" "$SPEC_DIR/ckpt.tok"; then
        echo "tiny-spec-gate: FAIL checkpoint rollback changed output (spec-decode must be lossless)"
        echo "  AR   : $(head -c 200 "$SPEC_DIR/ar.tok")"
        echo "  CKPT : $(head -c 200 "$SPEC_DIR/ckpt.tok")"
        exit 1
    fi
    # ⚠️ KNOWN BLIND SPOT — do not read this stage as full coverage.
    # The fixture target is RANDOM-INIT, so its next token is unpredictable by
    # construction and nothing can draft it: tau is 0.0 and accept_len is pinned
    # at 0 on every cycle (measured: plain/--pld/--ngram all give
    # histogram [(0,31),(1,0),(2,0)]). Since the restore reads slot
    # `accept_len`, and accept_len == 0 here, SLOT-INDEX BUGS ARE INVISIBLE to
    # this gate — mutating the restore to always read slot 0 still passes.
    #
    # What it DOES catch: the path silently not engaging, a non-lossless restore
    # at accept_len 0, conv-ring corruption, and crashes. That is worth having;
    # it is just not everything.
    #
    # Closing the gap needs a target whose output is predictable enough to
    # produce acceptances — i.e. a trained model, not a fixture. Until then,
    # slot indexing is only exercised on real models (Qwen3.5-35B-A3B, where
    # accept_len varies 0..b-1).
    echo "tiny-spec-gate: spec loop OK (checkpoint engaged ${CKPT_N}x, output == AR)"
    echo "tiny-spec-gate:   note: accept_len==0 on fixtures; slot indexing NOT covered here"
else
    echo "tiny-spec-gate: SKIP tiny DFlash train/export/smoke (HIPFIRE_TINY_SPEC_SKIP_DFLASH=1)"
fi

echo "tiny-spec-gate: PASS"
exit 0
