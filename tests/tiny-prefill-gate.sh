#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny BATCHED-PREFILL differential gate (GPU).
#
# The only tiny gate that executes a prefill. Before it existed,
# `tiny-affected-gate.sh` mapped every `crates/hipfire-arch-qwen35/*` edit onto
# tiny-quant / tiny-state / tiny-spec — all three of which feed ONE token per
# call to the decode path — so `--require-coverage` reported COVERED for a
# prefill change it never ran. See docs/plans/2026-08-22-prefill-lowering.md §4.
#
# WHAT IT CHECKS. For each qwen3.5 family: emit a seeded fixture, quantize it,
# then drive the SAME tokens through both
#   * `forward_prefill_batch` (the batched path), and
#   * per-token `forward_scratch` (the reference the serving path already takes
#     when `replay_as_generated_suffix || hier_enabled`, and that
#     `qwen35_prefill_owned_session_serial_segment` takes unconditionally),
# then decode past the prefill teacher-forced and compare the two decode
# distributions. Causal attention means the first decoded position reads every
# KV row the prefill wrote, so a prefill that is right at position n-1 and wrong
# at 0..n-2 — the failure `smoke-generate-batch-prefill.sh` cannot see, since it
# compares only the final-position logit — moves the number here.
#
# NO BASELINE FILE, deliberately. The oracle is in-tree and runs every time, so
# there is nothing to re-record, nothing keyed to a toolchain version, and no
# way for a stale-but-matching row to look like a pass. That also sidesteps every
# trap documented in tiny-state-gate.sh's header.
#
# NOT bit-exact, deliberately. The batched path runs GEMM over n rows and the
# reference runs GEMV per token, so float reassociation differs by construction;
# requiring equal hashes is the parity-oracle trap. The bar is "same
# distribution": max KL(ref || batched) over the decoded positions, plus argmax
# agreement.
#
# TOLERANCE, measured on nix1/gfx1103 against this gate's own qwen3_5 fp16
# fixture, over seeds {42,7,99,1234} x decode {4,16} at prefill 64:
#     clean reassociation noise   1.5e-6 .. 1.2e-5
#     corrupted KV prefix         5.3e-2 .. 1.1e-1   (4482x .. 68844x the noise)
# The default 1e-4 sits ~8x above the worst observed noise and ~530x below the
# weakest corruption signal. Raise it only with a measurement, not a hunch.
#
# PATH CHECK, before anything else. The batched and per-token paths write
# bit-different recurrent state, so if their state hashes come out EQUAL the
# "batched" run silently fell back to the reference and the cell compared it
# against itself. Such a cell reports SKIP, never OK. Measured: both tiny MoE
# fixtures show that signature today — they never reach `forward_prefill_batch`
# — so without this check they would have reported a clean pass for a path they
# never ran. (The real 35B-A3B does take the batched path; this is a fixture
# capability gap, not a product bug. It does mean the MoE arms of M2a4 have no
# tiny coverage yet.)
#
# SELF-CHECK. Every cell also runs with `--corrupt-kv-prefix`, which zeroes each
# layer's K buffer at every position EXCEPT the last — position 0 included. That
# is the plan's named failure ("correct at n-1, garbage at 0..n-2"), and that run
# MUST fail. A gate that cannot be made to fail is not evidence, and this
# milestone exists because the existing one could not.
#
# Corrupting position 0 ALONE was tried first and rejected on measurement: on a
# random-weight fixture attention is near-uniform, so one row of 64 moved max KLD
# by only 2x-44x and landed under tolerance for some token seeds. A self-check
# whose outcome depends on the seed is not a self-check.
#
# Env:
#   HIPFIRE_TINYQUANT_FAMILIES=qwen3_5,...   (non-qwen3.5 families are skipped)
#   HIPFIRE_TINYPREFILL_PREFILL=64
#   HIPFIRE_TINYPREFILL_DECODE=4
#   HIPFIRE_TINYPREFILL_TOL=1e-4
#   HIPFIRE_TINYPREFILL_KV=q8,kvarn
#
# Exit: 0 = all selected cells pass, 1 = divergence/crash/blind-probe, 2 = infra,
#       3 = nothing selected.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PREFILL="${HIPFIRE_TINYPREFILL_PREFILL:-64}"
DECODE="${HIPFIRE_TINYPREFILL_DECODE:-4}"
TOL="${HIPFIRE_TINYPREFILL_TOL:-1e-4}"
SEED="${HIPFIRE_TINYPREFILL_SEED:-42}"
# Both KV modes, because they select DIFFERENT FA arms and each leaves the other
# untested. `fa_kv_ok` (prefill_chunk.rs:2313) holds for Q8, so FA layers take the
# BATCHED arm; KVarN is `quantized` but none of q8/asym{2,3,4}, so it fails that
# test and FA layers take the PER-TOKEN FALLBACK. KVarN is also the default KV
# mode, so that fallback is production code. Verified by instrumentation, not
# assumed: a temporary print inside the fallback fired 0 times under q8 and once
# under kvarn.
IFS=',' read -r -a kv_modes <<<"${HIPFIRE_TINYPREFILL_KV:-q8,kvarn}"
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Only the qwen3.5 text families reach `forward_prefill_batch` through
# `TinyModel::Qwen35`. qwen3_5_vl loads the Qwen35Vl variant (its prefill goes
# through the vision seam) and every other arch has no batched prefill here, so
# they are skipped rather than failed.
ALL_FAMILIES=(qwen3_5 qwen3_5_moe qwen3_5_moe_indexed)
families=("${ALL_FAMILIES[@]}")
if [ -n "${HIPFIRE_TINYQUANT_FAMILIES:-}" ]; then
    IFS=',' read -r -a families <<<"$HIPFIRE_TINYQUANT_FAMILIES"
fi

echo "tiny-prefill-gate: building..."
cargo build --release \
    -p hipfire-quantize --bin hipfire-quantize \
    -p hipfire-serving-core --example tiny_quant_probe >/dev/null || {
    echo "build failed" >&2
    exit 2
}

"$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "tiny-prefill-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
TMP="$(mktemp -d)"
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$TMP"' EXIT

Q="$ROOT/target/release/hipfire-quantize"
P="$ROOT/target/release/examples/tiny_quant_probe"
fail=0
skip=0
ran=0

for raw_family in "${families[@]}"; do
    family="$(echo "$raw_family" | xargs)"
    [ -n "$family" ] || continue
    case " ${ALL_FAMILIES[*]} " in
        *" $family "*) ;;
        *)
            # Not a batched-prefill family; silently out of scope for this gate.
            skip=$((skip + 1))
            continue
            ;;
    esac
    hf_dir="$TMP/$family-hf"
    hfq="$TMP/$family-fp16.hfq"
    if ! "$Q" --emit-fixture "$family" --out "$hf_dir" --seed "$SEED" >/dev/null 2>&1; then
        echo "  FAIL emit fixture"
        fail=$((fail + 1))
        continue
    fi
    if ! "$Q" --input "$hf_dir" --output "$hfq" --format fp16 >/dev/null 2>&1; then
        echo "  SKIP fp16: quantize unsupported"
        skip=$((skip + 1))
        continue
    fi

    for kv in "${kv_modes[@]}"; do
        kv="$(echo "$kv" | xargs)"
        [ -n "$kv" ] || continue
        echo "== tiny-prefill: $family/$kv =="
        out="$TMP/$family-$kv.out"
        "$P" prefill-hash --arch "$family" --model "$hfq" --kv "$kv" \
            --prefill "$PREFILL" --decode "$DECODE" --tol "$TOL" --seed "$SEED" \
            >"$out" 2>"$TMP/$family-$kv.err"
        rc=$?
        if [ "$rc" -eq 3 ]; then
            # Both runs landed in the same code path, so this cell compared the
            # reference against itself. Inconclusive, and explicitly NOT a pass —
            # counting it would recreate the false-coverage bug in a new place.
            echo "  SKIP no batched prefill path reached (compared reference to itself)"
            skip=$((skip + 1))
            continue
        fi
        if [ "$rc" -ne 0 ]; then
            echo "  FAIL batched prefill diverges from the per-token reference"
            grep -E '^(max_kld|mean_kld|max_abs_diff|argmax_agree|tol):' "$out" | sed 's/^/    /'
            tail -2 "$TMP/$family-$kv.err" | sed 's/^/    /'
            fail=$((fail + 1))
            continue
        fi
        kld="$(awk '/^max_kld:/ {print $2}' "$out")"
        agree="$(awk '/^argmax_agree:/ {print $2}' "$out")"

        # The probe exits non-zero under --corrupt-kv-prefix precisely when it
        # FAILED to notice the corruption, so success here means the tripwire is
        # live. Not ceremony: it caught the KVarN case, where zeroing `k_gpu`
        # alone changed nothing because a 64-token prefill lives entirely in the
        # `k_window` recent-block ring.
        if "$P" prefill-hash --arch "$family" --model "$hfq" --kv "$kv" \
            --prefill "$PREFILL" --decode "$DECODE" --tol "$TOL" --seed "$SEED" \
            --corrupt-kv-prefix >"$TMP/$family-$kv.corrupt" 2>&1; then
            ckld="$(awk '/^max_kld:/ {print $2}' "$TMP/$family-$kv.corrupt")"
            echo "  OK  max_kld=$kld (tol=$TOL) argmax=$agree | corrupt-prefix caught at max_kld=$ckld"
        else
            echo "  FAIL falsifiability: probe did not catch a corrupted KV prefix"
            tail -2 "$TMP/$family-$kv.corrupt" | sed 's/^/    /'
            fail=$((fail + 1))
            continue
        fi
        ran=$((ran + 1))
    done
done

echo "tiny-prefill-gate: ran=$ran fail=$fail skip=$skip"
[ "$fail" -gt 0 ] && exit 1
[ "$ran" -eq 0 ] && exit 3
exit 0
