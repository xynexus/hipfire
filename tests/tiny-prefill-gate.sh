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
# PATH CHECK, before anything else, and measured POSITIVELY: the probe counts
# rows through `forward_prefill_chunk` (reachable only from
# `forward_prefill_batch*`, never from the per-token reference) and requires the
# batched run to move that counter while the reference run leaves it alone. A
# cell that fails it reports SKIP, never OK — without it a fixture that never
# reaches the batched path would report a clean pass for a path it never ran.
#
# It used to infer this from the two runs' recurrent-state hashes DIFFERING.
# That inference died with `0bbbfd08f`: once the duplicated MoE attention bodies
# were folded onto the shared lowered super-ops, both paths run the same
# per-token GDN kernels and a CORRECT implementation leaves the hashes equal.
# The old check called that "never ran", and because it was evaluated before the
# KLD comparison it also reported a real `max_kld 0.377, argmax 0/4` failure as
# INCONCLUSIVE. The state hashes are still printed, as diagnostics only.
#
# THE MoE CELLS. `qwen3_5_moe` SKIPs on EVERY arch, and that is correct, not an
# arch gap: `moe_preset` is top-2-of-8 and `moe_prefill_topk_shape_supported`
# requires k_top in {8,10}, so the refusal is `moe_topk_ok=false (K=2, E=8)` —
# a fixture shape, with no arch term in it. It is deliberately kept that way as
# the regression cover for the ADMISSION guard (see `moe_indexed_preset`'s
# docstring in hipfire-arch-qwen35-spec).
#
# `qwen3_5_moe_indexed` is top-8-of-16, reaches grouped path-2, and PASSES.
# Getting there took two fixes, both in `0bbbfd08f` and `75b718181`: raw
# F16/BF16 grouped MoE was dispatched to a gfx1151-only entry point, and — the
# real one — `DeltaNetMoe`/`FullAttnMoe` carried stale COPIES of the dense
# attention bodies that decoded F16 weights as HFQ4 nibble blocks on every arch.
# A red MoE cell here is not an arch gap to route around.
# See docs/plans/2026-08-24-raw-f16-moe-prefill-divergence.md.
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
    -p hipfire-serving-core --example tiny_quant_probe \
    -p hipfire-runtime --example compare_prefill_hidden_paths >/dev/null || {
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

    # ── HIDDEN-STATE arm (drafter fidelity) ──────────────────────────────
    # The KL checks below compare DECODE DISTRIBUTIONS, i.e. what the prefill
    # wrote into KV. A DFlash drafter consumes something else entirely: the
    # HiddenStateRingBuffer. Those can disagree — measured 2026-08-26, batched
    # vs per-token hidden states diverge under QUANTISED KV while greedy token
    # output stays byte-identical (the 2026-08-23 battery is a token-level
    # result and does not cover this):
    #
    #     target        q8        kvarn     fp32
    #     tiny fixture  1.27e-2   1.30e-2   EXACTLY 0
    #     27B dense     1.32e-2   2.28e-2   —
    #     35B-A3B MoE   1.36e-1   1.57e-1   EXACTLY 0
    #
    # Two assertions. The fp32 one is an INVARIANT: absent KV quantisation the
    # two prefill paths must agree exactly, and that is the positive control
    # proving the probe can see anything at all. The quantised one is a
    # regression ceiling pinned just above today's ~1.3e-2 — it does not claim
    # the current divergence is acceptable, only that it must not grow.
    if [ "${HIPFIRE_TINYPREFILL_SKIP_HIDDEN:-0}" != "1" ]; then
        HID="./target/release/examples/compare_prefill_hidden_paths"
        # The gate BUILDS this above. It used to be `if [ -x ]`, and nothing built
        # it — so on a clean checkout the whole arm silently did not run and the
        # gate reported fail=0, while a developer tree that happened to have it
        # from an earlier session reported fail=1 on the same commit. The check
        # most likely to catch a drafter-visible regression was the one CI never
        # ran. Absence is now an infra error, not a skip.
        if [ ! -x "$HID" ]; then
            echo "  FAIL $HID missing — it is built above; the hidden-state arm cannot be skipped" >&2
            exit 2
        fi
        # The probe reports one of TWO mutually exclusive summary lines:
        #   FIRST DIVERGING LAYER: 0   (worst overall 1.06e-1)
        #   IDENTICAL across all layers (worst 3.31e-4)
        # Both parenthesise the number; the old pattern matched only the first, so
        # the GOOD case parsed as empty and was skipped by `[ -n ]`. That made a
        # measurement that did not happen indistinguishable from one that passed.
        hid_worst() {
            "$HID" --model "$2" --n 32 --kv-mode "$1" 2>/dev/null |
                grep -oE '\(worst (overall )?[0-9.eE+-]+\)' |
                sed -E 's/.*[[:space:]]([0-9.eE+-]+)\)/\1/' | tail -1
        }
        w32="$(hid_worst fp32 "$hfq")"
        if [ -z "$w32" ]; then
            echo "  FAIL hidden-state probe emitted no parseable summary at fp32 KV —"
            echo "       treating an unmeasured invariant as a pass is how this arm"
            echo "       went unnoticed before; fix the probe or this parser"
            fail=$((fail + 1))
        elif [ "$(awk -v a="$w32" 'BEGIN{print (a>0)?1:0}')" = "1" ]; then
            echo "  FAIL hidden states differ at fp32 KV (worst $w32) — the prefill"
            echo "       paths must be exactly equivalent absent KV quantisation"
            fail=$((fail + 1))
        else
            echo "  hidden fp32: $w32 (invariant: must be exactly 0)"
        fi
        for hkv in "${kv_modes[@]}"; do
            hkv="$(echo "$hkv" | xargs)"; [ -n "$hkv" ] || continue
            wq="$(hid_worst "$hkv" "$hfq")"
            if [ -z "$wq" ]; then
                echo "  FAIL hidden-state probe emitted no parseable summary for $hkv"
                fail=$((fail + 1))
                continue
            fi
            if [ "$(awk -v a="$wq" -v c="${HIPFIRE_TINYPREFILL_HID_MAX:-5e-2}" 'BEGIN{print (a>c)?1:0}')" = "1" ]; then
                echo "  FAIL hidden-state divergence $hkv = $wq exceeds ceiling ${HIPFIRE_TINYPREFILL_HID_MAX:-5e-2}"
                echo "       a drafter reads these; see the note above this block"
                fail=$((fail + 1))
            else
                echo "  hidden $hkv: worst $wq (ceiling ${HIPFIRE_TINYPREFILL_HID_MAX:-5e-2})"
            fi
        done
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
            echo "  SKIP batched prefill did not execute for this fixture"
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
