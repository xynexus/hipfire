#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny-fixture golden tripwire (GPU).
#
# For each supported arch: emit a tiny random-init fixture → quantize → run the
# forward (fixture_golden) and capture the logit_hash. Two checks:
#   1. DETERMINISM (universal): run twice, hashes must match. Catches a kernel
#      that started producing nondeterministic output.
#   2. BASELINE (per gpu-arch × model-arch): compare to the committed hash. A
#      byte-exact logit_hash is GPU-arch + build specific, so baselines are keyed
#      by both; a missing entry is recorded (soft pass), a mismatch FAILS.
#
# A failure here is a TRIPWIRE — escalate to the 35B behavioral golden
# (tests/agentic-gate.sh) before rebaselining. See TODO.md "Tiny random-init
# fixtures + golden-output tripwire".
#
#   ./tests/fixture-golden-gate.sh            # check vs baselines
#   ./tests/fixture-golden-gate.sh --record   # (re)write baselines for this GPU
set -uo pipefail
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

BASELINES="tests/fixture-golden-baselines.txt"
# Format axis: each runtime quant format has its own dequant kernel path, so a
# regression in one shows up as that cell's hash drifting. Support is
# arch-conditional (e.g. mq3 is gfx11+), so an unrunnable cell is a soft SKIP,
# not a failure — only a runnable-but-drifted cell fails.
ARCHS=(qwen3_5 qwen3_5_vl qwen3_5_moe deepseek4 qwen3_legacy qwen2 dots_ocr gemma3 gemma3_vl gemma4_dense gemma4_ple gemma4_moe lfm2_moe llama minimax nemotron_h mamba2 zaya)
if [ -n "${HIPFIRE_FIXTURE_GOLDEN_ARCHS:-}" ]; then
    IFS=',' read -r -a ARCHS <<<"$HIPFIRE_FIXTURE_GOLDEN_ARCHS"
    echo "fixture-golden-gate: HIPFIRE_FIXTURE_GOLDEN_ARCHS=${HIPFIRE_FIXTURE_GOLDEN_ARCHS} → checking ${ARCHS[*]}"
fi
QWEN35_COMMON_FORMATS=(mq4 mq3 mq6 q8f16)
QWEN35_DENSE_OQ_FORMATS=(oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
QWEN35_VL_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
DEEPSEEK4_FORMATS=(deepseek4-source-precision oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
DEEPSEEK4_AUX_FORMATS=(deepseek4-source-precision oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
QWEN2_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
GEMMA3_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
DOTS_OCR_FORMATS=(hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
GEMMA3_VL_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
GEMMA4_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
GEMMA4_DENSE_PLE_FORMATS=(q8f16 hfq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
LFM2_FORMATS=(mq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
LLAMA_FORMATS=(hfq4 mq4 mq3 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
QWEN3_LEGACY_FORMATS=(hfq4 mq4 mq3 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
MINIMAX_FORMATS=(q8f16 mq4 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
NEMOTRON_FORMATS=(q8f16 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
MAMBA2_FORMATS=(q8f16 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
ZAYA_FORMATS=(q8f16 oq4 oq4+ oq4++ oq4.25++ oq8 oq8+ oq8++)
LEN=16
WARMUP=2
SEED=42

echo "fixture-golden-gate: building..."
cargo build --release -p hipfire-quantize -p hipfire-serving-core --example tiny_quant_probe --example fixture_golden -p hipfire-runtime >/dev/null || exit 2
Q="$ROOT/target/release/hipfire-quantize"
GOLD="$ROOT/target/release/examples/fixture_golden"
PROBE="$ROOT/target/release/examples/tiny_quant_probe"

"$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "fixture-golden-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true' EXIT

TMP="$(mktemp -d)"
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$TMP"' EXIT

run_golden() { # arch hfq -> prints "<gpu_arch> <logit_hash>"
    local hfq="$1"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
        "$GOLD" "$hfq" --len "$LEN" --warmup "$WARMUP" 2>"$TMP/err" >"$TMP/out"
    local ga hh
    ga="$(grep -oE 'gfx[0-9a-f]+' "$TMP/err" | head -1)"
    hh="$(grep -oE 'logit_hash: 0x[0-9a-f]+' "$TMP/out" | grep -oE '0x[0-9a-f]+')"
    echo "${ga:-unknown} ${hh:-MISSING}"
}

lookup_baseline() { # gpu_arch model_arch format
    [ -f "$BASELINES" ] || return 0
    awk -v g="$1" -v m="$2" -v f="$3" '$1==g && $2==m && $3==f {print $4}' "$BASELINES" | head -1
}

formats_for_arch() { # model_arch -> formats
    local arch="$1"
    if [ "$arch" = "gemma4_dense" ] || [ "$arch" = "gemma4_ple" ]; then
        printf '%s\n' "${GEMMA4_DENSE_PLE_FORMATS[@]}"
    elif [[ "$arch" == gemma4_* ]]; then
        printf '%s\n' "${GEMMA4_FORMATS[@]}"
    elif [ "$arch" = "deepseek4_compressed" ] || [ "$arch" = "deepseek4_mtp" ]; then
        printf '%s\n' "${DEEPSEEK4_AUX_FORMATS[@]}"
    elif [ "$arch" = "deepseek4" ]; then
        printf '%s\n' "${DEEPSEEK4_FORMATS[@]}"
    elif [ "$arch" = "qwen3_legacy" ]; then
        printf '%s\n' "${QWEN3_LEGACY_FORMATS[@]}"
    elif [ "$arch" = "qwen2" ]; then
        printf '%s\n' "${QWEN2_FORMATS[@]}"
    elif [ "$arch" = "gemma3" ]; then
        printf '%s\n' "${GEMMA3_FORMATS[@]}"
    elif [ "$arch" = "dots_ocr" ]; then
        printf '%s\n' "${DOTS_OCR_FORMATS[@]}"
    elif [ "$arch" = "gemma3_vl" ]; then
        printf '%s\n' "${GEMMA3_VL_FORMATS[@]}"
    elif [ "$arch" = "lfm2_moe" ]; then
        printf '%s\n' "${LFM2_FORMATS[@]}"
    elif [ "$arch" = "llama" ]; then
        printf '%s\n' "${LLAMA_FORMATS[@]}"
    elif [ "$arch" = "minimax" ]; then
        printf '%s\n' "${MINIMAX_FORMATS[@]}"
    elif [ "$arch" = "nemotron_h" ]; then
        printf '%s\n' "${NEMOTRON_FORMATS[@]}"
    elif [ "$arch" = "mamba2" ]; then
        printf '%s\n' "${MAMBA2_FORMATS[@]}"
    elif [ "$arch" = "zaya" ]; then
        printf '%s\n' "${ZAYA_FORMATS[@]}"
    elif [ "$arch" = "qwen3_5" ]; then
        printf '%s\n' "${QWEN35_COMMON_FORMATS[@]}"
        printf '%s\n' "${QWEN35_DENSE_OQ_FORMATS[@]}"
    elif [ "$arch" = "qwen3_5_vl" ]; then
        printf '%s\n' "${QWEN35_VL_FORMATS[@]}"
    elif [ "$arch" = "qwen3_5_moe" ]; then
        printf '%s\n' "${QWEN35_COMMON_FORMATS[@]}"
        printf '%s\n' "${QWEN35_DENSE_OQ_FORMATS[@]}"
    else
        printf '%s\n' "${QWEN35_COMMON_FORMATS[@]}"
    fi
}

blocked_format_reason() { # model_arch format -> optional reason
    local arch="$1"
    local fmt="$2"
    if [ "$arch" = "deepseek4_compressed" ] && [[ "$fmt" == oq* ]]; then
        echo "DeepSeek4 compressed-KV/indexer OQ needs compressor/indexer OQ dtype routing"
    elif [ "$arch" = "deepseek4_mtp" ] && [[ "$fmt" == oq* ]]; then
        echo "DeepSeek4 MTP OQ needs native MTP tensor inclusion and OQ dtype policy"
    fi
}

quant_flags_for_arch() { # model_arch -> extra hipfire-quantize flags
    local arch="$1"
    if [ "$arch" = "qwen2" ]; then
        printf '%s\n' "--arch-id" "7"
    elif [ "$arch" = "deepseek4" ] || [ "$arch" = "deepseek4_compressed" ] || [ "$arch" = "deepseek4_mtp" ]; then
        printf '%s\n' "--allow-mq2-lloyd"
    elif [ "$arch" = "dots_ocr" ]; then
        printf '%s\n' "--include-vision" "--vision-quant" "hfq4"
    elif [ "$arch" = "gemma3_vl" ]; then
        printf '%s\n' "--include-vision" "--vision-quant" "q8f16"
    elif [ "$arch" = "qwen3_5_vl" ]; then
        printf '%s\n' "--include-vision" "--vision-quant" "hfq4"
    fi
}

anchor_format_for_calib() { # model_arch -> anchor format, or empty if no calibrated fixture cell
    local arch="$1"
    case "$arch" in
        qwen2|dots_ocr|gemma3|gemma3_vl|qwen3_5_vl) echo "fp16" ;;
        llama|qwen3_legacy) echo "q8f16" ;;
        gemma4_dense|gemma4_ple|gemma4_moe) echo "fp16" ;;
        qwen3_5) echo "q8f16" ;;
        qwen3_5_moe) echo "fp16" ;;
        lfm2_moe) echo "mq6" ;;
        minimax) echo "q8f16" ;;
        deepseek4|deepseek4_compressed|deepseek4_mtp) echo "deepseek4-source-precision" ;;
        nemotron_h) echo "q8f16" ;;
        mamba2) echo "fp16" ;;
        zaya) echo "q8f16" ;;
    esac
}

requires_hessian_format() { # format -> true if fixture cell needs a collected Hessian
    local fmt="$1"
    case "$fmt" in
        oq4+|oq4++|oq4.25++|oq8+|oq8++) return 0 ;;
        *) return 1 ;;
    esac
}

fail=0       # drift or nondeterminism — a real regression signal
matched=0    # runnable cell == its committed baseline for this gpu-arch
nobaseline=0 # runnable cell, but no committed baseline for this gpu-arch
declare -a RECORDED=()
for arch in "${ARCHS[@]}"; do
    echo "== golden: $arch =="
    "$Q" --emit-fixture "$arch" --out "$TMP/$arch" --seed "$SEED" >/dev/null 2>&1
    mapfile -t qflags < <(quant_flags_for_arch "$arch")

    calib=""
    anchor_fmt="$(anchor_format_for_calib "$arch")"
    if [ -n "$anchor_fmt" ]; then
        anchor_hfq="$TMP/$arch-calib-anchor.hfq"
        calib="$TMP/$arch.calib.hfq"
        if ! HIPFIRE_OQ_RAGGED_Q8=1 "$Q" --input "$TMP/$arch" --output "$anchor_hfq" --format "$anchor_fmt" "${qflags[@]}" >/dev/null 2>&1; then
            echo "  NOTE calib: anchor $anchor_fmt quantize failed; calibrated fixture cells will skip"
            calib=""
        elif ! HIPFIRE_GPU_SLAB_LOAD=0 "$PROBE" collect --arch "$arch" --model "$anchor_hfq" --out "$calib" --len 24 >/dev/null 2>"$TMP/$arch-calib.err"; then
            echo "  NOTE calib: collect failed; calibrated fixture cells will skip"
            calib=""
        fi
    fi

    while IFS= read -r fmt; do
        if reason="$(blocked_format_reason "$arch" "$fmt")" && [ -n "$reason" ]; then
            echo "  SKIP $fmt: blocked: $reason"
            continue
        fi
        if requires_hessian_format "$fmt" && [ -z "$calib" ]; then
            echo "  SKIP $fmt: no calibration Hessian"
            continue
        fi
        hfq="$TMP/$arch-$fmt.hfq"
        qextra=("${qflags[@]}")
        if requires_hessian_format "$fmt"; then
            qextra+=(--hessian "$calib")
        fi
        if ! HIPFIRE_OQ_RAGGED_Q8=1 "$Q" --input "$TMP/$arch" --output "$hfq" --format "$fmt" "${qextra[@]}" >/dev/null 2>&1; then
            echo "  SKIP $fmt: quantize unsupported on this build/arch"
            continue
        fi

        read -r gpu_arch h1 <<<"$(run_golden "$hfq")"
        read -r _ h2 <<<"$(run_golden "$hfq")"
        if [ "$h1" = "MISSING" ]; then
            echo "  SKIP $fmt: not runnable here (no logit_hash)"
            continue
        fi

        # 1. Determinism (hard — a runnable cell that flips is a real bug).
        if [ "$h1" != "$h2" ]; then
            echo "  FAIL determinism $fmt: $h1 != $h2 (nondeterministic kernel)"
            fail=1
            continue
        fi
        RECORDED+=("$gpu_arch $arch $fmt $h1")

        # 2. Baseline (hard — a runnable cell that drifts from committed fails).
        base="$(lookup_baseline "$gpu_arch" "$arch" "$fmt")"
        if [ "$RECORD" = 1 ]; then
            echo "  $fmt: $h1 ($gpu_arch) [record]"
        elif [ -z "$base" ]; then
            echo "  NOTE $fmt: no baseline for $gpu_arch — observed $h1 (run --record to add)"
            nobaseline=$((nobaseline + 1))
        elif [ "$base" != "$h1" ]; then
            echo "  FAIL drift $fmt: $h1 != baseline $base → escalate to 35B golden (agentic-gate.sh)"
            fail=1
        else
            echo "  OK $fmt: matches baseline ($h1)"
            matched=$((matched + 1))
        fi
    done < <(formats_for_arch "$arch")
done

if [ "$RECORD" = 1 ]; then
    if [ "${#RECORDED[@]}" -eq 0 ]; then
        echo "fixture-golden-gate: nothing runnable to record on this box — $BASELINES left unchanged" >&2
        exit 0
    fi
    # MERGE, don't overwrite: baselines are keyed by (gpu_arch, model_arch,
    # format) and boxes only record their own runnable cells. A blind `>` here
    # would wipe every OTHER fleet arch's committed baselines. Keep the header +
    # all entries we did NOT just record, then append the fresh ones.
    tmp_base="$(mktemp)"
    echo "# gpu_arch  model_arch  format  logit_hash  (len=$LEN warmup=$WARMUP seed=$SEED mode=tf)" >"$tmp_base"
    {
        if [ -f "$BASELINES" ]; then
            # existing entries, minus the (gpu_arch, model_arch, format) cells
            # we just re-recorded (those get the fresh hash below)
            awk 'NR==FNR { key[$1" "$2" "$3]=1; next }
                 /^#/ { next }
                 !(($1" "$2" "$3) in key) { print }' \
                <(printf '%s\n' "${RECORDED[@]}") "$BASELINES"
        fi
        printf '%s\n' "${RECORDED[@]}"
    } | sort -k1,3 >>"$tmp_base"
    mv "$tmp_base" "$BASELINES"
    echo "fixture-golden-gate: merged ${#RECORDED[@]} baseline(s) for this box into $BASELINES"
    exit 0
fi

# Exit-code contract (consumed by .githooks/pre-commit's two-tier wiring):
#   0  CONFIRMED unchanged — every runnable cell matched a committed baseline
#      for this gpu-arch (>=1 match, no drift, no missing baseline). The covered
#      forward paths did not change → safe to skip the heavy coherence battery.
#   1  DRIFT / nondeterminism — a covered kernel changed output. Escalate.
#   3  INCONCLUSIVE — cells ran but ≥1 has no baseline for this gpu-arch
#      (e.g. fleet box not yet recorded). Do NOT skip the heavy battery.
#   2  could not run anything (build/GPU). Do NOT skip the heavy battery.
if [ "$fail" != 0 ]; then
    echo "fixture-golden-gate: FAIL (drift/nondeterminism) → escalate to coherence-gate.sh + rebaseline if intended"
    exit 1
fi
if [ "$nobaseline" -gt 0 ]; then
    echo "fixture-golden-gate: INCONCLUSIVE ($nobaseline cell(s) with no baseline for this gpu-arch; matched $matched) — run --record"
    exit 3
fi
if [ "$matched" -gt 0 ]; then
    echo "fixture-golden-gate: PASS ($matched cell(s) match committed baselines)"
    exit 0
fi
echo "fixture-golden-gate: INCONCLUSIVE (no cells ran on this box)"
exit 2
