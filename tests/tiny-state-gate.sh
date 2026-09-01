#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny autoregressive state/KV/logit-hash gate (GPU).
#
# Emits seeded tiny fixtures, quantizes to each family's anchor format, then
# free-runs greedy decode with growing KV/state via tiny_quant_probe ar-hash.
# This is a structural tripwire for forward/KV/position/long-tail drift; it is
# not a semantic quality test.
#
#   ./tests/tiny-state-gate.sh            # check vs committed baselines
#   ./tests/tiny-state-gate.sh --record   # rewrite baselines for THIS environment
#
# Baseline key: gpu_arch + family + format, PLUS the two trailing env fields
# `hip=` and `kcache=` when present. The arch alone is not a host: nix1 and nix2
# are both gfx1103, so a baseline recorded on one silently overwrote the other's
# row and the two hosts ping-ponged the value (mamba2/fp16 took three different
# values in 15 days with no code change between them).
#
# The key is deliberately what CAUSES numeric differences, not machine identity:
# the ROCm/HIP toolchain version, which determines kernel codegen. Two hosts on
# the same toolchain share one row, which is correct; a host on another gets "no
# baseline" (INCONCLUSIVE) rather than a false FAIL against a foreign row.
#
# A hash of the PRECOMPILED KERNEL CACHE was tried here and reverted before it
# shipped: the gate compiles kernels into that cache as it runs, so the hash
# mutates mid-run and even differs between cells (mamba2 saw kcache=8fdaba9f6e93
# where qwen3_5_moe_indexed saw c4aad18bdc7d in the same invocation). A key that
# the act of measuring invalidates can never match again, which would silently
# turn every re-recorded cell into a permanent INCONCLUSIVE — a disabled
# tripwire that still looks like a configured one. Do not reintroduce it.
#
# Machine identity was considered and rejected as the key. It is also mostly
# unavailable: x86 has no CPU serial (CPUID leaf 3 was Pentium-III-only), and
# `rocm-smi --showserial` reports "Not supported" on Phoenix/Strix APUs with
# `--showuniqueid` returning 0x0. `/etc/machine-id` exists but would fragment
# baselines across hosts that ought to agree.
#
# Rows written before this change have no env fields and match ANY environment,
# so nothing is invalidated; an exact env match is preferred when one exists.
#
# Env:
#   HIPFIRE_TINYQUANT_FAMILIES=llama,qwen2,...
#   HIPFIRE_TINYSTATE_LEN=128
#   HIPFIRE_TINYSTATE_PROMPT_LEN=4
#
# Exit: 0 = all selected cells pass, 1 = drift/crash, 2 = infra, 3 = skipped/no baseline.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

BASELINES="tests/tiny-state-baselines.txt"
LEN="${HIPFIRE_TINYSTATE_LEN:-128}"
PROMPT_LEN="${HIPFIRE_TINYSTATE_PROMPT_LEN:-4}"
SEED="${HIPFIRE_TINYSTATE_SEED:-42}"
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

ALL_FAMILIES=(llama qwen2 dots_ocr deepseek4 deepseek4_compressed deepseek4_mtp gemma3 gemma3_vl gemma4_dense gemma4_ple gemma4_moe minimax lfm2_moe mamba2 qwen3_5 qwen3_5_vl qwen3_5_moe qwen3_5_moe_indexed)
families=("${ALL_FAMILIES[@]}")
if [ -n "${HIPFIRE_TINYQUANT_FAMILIES:-}" ]; then
    IFS=',' read -r -a families <<<"$HIPFIRE_TINYQUANT_FAMILIES"
fi

anchor_for() {
    case "$1" in
        llama) echo q8f16 ;;
        deepseek4 | deepseek4_compressed | deepseek4_mtp) echo deepseek4-source-precision ;;
        minimax | lfm2_moe) echo mq6 ;;
        qwen2 | dots_ocr | gemma3 | gemma3_vl | gemma4_dense | gemma4_ple | gemma4_moe | mamba2 | qwen3_5 | qwen3_5_vl | qwen3_5_moe | qwen3_5_moe_indexed) echo fp16 ;;
        *) return 1 ;;
    esac
}

# Families whose recurrent state IS a low-precision accumulator: the state is
# held in fp16 and re-rounded every kernel invocation, with no error feedback, so
# divergence grows superlinearly (BUGS.md measures 13x error for 2.5x tokens on
# the DeltaNet analogue). For those, `logit_hash [exact]` asserts a property the
# arithmetic does not have — mamba2/fp16 has taken four different values across
# two architectures with no code change. `token_hash` stays blocking for every
# family; only the sub-argmax noise is downgraded. Add a family here only with a
# documented instance, not on suspicion: every other fp16 cell is stable.
logit_is_advisory() {
    case "$1" in
        mamba2) return 0 ;;
        *) return 1 ;;
    esac
}

# ROCm/HIP toolchain version — changes kernel codegen, so it changes logit bits.
hip_version() {
    hipconfig --version 2>/dev/null | head -1 | tr -d ' \n' || true
}

# Prefer a row matching this environment exactly; fall back to a legacy row that
# carries no env fields. Echoes "logit token scope".
lookup_baseline() {
    [ -f "$BASELINES" ] || return 0
    awk -v g="$1" -v f="$2" -v q="$3" -v h="$4" '
        $1==g && $2==f && $3==q {
            env_hip = ""
            for (i = 6; i <= NF; i++) {
                if ($i ~ /^hip=/) { env_hip = substr($i, 5) }
            }
            if (env_hip == "")     { legacy = $4" "$5" legacy" }
            else if (env_hip == h) { exact  = $4" "$5" exact" }
        }
        END { print (exact != "" ? exact : legacy) }
    ' "$BASELINES" | head -1
}

echo "tiny-state-gate: building..."
cargo build --release \
    -p hipfire-quantize --bin hipfire-quantize \
    -p hipfire-serving-core --example tiny_quant_probe >/dev/null || {
    echo "build failed" >&2
    exit 2
}

"$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "tiny-state-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
TMP="$(mktemp -d)"
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$TMP"' EXIT

Q="$ROOT/target/release/hipfire-quantize"
P="$ROOT/target/release/examples/tiny_quant_probe"
fail=0
skip=0
matched=0
advisory=0
declare -a RECORDED=()

for raw_family in "${families[@]}"; do
    family="$(echo "$raw_family" | xargs)"
    [ -n "$family" ] || continue
    fmt="$(anchor_for "$family")" || {
        echo "  SKIP $family: unknown family"
        skip=$((skip + 1))
        continue
    }
    echo "== tiny-state: $family/$fmt =="
    hf_dir="$TMP/$family-hf"
    hfq="$TMP/$family-$fmt.hfq"
    if ! "$Q" --emit-fixture "$family" --out "$hf_dir" --seed "$SEED" >/dev/null 2>&1; then
        echo "  FAIL emit fixture"
        fail=$((fail + 1))
        continue
    fi
    extra=()
    [ "$family" = "qwen2" ] && extra=(--arch-id 7)
    case "$family" in
        dots_ocr) extra=(--include-vision --vision-quant hfq4) ;;
        gemma3_vl) extra=(--include-vision --vision-quant q8f16) ;;
        qwen3_5_vl) extra=(--include-vision --vision-quant hfq4) ;;
        deepseek4*) extra=(--allow-mq2-lloyd) ;;
    esac
    if ! "$Q" --input "$hf_dir" --output "$hfq" --format "$fmt" "${extra[@]}" >/dev/null 2>&1; then
        echo "  SKIP $fmt: quantize unsupported"
        skip=$((skip + 1))
        continue
    fi

    out="$TMP/$family-ar.out"
    err="$TMP/$family-ar.err"
    if ! LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
        "$P" ar-hash --arch "$family" --model "$hfq" --len "$LEN" --prompt-len "$PROMPT_LEN" --seed "$SEED" \
        >"$out" 2>"$err"; then
        echo "  FAIL ar-hash: $(tail -2 "$err" | tr '\n' ' ')"
        fail=$((fail + 1))
        continue
    fi
    gpu_arch="$(grep -oE 'gfx[0-9a-f]+' "$err" | head -1)"
    logit_hash="$(awk '/^logit_hash:/ {print $2}' "$out" | head -1)"
    token_hash="$(awk '/^token_hash:/ {print $2}' "$out" | head -1)"
    if [ -z "$gpu_arch" ] || [ -z "$logit_hash" ] || [ -z "$token_hash" ]; then
        echo "  FAIL missing ar-hash output"
        fail=$((fail + 1))
        continue
    fi
    env_hip="$(hip_version)"
    RECORDED+=("$gpu_arch $family $fmt $logit_hash $token_hash hip=$env_hip")
    if [ "$RECORD" = 1 ]; then
        echo "  $fmt: $logit_hash $token_hash ($gpu_arch hip=$env_hip) [record]"
        matched=$((matched + 1))
        continue
    fi
    read -r base_logit base_token base_scope \
        <<<"$(lookup_baseline "$gpu_arch" "$family" "$fmt" "$env_hip")"
    if [ -z "$base_logit" ] || [ -z "$base_token" ]; then
        echo "  NOTE no baseline for $gpu_arch/$family/$fmt at hip=$env_hip — observed $logit_hash $token_hash"
        skip=$((skip + 1))
    elif [ "$base_token" != "$token_hash" ]; then
        echo "  FAIL token drift: observed $logit_hash/$token_hash baseline $base_logit/$base_token [$base_scope]"
        [ "$base_scope" = legacy ] && echo "         (matched a pre-env-key row; it may have been recorded under a different toolchain — hip=$env_hip here)"
        fail=$((fail + 1))
    elif [ "$base_logit" != "$logit_hash" ]; then
        if logit_is_advisory "$family"; then
            echo "  ADVISORY logit drift (token stream identical): observed $logit_hash baseline $base_logit [$base_scope]"
            advisory=$((advisory + 1))
        else
            echo "  FAIL logit drift: observed $logit_hash/$token_hash baseline $base_logit/$base_token [$base_scope]"
            [ "$base_scope" = legacy ] && echo "         (matched a pre-env-key row; it may have been recorded under a different toolchain — hip=$env_hip here)"
            fail=$((fail + 1))
        fi
    else
        echo "  OK $fmt: matches baseline ($logit_hash/$token_hash)"
        matched=$((matched + 1))
    fi
done

if [ "$RECORD" = 1 ]; then
    tmp_base="$(mktemp)"
    {
        echo "# tiny-state AR hash baselines — gpu_arch family format logit_hash token_hash [env...]"
        echo "# len=$LEN prompt_len=$PROMPT_LEN seed=$SEED"
    } >"$tmp_base"
    {
        if [ -f "$BASELINES" ]; then
            awk 'NR==FNR { key[$1" "$2" "$3]=1; next }
                 /^#/ { next }
                 !(($1" "$2" "$3) in key) { print }' \
                <(printf '%s\n' "${RECORDED[@]}") "$BASELINES"
        fi
        printf '%s\n' "${RECORDED[@]}"
    } | sort -k1,3 >>"$tmp_base"
    mv "$tmp_base" "$BASELINES"
    echo "tiny-state-gate: merged ${#RECORDED[@]} baseline(s) into $BASELINES"
    exit 0
fi
adv_note=""
[ "$advisory" -gt 0 ] && adv_note=", $advisory advisory logit drift"
if [ "$fail" -gt 0 ]; then
    echo "tiny-state-gate: FAIL ($fail cell(s)$adv_note)"
    exit 1
fi
if [ "$skip" -gt 0 ]; then
    echo "tiny-state-gate: INCONCLUSIVE ($skip skipped/no-baseline cell(s), $matched matched$adv_note)"
    exit 3
fi
if [ "$((matched + advisory))" -gt 0 ]; then
    echo "tiny-state-gate: PASS ($matched cell(s)$adv_note)"
    exit 0
fi
echo "tiny-state-gate: INCONCLUSIVE (no cells ran)"
exit 3
