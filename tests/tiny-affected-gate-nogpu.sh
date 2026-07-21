#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — no-GPU regression tests for the tiny affected-file selector.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

write_paths() {
    local file="$1"
    shift
    printf '%s\n' "$@" >"$file"
}

run_selector() {
    local file="$1"
    local out="$2"
    shift 2
    set +e
    ./tests/tiny-affected-gate.sh --files-from "$file" --dry-run "$@" >"$out" 2>&1
    local status=$?
    set -e
    return "$status"
}

assert_status() {
    local want="$1"
    local got="$2"
    local label="$3"
    if [ "$got" -ne "$want" ]; then
        echo "tiny-affected-gate-nogpu: $label: expected status $want, got $got" >&2
        return 1
    fi
}

assert_contains() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if ! grep -Fq "$pattern" "$file"; then
        echo "tiny-affected-gate-nogpu: $label: missing '$pattern'" >&2
        sed -n '1,80p' "$file" >&2
        return 1
    fi
}

case_file="$TMP/qwen2.txt"
out="$TMP/qwen2.out"
write_paths "$case_file" "crates/hipfire-arch-qwen2/src/qwen2.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "qwen2 mapping"
assert_contains "$out" "selected families: qwen2" "qwen2 mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "qwen2 mapping"

case_file="$TMP/deepseek4.txt"
out="$TMP/deepseek4.out"
write_paths "$case_file" "crates/hipfire-arch-deepseek4/src/forward.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "deepseek4 mapping"
assert_contains "$out" "selected families: deepseek4,deepseek4_compressed,deepseek4_mtp" "deepseek4 mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "deepseek4 mapping"

case_file="$TMP/spec.txt"
out="$TMP/spec.out"
write_paths "$case_file" "crates/hipfire-arch-qwen35/src/speculative.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "spec mapping"
assert_contains "$out" "selected families: qwen3_5,qwen3_5_moe" "spec mapping"
assert_contains "$out" "selected gates: quant=0 state=0 spec=1" "spec mapping"

case_file="$TMP/shared.txt"
out="$TMP/shared.out"
write_paths "$case_file" "crates/hipfire-quantize/src/fixture.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "shared fixture mapping"
assert_contains "$out" "selected families: deepseek4,deepseek4_compressed,deepseek4_mtp,dots_ocr,gemma3,gemma3_vl,gemma4_dense,gemma4_moe,gemma4_ple,lfm2_moe,llama,mamba2,minimax,qwen2,qwen3_5,qwen3_5_moe,qwen3_5_vl" "shared fixture mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "shared fixture mapping"

case_file="$TMP/dots_ocr.txt"
out="$TMP/dots_ocr.out"
write_paths "$case_file" "crates/hipfire-arch-dots-ocr/src/dots_ocr.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "dots-ocr mapping"
assert_contains "$out" "selected families: dots_ocr" "dots-ocr mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "dots-ocr mapping"

case_file="$TMP/qwen35_vl.txt"
out="$TMP/qwen35_vl.out"
write_paths "$case_file" "crates/hipfire-arch-qwen35-vl/src/arch.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "qwen35-vl mapping"
assert_contains "$out" "selected families: qwen3_5_vl" "qwen35-vl mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "qwen35-vl mapping"

case_file="$TMP/gemma3_vl.txt"
out="$TMP/gemma3_vl.out"
write_paths "$case_file" "crates/hipfire-arch-gemma3-vl/src/arch.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "gemma3-vl mapping"
assert_contains "$out" "selected families: gemma3,gemma3_vl" "gemma3-vl mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "gemma3-vl mapping"

case_file="$TMP/mamba2.txt"
out="$TMP/mamba2.out"
write_paths "$case_file" "crates/hipfire-runtime/src/mamba2_state.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "mamba2 mapping"
assert_contains "$out" "selected families: mamba2" "mamba2 mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "mamba2 mapping"

case_file="$TMP/lfm2.txt"
out="$TMP/lfm2.out"
write_paths "$case_file" "crates/hipfire-arch-lfm2moe/src/forward.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "lfm2 mapping"
assert_contains "$out" "selected families: lfm2_moe" "lfm2 mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "lfm2 mapping"

case_file="$TMP/uncovered.txt"
out="$TMP/uncovered.out"
write_paths "$case_file" "crates/hipfire-arch-gemma4/src/config.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "gemma4 mapping"
assert_contains "$out" "selected families: gemma4_dense,gemma4_moe,gemma4_ple" "gemma4 mapping"
assert_contains "$out" "selected gates: quant=1 state=1 spec=0" "gemma4 mapping"

case_file="$TMP/nemotron.txt"
out="$TMP/nemotron.out"
write_paths "$case_file" "crates/hipfire-arch-nemotron/src/model.rs"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 3 "$status" "nemotron mapping"
assert_contains "$out" "tiny-uncovered families/features" "nemotron mapping"

case_file="$TMP/irrelevant.txt"
out="$TMP/irrelevant.out"
write_paths "$case_file" "tools/unrelated.txt"
if run_selector "$case_file" "$out"; then status=0; else status=$?; fi
assert_status 0 "$status" "irrelevant mapping"
assert_contains "$out" "no tiny-model-relevant changed paths" "irrelevant mapping"

out="$TMP/irrelevant-required.out"
if run_selector "$case_file" "$out" --require-coverage; then status=0; else status=$?; fi
assert_status 3 "$status" "require coverage mapping"
assert_contains "$out" "no tiny coverage selected for changed paths" "require coverage mapping"

echo "tiny-affected-gate-nogpu: PASS"
