#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny-fixture round-trip check (CPU only, no GPU).
#
# Emits each tiny random-init gating fixture and round-trips it through the
# quantizer. Validates that `--emit-fixture` produces a model the real ingest
# path accepts (arch detect + name-mapper + per-tensor quantize), for every
# supported arch — the cheap half of the fixture tripwire. The GPU golden
# (forward + logit_hash) lives in tests/fixture-golden-gate.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Always build (cargo is a no-op when current; an existing binary may be stale).
echo "fixture-roundtrip: building hipfire-quantize..."
cargo build -p hipfire-quantize >/dev/null
Q="$ROOT/target/debug/hipfire-quantize"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# arch → expected ingest arch id + any extra quantize flags. Qwen2's model_type
# auto-detects to the LLaMA-family id=1 (which drops Q/K/V bias); --arch-id 7
# routes it to the dedicated hipfire-arch-qwen2 loader, so we assert the override
# message ("to 7") instead of an "id=" line.
ARCHS=(qwen3_5 qwen3_5_vl qwen3_5_moe deepseek4 deepseek4_compressed deepseek4_mtp qwen2 dots_ocr gemma3 gemma3_vl minimax lfm2_moe mamba2 llama gemma4_dense gemma4_ple gemma4_moe)
EXPECT_ID=("id=5" "id=5" "id=6" "id=9" "id=9" "id=9" "to 7" "id=8" "id=12" "id=13" "id=10" "id=11" "id=15" "id=0" "id=24" "id=24" "id=24")
ARCH_FLAGS=("" "--include-vision --vision-quant hfq4" "" "" "" "--allow-mq2-lloyd" "--arch-id 7" "--include-vision --vision-quant hfq4" "" "--include-vision --vision-quant q8f16" "" "" "" "" "" "" "")
ARCH_FORMATS=("mq4" "mq4" "mq4" "mq4" "mq4" "deepseek4-source-precision" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4" "mq4")

for i in "${!ARCHS[@]}"; do
    arch="${ARCHS[$i]}"
    want="${EXPECT_ID[$i]}"
    fmt="${ARCH_FORMATS[$i]}"
    echo "== fixture round-trip: $arch (expect $want) =="
    "$Q" --emit-fixture "$arch" --out "$TMP/$arch" --seed 42
    out="$("$Q" --input "$TMP/$arch" --output "$TMP/$arch.hfq" --format "$fmt" ${ARCH_FLAGS[$i]} 2>&1)"
    if ! grep -qiE "Architecture:.*$want" <<<"$out"; then
        echo "FAIL: $arch did not auto-detect $want" >&2
        grep -i architecture <<<"$out" >&2 || true
        exit 1
    fi
    if ! grep -qiE "^Writing:" <<<"$out"; then
        echo "FAIL: $arch produced no .hfq" >&2
        exit 1
    fi
    echo "  OK ($arch round-trips ingest→quantize, $want)"
done

echo "fixture-roundtrip-nogpu: PASS"
