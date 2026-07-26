#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HIPFIRE_R25_DYNAMIC_GEMM=1
export HIPFIRE_R25_COMPACT_FRAGMENT_RING=1
export HIPFIRE_R25_CANONICAL_GATE=1
export HIPFIRE_R25_INTERLEAVED_BF16X2_OUTPUT=1
export HIPFIRE_R25_OUTPUT_ROW_WORDS=1152
export HIPFIRE_R25_CXX_FLAGS=-DR15_DYNAMIC_ONLY
export R25_CACHE_DIR="${R99_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r99_canonical_bf16_w4_resident_ffn_combined_bf16x2_m256_k768_i1152_o768}"
R99_INPUT_LABEL=${R99_INPUT_LABEL:-canonical-bf16-pre-ffn-norm}
"$HERE/../r25/r25_cache.sh"
printf '%s\n' \
  "input=$R99_INPUT_LABEL" \
  'gate-prep=inline-r25-pack3' \
  'output=token-major-interleaved-bf16x2' \
  'output-row-words=1152' \
  'output-layout=post-ffn-combined-prefix' >> "$R25_CACHE_DIR/shape.txt"
