#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export R99_CACHE_DIR="${R106_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r106_unit_rms_bf16_w4_resident_ffn_combined_bf16x2_m256_k768_i1152_o768}"
export R99_INPUT_LABEL=canonical-bf16-unit-rms
"$HERE/../r99/build_r99.sh"
printf '%s\n' \
  'immutable-pre-ffn-norm=loader-folded' \
  'producer=embgemma-r105-direct-x-unit-rms' >> "$R99_CACHE_DIR/shape.txt"
echo "$R99_CACHE_DIR"
