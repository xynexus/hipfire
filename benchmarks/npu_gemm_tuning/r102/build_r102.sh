#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HIPFIRE_R25_DYNAMIC_GEMM=1
export HIPFIRE_R25_COMPACT_FRAGMENT_RING=1
export HIPFIRE_R25_CANONICAL_GATE=1
export HIPFIRE_R25_DIRECT_X_ROW_STATE=1
export HIPFIRE_R25_INTERLEAVED_BF16X2_OUTPUT=1
export HIPFIRE_R25_OUTPUT_ROW_WORDS=1152
# The 4,992-byte three-row activation object otherwise leaves insufficient
# tile memory for both copies of the 15,872-byte resident weight FIFO.
export HIPFIRE_R25_WEIGHT_CORE_DEPTH=1
export HIPFIRE_R25_CXX_FLAGS=-DR15_DYNAMIC_ONLY
export R25_CACHE_DIR="${R102_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r102_direct_x_row_state_w4_resident_ffn_combined_bf16x2_m256_k768_i1152_o768}"
"$HERE/../r25/r25_cache.sh"
printf '%s\n' \
  'input=canonical-direct-x-bf16-row-state' \
  'input-row-bytes=1664' \
  'input-state-offset=1536' \
  'input-state=pre-ffn-inverse-f32' \
  'gate-prep=inline-pre-ffn-norm-r25-pack3' \
  'output=token-major-interleaved-bf16x2' \
  'output-row-words=1152' >> "$R25_CACHE_DIR/shape.txt"
echo "$R25_CACHE_DIR"
