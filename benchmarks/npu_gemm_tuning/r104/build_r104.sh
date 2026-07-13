#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HIPFIRE_R25_DYNAMIC_GEMM=1
export HIPFIRE_R25_COMPACT_FRAGMENT_RING=1
export HIPFIRE_R25_COMPACT_DOWN_BRANCHES=1
export HIPFIRE_R25_CANONICAL_GATE=1
export HIPFIRE_R25_DIRECT_X_NORMALIZE=1
export HIPFIRE_R25_DIRECT_X_FULL_OBJECT=1
export HIPFIRE_R25_INTERLEAVED_BF16X2_OUTPUT=1
export HIPFIRE_R25_OUTPUT_ROW_WORDS=1152
export HIPFIRE_R25_CXX_FLAGS='-DR15_DYNAMIC_ONLY -DR104_FULL_X_OBJECT'
export HIPFIRE_R25_CXX_OPT=-O2
export R25_CACHE_DIR="${R104_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r104_direct_x_inline_norm_w4_resident_ffn_combined_bf16x2_m256_k768_i1152_o768}"
"$HERE/../r25/r25_cache.sh"
printf '%s\n' \
  'input=canonical-direct-x-bf16' \
  'normalization=inline-rms-pre-ffn-norm' \
  'rms-epsilon=1e-6' \
  'input-dma=single-full-row-object' \
  'gate-prep=inline-r25-pack3' \
  'output=token-major-interleaved-bf16x2' \
  'output-row-words=1152' >> "$R25_CACHE_DIR/shape.txt"
echo "$R25_CACHE_DIR"
