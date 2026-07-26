#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HIPFIRE_R25_DYNAMIC_GEMM=1
export HIPFIRE_R25_COMPACT_FRAGMENT_RING=1
export HIPFIRE_R25_CANONICAL_GATE=1
export HIPFIRE_R25_CXX_FLAGS=-DR15_DYNAMIC_ONLY
export R25_CACHE_DIR="${R97_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r97_canonical_bf16_w4_resident_ffn_m256_k768_i1152_o768}"
"$HERE/../r25/r25_cache.sh"
printf '%s\n' \
  'input=canonical-bf16-pre-ffn-norm' \
  'gate-prep=inline-r25-pack3' >> "$R25_CACHE_DIR/shape.txt"
