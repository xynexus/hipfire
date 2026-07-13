#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HIPFIRE_R25_DYNAMIC_GEMM=1
export HIPFIRE_R25_COMPACT_FRAGMENT_RING=1
export HIPFIRE_R25_CXX_FLAGS=-DR15_DYNAMIC_ONLY
export R25_CACHE_DIR="${R96_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r96_compact_ring_w4_resident_ffn_m256_k768_i1152_o768}"
"$HERE/../r25/r25_cache.sh"
