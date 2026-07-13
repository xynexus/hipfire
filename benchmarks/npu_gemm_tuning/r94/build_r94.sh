#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export R93_CACHE_DIR="${R94_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r94_vector_bf16_to_r25_w4_activation_m256_k768}"
export R93_VECTOR_PREP=1
"$HERE/../r93/build_r93.sh"
