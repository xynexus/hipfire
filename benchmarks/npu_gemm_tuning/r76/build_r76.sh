#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R75_WINDOW=3 \
R75_CACHE_DIR="${R76_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r76_enqueue_window3_attention_m256_k768_n1280}" \
  "$HERE/../r75/build_r75.sh"
