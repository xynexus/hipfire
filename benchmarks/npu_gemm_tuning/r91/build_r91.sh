#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export R90_CACHE_DIR="${R91_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r91_residual_norm_ffn_handoff_paired_w4_m256_k768_n1280}"
export R90_GENERATOR_FLAG=--direct-output-ffn-handoff
"$HERE/../r90/build_r90.sh"
printf 'handoff=staging-prefix-dmabuf\noutput=canonical-token-major-pre-ffn-norm-bf16-prefix\n' >> "$R90_CACHE_DIR/shape.txt"
