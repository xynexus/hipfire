#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export R85_CACHE_DIR="${R90_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r90_residual_norm_reuse_a_paired_w4_projection_pack_attention_o_m256_k768_n1280}"
export R85_OUTPUT_SOURCE="$HERE/../r89/r89_output_projection_bf16_stage.cc"
export R85_EXTRA_SOURCE="$HERE/r90_split_residual_norm.cc"
export R85_EXTRA_OPT=-O1
export R85_GENERATOR_FLAG="${R90_GENERATOR_FLAG:---direct-output-residual-norm}"
export R85_PACK_OPT=-Os
"$HERE/../r85/build_r85.sh"
printf 'output=token-major-bf16-pre-ffn-norm\nstaging=held-final-activation-fifo-8k-plus-local-tail-4k\ntail=post-attention-rms-residual-pre-ffn-rms\n' >> "$R85_CACHE_DIR/shape.txt"
