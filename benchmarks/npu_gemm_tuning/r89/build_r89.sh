#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export R85_CACHE_DIR="${R89_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r89_bf16_local_stage_reuse_a_paired_w4_projection_pack_attention_o_m256_k768_n1280}"
export R85_OUTPUT_SOURCE="$HERE/r89_output_projection_bf16_stage.cc"
export R85_GENERATOR_FLAG=--direct-output-bf16-stage
export R85_PACK_OPT=-Os
"$HERE/../r85/build_r85.sh"
printf 'output=token-major-bf16\nstaging=held-final-activation-fifo-10k-plus-local-tail-2k\n' >> "$R85_CACHE_DIR/shape.txt"
