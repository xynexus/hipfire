#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="${R88_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r88_owsh_depth3_reuse_a_paired_w4_projection_pack_attention_o_m256_k768_n1280}"
R85_CACHE_DIR="$OUT" \
R85_GENERATOR_FLAG=--output-weight-shim-depth3 \
  "$HERE/../r85/build_r85.sh" >/dev/null
printf 'op=resident-opus-compact-paired-qkv-projection-pack-attention-output-direct-reuse-a-owsh-depth3\nmode=w4-scaled\nm=256\nk=768\nn=1280\ncontexts=1\nprojection-columns=1,3,5,7\npack-output-columns=0,2,4,6\nattention-columns=1,3,5,7\nattention-handoff=adjacent-depth3-bf16\nattention-order=0,2,4,1,3,5\nweights=qkv-paired-whole-scaled-plus-o-direct\noutput=token-major-f32\noutput-kernel=reuse-activation-across-four-n-tiles\noutput-weight-shim-depth=3\noutput-weight-core-depth=1\n' > "$OUT/shape.txt"
echo "$OUT"
