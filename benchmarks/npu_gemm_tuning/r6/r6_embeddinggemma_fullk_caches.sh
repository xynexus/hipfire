#!/usr/bin/env bash
# Build the M=256 full-spectrum projection cache inventory used by the
# EmbeddingGemma resident Opus projector. Artifacts stay outside the checkout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# K values are padded to complete 256-element Opus groups.
shapes=(
  "768 256"
  "768 768"
  "768 1152"
  "768 3072"
  "1280 768"
  "3072 768"
)
for mode in w4 mixed w8; do
  for shape in "${shapes[@]}"; do
    read -r k n <<< "$shape"
    cache="$HOME/.hipfire/npu/embgemma_aie2p_fullk_submit_${mode}_m256_kg$((k / 256))_n${n}"
    if [[ -s "$cache/final.xclbin" && -s "$cache/insts.bin" ]]; then
      echo "$cache"
      continue
    fi
    "$HERE/r6_fullk_cache.sh" "$mode" 256 "$k" "$n" 8
  done
done
