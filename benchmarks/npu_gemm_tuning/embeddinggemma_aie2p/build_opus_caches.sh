#!/usr/bin/env bash
# Build AIE2P/NPU2 Opus GEMM caches for EmbeddingGemma-300M projection shapes.
#
# Outputs live under ~/.hipfire/npu:
#   embgemma_aie2p_w4_4x4x16_c8_nb{4,12,18,48}
#   embgemma_aie2p_sparse3_4x4x16_c8_nb{4,12,18,48}_sparse3
#   embgemma_aie2p_w8_4x4x32_c8_nb{4,12,18,48}_m8k8_w8 when
#     HIPFIRE_EMBGEMMA_NPU_BUILD_W8=1
#
# These are benchmark artifacts, not repo state.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R6="$HERE/../r6"
MT="${HIPFIRE_EMBGEMMA_NPU_MT:-4}"
NT=4
KCHUNK=16
W8_KCHUNK=32
W8_MTS="${HIPFIRE_EMBGEMMA_NPU_W8_MTS:-4}"
COLS="${HIPFIRE_EMBGEMMA_NPU_COLS:-8}"

build_one() {
  local tag="$1"
  local n="$2"
  local mt="${3:-$MT}"
  local nb=$((n / 64))
  [ $((n % 64)) -eq 0 ] || { echo "N=$n must be divisible by 64"; exit 1; }
  if [ "$tag" = sparse3 ]; then
    R6_SPARSE3=1 R6_GEN=r6_gen_mp.py R6_OUT_TAG=embgemma_aie2p_sparse3 \
      "$R6/r6_cache.sh" "$mt" "$NT" "$KCHUNK" "$COLS" "$nb"
  elif [ "$tag" = w8 ]; then
    R6_W8_M8=1 R6_GEN=r6_gen_mp.py R6_OUT_TAG=embgemma_aie2p_w8 \
      "$R6/r6_cache.sh" "$mt" "$NT" "$W8_KCHUNK" "$COLS" "$nb"
  else
    R6_KERNEL_SRC="$R6/r6_gemm_ts.cc" R6_GEN=r6_gen_mp.py R6_OUT_TAG=embgemma_aie2p_w4 \
      "$R6/r6_cache.sh" "$mt" "$NT" "$KCHUNK" "$COLS" "$nb"
  fi
}

for n in 256 768 1152 3072; do
  build_one w4 "$n"
  build_one sparse3 "$n"
  if [ "${HIPFIRE_EMBGEMMA_NPU_BUILD_W8:-0}" = 1 ]; then
    for w8_mt in $W8_MTS; do
      build_one w8 "$n" "$w8_mt"
    done
  fi
done
