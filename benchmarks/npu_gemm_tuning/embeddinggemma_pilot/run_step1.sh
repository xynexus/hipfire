#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a single-op IRON verification against the EmbeddingGemma oracle, using the
# pinned March mlir_aie overlay + pyxrt (single-op xclbin; no full-ELF needed).
set -euo pipefail
MARCH=/home/sadara/.cache/uv/archive-v0/g8BAAQbkdDCqINSV/mlir_aie/python
JUNE=/home/sadara/.cache/uv/archive-v0/WW__PzgdMSz4Uz7c/llvm-aie
IRON=/home/sadara/.hipfire/src/third_party/IRON
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  PYTHONUNBUFFERED=1 PEANO_INSTALL_DIR="$JUNE" \
  PYTHONPATH="$MARCH:/opt/xilinx/xrt/python:$IRON" \
  XILINX_XRT=/opt/xilinx/xrt \
  /home/sadara/.venv/bin/python3 -u "$HERE/${1:-step1_rmsnorm.py}"
