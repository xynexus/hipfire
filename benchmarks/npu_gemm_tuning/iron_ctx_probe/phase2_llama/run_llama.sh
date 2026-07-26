#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase-2 launcher: run the vendored IRON llama_3.2_1b whole-model app against
# the EXACT toolchain versions it was pinned to, overlaid non-destructively over
# ~/.venv (which carries a newer mlir_aie/Peano that break the vendored app).
#
# Pins (from third_party/IRON/requirements.txt), already extracted in the uv cache
# with cp314 ABIs — adjust the hashes if the cache is pruned/rebuilt:
#   mlir_aie==0.0.1.2026033104+e4f35d6   (has aie.iron.placers; old resolve_program)
#   llvm-aie==21.0.0.2026062201+7820460e (Peano)
#
# STATUS: compiles all 13 real aie2p operators + runs to the fused-decode ELF,
# which then fails aiecc --generate-full-elf NPU lowering ('aie.dma_bd' verifier).
# That failure is a toolchain codegen limitation, not this env overlay. See
# ../README.md "Phase 2 update".
set -euo pipefail

MARCH=/home/sadara/.cache/uv/archive-v0/g8BAAQbkdDCqINSV/mlir_aie/python
JUNE=/home/sadara/.cache/uv/archive-v0/WW__PzgdMSz4Uz7c/llvm-aie
IRON=/home/sadara/.hipfire/src/third_party/IRON
PY=/home/sadara/.venv/bin/python3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for p in "$MARCH/aie/iron/placers.py" "$JUNE/bin/clang++"; do
  [ -e "$p" ] || { echo "MISSING pinned toolchain: $p (uv cache pruned? re-pip-install IRON requirements)"; exit 1; }
done

exec env \
  PYTHONUNBUFFERED=1 \
  PEANO_INSTALL_DIR="$JUNE" \
  PYTHONPATH="$MARCH:/opt/xilinx/xrt/python:$IRON" \
  XILINX_XRT=/opt/xilinx/xrt \
  "$PY" -u "$HERE/run_llama.py" "$@"
