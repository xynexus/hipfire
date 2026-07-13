#!/usr/bin/env bash
set -euo pipefail

NPU_VENV="${HIPFIRE_NPU_VENV:-$HOME/.venv}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$NPU_VENV/bin/activate"
PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
export PEANO_INSTALL_DIR
export PATH="/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:$PATH"
# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1 || true
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export MLIR_AIE_INC="${MLIR_AIE_INC:-$MA_ROOT/include}"
PYXRT_DIR="$(dirname "$(find /opt/xilinx/xrt -name 'pyxrt*.so' 2>/dev/null | head -1)")"
export PYTHONPATH="$MA_ROOT/python${PYXRT_DIR:+:$PYXRT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v hipfire >/dev/null 2>&1; then
    echo "ERROR: hipfire command is required for NPU lock coordination" >&2
    exit 1
fi
hipfire lock acquire npu-r57-production-dma
trap 'hipfire lock release >/dev/null 2>&1' EXIT

cd "$HERE"
exec python run_matrix.py
