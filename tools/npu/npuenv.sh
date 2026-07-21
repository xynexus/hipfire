# shellcheck shell=bash
# Env for the mlir-aie fork venv on nix1 (working @iron.jit + CompileTime).
# Usage:  source tools/npu/npuenv.sh; npupy tools/npu/<script>.py ...
export PATH="/opt/xilinx/xrt/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export PEANO_INSTALL_DIR="$HOME/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie"
export PYTHONPATH="$HOME/mlir-aie-312/install/python:${PYTHONPATH:-}"
npupy() { "$HOME/mlir-aie-312/venv312/bin/python" "$@"; }
