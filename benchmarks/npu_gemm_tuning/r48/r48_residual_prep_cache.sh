#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_residual_prep_bf16x2_to_r34_records_m256_k768"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/r48_residual_prep.cc" -c -o "$OUT/r48prep.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
python "$HERE/r48_residual_prep_gen.py" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-residual-prep' \
  'm=256' \
  'k=768' \
  'input=shared-completed-bf16x2' \
  'output=shared-activation-tail-r34-bf16-records' \
  'residual-records=32' \
  'residual-record-bytes=16384' > "$OUT/shape.txt"
echo "$OUT"
