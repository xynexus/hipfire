#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R107_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r107_next_layer_prep_w8_bf16x2_with_residual_m256_k768}
rm -rf -- "$OUT"
mkdir -p -- "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$ROOT/r47/r47_next_layer_prep.cc" -c -o "$OUT/r47.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
"$PEANO/bin/clang++" "$ROOT/r48/r48_residual_prep.cc" -c -o "$OUT/r48prep.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
HIPFIRE_R47_FUSED_RESIDUAL=1 python "$ROOT/r47/r47_gen.py" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-next-layer-prep' \
  'mode=w8-scaled' \
  'm=256' \
  'k=768' \
  'input=shared-completed-bf16x2' \
  'output=shared-r34-activation-and-residual' \
  'prefix-bytes=6240' \
  'residual-records=32' \
  'residual-record-bytes=16384' \
  'input-passes=4' > "$OUT/shape.txt"
echo "$OUT"
