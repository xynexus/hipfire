#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R61="$HERE/../r61"
OUT="${R62_PHYSICAL_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r62_w4_native_physical_qkv_m256_k768_n1280}"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
COMMON=(
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf
)
"$PEANO/bin/clang++" "$HERE/../r15/r15_w4_scaled.cc" -c -o "$OUT/compute.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$R61/r61_weight_sink.cc" -c -o "$OUT/sink.o" "${COMMON[@]}"
python "$R61/r61_gen.py" w4-native physical qkv-only > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-opus-qkv\nmode=w4-scaled\nm=256\nk=768\nn=1280\ninput=w4-native\noutput=physical\n' > "$OUT/shape.txt"
echo "$OUT"
