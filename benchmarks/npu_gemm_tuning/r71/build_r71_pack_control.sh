#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${R71_PACK_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r71_pack_redistributed_control_m256_k768_n1280}"
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
"$PEANO/bin/clang++" "$HERE/../r70/r70_w4_scaled_group.cc" -c -o "$OUT/r70group.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r65/r65_w4_bf16_finish.cc" -c -o "$OUT/r65finish.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r29/r29_w8_qkv_attention_pack.cc" -c -o "$OUT/r66.o" "${COMMON[@]}" -Oz
python "$HERE/../r70/r70_gen.py" --r71-pack-free-4-7 > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-opus-qkv-projection-headnorm-rope-pack\nmode=w4-scaled\nm=256\nk=768\nn=1280\ncontexts=1\npack_owners=columns-0-3\n' > "$OUT/shape.txt"
echo "$OUT"
