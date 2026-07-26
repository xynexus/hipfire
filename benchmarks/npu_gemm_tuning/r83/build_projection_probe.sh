#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-both}
case "$MODE" in
  single-group) FLAGS=(--single-group-function) ;;
  dynamic-slices) FLAGS=(--dynamic-slice-loop) ;;
  both) FLAGS=(--single-group-function --dynamic-slice-loop) ;;
  *) echo "usage: $0 {single-group|dynamic-slices|both}" >&2; exit 2 ;;
esac
OUT="${R83_PROBE_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r83_${MODE}_projection_pack_probe_m256_k768_n1280}"
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
"$PEANO/bin/clang++" "$HERE/../r15/r15_w4_scaled.cc" -c -o "$OUT/r15.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r70/r70_w4_scaled_group.cc" -c -o "$OUT/r70group.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r65/r65_w4_bf16_finish.cc" -c -o "$OUT/r65finish.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r29/r29_w8_qkv_attention_pack.cc" -c -o "$OUT/r66.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/r83_control.cc" -c -o "$OUT/r83control.o" "${COMMON[@]}" -Oz
python "$HERE/../r81/r81_gen.py" "${FLAGS[@]}" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=r83-projection-pack-isolation\nmode=%s\n' "$MODE" > "$OUT/shape.txt"
echo "$OUT"
