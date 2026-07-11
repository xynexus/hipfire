#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAYS="${1:-3}"
[[ "$OVERLAYS" =~ ^[0-9]+$ ]] && (( OVERLAYS >= 1 && OVERLAYS <= 62 )) || {
  echo "usage: $0 [OVERLAYS:1..62]" >&2; exit 2;
}
OUT="$HOME/.hipfire/npu/embgemma_aie2p_token_pack_down_mixed_o${OVERLAYS}_m256_k1152_n768"
rm -rf "$OUT"; mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/r24_mixed_token_down.cc" -c -o "$OUT/r24.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf
python3 "$HERE/../r22/r22_gen.py" mixed "$OVERLAYS" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=token_pack_down\nmode=mixed\noverlays=%s\nm=256\npm=288\nk=1152\npk=1280\nn=768\npn=768\n' "$OVERLAYS" > "$OUT/shape.txt"
echo "$OUT"
