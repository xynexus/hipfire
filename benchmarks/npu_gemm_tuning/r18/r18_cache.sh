#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != w4 && "$1" != w8 ) ]]; then
  echo "usage: $0 w4|w8" >&2
  exit 2
fi
MODE=$1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_gateup_geglu_${MODE}_m256_k768_i1152"
rm -rf "$OUT"; mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/r18_${MODE}_fused.cc" -c -o "$OUT/r18.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -O2 -DNDEBUG --target=aie2p-none-unknown-elf
if [[ $MODE == w4 ]]; then OUTBLOCKS=9; else OUTBLOCKS=18; fi
if [[ $MODE == w4 ]]; then PADDED_N=1152; else PADDED_N=1536; fi
python3 "$HERE/r18_gen.py" "$MODE" 3 "$OUTBLOCKS" 8 256 1152 > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=gateup_geglu\nmode=%s\nm=256\nk=768\nintermediate=1152\npadded_n=%s\n' \
  "$MODE" "$PADDED_N" > "$OUT/shape.txt"
echo "$OUT"
