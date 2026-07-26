#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${1:?M}" K="${2:?K}" N="${3:?N}" MODE="${4:-w4}" COLS="${5:-4}"
(( M > 0 && K > 0 && N > 0 && K % 256 == 0 ))
(( COLS == 4 || COLS == 8 ))
case "$MODE" in
  w4) COLS_STRIPE=96; SOURCE="$HERE/r15_w4_scaled.cc" ;;
  w8) COLS_STRIPE=48; SOURCE="$HERE/r15_w8_scaled.cc" ;;
  *) echo "MODE must be w4 or w8" >&2; exit 2 ;;
esac
MACRO_N=$((COLS * COLS_STRIPE))
MM=$(((M + 95) / 96)); NM=$(((N + MACRO_N - 1) / MACRO_N)); KG=$((K / 256)); OB=$((MM * NM))
TAG="whole"; (( COLS == 8 )) && TAG="whole8"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_${TAG}_${MODE}-scaled_m${M}_kg${KG}_n${N}"
rm -rf "$OUT"; mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$SOURCE" -c -o "$OUT/r15.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -O2 -DNDEBUG --target=aie2p-none-unknown-elf
python3 "$HERE/r15_gen.py" "$MODE" "$KG" "$OB" "$COLS" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'mode=%s-scaled\ncols=%s\nm=%s\nk=%s\nn=%s\nmm=%s\nnm=%s\nkg=%s\noutblocks=%s\n' \
  "$MODE" "$COLS" "$M" "$K" "$N" "$MM" "$NM" "$KG" "$OB" > "$OUT/shape.txt"
echo "$OUT"
