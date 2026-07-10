#!/usr/bin/env bash
# Build a 4x4 AIE2P whole-array W4A8 cache for one padded full-K matrix.
# M is padded in 96-row macro tiles, N in mode-specific macro tiles, and K must
# already be padded to complete 256-element Opus groups.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${1:?M}"
K="${2:?K padded to 256}"
N="${3:?N}"
MODE="${4:-w4}"
(( M > 0 && K > 0 && N > 0 && K % 256 == 0 ))

case "$MODE" in
  w4)
    LM=6
    LN=6
    KT=16
    WBYTES=128
    CBASE=64
    DEPTH=2
    SOURCE="$HERE/r14_aie2p_gemm.cc"
    SYMBOL=r14_aie2p_gemm
    ;;
  w8)
    LM=3
    LN=3
    KT=32
    WBYTES=128
    CBASE=128
    DEPTH=2
    SOURCE="$HERE/r14_aie2p_gemm_w8.cc"
    SYMBOL=r14_aie2p_gemm_w8
    ;;
  *)
    echo "MODE must be w4 or w8" >&2
    exit 2
    ;;
esac

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"

MM=$(((M + 95) / 96))
MACRO_N=$((4 * LN * 16))
NM=$(((N + MACRO_N - 1) / MACRO_N))
KG=$((K / 256))
NBLK=$((MM * NM * KG))
OUT="$HOME/.hipfire/npu/embgemma_aie2p_whole_${MODE}_m${M}_kg${KG}_n${N}"
rm -rf "$OUT"
mkdir -p "$OUT"

"$PEANO/bin/clang++" "$SOURCE" -c -o "$OUT/r11.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -O2 -DNDEBUG \
  --target=aie2p-none-unknown-elf
python3 "$HERE/r14_gen.py" "$LM" "$LN" "$KT" "$NBLK" "$WBYTES" npu2 "$CBASE" "$DEPTH" \
  | sed "s/@r11_gemm/@${SYMBOL}/g" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'mode=%s\nm=%s\nk=%s\nn=%s\nlm=%s\nln=%s\nmm=%s\nnm=%s\nkg=%s\nnblk=%s\n' \
  "$MODE" "$M" "$K" "$N" "$LM" "$LN" "$MM" "$NM" "$KG" "$NBLK" > "$OUT/shape.txt"
echo "$OUT"
