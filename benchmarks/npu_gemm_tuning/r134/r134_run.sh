#!/usr/bin/env bash
# R134 -- sweep the R14C-class 8-stream W4A8 GEMM across W BO sizes at DFlash shapes.
# Usage: ./r134_run.sh NBLK [PAD_BYTES] [ITERS] [PASSES] [SYNC]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
NBLK="${1:-128}"; PAD="${2:-0}"; ITERS="${3:-100}"; PASSES="${4:-3}"; SYNC="${5:-1}"; INNER="${6:-0}"
LM=1; LN=4; KT=64; MT=1; NT=4; WBYTES=64; AVAL=3

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done

WB=$(( LN*KT*WBYTES )); CB=$(( LM*LN*32 )); CJ=$(( 4*CB )); K=$(( KT*16 ))
OUT="$HOME/.hipfire/npu/r134_nb${NBLK}_pad${PAD}_in${INNER}"
if [ ! -f "$OUT/final.xclbin" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  "$PEANO/bin/clang++" "$HERE/../r11/r11_gemm.cc" -c -o "$OUT/r11.o" -I"$MA_ROOT/include" \
    -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -O2 -DNDEBUG --target=aie2-none-unknown-elf -DLM="$LM" -DLN="$LN" -DKT="$KT" -DMT="$MT" -DNT="$NT"
  python3 "$HERE/r134_gen.py" "$LM" "$LN" "$KT" "$NBLK" "$WBYTES" npu1 32 2 "$AVAL" 2 4 "$PAD" "$INNER" \
    > "$OUT/aie.mlir" 2> "$OUT/gen.log"
  cat "$OUT/gen.log"
  if ! aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
       --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
       --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >"$OUT/aiecc.log" 2>&1; then
    echo "AIECC FAILED (nblk=$NBLK pad=$PAD):"; grep -iE "error|exceed" "$OUT/aiecc.log" | head -10; exit 1
  fi
else
  cat "$OUT/gen.log"
fi

WREAD=$(( 4*NBLK*WB ))
WSZ=$(( PAD > WREAD ? PAD : WREAD ))
CSZ=$(( 4*NBLK*CJ*4 ))
EXP=$(( AVAL*K ))
(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_bo_probe -- \
   "$OUT" 64 "$WSZ" "$CSZ" "$WREAD" "$ITERS" "$PASSES" "$SYNC" "$EXP")
