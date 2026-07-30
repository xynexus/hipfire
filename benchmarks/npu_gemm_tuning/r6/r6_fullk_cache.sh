#!/usr/bin/env bash
# Build one exact, one-dispatch full-K AIE2P cache under ~/.hipfire/npu.
#
# Usage: r6_fullk_cache.sh MODE M K N [COLS]
#   MODE = w4 | w4-scaled | mixed | w8
#   M defaults to the EmbeddingGemma target batch geometry; K must be padded to
#   a multiple of 256 and N to a multiple of 64. Mixed runs W4 base plus an
#   arbitrary dense W8 residual in the same runtime sequence.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?MODE w4|mixed|w8}"
M="${2:?M}"
K="${3:?K padded to 256}"
N="${4:?N}"
COLS="${5:-8}"
[[ "$MODE" =~ ^(w4|w4-scaled|w4-np-scaled|mixed|w8)$ ]] || { echo "MODE must be w4, w4-scaled, mixed, or w8" >&2; exit 2; }
(( M > 0 && K > 0 && N > 0 && COLS > 0 ))
if [[ "$MODE" != w4-np-scaled ]]; then (( M % (COLS * 8) == 0 )) || { echo "M must be divisible by COLS*8" >&2; exit 2; }; fi
(( K % 256 == 0 )) || { echo "K must be padded to a multiple of 256" >&2; exit 2; }
(( N % 64 == 0 )) || { echo "N must be divisible by 64" >&2; exit 2; }

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"

KGROUPS=$((K / 256))
NB=$((N / 64))
AW=$(((M / COLS) * 256))
CW=$(((M / COLS) * 64))
# Only the N-parallel mode encodes COLS. The M-parallel names are load-bearing
# (npu_linear.rs and npu_linear_oq4 resolve them), and their omission of COLS is
# a known trap: a COLS=4 build silently overwrites the COLS=8 one.
if [[ "$MODE" == w4-np-scaled ]]; then
  OUT="$HOME/.hipfire/npu/embgemma_aie2p_fullk_submit_${MODE}_m${M}_kg${KGROUPS}_n${N}_c${COLS}"
else
  OUT="$HOME/.hipfire/npu/embgemma_aie2p_fullk_submit_${MODE}_m${M}_kg${KGROUPS}_n${N}"
fi
rm -rf "$OUT"
mkdir -p "$OUT"

compile() {
  local source="$1" object="$2" mt="$3" kchunk="$4"
  shift 4
  "$PEANO/bin/clang++" "$source" -c -o "$object" -I"$MA_ROOT/include" \
    -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
    -Wno-empty-body -O2 -DNDEBUG --target=aie2p-none-unknown-elf \
    -DMT="$mt" -DNT=4 -DKCHUNK="$kchunk" "$@"
}

case "$MODE" in
  w4)
    compile "$HERE/r6_gemm_ts.cc" "$OUT/r6_mac.o" "$((M / COLS / 4))" 16
    python3 "$HERE/r6_gen_mp_fullk.py" "$COLS" "$NB" "$AW" 8192 "$CW" \
      "$KGROUPS" "$OUT/r6_mac.o" > "$OUT/aie.mlir"
    ;;
  w4-np-scaled)
    compile "$HERE/r6_gemm_ts.cc" "$OUT/r6_mac.o" "$((M / 4))" 16
    compile "$HERE/r6_scale_accum.cc" "$OUT/r6_scale.o" "$((M / 4))" 16
    python3 "$HERE/r6_gen_np_fullk_scaled.py" "$COLS" "$NB" "$((M * 256))" 8192 "$((M * 64))" \
      "$KGROUPS" "$OUT/r6_mac.o" "$OUT/r6_scale.o" > "$OUT/aie.mlir"
    ;;
  w4-scaled)
    compile "$HERE/r6_gemm_ts.cc" "$OUT/r6_mac.o" "$((M / COLS / 4))" 16
    compile "$HERE/r6_scale_accum.cc" "$OUT/r6_scale.o" "$((M / COLS / 4))" 16
    python3 "$HERE/r6_gen_mp_fullk_scaled.py" "$COLS" "$NB" "$AW" 8192 "$CW" \
      "$KGROUPS" "$OUT/r6_mac.o" "$OUT/r6_scale.o" > "$OUT/aie.mlir"
    ;;
  w8)
    compile "$HERE/r6_gemm_ts_w8m8_row.cc" "$OUT/r6_mac.o" "$((M / COLS / 8))" 32 \
      -DR6_W8_ENTRY=r6_mac
    python3 "$HERE/r6_gen_mp_fullk.py" "$COLS" "$NB" "$AW" 16384 "$CW" \
      "$KGROUPS" "$OUT/r6_mac.o" > "$OUT/aie.mlir"
    ;;
  mixed)
    compile "$HERE/r6_gemm_ts.cc" "$OUT/r6_base.o" "$((M / COLS / 4))" 16
    compile "$HERE/r6_gemm_ts_w8m8_row.cc" "$OUT/r6_residual.o" "$((M / COLS / 8))" 32 \
      -DR6_W8_ACCUMULATE=1
    python3 "$HERE/r6_gen_mp_fullk_mixed.py" "$COLS" "$NB" "$AW" "$CW" \
      "$KGROUPS" "$OUT/r6_base.o" "$OUT/r6_residual.o" > "$OUT/aie.mlir"
    ;;
esac

aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
if [[ "$MODE" == "w4-scaled" || "$MODE" == "w4-np-scaled" ]]; then
  echo scaled-f32-direct > "$OUT/output-layout.txt"
  # N-parallel always combines activations+scales into one stream per column
  # (shim budget: @fx and @fw in, @fc out).
  if (( COLS == 8 )) || [[ "$MODE" == "w4-np-scaled" ]]; then
    echo combined > "$OUT/input-layout.txt"
  else
    echo separate > "$OUT/input-layout.txt"
  fi
elif (( KGROUPS <= 5 )); then
  echo direct > "$OUT/output-layout.txt"
else
  echo physical > "$OUT/output-layout.txt"
fi
echo "$OUT"
