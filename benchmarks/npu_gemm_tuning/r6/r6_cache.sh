#!/usr/bin/env bash
# Offline: build + cache an R6 W4A8 GEMM xclbin for a tile config (MT NT KCHUNK), for
# the runtime to load via NpuGemm. The xclbin depends only on the TILE config, not the
# GEMM (K,N) — NpuGemm tiles arbitrary shapes over the fixed block — so a small set of
# configs covers a whole model. Python/aiecc stays OFFLINE here (AGENTS.md: no Python
# in the hot path; the inference binary only loads the cached bytes).
#
# Produces ~/.hipfire/npu/r6_<MT>x<NT>x<KCHUNK>/{final.xclbin,insts.bin}, the layout
# NpuGemm::load expects (single-core, one (MT*4)x(NT*16)x(KCHUNK*16) block/dispatch =
# r6_gen.py COLS=1 NB=1).
#
# Usage: r6_cache.sh [MT] [NT] [KCHUNK] [COLS] [NB]   (defaults 16 4 16 1 1)
# Set R6_W8=1 for the W8A8 tensor-stream kernel. Set R6_W8_M8=1 for the
# dense 8x8x8 W8 kernel. Set R6_SPARSE3=1 for the OQ4.25 three-outlier
# correction kernel. Cache dirs get `_w8`, `_m8k8_w8`, or `_sparse3` suffixes.
# COLS>1 / NB>1 build the streaming ARRAY (COLS cores × NB blocks = COLS*NB N-slabs
# per dispatch = NpuGemm `groups`); COLS=1 NB=1 is the single-core one-slab form.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT="${1:-16}"; NT="${2:-4}"; KCHUNK="${3:-16}"; COLS="${4:-1}"; NB="${5:-1}"
{ [ "$NT" = 4 ] || [ "$NT" = 8 ]; } || { echo "NT must be 4 (r6_gemm*.cc) or 8 (r6_gemm_ts8.cc)"; exit 1; }

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"

# Kernel source: default r6_gemm.cc (pre-tiled load_v/store_v). Set R6_KERNEL_SRC to
# r6_gemm_ts.cc for the tensor-buffer-stream variant (row-major A/C, no CPU marshaling);
# R6_OUT_TAG names its cache dir (default r6). Buffer sizes are identical either way, so
# the MLIR is unchanged.
# Kernel source: default r6_gemm.cc; set R6_KERNEL_SRC=r6_gemm_ts.cc for the tensor-stream
# variant. R6_GEN picks the array generator: r6_gen.py (N-parallel, default) or r6_gen_mp.py
# (M-parallel W-broadcast: COLS distinct M-blocks share one broadcast W). R6_OUT_TAG names
# the cache dir. Buffer sizes are identical, so the kernel object is shared.
W8="${R6_W8:-0}"
W8_M8="${R6_W8_M8:-0}"
SPARSE3="${R6_SPARSE3:-0}"
if [ "$W8_M8" = 1 ]; then W8=1; fi
if [ "$SPARSE3" = 1 ]; then
  [ "$NT" = 4 ] && [ "$KCHUNK" = 16 ] || { echo "R6_SPARSE3=1 requires NT=4 KCHUNK=16"; exit 1; }
  SRC="${R6_KERNEL_SRC:-$HERE/r6_sparse3_ts.cc}"
  WTAG="_sparse3"
  WTXT="OQ4.25-sparse3"
  MROWS=4
  KSTEP=16
  ATILE_BYTES=64
  WTILE_BYTES=6
  CTILE_ELEMS=64
  W_FIFO_DEPTH=2
elif [ "$W8" = 1 ]; then
  [ "$NT" = 4 ] || { echo "R6_W8=1 currently requires NT=4"; exit 1; }
  if [ "$W8_M8" = 1 ]; then
    SRC="${R6_KERNEL_SRC:-$HERE/r6_gemm_ts_w8m8.cc}"
    WTAG="_m8k8_w8"
    WTXT="W8A8-m8k8"
    MROWS=8
    KSTEP=8
    ATILE_BYTES=64
    WTILE_BYTES=128
    CTILE_ELEMS=128
    W_FIFO_DEPTH=1
  else
    [ "$KCHUNK" = 1 ] || { echo "R6_W8=1 currently requires KCHUNK=1 unless R6_W8_M8=1"; exit 1; }
    SRC="${R6_KERNEL_SRC:-$HERE/r6_gemm_ts_w8.cc}"
    WTAG="_w8"
    WTXT="W8A8"
    MROWS=4
    KSTEP=16
    ATILE_BYTES=64
    WTILE_BYTES=256
    CTILE_ELEMS=64
    W_FIFO_DEPTH=2
  fi
else
  SRC="${R6_KERNEL_SRC:-$HERE/r6_gemm.cc}"
  WTAG=""
  WTXT="W4A8"
  MROWS=4
  KSTEP=16
  ATILE_BYTES=64
  WTILE_BYTES=128
  CTILE_ELEMS=64
  W_FIFO_DEPTH=2
fi
TAG="${R6_OUT_TAG:-r6}"; GEN="${R6_GEN:-r6_gen.py}"
ROUNDS="${R6_ROUNDS:-1}"  # r6_gen_mp.py only: M-blocks/core streamed in ONE dispatch
RSFX=""; [ "$ROUNDS" != 1 ] && RSFX="_r${ROUNDS}"
OUT="$HOME/.hipfire/npu/${TAG}_${MT}x${NT}x${KCHUNK}_c${COLS}_nb${NB}${WTAG}${RSFX}"
rm -rf "$OUT"; mkdir -p "$OUT"

# Tile buffer sizes: A = MT*KCHUNK tiles, W = NT*KCHUNK tiles,
# C = MT*NT tiles (MR*MN=64 int32). groups = COLS*NB N-slabs/dispatch.
AW=$((MT * KCHUNK * ATILE_BYTES)); WW=$((NT * KCHUNK * WTILE_BYTES)); CW=$((MT * NT * CTILE_ELEMS))
"$PEANO/bin/clang++" "$SRC" -c -o "$OUT/r6_mac.o" -I"$MA_ROOT/include" \
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf -DMT="$MT" -DNT="$NT" -DKCHUNK="$KCHUNK"
python3 "$HERE/$GEN" "$COLS" "$NB" "$AW" "$WW" "$CW" "$ROUNDS" "$W_FIFO_DEPTH" > "$OUT/aie.mlir"

aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT/final.xclbin  $OUT/insts.bin  ($WTXT block ${MT}x${MROWS} x ${NT}x16 x ${KCHUNK}x${KSTEP} => M=$((MT*MROWS)) N=$((NT*16)) K=$((KCHUNK*KSTEP)))"
