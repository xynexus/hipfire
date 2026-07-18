#!/usr/bin/env bash
# Build the R133 region-spread weight-feed-only probe.
# Usage: build_r133.sh WB NBLK REGIONS SPREAD_MB
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WB="${1:-16384}"; NBLK="${2:-128}"; RG="${3:-8}"; SP="${4:-8}"; PAD="${5:-0}"; SPL="${6:-1}"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
OUT="$HOME/.hipfire/npu/r133_wb${WB}_nb${NBLK}_r${RG}_sp${SP}${PAD:+_pad$PAD}_bo${SPL}"
rm -rf "$OUT"; mkdir -p "$OUT"
python3 "$HERE/r133_gen.py" "$WB" "$NBLK" "$RG" "$SP" npu1 2 4 "$PAD" "$SPL" > "$OUT/aie.mlir" 2>"$OUT/gen.log"
cat "$OUT/gen.log"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT"
