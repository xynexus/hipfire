#!/usr/bin/env bash
# Build the R132 weight-feed-only probe. Usage: build_r132.sh WB NBLK STREAMS
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WB="${1:-16384}"; NBLK="${2:-128}"; ST="${3:-1}"; NC="${4:-4}"; BURST="${5:-0}"; SPLIT="${6:-0}"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
OUT="$HOME/.hipfire/npu/r132_wb${WB}_nb${NBLK}_s${ST}_c${NC}_b${BURST}_sp${SPLIT}"
rm -rf "$OUT"; mkdir -p "$OUT"
python3 "$HERE/r132_gen.py" "$WB" "$NBLK" "$ST" npu1 2 "$NC" "$BURST" "$SPLIT" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT  (W total = $((4*NBLK*WB)) bytes, $ST stream(s)/column)"
