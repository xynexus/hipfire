#!/usr/bin/env bash
# Build the R133 dual-topology weight-feed probe. Usage: build_r133.sh WB NBLK TOPO BURST
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WB="${1:-16384}"; NBLK="${2:-128}"; TOPO="${3:-1}"; BURST="${4:-64}"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
OUT="$HOME/.hipfire/npu/r133_wb${WB}_nb${NBLK}_t${TOPO}_b${BURST}"
rm -rf "$OUT"; mkdir -p "$OUT"
python3 "$HERE/r133_gen.py" "$WB" "$NBLK" "$TOPO" npu1 2 "$BURST" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
NB=$([ "$TOPO" = 1 ] && echo "$NBLK" || echo $((NBLK/2)))
TOT=$([ "$TOPO" = 1 ] && echo $((4*NB*WB)) || echo $((8*NB*WB)))
echo "cached: $OUT  (TOPO=$TOPO, W total = $TOT bytes)"
