#!/usr/bin/env bash
# Build + run one R135 config. Usage: r135_run.sh WB N VMODE HMODE [ITERS] [PASSES] [BURST]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WB="${1:-16384}"; N="${2:-128}"; VMODE="${3:-b}"; HMODE="${4:-off}"
ITERS="${5:-100}"; PASSES="${6:-3}"; BURST="${7:-64}"

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done

OUT="$HOME/.hipfire/npu/r135_wb${WB}_n${N}_v${VMODE}_h${HMODE}_b${BURST}"
if [ ! -f "$OUT/final.xclbin" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  python3 "$HERE/r135_gen.py" "$WB" "$N" "$VMODE" "$HMODE" npu1 2 "$BURST" \
    > "$OUT/aie.mlir" 2> "$OUT/acct.log"
  if ! aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
       --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
       --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >"$OUT/aiecc.log" 2>&1; then
    echo "AIECC FAILED (v=$VMODE h=$HMODE N=$N):"; grep -iE "error|exceed" "$OUT/aiecc.log" | head -20; exit 1
  fi
fi
cat "$OUT/acct.log"
TOT=$(sed -n 's/.*ddr_read=\([0-9]*\).*/\1/p' "$OUT/acct.log")
(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_bo_probe -- \
   "$OUT" 64 "$TOT" 256 "$TOT" "$ITERS" "$PASSES" 1)
