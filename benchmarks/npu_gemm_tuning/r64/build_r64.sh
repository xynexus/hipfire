#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R15="$HERE/../r15"
R61="$HERE/../r61"
OUT="${R64_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r64_trace_compact_qkv_m256_k768_n1280}"
: "${R64_TRACE_START:=0}"
: "${R64_TRACE_COLS:=4}"
: "${R64_TRACE_TILE:=shim}"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
COMMON=(
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf
)
"$PEANO/bin/clang++" "$R15/r15_w4_scaled.cc" -c -o "$OUT/r15.o" "${COMMON[@]}"
python "$R61/r61_gen.py" w4-native physical qkv-compact \
  | sed 's/link_with = "compute.o"/link_with = "r15.o"/' \
  | sed '/r61_weight_sink/d' > "$OUT/base.mlir"
python "$R15/r15_gen.py" w4 3 6 8 > "$OUT/canonical.mlir"
diff -u "$OUT/canonical.mlir" "$OUT/base.mlir"
python "$HERE/r64_inject_trace.py" "$OUT/base.mlir" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
PHYSICAL="$OUT/input_with_addresses.mlir"
if [[ ! -f "$PHYSICAL" ]]; then
  PHYSICAL="$(find "$OUT" -type f -name '*physical*.mlir' -print | head -n 1)"
fi
if [[ -z "$PHYSICAL" ]]; then
  echo "R64 build did not retain physical MLIR" >&2
  exit 1
fi
printf '%s\n' "$PHYSICAL" > "$OUT/trace_physical_mlir.txt"
printf 'op=resident-opus-qkv-trace\nmode=w4-scaled\ncols=8\nm=256\nk=768\nn=1280\nmm=3\nnm=2\nkg=3\noutblocks=6\ninput=w4-native\nweights=qkv-compact-rdna2-hfp\noutput=physical\ntrace_bytes=16777216\ntrace_start=%s\ntrace_cols=%s\ntrace_tile=%s\n' "$R64_TRACE_START" "$R64_TRACE_COLS" "$R64_TRACE_TILE" > "$OUT/shape.txt"
echo "$OUT"
