#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${R105_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r105_direct_x_unit_rms_bf16_m256_k768}
rm -rf -- "$OUT"
mkdir -p -- "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT=$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"

"$PEANO/bin/clang++" "$HERE/r105_apply_pre_inverse.cc" -c -o "$OUT/r105.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
python "$HERE/r105_gen.py" > "$OUT/aie.mlir"

LOG="$OUT/aiecc.log"
if ! aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >"$LOG" 2>&1; then
  sed -n '1,240p' "$LOG" >&2
  exit 1
fi
test -s "$OUT/final.xclbin"
test -s "$OUT/insts.bin"

max_text=0
for elf in "$OUT"/main_core_*.elf; do
  text=$("$PEANO/bin/llvm-size" "$elf" | tail -1 | awk '{print $1}')
  (( text > max_text )) && max_text=$text
done
(( max_text <= 16384 )) || { echo "core text $max_text exceeds 16384" >&2; exit 1; }

printf '%s\n' \
  'op=embeddinggemma-direct-x-unit-rms' \
  'mode=bf16' \
  'm=256' \
  'k=768' \
  'input=r44-canonical-direct-x' \
  'output=canonical-bf16-unit-rms' \
  'output-bytes=442368' \
  'immutable-pre-ffn-norm=loader-folded' \
  "max-core-text=$max_text" > "$OUT/shape.txt"
echo "$OUT"
