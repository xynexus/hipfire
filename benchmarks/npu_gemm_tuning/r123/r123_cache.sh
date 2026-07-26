#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH="${1:-1}"
if ! [[ "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-batch-count]" >&2
  exit 2
fi
M=$((256 * BATCH))
OUT="${HIPFIRE_R123_OUT:-$HOME/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16_weight_stationary_m${M}_k768_i1152_o768}"
rm -rf "$OUT"
mkdir -p "$OUT"

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"

"$PEANO/bin/clang++" "$HERE/../r26/r26_w8_resident_ffn.cc" \
  -c -o "$OUT/r26.o" -I"$MA_ROOT/include" -std=c++20 \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
  -Wno-empty-body -Wno-deprecated-declarations -Os -DNDEBUG \
  -DR35_CANONICAL_BF16 -DR123_WEIGHT_STATIONARY \
  --target=aie2p-none-unknown-elf

python "$HERE/../r26/r26_gen.py" --canonical-bf16-input \
  --weight-reuse-across-macros --batch="$BATCH" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT"

printf '%s\n' \
  'op=resident_ffn' \
  'mode=dense-w8-canonical-bf16' \
  'input=token-major-bf16' \
  'weight-reuse=core-weight-stationary' \
  'weight-block-bytes=15552' \
  "m=$M" \
  'k=768' \
  'intermediate=1152' \
  'out=768' > "$OUT/shape.txt"
echo "$OUT"
