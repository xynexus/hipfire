#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
BATCH="${1:-1}"
if ! [[ "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-batch-count]" >&2
  exit 2
fi
M=$((256 * BATCH))
COMPLETED_BYTES=$((884736 * BATCH))
ACTIVE_BYTES=$((786432 * BATCH))
OUT=${R111_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r111_next_layer_prep_w8_bf16x2_one_pass_m${M}_k768}
rm -rf -- "$OUT"
mkdir -p -- "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$ROOT/r47/r47_next_layer_prep.cc" -c -o "$OUT/r47.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  -DR111_ALIGN_SAFE_CHUNK_COPY \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
HIPFIRE_R47_OUTPUT_BASE="$COMPLETED_BYTES" HIPFIRE_R47_IN_PLACE=1 \
HIPFIRE_R47_ONE_PASS_COMPLETED=1 \
  python "$ROOT/r47/r47_gen.py" --batch="$BATCH" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-next-layer-prep' \
  'mode=w8-scaled' \
  "m=$M" \
  'k=768' \
  'input=shared-completed-bf16x2' \
  'output=shared-r34-activation-prefix' \
  "output-prefix-offset=$COMPLETED_BYTES" \
  'buffer-mode=in-place-disjoint-prefix-suffix' \
  'completed-input-passes=1' \
  "completed-allocation-bytes=$COMPLETED_BYTES" \
  "completed-active-bytes-per-pass=$ACTIVE_BYTES" \
  'prefix-bytes=6240' > "$OUT/shape.txt"
echo "$OUT"
