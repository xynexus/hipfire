#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
BATCH=${1:-2}
if ! [[ "$BATCH" =~ ^(2|4)$ ]]; then
  echo "usage: $0 [2|4]" >&2
  exit 2
fi
M=$((256 * BATCH))
N_BLOCKS=40
OUTPUT_OBJECTS=$((BATCH * N_BLOCKS))
OUTPUT_TASKS=1
ACTIVATION_BYTES=$((BATCH * 589824))
ACTIVATION_UNIQUE_BYTES=$((BATCH * 199680))
STAGE_BYTES=$((BATCH * 6336))
OUT=${R129_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r129_r121_staged_fullk_weight_reuse_m${M}_k768_n1280}
rm -rf -- "$OUT"
mkdir -p -- "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
COMMON=(
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
)
"$PEANO/bin/clang++" "$ROOT/r118/r118_stage_compact_fullk_n64.cc" -c \
  -o "$OUT/r118.o" "${COMMON[@]}"
python "$ROOT/r118/r118_gen.py" --repeat-output-task --n-blocks="$N_BLOCKS" \
  --batch="$BATCH" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-r129-staged-fullk-weight-reuse-n1280' \
  'mode=w8-scaled' \
  "m=$M" \
  "batch=$BATCH" \
  'segment-rows=256' \
  'k=768' \
  'n=1280' \
  'activation-input=r113-per-core-diagnostic-slots' \
  'activation-groups=3' \
  'activation-stage-bytes-per-core=6336' \
  "activation-stage-documents=$BATCH" \
  "activation-stage-total-bytes-per-core=$STAGE_BYTES" \
  'activation-stage-group-stride=2112' \
  'activation-stage-group-alignment=64' \
  "activation-physical-bytes=$ACTIVATION_BYTES" \
  "activation-unique-chunk-bytes=$ACTIVATION_UNIQUE_BYTES" \
  'activation-dma-passes=1' \
  'n32-output-blocks=40' \
  "logical-output-objects-per-stream=$OUTPUT_OBJECTS" \
  "output-objects-per-stream=$N_BLOCKS" \
  "output-documents-per-object=$BATCH" \
  'output-task-max-objects=64' \
  "output-task-count-per-stream=$OUTPUT_TASKS" \
  'output-layout=stream-nblock-core-document-row-n32' \
  'accumulation=local-f32-staged' \
  'nmacro-materialized-replicas=0' \
  'weight-layout=offline-rdna2-hfp-records' \
  'weight-bytes=7987200' \
  'weight-records-per-column=120' \
  'weight-dma-passes=1' \
  'weight-batch-replicas=0' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
