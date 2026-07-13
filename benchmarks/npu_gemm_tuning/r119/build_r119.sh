#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R119_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r119_r113_staged_fullk_n64_repeat_output_m256_k768}
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
python "$ROOT/r118/r118_gen.py" --repeat-output-task > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-r113-staged-fullk-n64-repeat-output' \
  'mode=w8-scaled' \
  'm=256' \
  'k=768' \
  'n=64' \
  'activation-input=r113-per-core-diagnostic-slots' \
  'activation-groups=3' \
  'activation-stage-bytes-per-core=6336' \
  'activation-stage-group-stride=2112' \
  'activation-stage-group-alignment=64' \
  'activation-physical-bytes=589824' \
  'activation-unique-chunk-bytes=199680' \
  'activation-dma-passes=1' \
  'n32-output-blocks=2' \
  'output-task-repeat-count=1' \
  'output-bd-outer-dimension=2' \
  'accumulation=local-f32-staged' \
  'nmacro-materialized-replicas=0' \
  'weight-layout=offline-rdna2-hfp-records' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
