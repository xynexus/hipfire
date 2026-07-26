#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R117_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r117_r113_compact_fullk_n32_m256_k768}
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
"$PEANO/bin/clang++" "$HERE/r117_compact_fullk_n32.cc" -c \
  -o "$OUT/r117.o" "${COMMON[@]}"
python "$ROOT/r115/r115_gen.py" --active-groups=3 --n-slices=2 > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-r113-compact-fullk-n32' \
  'mode=w8-scaled' \
  'm=256' \
  'k=768' \
  'n=32' \
  'activation-input=r113-per-core-diagnostic-slots' \
  'activation-groups=3' \
  'activation-physical-bytes=589824' \
  'activation-unique-chunk-bytes=199680' \
  'accumulation=local-f32-staged' \
  'activation-load-reuse=n32-per-k-tile' \
  'nmacro-materialized-replicas=0' \
  'weight-layout=offline-rdna2-hfp-records' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
