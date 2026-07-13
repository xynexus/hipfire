#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R116_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r116_r113_compact_fullk_n16_m256_k768}
ACTIVE_GROUPS=${R116_ACTIVE_GROUPS:-3}
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
"$PEANO/bin/clang++" "$HERE/r116_compact_fullk_n16.cc" -c \
  -o "$OUT/r116.o" "${COMMON[@]}"
python "$ROOT/r115/r115_gen.py" --active-groups="$ACTIVE_GROUPS" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-r113-compact-fullk-n16' \
  'mode=w8-scaled' \
  'm=256' \
  "k=$((256 * ACTIVE_GROUPS))" \
  'n=16' \
  'activation-input=r113-per-core-diagnostic-slots' \
  "activation-groups=$ACTIVE_GROUPS" \
  "activation-physical-bytes=$((196608 * ACTIVE_GROUPS))" \
  "activation-unique-chunk-bytes=$((66560 * ACTIVE_GROUPS))" \
  'accumulation=local-f32' \
  'nmacro-materialized-replicas=0' \
  'weight-layout=offline-rdna2-hfp-records' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
