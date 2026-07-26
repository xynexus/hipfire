#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R113_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r113_post_ffn_tail_next_pack_m256_k768}
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
"$PEANO/bin/clang++" "$ROOT/r43/r43_post_ffn_direct_tail_bf16x2.cc" -c \
  -o "$OUT/r43_tail.o" "${COMMON[@]}" -DR100_INTERLEAVED_FFN
"$PEANO/bin/clang++" "$HERE/r113_tail_pack.cc" -c \
  -o "$OUT/r113.o" "${COMMON[@]}" -DR111_ALIGN_SAFE_CHUNK_COPY
python "$ROOT/r43/r43_tail_gen.py" --fusion-ready --fused-next-pack > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-post-ffn-tail-next-pack' \
  'mode=bf16x2-resident' \
  'm=256' \
  'k=768' \
  'input=shared-y-interleaved-bf16x2-and-residual-bf16' \
  'output=shared-completed-bf16x2-and-diagnostic-next-pack' \
  'token-owner-order=contiguous-eight-per-core' \
  'x-route=joined-row-suffix' \
  'completed-input-passes-for-pack=0' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
