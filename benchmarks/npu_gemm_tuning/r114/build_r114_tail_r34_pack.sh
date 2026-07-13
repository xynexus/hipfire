#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
OUT=${R114_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r114_post_ffn_tail_r34_pack_m256_k768}
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
"$PEANO/bin/clang++" "$HERE/r114_tail_r34_pack.cc" -c \
  -o "$OUT/r114.o" "${COMMON[@]}" -DR111_ALIGN_SAFE_CHUNK_COPY
python "$ROOT/r43/r43_tail_gen.py" --fusion-ready --fused-r34-pack > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=embeddinggemma-post-ffn-tail-r34-compact-pack' \
  'mode=bf16x2-resident' \
  'm=256' \
  'k=768' \
  'input=shared-y-interleaved-bf16x2-and-residual-bf16' \
  'output=shared-completed-bf16x2-and-compact-r34-q-scale-planes' \
  'r34-output-bytes=589824' \
  'r34-prefix-bytes=6240' \
  'r34-materialized-nmacro-replicas=0' \
  'r34-compact-memory-tiles=2,3,6,7' \
  'r34-shim-route=reuse-completed-output' \
  'token-owner-order=contiguous-eight-per-core' \
  'token-owner-routing=adjacent-triomino-dma' \
  'chunk-assembly=neighbor-memory-objectfifo' \
  'x-route=joined-row-suffix' \
  'completed-input-passes-for-pack=0' \
  'immutable-tensor-reorder=none' > "$OUT/shape.txt"
echo "$OUT"
