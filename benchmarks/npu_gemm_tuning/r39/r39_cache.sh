#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_post_ffn_tail_m256_k768"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
read -r -a CXX_FLAGS <<< "${HIPFIRE_R39_CXX_FLAGS:--Os}"
"$PEANO/bin/clang++" "$HERE/r39_post_ffn_tail.cc" -c -o "$OUT/r39.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  "${CXX_FLAGS[@]}" -DNDEBUG --target=aie2p-none-unknown-elf
python "$HERE/r39_gen.py" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=embeddinggemma-post-ffn-tail\nmode=bf16-resident\nm=256\nk=768\ninput=shared-h-and-y\noutput=shared-y-overwrite\nstate=pre-ffn-inverse-f32\n' > "$OUT/shape.txt"
echo "$OUT"
