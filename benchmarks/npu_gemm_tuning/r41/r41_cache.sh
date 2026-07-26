#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16x2_m256_k768_i1152_o768"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
read -r -a CXX_FLAGS <<< "${HIPFIRE_R41_CXX_FLAGS:--Os}"
"$PEANO/bin/clang++" "$HERE/../r26/r26_w8_resident_ffn.cc" -c -o "$OUT/r41.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  "${CXX_FLAGS[@]}" -DNDEBUG -DR35_CANONICAL_BF16 \
  -DR41_CANONICAL_BF16X2_OUTPUT --target=aie2p-none-unknown-elf
python "$HERE/../r26/r26_gen.py" --canonical-bf16-input --canonical-bf16x2-output \
  > "$OUT/aie.mlir"
sed -i 's/link_with = "r26.o"/link_with = "r41.o"/g' "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=resident_ffn' \
  'mode=dense-w8-canonical-bf16-bf16x2-output' \
  'm=256' \
  'k=768' \
  'intermediate=1152' \
  'out=768' \
  'input=token-major-bf16' \
  'output=token-major-bf16x2' > "$OUT/shape.txt"
echo "$OUT"
