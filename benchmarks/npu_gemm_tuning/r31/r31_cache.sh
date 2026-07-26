#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX=""
GEN_ARGS=(--attention --output-projection)
if [[ "${HIPFIRE_R31_NO_OUTPUT_EXECUTION:-0}" == 1 ]]; then
  SUFFIX="_no_output_execution"
  GEN_ARGS+=(--no-output-execution)
fi
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_w8_qkv_attention_o_m256_k768_n1280${SUFFIX}"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/../r30/r30_w8_qkv_attention.cc" -c -o "$OUT/r30.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -Os -DNDEBUG --target=aie2p-none-unknown-elf
"$PEANO/bin/clang++" "$HERE/r31_w8_qkv_attention_o.cc" -c -o "$OUT/r31.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  "${HIPFIRE_R31_CXX_OPT:--Os}" -DNDEBUG --target=aie2p-none-unknown-elf
python "$HERE/../r29/r29_gen.py" "${GEN_ARGS[@]}" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-qkv-attention-o\nmode=w8-qkv-bf16-o\nm=256\nk=768\nn=1280\nroles=q0,q1,q2,k,v,o\noutput=token-major-bf16\n' > "$OUT/shape.txt"
echo "$OUT"
