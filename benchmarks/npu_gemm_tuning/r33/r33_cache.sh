#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_w8_qkv_paired_attention_o_direct_m256_k768_n1280"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
COMMON=(
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations
  -Os -DNDEBUG --target=aie2p-none-unknown-elf
)
"$PEANO/bin/clang++" "$HERE/../r30/r30_w8_qkv_attention.cc" -c -o "$OUT/r30.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r32/r32_attention_finish_pair_packed.cc" -c -o "$OUT/r32att.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r32/r32_output_projection_m8.cc" -c -o "$OUT/r32out.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/r33_paired_qkv.cc" -c -o "$OUT/r33pair.o" "${COMMON[@]}"
python "$HERE/../r29/r29_gen.py" --paired-qkv > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-qkv-paired-attention-output-direct\nmode=w8-scaled\nm=256\nk=768\nn=1280\nroles=q0,q1,q2,k,v,o\noutput=token-major-f32\n' > "$OUT/shape.txt"
echo "$OUT"
