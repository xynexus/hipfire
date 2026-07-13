#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="${R84_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r84_compact_paired_w4_projection_pack_attention_o_m256_k768_n1280}"
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
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf
)
"$PEANO/bin/clang++" "$HERE/../r70/r70_w4_scaled_group.cc" -c -o "$OUT/r70group.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r65/r65_w4_bf16_finish.cc" -c -o "$OUT/r65finish.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r29/r29_w8_qkv_attention_pack.cc" -c -o "$OUT/r66.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/../r30/r30_w8_qkv_attention.cc" -c -o "$OUT/r71att.o" "${COMMON[@]}" -Oz -DR30_ATTENTION_ONLY
"$PEANO/bin/clang++" "$HERE/../r32/r32_attention_finish_pair_packed.cc" -c -o "$OUT/r32att.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/../r32/r32_output_projection_m8.cc" -c -o "$OUT/r32out.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/../r83/r83_control.cc" -c -o "$OUT/r83control.o" "${COMMON[@]}" -Oz
python "$HERE/../r71/r71_gen.py" --direct-output-projection --enqueue-window3 > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-opus-compact-paired-qkv-projection-pack-attention-output-direct\nmode=w4-scaled\nm=256\nk=768\nn=1280\ncontexts=1\nprojection-columns=1,3,5,7\npack-output-columns=0,2,4,6\nattention-columns=1,3,5,7\nattention-handoff=adjacent-depth3-bf16\nattention-order=0,2,4,1,3,5\nweights=qkv-paired-whole-scaled-plus-o-direct\noutput=token-major-f32\n' > "$OUT/shape.txt"
echo "$OUT"
