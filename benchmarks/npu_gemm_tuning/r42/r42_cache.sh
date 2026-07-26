#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 -ne 39 && $1 -ne 731 ) ]]; then
  echo "usage: $0 39|731" >&2
  exit 2
fi

EXCEPTION_COLUMN=$1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_exception_c${EXCEPTION_COLUMN}_m256_k768_n1280"
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
"$PEANO/bin/clang++" "$HERE/../r33/r33_paired_qkv.cc" -c -o "$OUT/r33pair.o" "${COMMON[@]}"
R42_COMMON=(
  "${COMMON[@]}" -O1 -DR42_EXCEPTION_COLUMN="$EXCEPTION_COLUMN"
  -DR42_SPLIT_OBJECTS -mllvm -aie-premisched-near-critical-regs=0
  -mllvm -aie-bottomup-cycles=0
)
"$PEANO/bin/clang++" "$HERE/../r34/r34_residual_norm.cc" -c -o "$OUT/r42even.o" \
  "${R42_COMMON[@]}" -DR42_BUILD_OUTPUT -DR42_BUILD_POST -DR42_BUILD_EMIT
"$PEANO/bin/clang++" "$HERE/../r34/r34_residual_norm.cc" -c -o "$OUT/r42relay.o" \
  "${R42_COMMON[@]}" -DR42_BUILD_RELAY
python "$HERE/../r29/r29_gen.py" --residual-norm > "$OUT/aie.mlir"
sed -i '/@r34_output_projection_finish_pair_bf16/ s/r34norm.o/r42even.o/' "$OUT/aie.mlir"
sed -i '/@r34_post_residual_pre_ffn/ s/r34norm.o/r42even.o/' "$OUT/aie.mlir"
sed -i '/@r34_emit_norm_half/ s/r34norm.o/r42even.o/' "$OUT/aie.mlir"
sed -i '/@r38_relay_pre_inverse/ s/r34norm.o/r42relay.o/' "$OUT/aie.mlir"
sed -i 's/memref<64xi8>/memref<128xi8>/g' "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf '%s\n' \
  'op=resident-qkv-paired-attention-output-norm' \
  'mode=w8-scaled' \
  'm=256' \
  'k=768' \
  'n=1280' \
  'roles=q0,q1,q2,k,v,o' \
  'tails=post-attn-norm,residual,pre-ffn-norm' \
  'output=canonical-token-major-bf16' \
  'handoff=staging-prefix-dmabuf' \
  'state=pre-ffn-inverse-f32-x-bf16' \
  "exception-column=$EXCEPTION_COLUMN" \
  'state-layout=core-row,wave-active-column,row' \
  'state-record-bytes=12288' > "$OUT/shape.txt"
echo "$OUT"
