#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="${R85_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r85_reuse_a_paired_w4_projection_pack_attention_o_m256_k768_n1280}"
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
OUTPUT_SOURCE="${R85_OUTPUT_SOURCE:-$HERE/r85_output_projection_m8_reuse_a.cc}"
GENERATOR_ARGS=(--direct-output-projection --enqueue-window3)
if [[ -n "${R85_GENERATOR_FLAG:-}" ]]; then
  GENERATOR_ARGS+=("$R85_GENERATOR_FLAG")
fi
"$PEANO/bin/clang++" "$HERE/../r70/r70_w4_scaled_group.cc" -c -o "$OUT/r70group.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r65/r65_w4_bf16_finish.cc" -c -o "$OUT/r65finish.o" "${COMMON[@]}"
"$PEANO/bin/clang++" "$HERE/../r29/r29_w8_qkv_attention_pack.cc" -c -o "$OUT/r66.o" "${COMMON[@]}" "${R85_PACK_OPT:--Oz}"
"$PEANO/bin/clang++" "$HERE/../r30/r30_w8_qkv_attention.cc" -c -o "$OUT/r71att.o" "${COMMON[@]}" -Oz -DR30_ATTENTION_ONLY
"$PEANO/bin/clang++" "$HERE/../r32/r32_attention_finish_pair_packed.cc" -c -o "$OUT/r32att.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$OUTPUT_SOURCE" -c -o "$OUT/r32out.o" "${COMMON[@]}" -Oz
if [[ -n "${R85_EXTRA_SOURCE:-}" ]]; then
  "$PEANO/bin/clang++" "$R85_EXTRA_SOURCE" -c -o "$OUT/r90norm.o" "${COMMON[@]}" "${R85_EXTRA_OPT:--Oz}"
fi
"$PEANO/bin/clang++" "$HERE/../r83/r83_control.cc" -c -o "$OUT/r83control.o" "${COMMON[@]}" -Oz
python "$HERE/../r71/r71_gen.py" "${GENERATOR_ARGS[@]}" > "$OUT/aie.mlir"
AIECC_OPT=()
if [[ -n "${R85_AIECC_OPT:-}" ]]; then
  AIECC_OPT=(-O "$R85_AIECC_OPT")
fi
if ! aiecc "$OUT/aie.mlir" "${AIECC_OPT[@]}" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >"$OUT/aiecc.log" 2>&1; then
  cat "$OUT/aiecc.log" >&2
  exit 1
fi
for artifact in "$OUT/final.xclbin" "$OUT/insts.bin"; do
  if [[ ! -s "$artifact" ]]; then
    tail -80 "$OUT/aiecc.log" >&2
    echo "missing or empty AIE artifact: $artifact" >&2
    exit 1
  fi
done
program_limit=16384
program_max=0
: > "$OUT/core-text-sizes.txt"
for elf in "$OUT"/main_core_*.elf; do
  text_size=$("$PEANO/bin/llvm-size" -A "$elf" | awk '$1 == ".text" {print $2}')
  printf '%s %s\n' "$(basename "$elf")" "$text_size" >> "$OUT/core-text-sizes.txt"
  if (( text_size > program_max )); then
    program_max=$text_size
  fi
done
sort -k2,2nr -o "$OUT/core-text-sizes.txt" "$OUT/core-text-sizes.txt"
if (( program_max > program_limit )); then
  head -20 "$OUT/core-text-sizes.txt" >&2
  echo "AIE core program text exceeds ${program_limit} bytes: ${program_max}" >&2
  exit 1
fi
printf 'op=resident-opus-compact-paired-qkv-projection-pack-attention-output-direct-reuse-a\nmode=w4-scaled\nm=256\nk=768\nn=1280\ncontexts=1\nprojection-columns=1,3,5,7\npack-output-columns=0,2,4,6\nattention-columns=1,3,5,7\nattention-handoff=adjacent-depth3-bf16\nattention-order=0,2,4,1,3,5\nweights=qkv-paired-whole-scaled-plus-o-direct\noutput=token-major-f32\noutput-kernel=reuse-activation-across-four-n-tiles\n' > "$OUT/shape.txt"
echo "$OUT"
