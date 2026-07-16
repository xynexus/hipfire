#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH="${1:-1}"
if ! [[ "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-batch-count]" >&2
  exit 2
fi
M=$((256 * BATCH))
OUT="$HOME/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_direct_x_gate_reuse_bf16x2_m${M}_k768_i1152_o768"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
read -r -a CXX_FLAGS <<< "${HIPFIRE_R55_CXX_FLAGS:--Os}"
"$PEANO/bin/clang++" "$HERE/../r26/r26_w8_resident_ffn.cc" -c -o "$OUT/r26.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  "${CXX_FLAGS[@]}" -DNDEBUG --target=aie2p-none-unknown-elf \
  -DR35_CANONICAL_BF16 -DR41_CANONICAL_BF16X2_OUTPUT \
  -DR45_DIRECT_X_PRE_NORM -DR55_REUSE_GATE_ACTIVATION
python "$HERE/../r26/r26_gen.py" --canonical-bf16-input \
  --canonical-bf16x2-output --direct-x-pre-norm \
  --reuse-gate-activation --batch="$BATCH" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT"
printf '%s\n' \
  'op=resident_ffn' \
  'mode=dense-w8-direct-x-pre-norm-bf16x2-output' \
  'input=token-major-x-bf16' \
  'input-state=pre-ffn-inverse-f32-physical-records' \
  'output=token-major-bf16x2' \
  'gate-activation=reuse-quantized-local-fragments' \
  "m=$M" \
  'k=768' \
  'intermediate=1152' \
  'out=768' > "$OUT/shape.txt"
echo "$OUT"
