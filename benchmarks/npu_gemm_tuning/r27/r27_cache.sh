#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH="${1:-1}"
if ! [[ "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-batch-count]" >&2
  exit 2
fi
M=$((256 * BATCH))
OUT="$HOME/.hipfire/npu/embgemma_aie2p_attention_bf16_m${M}_h3_kv1_d256"
rm -rf "$OUT"
mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/r27_attention.cc" -c -o "$OUT/r27.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  "${HIPFIRE_R27_CXX_OPT:--O2}" -DNDEBUG --target=aie2p-none-unknown-elf
python "$HERE/r27_gen.py" --batch="$BATCH" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=attention\nmode=bf16\nm=%s\nheads=3\nkv_heads=1\nhead_dim=256\nq_layout=mmul-packed\nkv_layout=mmul-packed-single-replay\nattention=block-diagonal-documents\nsegment-rows=256\n' "$M" > "$OUT/shape.txt"
echo "$OUT"
