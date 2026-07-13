#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${R68_PACK_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r68_overlap_joined_stage_to_qkv_m256}"
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
"$PEANO/bin/clang++" "$HERE/../r28/r28_qkv_pack.cc" -c -o "$OUT/r28.o" "${COMMON[@]}"
python "$HERE/../r67/r67_pack_gen.py" 37 > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=r68-overlap-joined-stage-headnorm-rope-pack\nmode=bf16-overlap-joined\nm=256\nheads=3\nkv_heads=1\nhead_dim=256\nrecord_bytes=8192\nrecords_per_role=37\nq_layout=mmul-packed\nkv_layout=mmul-packed-single-replay\n' > "$OUT/shape.txt"
echo "$OUT"
