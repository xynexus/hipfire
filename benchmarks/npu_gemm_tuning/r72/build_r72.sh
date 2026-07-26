#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${R72_Q_MODE:-vector-stream}"
case "$MODE" in
  vector-stream)
    DEFAULT_OUT="$HOME/.hipfire/npu/embgemma_r72_vector_direct_q_projection_pack_attention_m256_k768_n1280"
    GEN_ARGS=(--direct-q)
    Q_HANDOFF="core-to-core-vector-load-store-stream"
    ;;
  local)
    DEFAULT_OUT="$HOME/.hipfire/npu/embgemma_r72_local_q_projection_pack_attention_m256_k768_n1280"
    GEN_ARGS=(--local-q)
    Q_HANDOFF="producer-tile-local-cache"
    ;;
  adjacent)
    DEFAULT_OUT="$HOME/.hipfire/npu/embgemma_r73_adjacent_q_projection_pack_attention_m256_k768_n1280"
    GEN_ARGS=(--adjacent-q)
    Q_HANDOFF="adjacent-core-shared-objectfifo"
    ;;
  *)
    echo "unknown R72_Q_MODE=$MODE (want vector-stream, local, or adjacent)" >&2
    exit 2
    ;;
esac
OUT="${R72_CACHE_DIR:-$DEFAULT_OUT}"
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
"$PEANO/bin/clang++" "$HERE/../r70/r70_w4_scaled_group.cc" -c -o "$OUT/r72group.o" "${COMMON[@]}" -Dr70_w4_scaled_group=r72_w4_scaled_group_cache
"$PEANO/bin/clang++" "$HERE/../r65/r65_w4_bf16_finish.cc" -c -o "$OUT/r72finish.o" "${COMMON[@]}" -Dr65_w4_finish_bf16_slice=r72_w4_finish_bf16_slice_cache
"$PEANO/bin/clang++" "$HERE/../r29/r29_w8_qkv_attention_pack.cc" -c -o "$OUT/r66.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/r72_q_stream.cc" -c -o "$OUT/r72stream.o" "${COMMON[@]}" -Oz
"$PEANO/bin/clang++" "$HERE/../r30/r30_w8_qkv_attention.cc" -c -o "$OUT/r71att.o" "${COMMON[@]}" -Oz -DR30_ATTENTION_ONLY -DR72_Q_CACHE
python "$HERE/../r71/r71_gen.py" "${GEN_ARGS[@]}" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident-opus-qkv-projection-direct-q-attention\nmode=w4-scaled\nm=256\nk=768\nn=1280\ncontexts=1\nq_handoff=%s\nkv_handoff=external-packed\noutput=head-major-bf16\n' "$Q_HANDOFF" > "$OUT/shape.txt"
echo "$OUT"
