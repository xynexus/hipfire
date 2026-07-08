#!/usr/bin/env bash
# Offline: build + cache the R14 whole-array (4×4) W4A8 GEMM xclbin for the Phoenix
# NPU (aie2 / npu1 / XDNA1) — the aie2 analog of r6/r6_cache.sh (which is aie2p-only).
# Produces ~/.hipfire/npu/r14_<LM>x<LN>x<KT>_nb<N_BLK>/{final.xclbin,insts.bin}, which
# crates/hipfire-xdna's npu_gemm_bench / npu_embeddinggemma_bench load directly.
#
# The r14 kernel (r11_gemm.cc, mmul<4,16,8,int8,int4>) + generator (r14_gen.py,
# aie.device(npu1)) already target aie2; this just wraps the r11_run.sh build block
# into a cached artifact. N_BLK is baked into the xclbin and must be ≤ 1023 (npu1 DMA
# BD dimension cap; r14_gen.py splits the contiguous stripe to stay in range).
#
# Usage: r14_cache.sh [LM] [LN] [KT] [N_BLK]   (defaults 6 12 16 512)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM="${1:-6}"; LN="${2:-12}"; KT="${3:-16}"; NBLK="${4:-512}"
[ "$NBLK" -le 1023 ] || { echo "N_BLK must be ≤ 1023 (npu1 DMA BD cap)"; exit 1; }

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done

OUT="$HOME/.hipfire/npu/r14_${LM}x${LN}x${KT}_nb${NBLK}"
rm -rf "$OUT"; mkdir -p "$OUT"
"$PEANO/bin/clang++" "$HERE/../r11/r11_gemm.cc" -c -o "$OUT/r11.o" -I"$MA_ROOT/include" \
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -O2 -DNDEBUG --target=aie2-none-unknown-elf -DLM="$LM" -DLN="$LN" -DKT="$KT"
python3 "$HERE/r14_gen.py" "$LM" "$LN" "$KT" "$NBLK" > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT/final.xclbin  $OUT/insts.bin"
echo "  block: A=4·N·$((LM*KT*64))  W=4·N·$((LN*KT*64))  C=4·N·$((LM*LN*32))·4B  macs=16·N·${LM}·${LN}·${KT}·512  expect=$((KT*16))  (N=$NBLK)"
