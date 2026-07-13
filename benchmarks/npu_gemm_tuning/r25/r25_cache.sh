#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${R25_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_aie2p_resident_ffn_w4_m256_k768_i1152_o768}"
rm -rf "$OUT"; mkdir -p "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"
"$PEANO/bin/clang++" "$HERE/r25_w4_resident_ffn.cc" -c -o "$OUT/r25.o" \
  -I"$MA_ROOT/include" -std=c++20 -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  ${HIPFIRE_R25_CXX_OPT:--O2} ${HIPFIRE_R25_CXX_FLAGS:-} -DNDEBUG \
  -DR25_PROBE_ROW="${HIPFIRE_R25_PROBE_ROW:-0}" \
  -DR25_RAW_SOURCE_JN="${HIPFIRE_R25_RAW_SOURCE_JN:--1}" \
  --target=aie2p-none-unknown-elf
if [[ -n "${HIPFIRE_R25_PROBE_BUILD:-}" ]]; then PROBE_ARG=probe; else PROBE_ARG=; fi
python3 "$HERE/r25_gen.py" $PROBE_ARG > "$OUT/aie.mlir"
aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
printf 'op=resident_ffn\nmode=w4\nm=256\nk=768\nintermediate=1152\nout=768\nnmacros=%s\nnmacro_base=%s\n' \
  "${HIPFIRE_R25_N_MACROS:-3}" "${HIPFIRE_R25_NMACRO_BASE:-0}" > "$OUT/shape.txt"
echo "$OUT"
