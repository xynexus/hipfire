#!/usr/bin/env bash
# R14B — build + measure the activation-folded dual-shim-channel GEMM against a
# same-flags R14 control, at DFlash shapes (M=16, K-blocked, N-blocked).
#
# Usage: ./r14b_run.sh [LM] [LN] [KT] [N_BLK] [ITERS] [MT] [NT]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
LM="${1:-1}"; LN="${2:-4}"; KT="${3:-64}"; NBLK="${4:-128}"; ITERS="${5:-200}"
MT="${6:-1}"; NT="${7:-4}"

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done

WBYTES=64                      # int4 weights (W4A8)
AB=$(( LM*KT*64 )); WB=$(( LN*KT*WBYTES )); OB=$(( WB+AB ))
CB=$(( LM*LN*32 )); CJ=$(( 4*CB ))
MACS=$(( 16*NBLK*LM*LN*KT*512 ))
K=$(( KT*16 ))

build_o () {  # $1 outdir
  "$PEANO/bin/clang++" "$HERE/../r11/r11_gemm.cc" -c -o "$1/r11.o" -I"$MA_ROOT/include" \
    -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -O2 -DNDEBUG --target=aie2-none-unknown-elf -DLM="$LM" -DLN="$LN" -DKT="$KT" -DMT="$MT" -DNT="$NT"
}
aiecc_it () { # $1 dir
  aiecc "$1/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
    --aie-generate-npu-insts --npu-insts-name="$1/insts.bin" \
    --aie-generate-xclbin --xclbin-name="$1/final.xclbin" --tmpdir="$1" >"$1/aiecc.log" 2>&1
}

TAG="${LM}x${LN}x${KT}_nb${NBLK}_mt${MT}n${NT}"

# ---- control: stock r14 dataflow, same kernel flags -------------------------------
CTL="$HOME/.hipfire/npu/r14ctl_$TAG"; rm -rf "$CTL"; mkdir -p "$CTL"
build_o "$CTL"
python3 "$HERE/../r14/r14_gen.py" "$LM" "$LN" "$KT" "$NBLK" "$WBYTES" > "$CTL/aie.mlir"
if aiecc_it "$CTL"; then
  ASZ=$(( 4*NBLK*AB )); WSZ=$(( 4*NBLK*WB )); CSZ=$(( 4*NBLK*CJ*4 ))
  echo "== control (r14): A=${ASZ}B W=${WSZ}B  expect C[0]=$K"
  (cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
     "$CTL" "$ASZ" "$WSZ" "$CSZ" "$MACS" "$ITERS" "$K")
else
  echo "control AIECC FAILED:"; grep -iE "error|exceed" "$CTL/aiecc.log" | head -5
fi

# ---- variant: activation-folded, 8 weight streams ---------------------------------
VAR="$HOME/.hipfire/npu/r14b_$TAG"; rm -rf "$VAR"; mkdir -p "$VAR"
build_o "$VAR"
python3 "$HERE/r14b_gen.py" "$LM" "$LN" "$KT" "$NBLK" "$WBYTES" > "$VAR/aie.mlir"
if aiecc_it "$VAR"; then
  ASZ=64; WSZ=$(( 4*NBLK*OB )); CSZ=$(( 4*NBLK*CJ*4 ))
  EXP=$(( 17*K ))
  echo "== r14b: fused W|A=${WSZ}B (W part $(( 4*NBLK*WB ))B)  expect C[0]=$EXP"
  (cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
     "$VAR" "$ASZ" "$WSZ" "$CSZ" "$MACS" "$ITERS" "$EXP")
else
  echo "r14b AIECC FAILED:"; grep -iE "error|exceed" "$VAR/aiecc.log" | head -10
fi

# ---- r14c: 8 weight streams, core-resident A --------------------------------------
AVAL="${AVAL:-3}"
VC="$HOME/.hipfire/npu/r14c_$TAG"; rm -rf "$VC"; mkdir -p "$VC"
build_o "$VC"
python3 "$HERE/r14c_gen.py" "$LM" "$LN" "$KT" "$NBLK" "$WBYTES" npu1 32 2 "$AVAL" > "$VC/aie.mlir"
if aiecc_it "$VC"; then
  ASZ=64; WSZ=$(( 4*NBLK*WB )); CSZ=$(( 4*NBLK*CJ*4 ))
  EXP=$(( AVAL*K ))
  echo "== r14c: W=${WSZ}B on 8 shim channels, A resident (=$AVAL)  expect C[0]=$EXP"
  (cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
     "$VC" "$ASZ" "$WSZ" "$CSZ" "$MACS" "$ITERS" "$EXP")
else
  echo "r14c AIECC FAILED:"; grep -iE "error|exceed" "$VC/aiecc.log" | head -10
fi
