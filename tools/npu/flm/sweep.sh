#!/usr/bin/env bash
# The regression sweep. `kernels/npu/flm_kv_pair.h` and `flm_q4_1_tile.h` are
# linked by every design in this tree, so a change to either is not local to the
# fused one even when it is meant to be -- and "this change is inert for other
# callers" is the claim that has proved false here before.
#
# `fused` is NOT in this list: it costs ~25 minutes of weight packing, so run it
# separately (`fused.py --tokens ...` for a single step, `decode.py` for the loop).
#
# Two things this script has to get right to be worth running at all:
#
#   PYTHONPATH, BOTH halves. A stale mlir_aie wheel in site-packages SHADOWS the
#   build tree -- the same trap txn_check.py names -- and every design dies on
#   `cannot import name 'CompileTime' from 'aie.iron'`. Adding only the build
#   tree then fails DIFFERENTLY and much more quietly: with no importable pyxrt,
#   `iron.tensor(..., device="npu")` resolves to CPUOnlyTensor, whose DEVICES is
#   ["cpu"], and the sweep dies on `Unsupported device: npu`. XRT is not optional
#   here either. MLIR_AIE_BUILD and XRT_DIR override; the mlir-aie default
#   matches txn_check.py's.
#
#   The exit status. `python3 ... | tail -6` reports TAIL's status, so the first
#   version exited 0 with all six designs failing to import. A regression sweep
#   that cannot fail is not a regression sweep. PIPESTATUS is the fix.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${MLIR_AIE_BUILD:=$HOME/build/mlir-aie/build}"
: "${XRT_DIR:=/opt/xilinx/xrt}"
[ -d "$MLIR_AIE_BUILD/python" ] ||
    { echo "no aie package at $MLIR_AIE_BUILD/python -- set MLIR_AIE_BUILD" >&2; exit 2; }
export PYTHONPATH="$MLIR_AIE_BUILD/python:$XRT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import pyxrt" 2>/dev/null ||
    { echo "no importable pyxrt via $XRT_DIR/python -- set XRT_DIR" >&2; exit 2; }

fails=0
# run_n takes the tail depth: the gate's verdict lines (48/48, outside top-64) sit
# ABOVE its timing block, so at 6 the sweep showed the gate running and none of
# what it decided.
run_n() {
    local n=$1; shift
    echo; echo "===== $* ====="
    python3 -u "$@" 2>&1 | tail -"$n"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "  ^^ FAILED (exit ${PIPESTATUS[0]})"; fails=$((fails + 1)); }
}
run() { run_n 6 "$@"; }
run group_a.py
run groups_ab.py --pos 0
run groups_ab.py --pos 30
run group_c.py --seq 31
run p5_pass.py
run p1p2_chain.py --seq 31
# The gate is the ONLY device-side recall evidence for the two-pass head, and the
# bug it caught (PyxrtDesign dispatching stale buffers when the arguments change)
# was silent everywhere else: correct timing, correct-looking logits, wrong token.
# Skipped rather than failed when the coarse tier has not been built, because a
# missing artifact is not a regression.
# Path comes from the module, not a second copy of it here -- lmhead_coarse.CACHE
# is /tmp/lmhead2p, so the tier does not survive a reboot and the skip is a normal
# state, not a broken checkout.
tier=$(python3 -c "import lmhead_coarse as lc; print(lc.CACHE / 'lmhead_coarse.npz')" 2>/dev/null || true)
if [ -n "${tier:-}" ] && [ -f "$tier" ]; then
    run_n 12 lmhead_twostage.py --gate
else
    echo; echo "===== lmhead_twostage.py --gate ====="
    echo "  SKIP: no coarse tier at ${tier:-<unresolved>} -- lmhead_coarse.py --build"
fi

echo
# "no design failed", NOT "all designs PASS" -- p1p2_chain.py returns 0 with an
# ADVISORY verdict ("within the empirical envelope; floor model unresolved"), so
# a zero exit here means nothing regressed, not that every design asserted.
if [ "$fails" -eq 0 ]; then echo "sweep: no design FAILED"; else echo "sweep: $fails design(s) FAILED"; fi
exit "$fails"
