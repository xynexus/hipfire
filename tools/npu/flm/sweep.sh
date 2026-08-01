#!/usr/bin/env bash
# The regression sweep. `kernels/npu/flm_kv_pair.h` and `flm_q4_1_tile.h` are
# linked by every design in this tree, so a change to either is not local to the
# fused one even when it is meant to be -- and "this change is inert for other
# callers" is the claim that has proved false here before.
#
# `fused` is NOT in this list: it costs ~25 minutes of weight packing, so run it
# separately (`fused.py --tokens ...` for a single step, `decode.py` for the loop).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
run() { echo; echo "===== $* ====="; python3 -u "$@" 2>&1 | tail -6; }
run group_a.py
run groups_ab.py --pos 0
run groups_ab.py --pos 30
run group_c.py --seq 31
run p5_pass.py
run p1p2_chain.py --seq 31
