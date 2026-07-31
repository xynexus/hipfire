#!/usr/bin/env python3
"""What actually limits decode: MAC width, or weight supply?

Phase 3a is argued in the plan as "symmetric int4 straight into
mac_4x16_16x16" — a MAC-width story. This checks that framing against the
bandwidth the array actually gets, and the framing does not survive.

    python3 gemm_shapes.py            # native int4 loop cost + the supply ceiling
    python3 gemm_shapes.py --asm oq4  # dump the inner loop

Two independently measured facts, put side by side:

  mac_4x16_16x16 issues 512 MACs/cycle/core  (macbench_hw.py, hardware)
  FLM decodes at 46.2 GB/s of 5.00 bpw weights  (flm_bench.py, hardware)

The second implies **2.57 MACs/cycle/core** of weight supply across 16 cores at
1.8 GHz. The MAC unit offers 199x that. Even mac_elem_16 — the 16-lane mode FLM
actually uses, and the one the plan treats as its handicap — is 6x more than the
supply can feed.

**Decode is bandwidth-bound, so MAC width is nearly irrelevant to it.** oq4++
still wins, but for a different reason than the plan gives: 4.125 bpw against
q4_1's 5.00 is 21% more weights per second at the same bandwidth. That is the
number to design toward.

This deliberately does not rebuild FLM's dequant chain from source. Its cost is
already measured from the shipped binary (~117 bundles per K-group of 32,
4.2 MACs/cycle/core), and a reconstruction would compare our codegen against
theirs rather than the two formats.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = Path.home() / ".venv/lib/python3.14/site-packages/llvm-aie/bin"
INC = Path.home() / ".venv/lib/python3.14/site-packages/llvm-aie/lib/clang/21/include"

# FLM's decode tile.
K_GROUP = 32          # weights per scale group
N_TILE = 16           # output columns per core

# NOTE: this deliberately does NOT rebuild FLM's dequant chain. Its cost is
# already measured from the shipped binary (phase 1: ~117 bundles per K-group of
# 32, 4.2 MACs/cycle/core), and a from-source reconstruction would compare our
# codegen against theirs rather than the two *formats*. What matters is the
# native int4 path's cost, measured here, set against the weight-supply ceiling.
# --- oq4++'s shape: native int4 MAC, no dequant ----------------------------
#
# v256int4 is 128 B = exactly one oq4++ group's nibble payload. Symmetric, so
# there is no zero-point term, and the scale multiply happens once per group at
# the accumulator boundary rather than per element.
SRC_OQ4 = """
#include <aie2pintrin.h>
#include <stdint.h>

extern "C" void gemm_oq4(int8_t *w, int8_t *act, int32_t *out, int32_t n) {
  v256int4 *wp = (v256int4 *)w;
  v64int8 a = *(v64int8 *)act;
  v64acc32 acc = undef_v64acc32();
  for (int g = 0; g < %(groups)d; ++g) {
    acc = mac_4x16_16x16(a, wp[g], acc);
  }
  *(v64acc32 *)out = acc;
}
"""


def compile_and_count(name, src, groups):
    """Compile one shape and return (bundles, ops, vmacs) for its inner loop."""
    text = src % dict(groups=groups)
    with tempfile.TemporaryDirectory() as t:
        cpp, obj = Path(t) / f"{name}.cpp", Path(t) / f"{name}.o"
        cpp.write_text(text)
        r = subprocess.run(
            [str(BIN / "clang++"), "--target=aie2p-none-unknown-elf", "-O2",
             "-I", str(INC), "-c", str(cpp), "-o", str(obj)],
            capture_output=True, text=True)
        if r.returncode:
            first = next((l for l in r.stderr.splitlines() if "error" in l), r.stderr[:120])
            return None, first.strip()[:140], None
        d = subprocess.run([str(BIN / "llvm-objdump"), "-d", "--no-show-raw-insn",
                            str(obj)], capture_output=True, text=True).stdout

    # The AIE hardware loop body runs from the loop-head label through the
    # .L_LEnd bundle inclusive -- the same convention macbench.py uses.
    lines = d.splitlines()
    start = next((i for i, l in enumerate(lines) if re.search(r"<\.LBB\d+_\d+>", l)), None)
    end = next((i for i, l in enumerate(lines) if "<.L_LEnd" in l), None)
    if start is None or end is None or end < start:
        return None, "no hardware loop found (fully unrolled?)", d
    body = [l for l in lines[start:end + 2] if re.match(r"\s*[0-9a-f]+:", l)]
    ops = sum(1 for l in body for s in l.split(";")
              if s.strip() and not s.strip().split()[0].startswith("nop"))
    vmacs = sum(1 for l in body if re.search(r"\bvmac", l))
    return len(body), ops, (vmacs, d)


def main():
    p = argparse.ArgumentParser(description="Compare GEMM inner-loop schedules")
    p.add_argument("--groups", type=int, default=8,
                   help=f"K-groups of {K_GROUP} per iteration (default 8 = K256)")
    p.add_argument("--asm", choices=("flm", "oq4"), help="dump that shape's disassembly")
    o = p.parse_args()

    shapes = [("oq4", SRC_OQ4)]
    results = {}
    print(f"{'shape':6s} {'bundles':>8s} {'ops':>6s} {'vmac':>5s}  note")
    print("-" * 64)
    for name, src in shapes:
        bundles, ops, extra = compile_and_count(name, src, o.groups)
        if bundles is None:
            print(f"{name:6s} {'—':>8s} {'—':>6s} {'—':>5s}  FAIL: {ops}")
            continue
        vmacs, dis = extra
        results[name] = bundles
        print(f"{name:6s} {bundles:8d} {ops:6d} {vmacs:5d}")
        if o.asm == name:
            print("\n".join(dis.splitlines()[:60]))

    if "oq4" in results:
        # The number that actually decides phase 3a is not the MAC rate but how
        # many MACs the weight supply can feed.
        clk, cores, bw = 1.8e9, 16, 46.2e9
        for label, bpw in (("q4_1 (FLM)", 5.00), ("oq4++", 4.125)):
            sup = bw / (bpw / 8) / cores / clk
            print(f"  supply at {bpw:5.3f} bpw: {sup:5.2f} MACs/cycle/core")
        print(f"  mac_4x16_16x16 offers 512 MACs/cycle -> "
              f"{512 / (bw / (5.00 / 8) / cores / clk):.0f}x more than can be fed.")
        print("Decode is bandwidth-bound. The oq4++ win is FEWER BYTES "
              "(5.00 -> 4.125 bpw = 21% more weights/s), not MAC width.")


if __name__ == "__main__":
    main()
