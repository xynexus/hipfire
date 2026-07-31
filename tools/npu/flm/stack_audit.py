#!/usr/bin/env python3
"""Report every kernel's stack frame. Overflow on this device is SILENT.

IRON's default stack is 1024 B and an overflow does not fault — it corrupts,
and the corruption looks like a logic bug. `static_persist_probe.py` spent two
rounds reading "core state does not persist" out of a 7232 B frame in a 4096 B
stack.

Frame size is not proportional to anything obvious. It is driven by how far the
Peano backend unrolls, and a **fully unrolled loop doing scalar
`float -> bfloat16` conversion spills the whole accumulator register file**:

    trips     8     16     24     32     48     64
    frame  1024   3136   5184   7232      0      0

— ~2 KB per 8 iterations while it unrolls, then 0 once it gives up and emits a
real loop. The same loop writing `float` instead of `bfloat16` costs 64 B. Every
kernel in `kernels/npu/` escapes this because it indexes with a dynamic base
(`slot + r`, `base + r`), which the backend will not unroll that way — but that
is a property of how they happen to be written, not a guarantee, so this script
exists to check rather than assume.

    python3 stack_audit.py
    python3 stack_audit.py --limit 4096

Needs the Peano toolchain (`clang++`, `llvm-objdump`) and the mlir-aie headers.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
# the shapes the harnesses actually build with
DEFS = ["-DDIM_K=2048", "-DDIM_NROWS=16", "-DDIM_HEAD=64", "-DDIM_ACT=2048",
        "-DDIM_GQA=4", "-DDIM_TSEQ=32", "-DDIM_KVPER=2", "-DDIM_ACCN=256",
        "-DDIM_RESN=256", "-DRESID_FROM_STASH=1", "-DDIM_N=32"]
FRAME = re.compile(r"paddxm\s+\[sp\], #(0x[0-9a-f]+)")


def frame_bytes(src, incs, tmp):
    obj = tmp / (src.stem + ".o")
    r = subprocess.run(
        ["clang++", "--target=aie2p-none-unknown-elf", "-std=c++20", "-O2",
         "-w", "-c", str(src), "-o", str(obj), *incs, *DEFS],
        capture_output=True, text=True)
    if r.returncode:
        return None
    d = subprocess.run(["llvm-objdump", "-d", str(obj)],
                       capture_output=True, text=True)
    found = [int(m, 16) for m in FRAME.findall(d.stdout)]
    return max(found) if found else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=4096,
                   help="the stack_size the harnesses pass (default 4096)")
    o = p.parse_args()

    mlir = os.environ.get("MLIR_AIE_DIR")
    if not mlir:
        print("MLIR_AIE_DIR is not set", file=sys.stderr)
        return 2
    incs = [f"-I{mlir}/install/include", f"-I{mlir}/runtime_lib/AIE2P"]

    srcs = sorted(KDIR.glob("flm_*.cc"))
    worst, bad = 0, []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print(f"{'kernel':<26s} {'frame B':>8s}  {'of ' + str(o.limit):>9s}")
        print("-" * 48)
        for s in srcs:
            fb = frame_bytes(s, incs, tmp)
            if fb is None:
                print(f"{s.stem:<26s} {'(no build)':>8s}")
                continue
            worst = max(worst, fb)
            flag = "" if fb <= o.limit else "   <-- OVERFLOWS"
            if fb > o.limit:
                bad.append((s.stem, fb))
            print(f"{s.stem:<26s} {fb:8d}  {100*fb/o.limit:8.0f}%{flag}")

    print(f"\nworst {worst} B against a {o.limit} B stack "
          f"({o.limit/max(worst,1):.1f}x margin)")
    if bad:
        for n, fb in bad:
            print(f"  {n} needs at least {fb} B — raise stack_size")
        return 1
    print("all kernels fit. Note this is per-kernel; a Worker's stack must "
          "cover the deepest\nkernel it calls, not the sum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
