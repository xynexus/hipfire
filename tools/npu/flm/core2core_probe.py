#!/usr/bin/env python3
"""Does a core-to-core ObjectFifo avoid a MemTile?

32 cores cannot be reached while every core emits its result through a memtile
join: a join costs `w` inputs and there are `cores / w` of them, so the join side
costs exactly `cores` input channels at any width. At 32 cores that is 40 of a
~48 budget before a single operand split is placed.

The way out — and what FLM's `mvm_tiles` / `proj_tiles` split implies — is that
intermediate results move core to core instead of out through a memtile. That
only helps if a core-to-core fifo genuinely does not consume memtile DMA. This
checks it by building one and reading the generated MLIR:

    worker A --f_mid--> worker B --f_out--> shim

If `f_mid` is placed on the two core tiles with no `logical_tile<MemTile>` for it,
the premise holds and the architecture is worth building. If every fifo drags in a
memtile regardless, the budget is unchanged and that direction is dead too.

    python3 core2core_probe.py

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

N = 64


def build():
    ty = np.ndarray[(N,), np.dtype[np.int32]]

    def _design(a: In, o: Out):
        f_in = ObjectFifo(ty, depth=1, name="c2c_in")
        f_mid = ObjectFifo(ty, depth=1, name="c2c_mid")     # the one under test
        f_out = ObjectFifo(ty, depth=1, name="c2c_out")

        def stage_a(ic, oc):
            e = ic.acquire(1)
            r = oc.acquire(1)
            for k in range(N):
                r[k] = e[k] + 1
            oc.release(1)
            ic.release(1)

        def stage_b(ic, oc):
            e = ic.acquire(1)
            r = oc.acquire(1)
            for k in range(N):
                r[k] = e[k] * 2
            oc.release(1)
            ic.release(1)

        wa = Worker(stage_a, fn_args=[f_in.cons(), f_mid.prod()], stack_size=2048)
        wb = Worker(stage_b, fn_args=[f_mid.cons(), f_out.prod()], stack_size=2048)

        def seq(ab, ob, ah, oh):
            tg = TaskGroup()
            ah.fill(ab, group=tg)
            oh.drain(ob, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [ty, ty, f_in.prod(tile=AnyShimTile),
                           f_out.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt, workers=[wa, wb]).resolve_program()

    return iron.jit(_design)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    src = np.arange(N, dtype=np.int32)
    a = iron.tensor(src, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    build()(a, b)

    got, want = b.numpy(), (src + 1) * 2
    ok = bool((got == want).all())
    print(f"core -> core -> shim: {'computes correctly' if ok else 'WRONG RESULT'}")
    if not ok:
        i = int(np.argmax(got != want))
        print(f"  first mismatch at {i}: got {got[i]} want {want[i]}")
        return 1

    # The point of the probe: what did f_mid get placed on?
    cache = sorted(Path.home().glob(".npu/cache/*/aie.mlir"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    mlir = next((p for p in cache if "c2c_mid" in p.read_text()), None)
    if mlir is None:
        print("  could not find this design's MLIR to inspect")
        return 2
    text = mlir.read_text()
    line = next((l.strip() for l in text.splitlines()
                 if "@c2c_mid" in l and "objectfifo" in l), "")
    memtiles = len(re.findall(r"logical_tile<MemTile>", text))
    print(f"  f_mid: {line[:110]}")
    print(f"  MemTiles declared in the whole design: {memtiles}")
    if "mem_" in line:
        print("  -> f_mid DOES traverse a memtile; the core-to-core premise fails")
        return 1
    print("  -> f_mid is core-to-core with no memtile: the premise holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
