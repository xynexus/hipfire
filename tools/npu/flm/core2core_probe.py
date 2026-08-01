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

N = 64                          # default elements; --elems overrides


def build(stages, n=N):
    """A chain of `stages` workers, each handing off to the next core to core.

    Built through exec so `stages` lands in the design SOURCE — iron.jit keys its
    cache on the function's text, and a closure over it silently reuses a stale
    build (that trap has cost time four times in this session).
    """
    ty = np.ndarray[(n,), np.dtype[np.int32]]

    src = f"""
def _design(a: In, o: Out):
    f_in = ObjectFifo(ty, depth=1, name="c2c_in{stages}_{n}")
    mids = [ObjectFifo(ty, depth=1, name=f"c2c_mid{stages}_{n}_{{i}}")
            for i in range({stages} - 1)]
    f_out = ObjectFifo(ty, depth=1, name="c2c_out{stages}_{n}")

    def stage(ic, oc):
        e = ic.acquire(1)
        r = oc.acquire(1)
        for k in range({n}):
            r[k] = e[k] + 1
        oc.release(1)
        ic.release(1)

    ins = [f_in.cons()] + [m.cons() for m in mids]
    outs = [m.prod() for m in mids] + [f_out.prod()]
    workers = [Worker(stage, fn_args=[ins[i], outs[i]], stack_size=2048)
               for i in range({stages})]

    def seq(ab, ob, ah, oh):
        tg = TaskGroup()
        ah.fill(ab, group=tg)
        oh.drain(ob, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty, ty, f_in.prod(tile=AnyShimTile),
                       f_out.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
"""
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, ty=ty,
              __name__=f"c2c{stages}_{n}")
    exec(src, ns)
    return iron.jit(ns["_design"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--elems", type=int, default=N,
                    help="int32 elements per handoff; a real intermediate is\n"
                         "K_DIM bf16 = 4 KB = 1024 int32-equivalents")
    ap.add_argument("--bench", action="store_true",
                    help="time it; the slope over --stages is the per-handoff cost")
    o = ap.parse_args()
    src = np.arange(o.elems, dtype=np.int32)
    a = iron.tensor(src, dtype=np.int32, device="npu")
    b = iron.zeros(o.elems, dtype=np.int32, device="npu")
    design = build(o.stages, o.elems)
    design(a, b)

    got, want = b.numpy(), src + o.stages
    ok = bool((got == want).all())
    print(f"{o.stages}-stage core-to-core chain: {'computes correctly' if ok else 'WRONG RESULT'}")
    if not ok:
        i = int(np.argmax(got != want))
        print(f"  first mismatch at {i}: got {got[i]} want {want[i]}")
        return 1

    # The point of the probe: what did f_mid get placed on?
    cache = sorted(Path.home().glob(".npu/cache/*/aie.mlir"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    mlir = next((p for p in cache
                 if f"c2c_mid{o.stages}_{o.elems}_" in p.read_text()), None)
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
    if o.bench:
        from aie.utils.benchmark import run_iters
        r = run_iters(design, a, b, warmup=2, iters=20)
        us = r.npu.min_us if r.npu else r.e2e.min_us
        kb = o.elems * 4 / 1024
        print(f"  {o.stages} stages, {kb:.1f} KB/handoff: {us:.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
