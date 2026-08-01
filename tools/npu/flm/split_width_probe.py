#!/usr/bin/env python3
"""Does an N-way memtile split actually carry data, for N > 2?

The layer design splits each pair's operand fifo two ways, and every split lowers
to a memtile-hosted `aie.objectfifo.link`. At 16 cores that is 18 logical
memtiles against 8 physical; at 32 cores it is 36, and the build fails with "no
MemTile has sufficient DMA capacity".

`ObjectFifoHandle.split` takes an arbitrary-length offsets list, so a 4- or 8-way
split would cut the link count by the same factor and put 32 cores back inside
budget. That is the plan — but the plan rests on an API signature, and a signature
is not a measurement. Restructuring the host's weight packing to match a wider
split is a large change to make on faith.

So this checks the mechanism in isolation first: one fifo split `--way` ways at a
memtile, one worker per slice, each writing a value only it knows. If the drained
result carries every worker's mark in the right place, the split delivers.

    python3 split_width_probe.py --way 2      # what the layer does today
    python3 split_width_probe.py --way 4
    python3 split_width_probe.py --way 8

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

SLICE = 64                      # elements each worker owns


def build(way):
    """Built through exec so `way` lands in the design SOURCE.

    iron.jit keys its cache on the design function's source text. A closure over
    `way` is invisible to that: --way 4 silently reused --way 2's build and failed
    with "argument 'a' has 256 elements but the kernel was compiled for 128".
    Interpolating the width into the source is what makes the cache see it.
    """
    whole = np.ndarray[(way * SLICE,), np.dtype[np.int32]]
    part = np.ndarray[(SLICE,), np.dtype[np.int32]]

    src = f"""
def _design(a: In, o: Out):
    f_in = ObjectFifo(whole, depth=1, name="sw_in{way}")
    f_out = ObjectFifo(whole, depth=1, name="sw_out{way}")
    subs = f_in.cons().split(
        [i * {SLICE} for i in range({way})],
        obj_types=[part] * {way},
        names=[f"sw_s{way}_{{i}}" for i in range({way})])
    outs = f_out.prod().join(
        [i * {SLICE} for i in range({way})],
        obj_types=[part] * {way},
        names=[f"sw_j{way}_{{i}}" for i in range({way})])

    def core(ic, oc, mark):
        e = ic.acquire(1)
        r = oc.acquire(1)
        for k in range({SLICE}):
            r[k] = e[k] + mark
        oc.release(1)
        ic.release(1)

    workers = [Worker(core,
                      fn_args=[subs[i].cons(), outs[i].prod(), 1000 * (i + 1)],
                      stack_size=2048)
               for i in range({way})]

    def seq(ab, ob, ah, oh):
        tg = TaskGroup()
        ah.fill(ab, group=tg)
        oh.drain(ob, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [whole, whole, f_in.prod(tile=AnyShimTile),
                       f_out.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
"""
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, whole=whole, part=part,
              __name__=f"sw{way}")
    exec(src, ns)
    return iron.jit(ns["_design"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--way", type=int, default=4, choices=[2, 4, 8])
    o = p.parse_args()

    n = o.way * SLICE
    src = np.arange(n, dtype=np.int32)
    a = iron.tensor(src, dtype=np.int32, device="npu")
    b = iron.zeros(n, dtype=np.int32, device="npu")
    build(o.way)(a, b)

    got = b.numpy()
    want = src + np.repeat([1000 * (i + 1) for i in range(o.way)], SLICE)
    bad = int((got != want).sum())
    print(f"{o.way}-way memtile split, {n} elements")
    if bad:
        i = int(np.argmax(got != want))
        print(f"  MISMATCH in {bad}/{n}; first at {i}: got {got[i]} want {want[i]}"
              f"  (slice {i // SLICE})")
        return 1
    print(f"  every slice reached its own worker -> PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
