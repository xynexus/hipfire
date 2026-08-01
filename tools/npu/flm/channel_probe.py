#!/usr/bin/env python3
"""Does the FUSED design's MEMTILE channel demand actually place?

**This probe answers one budget and there are two.** It builds 12 fills and 12
drains, so it never exercised the SHIM, which on NPU2 is 8 shim tiles x 2
channels each way = a hard 16 in / 16 out (measured). The naive union of
`groups_ab` and `group_c` asks the shim for 22 in / 18 out and the placer refuses
with "no ShimNOCTile has sufficient DMA capacity". Passing here says nothing
about that.

`fused.py` fits both budgets, at 16 shim in / 14 out and 46 memtile in / 46 out
— tighter on the memtile side than the 40/36 below, and with room for no further
fifo at all.


The 64.5 tok/s projection assumes `groups_ab` and `group_c` fuse into one
dispatch. Everything else about that has been checked — placement and routing at
full operand size, per-core program memory (A+B 57%, C 90% of 16 KB, and fusing
adds code to no core), and the C->A seam needing no new core code. The one thing
left is a number I computed and never built:

    A+B      4 splits 1->2, 2 joins 4->1   ->  12 in, 10 out
    C        8 splits 1->2, 10 joins 2->1  ->  28 in, 26 out
    union                                      40 in, 36 out   (budget 48 / 48)

`layer_roles.py --skeleton` loads at full operand size but routes intermediates
core to core, so it only asks for 24 in / 6 out. Passing at 24 says nothing about
40, and the earlier 32-core attempt died at exactly this kind of margin — "no
MemTile has sufficient DMA capacity" at 40 inputs from joins alone.

So this builds the union's link structure and nothing else: the same counts and
widths, 32 cores, full-size objects, and bodies that touch 8 elements so program
memory cannot be the thing that fails. It answers one question.

    python3 channel_probe.py                 # the union, 40 in / 36 out
    python3 channel_probe.py --c-joins 6     # trim until it fits, if it does not

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

OPERAND = 2576                  # int32; the real 10304 B weight tile
TOUCH = 8


def build(ab_splits, ab_joins, c_splits, c_joins, elems, touch):
    """Splits are 1->2, A+B's joins are 4->1, C's are 2->1 — as measured."""
    whole = np.ndarray[(elems,), np.dtype[np.int32]]
    half = np.ndarray[(elems // 2,), np.dtype[np.int32]]
    quarter = np.ndarray[(elems // 4,), np.dtype[np.int32]]

    src = f"""
def _design(a: In, o: Out):
    tag = "cp{ab_splits}_{ab_joins}_{c_splits}_{c_joins}_{elems}_{touch}"
    outs, cons, prods = [], [], []

    for i in range({ab_splits} + {c_splits}):
        f = ObjectFifo(whole, depth=1, name=f"{{tag}}_s{{i}}")
        subs = f.cons().split([0, {elems} // 2], obj_types=[half, half],
                              names=[f"{{tag}}_sa{{i}}", f"{{tag}}_sb{{i}}"])
        cons += [sub.cons() for sub in subs]
        outs.append(("in", f))

    for i in range({ab_joins}):
        f = ObjectFifo(whole, depth=1, name=f"{{tag}}_j4_{{i}}")
        ins = f.prod().join([k * ({elems} // 4) for k in range(4)],
                            obj_types=[quarter] * 4,
                            names=[f"{{tag}}_j4a{{i}}_{{k}}" for k in range(4)])
        prods += [h.prod() for h in ins]
        outs.append(("out", f))

    for i in range({c_joins}):
        f = ObjectFifo(whole, depth=1, name=f"{{tag}}_j2_{{i}}")
        ins = f.prod().join([0, {elems} // 2], obj_types=[half, half],
                            names=[f"{{tag}}_j2a{{i}}", f"{{tag}}_j2b{{i}}"])
        prods += [h.prod() for h in ins]
        outs.append(("out", f))

    # One core carries a split consumer AND a join producer, which is what the
    # real designs do — group C's cores each read a weight sub-fifo and write
    # into a result join. Giving each endpoint its own core asks for 52 and the
    # placer refuses at 32.
    def both(ic, oc):
        e = ic.acquire(1); r = oc.acquire(1)
        for k in range({touch}):
            r[k] = e[k]
        oc.release(1); ic.release(1)

    def only_out(oc):
        r = oc.acquire(1)
        for k in range({touch}):
            r[k] = k
        oc.release(1)

    def only_in(ic):
        e = ic.acquire(1)
        for k in range({touch}):
            pass
        ic.release(1)

    workers = []
    n = min(len(cons), len(prods))
    for i in range(n):
        workers.append(Worker(both, fn_args=[cons[i], prods[i]], stack_size=1024))
    for h in cons[n:]:
        workers.append(Worker(only_in, fn_args=[h], stack_size=1024))
    for h in prods[n:]:
        workers.append(Worker(only_out, fn_args=[h], stack_size=1024))

    def seq(ab, ob, *hs):
        tg = TaskGroup()
        for h, (kind, _) in zip(hs, outs):
            if kind == "in":
                h.fill(ab, group=tg)
            else:
                h.drain(ob, wait=True, group=tg)
        tg.finish()

    at = [whole, whole]
    at += [f.prod(tile=AnyShimTile) if k == "in" else f.cons(tile=AnyShimTile)
           for k, f in outs]
    rt = Runtime(seq, at)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
"""
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile,
              whole=whole, half=half, quarter=quarter,
              __name__=f"cp{ab_splits}_{ab_joins}_{c_splits}_{c_joins}_{elems}_{touch}")
    exec(src, ns)
    return iron.jit(ns["_design"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ab-splits", type=int, default=4)
    p.add_argument("--ab-joins", type=int, default=2)
    p.add_argument("--c-splits", type=int, default=8)
    p.add_argument("--c-joins", type=int, default=10)
    p.add_argument("--elems", type=int, default=OPERAND)
    p.add_argument("--touch", type=int, default=TOUCH)
    o = p.parse_args()

    nin = o.ab_splits + o.c_splits + 4 * o.ab_joins + 2 * o.c_joins
    nout = 2 * (o.ab_splits + o.c_splits) + o.ab_joins + o.c_joins
    _c = 2 * (o.ab_splits + o.c_splits)
    _p = 4 * o.ab_joins + 2 * o.c_joins
    ncore = max(_c, _p)          # a core carries one of each where it can
    print(f"links: {o.ab_splits + o.c_splits} splits 1->2, {o.ab_joins} joins 4->1, "
          f"{o.c_joins} joins 2->1")
    print(f"memtile channels: {nin} in, {nout} out   (budget 48 / 48)")
    print(f"cores: {ncore}   object: {o.elems} int32 = {o.elems * 4} B")

    a = iron.tensor(np.arange(o.elems, dtype=np.int32), dtype=np.int32, device="npu")
    b = iron.zeros(o.elems, dtype=np.int32, device="npu")
    nfifo = o.ab_splits + o.c_splits + o.ab_joins + o.c_joins
    build(o.ab_splits, o.ab_joins, o.c_splits, o.c_joins, o.elems, o.touch)(
        a, b, *([a] * 0))
    print("  -> PLACES, ROUTES AND LOADS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
