#!/usr/bin/env python3
"""Can a drain offset be a RUNTIME value instead of a build constant?

Every design here bakes the position into the descriptors —
`offset = 2 * (base * KVSTRIDE + pos)` is interpolated into the design source, so
each position is a separate xclbin. That collides with `g_kprev`, which closes
the K column pair using the previous token's k' and lives in core .bss: loading
the next position's design clears it.

`ObjectFifoHandle.fill/drain` take an `offset_parameter` that accepts a
`ScratchpadParameter` — "a named runtime value set from the host and read by
Workers". If that works for a drain offset, one xclbin serves every position and
the collision disappears. Nothing in this tree has used it, so it is an API
signature, and a signature is not a measurement — the same standard applied to
the 4-way memtile split before the layer design was built on it.

The check: drain a fixed pattern into a buffer at a runtime-chosen offset, twice
with different values, and confirm the data lands where the parameter said and
nowhere else.

    python3 runtime_offset_probe.py
    python3 runtime_offset_probe.py --offsets 0,64,192

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

N = 64                          # elements per object
SPAN = 256                      # destination buffer


def build(n, span):
    obj = np.ndarray[(n,), np.dtype[np.int32]]
    dst = np.ndarray[(span,), np.dtype[np.int32]]

    src = f"""
def _design(a: In, o: Out):
    from aie.iron import ScratchpadParameter
    off = ScratchpadParameter("rt_off_{n}_{span}", np.int32)
    f_in = ObjectFifo(obj, depth=1, name="rop_in_{n}_{span}")
    f_out = ObjectFifo(obj, depth=1, name="rop_out_{n}_{span}")

    def core(ic, oc):
        e = ic.acquire(1)
        r = oc.acquire(1)
        for k in range({n}):
            r[k] = e[k] + 1000
        oc.release(1)
        ic.release(1)

    w = Worker(core, fn_args=[f_in.cons(), f_out.prod()], stack_size=2048)

    def seq(ab, ob, ah, oh):
        tg = TaskGroup()
        ah.fill(ab, group=tg)
        # THE POINT: the destination offset is a runtime parameter, not a
        # constant folded into the descriptor at build time.
        oh.drain(ob, wait=True, group=tg, offset_parameter=off)
        tg.finish()

    rt = Runtime(seq, [obj, dst, f_in.prod(tile=AnyShimTile),
                       f_out.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
"""
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, obj=obj, dst=dst,
              __name__=f"rop{n}_{span}")
    exec(src, ns)
    return iron.jit(ns["_design"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offsets", default="0,128",
                   help="runtime offsets to try, in ELEMENTS")
    p.add_argument("--elems", type=int, default=N)
    p.add_argument("--span", type=int, default=SPAN)
    o = p.parse_args()

    offs = [int(t) for t in o.offsets.split(",")]
    design = build(o.elems, o.span)
    src = np.arange(o.elems, dtype=np.int32)
    a = iron.tensor(src, dtype=np.int32, device="npu")
    want = src + 1000
    print(f"runtime drain offset, object {o.elems} into a {o.span}-element buffer")
    ok = True
    for off in offs:
        b = iron.zeros(o.span, dtype=np.int32, device="npu")
        design(a, b, off)
        got = b.numpy()
        hit = np.array_equal(got[off:off + o.elems], want)
        rest = got.copy()
        rest[off:off + o.elems] = 0
        clean = not rest.any()
        print(f"  offset {off:4d}: data lands {'YES' if hit else 'NO'}, "
              f"rest of buffer {'clean' if clean else 'DIRTY'}")
        ok &= hit and clean
    print(f"  -> {'runtime offsets WORK' if ok else 'runtime offsets DO NOT work'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
