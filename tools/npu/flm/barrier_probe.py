#!/usr/bin/env python3
"""What does a phase barrier INSIDE one dispatch cost? (fused-layer falsifier #0)

`docs/npu/flm-fused-layer-plan.md` proposes running a whole decoder layer as one
dispatch with 5 internal phase barriers, replacing 5 separate dispatches. That
trade is only a win if an in-sequence barrier is cheaper than the **92.9 us**
per-dispatch cost measured in `docs/npu/flm-refe-log.md`. That number does not
exist yet, and the entire plan rests on it.

The probe strips the question to its bones: same 16-core paired topology as
`gemv_bench.py`, a no-op kernel, and N sequential `fill` / `drain(wait=True)`
round trips in ONE runtime sequence. Fit `time = a + b*N`; `b` is the barrier.

    GATE: b < 93 us  -> fusing phases beats separate dispatches
          b >= 93 us -> the fused layer is not worth building; keep one dispatch
                        per phase and pursue the 16-layer unroll instead

    python3 barrier_probe.py
    python3 barrier_probe.py --cores 8 --sweep 1,5,20,80

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      TaskGroup, Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

SRC = Path(__file__).parent / "_barrier_noop.cc"
SRC.write_text("""// no-op body: this probe measures the dataflow, not arithmetic
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" __attribute__((noinline)) void
barrier_noop(const bfloat16 *restrict in, bfloat16 *restrict out) {
  out[0] = in[0];
}
""")
BCAST = 8448 // 2      # bf16 elements, the plan's broadcast object
RESULT = 64            # bf16 elements per core result


def build(ncores, nphases):
    """N phases, each a broadcast acquire + a result produce, in ONE dispatch."""
    npairs = ncores // 2
    b_ty = np.ndarray[(BCAST,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(RESULT,), np.dtype[bfloat16]]
    opair_ty = np.ndarray[(2 * RESULT,), np.dtype[bfloat16]]
    # ONE phase's worth: each barrier drains this same buffer. Sizing it for
    # all phases makes the drain wait for more data than the fifo produces per
    # phase, and the dispatch hangs (ERT_CMD_STATE_TIMEOUT).
    o_all_ty = np.ndarray[(2 * RESULT,), np.dtype[bfloat16]]

    params = ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(b: In, {params}):
    kern = ExternalFunction("barrier_noop", source_file=str(SRC),
                            arg_types=[b_ty, o_ty])
    f_b = ObjectFifo(b_ty, name="bc")
    b_cons = [f_b.cons() for _ in range({ncores})]
    f_op = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {RESULT}], obj_types=[o_ty, o_ty]) for f in f_op]

    def core(bc, op, k):
        # one acquire/release pair per phase: this is the barrier under test
        for _ in range_({nphases}):
            eb = bc.acquire(1)
            eo = op.acquire(1)
            k(eb, eo)
            op.release(1)
            bc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[b_cons[2 * p + j], o_sub[p][j].prod(), kern],
                stack_size=4096))

    def sequence(*args):
        bb = args[0]
        ob = [args[1 + i] for i in range({npairs})]
        bh = args[1 + {npairs}]
        oh = [args[2 + {npairs} + i] for i in range({npairs})]
        # N barriers: each phase refills the broadcast and waits for results.
        # Each phase is its own TaskGroup so its BDs are freed before the next
        # opens — without that, N phases x npairs fills/drains exceed the 16
        # simultaneously-active BDs a shim tile supports and the design fails to
        # compile. IRON also forbids mixing explicit groups with the implicit
        # default one, so every fill and drain here is grouped.
        for _ in range({nphases}):
            tg = TaskGroup()
            bh.fill(bb, group=tg)
            for i in range({npairs}):
                oh[i].drain(ob[i], wait=True, group=tg)
            tg.finish()

    arg_types = [b_ty] + [o_all_ty] * {npairs}
    arg_types += [f_b.prod(tile=AnyShimTile)]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_op]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, SRC=SRC, TaskGroup=TaskGroup,
              b_ty=b_ty, o_ty=o_ty, opair_ty=opair_ty, o_all_ty=o_all_ty,
              __name__="flm_barrier_probe")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[str(SRC)], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--sweep", default="1,5,20,80")
    p.add_argument("--iters", type=int, default=10)
    o = p.parse_args()
    points = [int(x) for x in o.sweep.split(",")]

    print(f"barrier probe: {o.cores} cores, broadcast {BCAST*2} B, "
          f"phases per dispatch {points}\n")
    print(f"{'phases':>7s} {'us':>10s} {'us/phase':>10s}")
    print("-" * 30)
    xs, ys = [], []
    for n in points:
        design = build(o.cores, n)
        b_t = iron.zeros(BCAST, dtype=bfloat16, device="npu")
        o_ts = [iron.zeros(2 * RESULT, dtype=bfloat16, device="npu")
                for _ in range(o.cores // 2)]
        bench = run_iters(design, b_t, *o_ts, warmup=2, iters=o.iters)
        npu = bench.npu
        us = npu.min_us if npu else bench.e2e.min_us
        print(f"{n:7d} {us:10.1f} {us/n:10.1f}")
        xs.append(n); ys.append(us)

    A = np.vstack([np.ones(len(xs)), np.array(xs, float)]).T
    fixed, per = np.linalg.lstsq(A, np.array(ys, float), rcond=None)[0]
    pred = A @ np.array([fixed, per])
    r2 = 1 - ((np.array(ys) - pred) ** 2).sum() / max(
        ((np.array(ys) - np.mean(ys)) ** 2).sum(), 1e-30)
    print(f"\nfit: time_us = {fixed:.1f} + {per:.2f} * phases   (R^2 = {r2:.5f})")
    print(f"  in-dispatch barrier   = {per:.2f} us")
    print(f"  per-dispatch cost     = 92.90 us  (docs/npu/flm-refe-log.md)")
    ok = per < 92.9
    print(f"  -> {'PASS' if ok else 'FAIL'}: fusing phases is "
          f"{'CHEAPER' if ok else 'NOT cheaper'} than separate dispatches"
          f"  ({92.9/per:.1f}x)" if per > 0 else "")

    if ok:
        # These assume every byte moves at the 57.0 GB/s fabric roof and ignore
        # the KV cache, so they are an UPPER BOUND, not a forecast. The measured
        # projection, built from the phases' real rates (89-97% of ceiling) and
        # real KV traffic, is 59.7 tok/s at S=512 — see docs/npu/flm-refe-log.md,
        # "projection from measured phases". Quote that one, not this one.
        B, RATE, L = 772.3, 57.0, 16
        for nb, nd, lab in ((5 * L, L + 1, "fused layer: 17 dispatches, 80 barriers"),
                            (0, 5 * L + 1, "today: 81 dispatches, 0 barriers")):
            t = B / RATE + nd * 92.9 / 1000 + nb * per / 1000
            print(f"  {lab:42s} {t:6.2f} ms -> {1000/t:5.1f} tok/s "
                  f"= {(1000/t)/59.86:.2f}x FLM   [upper bound]")
        print("  (upper bound: fabric roof, no KV. Measured: 59.7 tok/s at S=512.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
