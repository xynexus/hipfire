#!/usr/bin/env python3
"""Can ONE dispatch stream a whole model's weights at FLM's rate? Phase 2 milestone 1.

FLM decodes llama3.2:1b with **2 commands per token**, streaming 772.3 MB/s worth
of weights at **46.2 GB/s** (`docs/npu/flm-layer-dataflow.md`). hipfire's own
decode path issues ~96 dispatches per token and delivers ~10 GB/s effective
(`docs/npu/decoder-layer-npu-scope.md`). That gap is the phase-2 problem, and it
is a *dispatch-structure* problem before it is a kernel problem — so it is worth
measuring on its own, before any GEMM arithmetic exists to confound it.

This probe strips the question to its bones: bind N host buffers to ONE dispatch,
stream every byte into the array, and report the achieved rate. The cores acquire
and release without reading — the DMA moves the bytes either way, and what is
being measured is the delivery structure, not the arithmetic.

`manybuf_probe.py` established the two requirements
(`docs/npu/flm-refe-log.md`): raise `aiecc`'s `kMaxHostBOs` (done, 16 -> 64), and
**reuse BDs instead of allocating one per buffer**, since a shim tile supports
only 16 simultaneously active. This probe does the second with IRON `TaskGroup`s:
each group of `--group` fills is awaited and freed before the next opens, so
active BDs stay bounded no matter how many buffers are bound.

    python3 dispatch_bw_probe.py                       # 50 buffers, FLM's count
    python3 dispatch_bw_probe.py --sweep-workers       # find the worker count that feeds

Reference points this prints against:
  46.2 GB/s   FLM decode, measured end to end
  56.5 GB/s   fabric roof, 8 columns (npu-memory-bandwidth-cache-characterization.md)
  14.4 GB/s   ONE compute-tile receive stream while active -- so >=4 concurrent
              streams are needed to pass FLM, which is why FLM runs 16.

Needs PYTHONPATH=<mlir-aie>/build/python and the Peano/XRT env, same as the
other tools here.
"""

import argparse
import sys

import numpy as np

FLM_DECODE_GBS = 46.2
FABRIC_ROOF_GBS = 56.5


def build_and_run(nbufs, buf_kb, workers, tile_kb, group, warmup, iters,
                  verify=False):
    """Stream nbufs buffers through `workers` parallel fifos in ONE dispatch.

    With ``verify``, one extra buffer rides the same dispatch through a
    forwarding fifo and is checked byte for byte on the way out. Without it the
    cores never read what arrives, so a design whose DMAs silently did nothing
    would report a spectacular bandwidth -- which is the exact failure mode
    `docs/npu/flm-refe-log.md` keeps recording ("suspect clean-looking results").
    """
    import aie.iron as iron
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyShimTile
    from aie.iron.worker import Worker
    from aie.utils.benchmark import run_iters

    if nbufs % workers:
        raise ValueError(f"--bufs {nbufs} must divide by --workers {workers}")
    if buf_kb % tile_kb:
        raise ValueError(f"--buf-kb {buf_kb} must divide by --tile-kb {tile_kb}")

    buf_elems = buf_kb * 1024 // 4
    tile_elems = tile_kb * 1024 // 4
    per_worker = nbufs // workers
    tiles_per_worker = per_worker * (buf_elems // tile_elems)
    # Host buffers come first in the arg list, fifo handles after, so the
    # handles' offset moves when --verify adds its two.
    nhost = nbufs + (2 if verify else 0)

    # iron.jit binds host buffers to *named* parameters, so the design is
    # generated -- the whole point is that N is large.
    params = ", ".join(f"b{i}: In" for i in range(nbufs))
    if verify:
        params += ", vin: In, vout: Out"
    src = f'''
def _design({params}, *, tile_elems: CompileTime[int]):
    buf_ty = np.ndarray[({buf_elems},), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_elems,), np.dtype[np.int32]]
    check_ty = np.ndarray[({tile_elems},), np.dtype[np.int32]]

    fifos = [ObjectFifo(tile_ty, name=f"feed{{w}}") for w in range({workers})]
    # One tile rides the same dispatch and comes back out, so the run proves
    # bytes really moved rather than only that time elapsed.
    f_chk = ObjectFifo(check_ty, name="check") if {verify} else None
    f_chk_out = f_chk.cons().forward() if {verify} else None

    # The core acquires and releases without touching the data. The DMA has
    # already moved it into L1 by then, so the byte rate is real; adding a read
    # would measure the core's load unit instead of the delivery path.
    def core_body(cons):
        for _ in range_({tiles_per_worker}):
            cons.acquire(1)
            cons.release(1)

    ws = [Worker(core_body, fn_args=[f.cons()]) for f in fifos]

    def sequence(*args):
        # Indexed, not sliced: on Python 3.14 a constant slice folds into
        # co_consts, and mlir-aie's jit cache hashes the generator with
        # marshal.dumps(code, 4), which cannot serialize slice objects. The
        # symptom is a bare "ValueError: unmarshallable object" at compile time.
        bufs = [args[i] for i in range({nbufs})]
        handles = [args[{nhost} + i] for i in range({workers})]
        # BD REUSE: each group is awaited and freed before the next opens, so
        # active BDs per shim tile stay bounded however many buffers are bound.
        # Without this the design fails to compile past 16 active descriptors.
        for start in range(0, {nbufs}, {group}):
            tg = TaskGroup()
            for i in range(start, min(start + {group}, {nbufs})):
                handles[i % {workers}].fill(bufs[i], wait=True, group=tg)
            tg.finish()
        if {verify}:
            # Its own group: IRON forbids mixing explicit groups with the
            # implicit default one, and every fill above is already grouped.
            vtg = TaskGroup()
            args[{nhost} + {workers}].fill(args[{nbufs}], group=vtg)
            args[{nhost} + {workers} + 1].drain(args[{nbufs} + 1], wait=True, group=vtg)
            vtg.finish()

    arg_types = [buf_ty] * {nbufs}
    if {verify}:
        arg_types += [check_ty, check_ty]
    arg_types += [f.prod(tile=AnyShimTile) for f in fifos]
    if {verify}:
        arg_types += [f_chk.prod(tile=AnyShimTile), f_chk_out.cons(tile=AnyShimTile)]

    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, ws).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_, TaskGroup=TaskGroup,
              # exec()'d functions get __module__ = None, which mlir-aie's jit
              # cache hashes with .encode() and dies on. Same trap as
              # manybuf_probe.py.
              __name__="dispatch_bw_probe")
    exec(src, ns)
    design = iron.jit(ns["_design"])

    bufs = [iron.zeros(buf_elems, dtype=np.int32, device="npu") for _ in range(nbufs)]
    extra = []
    if verify:
        extra = [iron.arange(1, tile_elems + 1, dtype=np.int32, device="npu"),
                 iron.zeros(tile_elems, dtype=np.int32, device="npu")]
    bench = run_iters(design, *bufs, *extra, tile_elems=tile_elems,
                      warmup=warmup, iters=iters)

    if verify and not np.array_equal(extra[1].numpy(), extra[0].numpy()):
        raise RuntimeError("verification tile did not survive the dispatch — "
                           "the measured rate is not moving real bytes")

    total_bytes = nbufs * buf_kb * 1024
    npu = bench.npu
    us = npu.min_us if npu else bench.e2e.min_us
    return total_bytes / (us * 1e-6) / 1e9, us, bench


def main():
    p = argparse.ArgumentParser(description="One-dispatch weight streaming rate")
    p.add_argument("--bufs", type=int, default=50, help="host buffers on ONE dispatch")
    p.add_argument("--buf-kb", type=int, default=1024, help="KiB per buffer")
    p.add_argument("--workers", type=int, default=8, help="parallel feed streams")
    p.add_argument("--tile-kb", type=int, default=16, help="KiB per ObjectFifo tile")
    p.add_argument("--group", type=int, default=8, help="fills per TaskGroup (BD reuse)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--verify", action="store_true",
                   help="ride one tile through the same dispatch and check it")
    p.add_argument("--sweep-workers", default=None,
                   help="comma-separated worker counts to compare, e.g. 1,2,4,8,16")
    o = p.parse_args()

    points = ([int(x) for x in o.sweep_workers.split(",")]
              if o.sweep_workers else [o.workers])

    print(f"{o.bufs} buffers x {o.buf_kb} KiB = "
          f"{o.bufs * o.buf_kb / 1024:.0f} MiB on ONE dispatch\n")
    print(f"{'workers':>7s} {'GB/s':>8s} {'us':>10s}  vs FLM 46.2")
    print("-" * 44)
    best = 0.0
    for w in points:
        try:
            gbs, us, _ = build_and_run(o.bufs, o.buf_kb, w, o.tile_kb,
                                       o.group, o.warmup, o.iters, o.verify)
        except Exception as e:
            first = str(e).strip().splitlines()[0][:70] if str(e).strip() else type(e).__name__
            print(f"{w:7d} {'FAIL':>8s} {'':>10s}  {first}")
            continue
        print(f"{w:7d} {gbs:8.1f} {us:10.1f}  {gbs / FLM_DECODE_GBS:.2f}x")
        best = max(best, gbs)

    print(f"\nbest {best:.1f} GB/s = {best / FLM_DECODE_GBS:.2f}x FLM decode, "
          f"{100 * best / FABRIC_ROOF_GBS:.0f}% of the {FABRIC_ROOF_GBS} GB/s fabric roof")
    sys.exit(0 if best >= FLM_DECODE_GBS else 1)


if __name__ == "__main__":
    main()
