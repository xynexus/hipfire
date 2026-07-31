#!/usr/bin/env python3
"""Can one NPU dispatch bind many host buffers? Phase 2 groundwork.

FLM's decode command binds **50** buffers to a single dispatch, against an
xclbin declaring the ordinary 5-buffer DPU signature. That works because the
`args` array of `amdxdna_drm_exec_cmd` is a driver-level buffer table indexed by
`DDR_PATCH`'s `arg_idx`, not the kernel signature — and neither limit is tight
(driver `MAX_ARG_COUNT` = 4095; `aiex.npu.address_patch`'s `arg_idx` is I32Attr).

That establishes the *mechanism*. This probe tests whether mlir-aie's host
helpers and IRON's runtime plumbing actually pass that many BOs, which is the
part no document promises.

Each of N input buffers is forwarded to its own output buffer through its own
ObjectFifo, so **2N buffers are bound and every one is individually verified** —
a buffer patched to the wrong address shows up as a mismatch, not as a crash.

    python3 manybuf_probe.py            # sweep until it breaks
    python3 manybuf_probe.py -n 25      # one point (25 pairs = 50 buffers)

Needs PYTHONPATH=<mlir-aie>/build/python and the Peano/XRT env, same as the
other tools here.
"""

import argparse
import sys

import numpy as np

LINE = 1024          # elements per transfer; also the buffer length


def build_and_run(npairs, verbose=False):
    """Forward npairs inputs to npairs outputs in ONE dispatch. Returns (ok, msg)."""
    import aie.iron as iron
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
    from aie.iron.device import AnyShimTile

    # iron.jit needs real named parameters, so the design is generated. Params
    # are in0..in{N-1}, out0..out{N-1} — 2N host buffers on one dispatch.
    params = ", ".join(f"in{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"out{i}: Out" for i in range(npairs))
    # ONE ObjectFifo, reused for all N transfers. An earlier version gave each
    # pair its own fifo, which binds 2N buffers but also demands 2N shim DMA
    # channels — 8 shim tiles supply only 16 per direction, so it died at 32
    # buffers with ERT_CMD_STATE_ERROR. That measured channel exhaustion, not
    # buffer binding. FLM binds 50 buffers through a handful of channels used
    # sequentially, which is what this now models.
    src = f'''
def _design({params}, *, n: CompileTime[int]):
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]

    f_in = ObjectFifo(line_ty, name="shared_in")
    f_out = f_in.cons().forward()

    def sequence(*args):
        bufs = args[:2 * {npairs}]
        in_h, out_h = args[2 * {npairs}], args[2 * {npairs} + 1]
        for i in range({npairs}):
            in_h.fill(bufs[i])
            out_h.drain(bufs[{npairs} + i], wait=True)

    arg_types = [vector_ty] * (2 * {npairs})
    arg_types.append(f_in.prod(tile=AnyShimTile))
    arg_types.append(f_out.cons(tile=AnyShimTile))

    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              AnyShimTile=AnyShimTile, LINE=LINE,
              # A function created by exec() in a namespace without __name__ gets
              # __module__ = None, and mlir-aie's jit cache hashes it with
              # `getattr(generator, "__module__", "").encode()` — the default does
              # not apply when the attribute exists and is None, so it raises
              # AttributeError. Give the namespace a __name__.
              __name__="manybuf_probe")
    exec(src, ns)
    design = iron.jit(ns["_design"])

    ins = [iron.arange(i * LINE + 1, (i + 1) * LINE + 1, dtype=np.int32, device="npu")
           for i in range(npairs)]
    outs = [iron.zeros_like(ins[0]) for _ in range(npairs)]

    design(*ins, *outs, n=LINE)

    # Every buffer verified separately: a mis-patched address shows as a
    # mismatch on that specific buffer, which is the whole point of the probe.
    bad = []
    for i in range(npairs):
        got, want = outs[i].numpy(), ins[i].numpy()
        if not np.array_equal(got, want):
            bad.append(i)
    if bad:
        return False, f"{len(bad)}/{npairs} buffers wrong (first: {bad[:5]})"
    return True, f"{2 * npairs} buffers bound and verified"


def main():
    p = argparse.ArgumentParser(description="Probe max host buffers per dispatch")
    p.add_argument("-n", "--pairs", type=int, default=None,
                   help="one measurement at this many in/out pairs")
    p.add_argument("--sweep", default="2,4,8,16,25,32",
                   help="pair counts to try in order (default sweeps to 32 = 64 buffers)")
    o = p.parse_args()

    points = [o.pairs] if o.pairs else [int(x) for x in o.sweep.split(",")]
    last_ok = 0
    for npairs in points:
        try:
            ok, msg = build_and_run(npairs)
        except Exception as e:
            first = str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__
            print(f"{npairs:3d} pairs ({2*npairs:3d} buffers): FAIL  {first}")
            break
        print(f"{npairs:3d} pairs ({2*npairs:3d} buffers): {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            break
        last_ok = npairs
    print(f"\nmax verified: {last_ok} pairs = {2 * last_ok} buffers on one dispatch")
    print("FLM's decode command binds 50.")
    sys.exit(0 if last_ok * 2 >= 50 else 1)


if __name__ == "__main__":
    main()
