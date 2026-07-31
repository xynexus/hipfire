#!/usr/bin/env python3
"""Decode attention as phase P2 of the fused layer — 8 cores, one dispatch.

`attn_verify.py` proves the attention arithmetic on ONE core. This runs it at
the phase's real shape: **8 cores, one KV group each**, in the fused layer's
paired topology (operand fifo per pair, split to two cores; result fifo per
pair, joined). llama-3.2-1B has 8 KV heads and GQA=4, so 8 cores cover all 32
query heads with **no broadcast of KV and no cross-core softmax merge** — each
core's cache is private. That is why P2 runs on 8 of the 16 cores and pairs 4-7
sit the phase out.

**KV rides the same 20544 B operand object every other phase uses**, which is
what makes one dispatch per layer legal — the topology cannot change between
phases. A KV tile is 2 x TSEQ x HEAD bf16 = 8192 B, so a lone tile would waste
61% of the object. Two fill 16384 of 20544: 20% waste on the KV stream, which
at S=2048 is **+2.8% of a layer's bytes against +16.6%** for one. `DIM_KVPER=2`
is what makes that possible; TSEQ itself cannot be doubled because the score
vector is one 32-lane register.

Padding falls out of the same mechanism `attn_verify.py` checks: whatever the
last object is short of, K=0/V=0 supplies, and `npad` subtracts its
`exp2(-m)` contribution in `flm_attn_finish`. Here npad covers the tail of the
last *object*, not just the last tile.

    python3 attn_phase.py                    # S=512
    python3 attn_phase.py --seq 2048 --bench
    python3 attn_phase.py --seq 500          # exercises the pad correction

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from attn_verify import GQA, HEAD, ROPE_INV_FREQ, TSEQ  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
SRC = str(KDIR / "flm_attn_decode.cc")
BEGIN_SRC = str(KDIR / "flm_attn_begin.cc")
FIN_SRC = str(KDIR / "flm_attn_finish.cc")
OPERAND = 20544          # the fused layer's universal operand object
KVPER = 2                # whole KV tiles per operand object
KVELEMS = 2 * TSEQ * HEAD                  # bf16 elements in one [K][V] tile
FIXED_US = 92.9

rnd = lambda x: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(x, np.float32)))


def build(ncores, nobj):
    npairs = ncores // 2
    # the operand object is the fused layer's, in bf16 elements
    kv_ty = np.ndarray[(OPERAND // 2,), np.dtype[bfloat16]]
    kvpair_ty = np.ndarray[(2 * (OPERAND // 2),), np.dtype[bfloat16]]
    kv_all_ty = np.ndarray[(2 * nobj * (OPERAND // 2),), np.dtype[bfloat16]]
    q_ty = np.ndarray[(GQA * HEAD + 2,), np.dtype[bfloat16]]     # + f32 npad
    qpair_ty = np.ndarray[(2 * (GQA * HEAD + 2),), np.dtype[bfloat16]]
    o_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]
    opair_ty = np.ndarray[(2 * GQA * HEAD,), np.dtype[bfloat16]]

    flags = [f"-DDIM_GQA={GQA}", f"-DDIM_HEAD={HEAD}", f"-DDIM_TSEQ={TSEQ}",
             f"-DDIM_KVPER={KVPER}"]
    params = ", ".join(f"q{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"kv{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design({params}):
    kb = ExternalFunction("flm_attn_begin", source_file=BEGIN_SRC,
                          arg_types=[q_ty], compile_flags=FLAGS)
    kt = ExternalFunction("flm_attn_tile", source_file=SRC,
                          arg_types=[q_ty, kv_ty], compile_flags=FLAGS)
    kf = ExternalFunction("flm_attn_finish", source_file=FIN_SRC,
                          arg_types=[o_ty, q_ty], compile_flags=FLAGS)

    f_q = [ObjectFifo(qpair_ty, name=f"q{{i}}") for i in range({npairs})]
    q_sub = [f.cons().split([0, {GQA * HEAD + 2}], obj_types=[q_ty, q_ty])
             for f in f_q]
    f_kv = [ObjectFifo(kvpair_ty, name=f"kv{{i}}") for i in range({npairs})]
    kv_sub = [f.cons().split([0, {OPERAND // 2}], obj_types=[kv_ty, kv_ty])
              for f in f_kv]
    f_o = [ObjectFifo(opair_ty, name=f"o{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {GQA * HEAD}], obj_types=[o_ty, o_ty])
             for f in f_o]

    def core(qc, kvc, op, kbegin, ktile, kfin):
        eq = qc.acquire(1)
        kbegin(eq)                              # zero the online-softmax state
        for _ in range_({nobj}):
            ekv = kvc.acquire(1)
            ktile(eq, ekv)                      # KVPER tiles per acquire
            kvc.release(1)
        eo = op.acquire(1)
        kfin(eo, eq)                            # npad rides in the Q tail
        op.release(1)
        qc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[q_sub[p][j].cons(), kv_sub[p][j].cons(),
                         o_sub[p][j].prod(), kb, kt, kf], stack_size=4096))

    def sequence(*args):
        qb = [args[i] for i in range({npairs})]
        kvb = [args[{npairs} + i] for i in range({npairs})]
        ob = [args[2 * {npairs} + i] for i in range({npairs})]
        qh = [args[3 * {npairs} + i] for i in range({npairs})]
        kvh = [args[4 * {npairs} + i] for i in range({npairs})]
        oh = [args[5 * {npairs} + i] for i in range({npairs})]
        for i in range({npairs}):
            qh[i].fill(qb[i])
        for i in range({npairs}):
            kvh[i].fill(kvb[i])
        for i in range({npairs}):
            oh[i].drain(ob[i], wait=True)

    arg_types = [qpair_ty] * {npairs} + [kv_all_ty] * {npairs} + [opair_ty] * {npairs}
    arg_types += [f.prod(tile=AnyShimTile) for f in f_q]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_kv]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_o]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, SRC=SRC, BEGIN_SRC=BEGIN_SRC,
              FIN_SRC=FIN_SRC, FLAGS=flags, q_ty=q_ty, qpair_ty=qpair_ty,
              kv_ty=kv_ty, kvpair_ty=kvpair_ty, kv_all_ty=kv_all_ty,
              o_ty=o_ty, opair_ty=opair_ty, __name__="flm_attn_phase")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[SRC, BEGIN_SRC, FIN_SRC],
                    full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--pos", type=int, default=1000)
    p.add_argument("--cores", type=int, default=8, help="one KV group per core")
    p.add_argument("--bench", action="store_true")
    o = p.parse_args()

    SEQ, ncores = o.seq, o.cores
    ntiles = -(-SEQ // TSEQ)
    nobj = -(-ntiles // KVPER)
    NPAD = nobj * KVPER * TSEQ - SEQ           # pad to whole OBJECTS now
    npairs = ncores // 2

    rng = np.random.default_rng(0)
    q = rnd(rng.standard_normal((ncores, GQA, HEAD)) * 0.3)
    Kc = rnd(rng.standard_normal((ncores, SEQ, HEAD)) * 0.3)
    Vc = rnd(rng.standard_normal((ncores, SEQ, HEAD)) * 0.3)

    # Q pre-scaled and pre-rotated on the host: phase P1 does both on device.
    pos_ang = o.pos * ROPE_INV_FREQ
    cs = rnd(np.concatenate([np.cos(pos_ang), np.sin(pos_ang)]))
    cb, sb = cs[:HEAD // 2], cs[HEAD // 2:]
    qs = rnd(q * (1.0 / math.sqrt(HEAD)) * math.log2(math.e))
    qrot = rnd(np.concatenate(
        [qs[:, :, :HEAD // 2] * cb - qs[:, :, HEAD // 2:] * sb,
         qs[:, :, HEAD // 2:] * cb + qs[:, :, :HEAD // 2] * sb], axis=2))

    design = build(ncores, nobj)

    q_ts, kv_ts, o_ts = [], [], []
    npad_u16 = np.array([NPAD], np.float32).view(np.uint16)
    for pr in range(npairs):
        qp = np.concatenate([
            np.concatenate([qrot[2 * pr + j].reshape(-1).astype(bfloat16)
                            .view(np.uint16), npad_u16]) for j in range(2)])
        q_ts.append(iron.tensor(qp.view(bfloat16), dtype=bfloat16, device="npu"))
        # one operand object per acquire: KVPER tiles then pad to 20544 B
        buf = np.zeros((2, nobj, OPERAND // 2), np.float32)
        for j in range(2):
            c = 2 * pr + j
            Kp = np.zeros((nobj * KVPER * TSEQ, HEAD), np.float32); Kp[:SEQ] = Kc[c]
            Vp = np.zeros((nobj * KVPER * TSEQ, HEAD), np.float32); Vp[:SEQ] = Vc[c]
            for t in range(nobj * KVPER):
                sl = slice(t * TSEQ, (t + 1) * TSEQ)
                base = (t % KVPER) * KVELEMS
                buf[j, t // KVPER, base:base + TSEQ * HEAD] = Kp[sl].T.reshape(-1)
                buf[j, t // KVPER, base + TSEQ * HEAD:base + KVELEMS] = Vp[sl].reshape(-1)
        # interleave the pair's objects the way the memtile split consumes them
        inter = np.empty((nobj, 2, OPERAND // 2), np.float32)
        inter[:, 0], inter[:, 1] = buf[0], buf[1]
        kv_ts.append(iron.tensor(inter.reshape(-1).astype(bfloat16),
                                 dtype=bfloat16, device="npu"))
        o_ts.append(iron.zeros(2 * GQA * HEAD, dtype=bfloat16, device="npu"))

    if o.bench:
        bench = run_iters(design, *q_ts, *kv_ts, *o_ts, warmup=2, iters=10)
        us = bench.npu.min_us if bench.npu else bench.e2e.min_us
    else:
        design(*q_ts, *kv_ts, *o_ts)
        us = None

    kv_bytes = ncores * nobj * OPERAND
    print(f"attention as phase P2: {ncores} cores x 1 KV group, GQA={GQA}, "
          f"seq={SEQ}")
    print(f"  {ntiles} KV tiles -> {nobj} operand objects of {OPERAND} B "
          f"({KVPER} tiles = {KVPER*KVELEMS*2} B used, "
          f"{100*(1-KVPER*KVELEMS*2/OPERAND):.0f}% pad), npad={NPAD}")

    worst, scale = 0.0, 0.0
    for pr in range(npairs):
        got = o_ts[pr].numpy().astype(np.float64).reshape(2, GQA, HEAD)
        for j in range(2):
            c = 2 * pr + j
            qd = q[c].astype(np.float64)
            cbd, sbd = cb.astype(np.float64), sb.astype(np.float64)
            qr = np.concatenate(
                [qd[:, :HEAD // 2] * cbd - qd[:, HEAD // 2:] * sbd,
                 qd[:, HEAD // 2:] * cbd + qd[:, :HEAD // 2] * sbd], axis=1)
            sc = (qr @ Kc[c].astype(np.float64).T) / math.sqrt(HEAD)
            e = np.exp(sc - sc.max(1, keepdims=True))
            ref = (e / e.sum(1, keepdims=True)) @ Vc[c].astype(np.float64)
            worst = max(worst, np.abs(got[j] - ref).max())
            scale = max(scale, np.abs(ref).mean())

    if us:
        print(f"  {kv_bytes/1e6:.2f} MB KV  {kv_bytes/(us*1e-6)/1e9:.1f} GB/s  "
              f"{us:.1f} us (marginal {us-FIXED_US:.1f})")
    # the floor is AIE2P's hardware exp2 (3.54% mean / 5.86% max), not bf16
    tol = 8e-2 * scale
    print(f"  max err vs float64 {worst:.4e}   mean|ref| {scale:.5f}")
    print(f"  tolerance {tol:.4e} (exp2 NLF floor) -> "
          f"{'PASS' if worst <= tol else 'FAIL'}")
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
