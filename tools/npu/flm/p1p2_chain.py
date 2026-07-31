#!/usr/bin/env python3
"""P1 -> P2 chained in one dispatch: qkv+RoPE feeding attention over the cache.

The first half of a decoder layer, end to end on device:

    P1  16 cores  norm + qkv + RoPE -> q', and k'/v' appended to the KV cache
    P2   8 cores  attention over the cache INCLUDING the token P1 just appended

`p1_route.py` verifies P1's three destinations and `attn_phase.py` verifies
attention at its phase shape; this is the seam between them, which is where the
constraints live.

**Cores 0-7 run both phases, cores 8-15 only P1** — two Worker bodies in one
design. 8 KV heads at GQA=4 is exactly 8 attention cores.

Four things the seam forced, each measured rather than assumed:

  * **q' rides the operand fifo**, not the broadcast. A core's DMA input
    channels are allocated over the union of every fifo it consumes, not per
    phase, and broadcast+operand already spends both. P2's first operand acquire
    is q'; the rest are KV tiles.
  * **The operand fifo is `uint8`.** One fifo carries q4_1 tiles and q'/KV, a
    fifo has one object type, and IRON requires the kernel arg type to match it
    exactly. Attention casts on entry.
  * **q' is strided.** P1's result object is 2*HEAD per head and a drain cannot
    skip source elements, so the query block arrives with 128 elements per head.
    `-DDIM_QSTRIDE` lets attention read it in place.
  * **P2 gets its own result fifo.** P1 emits 128-element objects and P2 emits
    256; one fifo cannot do both. A core has two output DMA channels and P1 uses
    one, so this is free — 12 of 16 shim outputs.

    python3 p1p2_chain.py                 # S=32, one KV tile
    python3 p1p2_chain.py --seq 64

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from qkv_verify import HEAD, K_DIM, NROWS, TPH, EPS, ROPE_THETA, qkv_rows, rope_ref  # noqa: E402
from p1_route import NQ, NK, NV, NCORES, HPC, heads_of, rnd  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
QKV_SRC = str(KDIR / "flm_gemv_qkv.cc")
EMIT_SRC = str(KDIR / "flm_p1_emit.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
ATT_SRC = str(KDIR / "flm_attn_decode.cc")
BEG_SRC = str(KDIR / "flm_attn_begin.cc")
FIN_SRC = str(KDIR / "flm_attn_finish.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
TSEQ, GQA, KVPER = 32, 4, 2
OPERAND = 20544
NATT = 8                       # attention cores = KV heads


def build(pos, nobj):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = NCORES // 2
    apairs = NATT // 2
    BC = 2 * K_DIM + 2 * HEAD
    OBJ = 2 * HEAD                                  # P1 result object, bf16
    KTILE, VTILE = HEAD * TSEQ, TSEQ * HEAD
    KVSTRIDE = KTILE + VTILE
    off = pos - (pos & 1)

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    op_ty = np.ndarray[(OPERAND,), np.dtype[np.uint8]]      # ONE operand type
    oppair_ty = np.ndarray[(2 * OPERAND,), np.dtype[np.uint8]]
    p1o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    p1opair_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    p2o_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]
    p2opair_ty = np.ndarray[(2 * GQA * HEAD,), np.dtype[bfloat16]]
    # P1's weights, then P2's q'+KV objects, on the same fifo
    w_all_ty = np.ndarray[(2 * HPC * TPH * wt,), np.dtype[np.uint8]]
    kvin_ty = np.ndarray[(2 * (1 + nobj) * OPERAND,), np.dtype[np.uint8]]
    q_ty = np.ndarray[(4 * OBJ,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(NATT * KVSTRIDE,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS={NQ}", f"-DDIM_QKHEADS={NK}",
             f"-DDIM_GQA={GQA}", f"-DDIM_TSEQ={TSEQ}", f"-DDIM_KVPER={KVPER}",
             f"-DDIM_QSTRIDE={OBJ}"]
    P = ", ".join(f"w{i}: In" for i in range(npairs))
    P += ", " + ", ".join(f"kvin{i}: In" for i in range(apairs))
    P += ", " + ", ".join(f"q{i}: Out" for i in range(npairs))
    P += ", " + ", ".join(f"cache{i}: Out" for i in range(npairs))
    P += ", " + ", ".join(f"attn{i}: Out" for i in range(apairs))
    src = f'''
def _design(bc: In, {P}):
    kq = ExternalFunction("flm_gemv_qkv", source_file=QKV_SRC,
                          arg_types=[bc_ty, op_ty], compile_flags=FLAGS)
    ke = ExternalFunction("flm_p1_emit", source_file=EMIT_SRC,
                          arg_types=[op_ty, p1o_ty], compile_flags=FLAGS)
    kn = ExternalFunction("flm_norm_prepare", source_file=NORM_SRC,
                          arg_types=[bc_ty], compile_flags=FLAGS)
    kab = ExternalFunction("flm_attn_begin", source_file=BEG_SRC,
                           arg_types=[op_ty], compile_flags=FLAGS)
    kat = ExternalFunction("flm_attn_tile", source_file=ATT_SRC,
                           arg_types=[op_ty, op_ty], compile_flags=FLAGS)
    kaf = ExternalFunction("flm_attn_finish", source_file=FIN_SRC,
                           arg_types=[p2o_ty, op_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name="bc")
    bc_cons = [f_bc.cons() for _ in range({NCORES})]
    f_w = [ObjectFifo(oppair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {OPERAND}], obj_types=[op_ty, op_ty]) for f in f_w]
    f_p1 = [ObjectFifo(p1opair_ty, name=f"p1o{{i}}") for i in range({npairs})]
    p1_sub = [f.prod().join([0, {OBJ}], obj_types=[p1o_ty, p1o_ty]) for f in f_p1]
    # P2's own result fifo — P1 emits {OBJ} bf16 objects and P2 emits
    # {GQA * HEAD}; one fifo cannot carry both sizes. Only pairs 0..{apairs - 1}
    # need it, and a core has 2 output channels.
    f_p2 = [ObjectFifo(p2opair_ty, name=f"p2o{{i}}") for i in range({apairs})]
    p2_sub = [f.prod().join([0, {GQA * HEAD}], obj_types=[p2o_ty, p2o_ty])
              for f in f_p2]

    def p1_body(bcc, wc, op, kqkv, kemit, kprep):
        eb = bcc.acquire(1)
        kprep(eb)
        for _ in range_({HPC}):
            for _ in range_({TPH} - 1):
                ew = wc.acquire(1)
                kqkv(eb, ew)
                wc.release(1)
            ew = wc.acquire(1)
            eo = op.acquire(1)
            kqkv(eb, ew)
            kemit(ew, eo)          # reuses the head's last tile for row_base
            op.release(1)
            wc.release(1)
        bcc.release(1)

    def core_p1(bcc, wc, op, kqkv, kemit, kprep):
        p1_body(bcc, wc, op, kqkv, kemit, kprep)

    def core_p1p2(bcc, wc, op, ap, kqkv, kemit, kprep, kbeg, ktile, kfin):
        p1_body(bcc, wc, op, kqkv, kemit, kprep)
        # ---- P2: first operand object is q', the rest are KV tiles ----
        eq = wc.acquire(1)
        kbeg(eq)
        for _ in range_({nobj}):
            ekv = wc.acquire(1)
            ktile(eq, ekv)
            wc.release(1)
        eo = ap.acquire(1)
        kfin(eo, eq)
        ap.release(1)
        wc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            c = 2 * p + j
            if p < {apairs}:
                workers.append(Worker(core_p1p2,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             p2_sub[p][j].prod(), kq, ke, kn, kab, kat, kaf],
                    stack_size=8192))
            else:
                workers.append(Worker(core_p1,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             kq, ke, kn], stack_size=8192))

    def sequence(*args):
        n, a = {npairs}, {apairs}
        bcb = args[0]
        wb = [args[1 + i] for i in range(n)]
        kvb = [args[1 + n + i] for i in range(a)]
        qb = [args[1 + n + a + i] for i in range(n)]
        cb = [args[1 + 2 * n + a + i] for i in range(n)]
        ab = [args[1 + 3 * n + a + i] for i in range(a)]
        base = 1 + 3 * n + 2 * a
        bch = args[base]
        wh = [args[base + 1 + i] for i in range(n)]
        p1h = [args[base + 1 + n + i] for i in range(n)]
        p2h = [args[base + 1 + 2 * n + i] for i in range(a)]

        tg = TaskGroup()
        bch.fill(bcb, group=tg)
        for i in range(n):
            wh[i].fill(wb[i], group=tg)
        for i in range(n):
            p1h[i].drain(qb[i], wait=True, group=tg,
                         sizes=[1, 1, 1, 4 * {OBJ}], strides=[0, 0, 0, 1])
            if i < n // 2:
                p1h[i].drain(cb[i], wait=True, group=tg,
                             offset=2 * i * {KVSTRIDE} + {off},
                             sizes=[1, 2, {HEAD}, 2],
                             strides=[0, {KVSTRIDE}, {TSEQ}, 1])
            else:
                p1h[i].drain(cb[i], wait=True, group=tg,
                             offset=2 * (i - n // 2) * {KVSTRIDE} + {KTILE}
                                    + {pos} * {HEAD},
                             sizes=[1, 2, 1, {OBJ}],
                             strides=[0, {KVSTRIDE}, 0, 1])
        tg.finish()

        # ---- P2 ----------------------------------------------------------
        tg = TaskGroup()
        for i in range(a):
            wh[i].fill(kvb[i], group=tg)
        for i in range(a):
            p2h[i].drain(ab[i], wait=True, group=tg)
        tg.finish()

    at = [bc_ty] + [w_all_ty] * {npairs} + [kvin_ty] * {apairs}
    at += [q_ty] * {npairs} + [cache_ty] * {npairs} + [p2opair_ty] * {apairs}
    at += [f_bc.prod(tile=AnyShimTile)]
    at += [f.prod(tile=AnyShimTile) for f in f_w]
    at += [f.cons(tile=AnyShimTile) for f in f_p1]
    at += [f.cons(tile=AnyShimTile) for f in f_p2]
    rt = Runtime(sequence, at)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, QKV_SRC=QKV_SRC,
              EMIT_SRC=EMIT_SRC, NORM_SRC=NORM_SRC, ATT_SRC=ATT_SRC,
              BEG_SRC=BEG_SRC, FIN_SRC=FIN_SRC, FLAGS=flags, bc_ty=bc_ty,
              op_ty=op_ty, oppair_ty=oppair_ty, p1o_ty=p1o_ty,
              p1opair_ty=p1opair_ty, p2o_ty=p2o_ty, p2opair_ty=p2opair_ty,
              w_all_ty=w_all_ty, kvin_ty=kvin_ty, q_ty=q_ty,
              cache_ty=cache_ty, __name__="flm_p1p2")
    exec(src, ns)
    return iron.jit(ns["_design"],
                    source_files=[QKV_SRC, EMIT_SRC, NORM_SRC, ATT_SRC,
                                  BEG_SRC, FIN_SRC], full_elf=True), wt, KVSTRIDE


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=32,
                   help="cache length INCLUDING the token P1 appends")
    o = p.parse_args()
    pos = o.seq - 1                       # P1 appends at the end
    ntiles = -(-o.seq // TSEQ)
    nobj = -(-ntiles // KVPER)
    npad = nobj * KVPER * TSEQ - o.seq
    npairs, apairs = NCORES // 2, NATT // 2

    c = q4nx.Q4nx(str(Q4NX))
    nw = c.bf16(f"model.layers.{o.layer}.input_layernorm.weight").astype(np.float32)[:K_DIM]
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)[:HEAD // 2]
    inv_freq = (1.0 / ROPE_THETA ** (np.arange(0, HEAD, 2) / HEAD)) / divisor
    ang = pos * inv_freq
    cs_k = rnd(np.concatenate([np.cos(ang), np.sin(ang)]))
    cs_q = rnd(cs_k * (HEAD ** -0.5) * np.log2(np.e))

    design, wt, KVSTRIDE = build(pos, nobj)
    OBJ = 2 * HEAD
    KTILE = HEAD * TSEQ

    rng = np.random.default_rng(0)
    x = rnd(rng.standard_normal(K_DIM) * 0.05)
    bc = np.zeros(2 * K_DIM + 2 * HEAD, np.float32)
    bc[:K_DIM] = x
    bc[K_DIM:2 * K_DIM] = nw
    bc[2 * K_DIM:2 * K_DIM + HEAD] = cs_q
    bc[2 * K_DIM + HEAD:] = cs_k
    xd = x.astype(np.float64)
    inv = np.float32(1.0 / np.sqrt((xd * xd).mean() + EPS))
    xn = rnd(rnd(x * rnd(inv)) * nw)

    w_ts, ref = [], {}
    for pr in range(npairs):
        per = []
        for j in range(2):
            blob = []
            for h in heads_of(2 * pr + j):
                first = h * HEAD
                d, m, q = qkv_rows(c, o.layer, first, HEAD)
                blob.append(np.concatenate([
                    q4nx.pack_tile(d[i:i+NROWS], m[i:i+NROWS], q[i:i+NROWS],
                                   row_base=first + i, flags=float(pos))
                    for i in range(0, HEAD, NROWS)]))
                v = rnd(np.concatenate([
                    q4nx.gemv_reference_bf16(xn, d[i:i+NROWS], m[i:i+NROWS],
                                             q[i:i+NROWS])
                    for i in range(0, HEAD, NROWS)]))
                if h < NQ:
                    v = rope_ref(v, cs_q)
                elif h < NK:
                    v = rope_ref(v, cs_k)
                ref[h] = rnd(v).astype(np.float64)
            per.append(np.concatenate(blob))
        b = np.empty((HPC * TPH, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w_ts.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))

    # prior cache contents for positions 0..pos-1, and the q'/KV operand stream
    Kc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    Vc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    cache = np.zeros((NATT, KVSTRIDE), np.float32)
    for g in range(NATT):
        K = cache[g, :KTILE].reshape(HEAD, TSEQ)
        V = cache[g, KTILE:].reshape(TSEQ, HEAD)
        if pos:
            K[:, :pos] = Kc[g].T
            V[:pos] = Vc[g]
    cache_t = iron.tensor(cache.reshape(-1).astype(bfloat16), dtype=bfloat16,
                          device="npu")

    # P2's operand stream per pair: [q' object][KV objects]
    kv_ts = []
    for pr in range(apairs):
        stream = np.zeros((1 + nobj, 2, OPERAND // 2), np.float32)
        for j in range(2):
            a = 2 * pr + j                    # attention core == KV head
            qh = np.zeros(OPERAND // 2, np.float32)
            for s in range(GQA):
                qh[s * OBJ:s * OBJ + HEAD] = ref[4 * a + s][:HEAD]
            qh[GQA * OBJ:GQA * OBJ + 2] = np.array([float(npad)],
                                                   np.float32).view(np.float32)
            stream[0, j] = qh
        kv_ts.append(stream)                 # KV objects filled after the run
    q_ts = [iron.zeros(4 * OBJ, dtype=bfloat16, device="npu") for _ in range(npairs)]

    print(f"P1 -> P2 in one dispatch: seq {o.seq} (P1 appends at pos {pos}), "
          f"{nobj} KV objects, npad {npad}")
    print(f"  design placed and compiled: {NCORES} cores, "
          f"{NATT} running both phases")
    print(f"  channels: 2 in/core (broadcast + operand), 2 out/core "
          f"(P1 + P2 results), {npairs + apairs} of 16 shim outputs")
    print("  NOT YET RUN: P2's KV operand must be FILLED from the cache BO that")
    print("  P1 drains into — a strided gather, the mirror of the drain")
    print("  patterns. That is the remaining piece.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
