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

  * **q' rides the BROADCAST fifo**, not the operand fifo. It was the other way
    round originally, on the reasoning that a core's DMA input channels are
    allocated over the union of every fifo it consumes, so broadcast+operand
    already spends both. But an object held across other acquire/release cycles
    on the SAME fifo does not stay valid, which is what forced q' off the
    operand fifo, and the sequence was changed to broadcast it.

    **The core body was not changed with it, and that was the P2 fault**: the
    sequence filled the broadcast with q' while `core_p1p2` still acquired q'
    from the weight fifo, so attention was handed KV-cache bytes as its query.
    A wrong input explains every elimination — invariant to core count, q
    stride, sequence length, and surviving both a host-built cache and a
    host-built q'. Fixing it took the error from 1.0496e-01 to 3.5241e-03.

    The lesson is narrow and worth keeping: this docstring described the old
    design for several ticks after the code changed, and I read it as a
    statement of what the code does. Prose that outlives its code is worse than
    no prose.
  * **The operand fifo is `uint8`.** One fifo carries q4_1 tiles and q'/KV, a
    fifo has one object type, and IRON requires the kernel arg type to match it
    exactly. Attention casts on entry.
  * **q' is strided.** P1's result object is 2*HEAD per head and a drain cannot
    skip source elements, so the query block arrives with 128 elements per head.
    `-DDIM_QSTRIDE` lets attention read it in place.
  * **P2 gets its own result fifo.** P1 emits 128-element objects and P2 emits
    256; one fifo cannot do both. A core has two output DMA channels and P1 uses
    one, so this is free — 12 of 16 shim outputs.

**STATUS: P1 verifies inside the chain; P2 emits zeros.** Run it with
`NATT = 2` — at 2 or 4 attention pairs the design fails to route.

**Append at an EVEN position only.** The k′ pair-write emits `(g_kprev, k_t)` at
column `t-1` when `t` is odd, and `g_kprev` is empty on a design's first
dispatch — so an odd append zeroes the previous column, which is correct only if
the same design processed the previous token. `--seq 31` appends at 30 and the
cache verifies (k′ one bf16 ulp, v′ exact); `--seq 32` appends at 31 and shows
8.9e-01 on K, which is the test setup, not the kernel.

    python3 p1p2_chain.py --seq 31        # even append position
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
from aie.utils.benchmark import run_iters  # noqa: E402
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
NATT = 4                       # attention cores = KV heads


def build(pos, nobj):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = NCORES // 2
    apairs = NATT // 2
    BC = 2 * K_DIM + 2 * HEAD
    OBJ = 2 * HEAD                                  # P1 result object, bf16
    KTILE, VTILE = HEAD * TSEQ, TSEQ * HEAD
    # One slot per KV head, OPERAND bytes wide — not KVSTRIDE. P2's fill has
    # to deliver whole operand objects, and the fifo side of a transfer is
    # linear, so a tightly-packed cache would land head g+1's data at the
    # wrong offset inside the pair object. The slack inside each slot is the
    # same 4160 B attn_phase pads with, and must be zero for npad.
    SLOT = OPERAND // 2                 # bf16 elements per head slot
    KVSTRIDE = SLOT
    off = pos - (pos & 1)
    import os as _osk
    SKIP_P1 = 1 if _osk.environ.get("CHAIN_P2_ONLY") else 0
    # bisect: P2 reads a host-built cache instead of the one P1 drained into.
    # P1 still runs. Separates "P2 after P1" from "P2 reading P1's output".
    HOSTKV = 1 if _osk.environ.get("CHAIN_HOST_KV") else 0
    HOSTNORM = 1 if _osk.environ.get("CHAIN_HOST_NORM") else 0
    PREP = "flm_asum_prepare" if HOSTNORM else "flm_norm_prepare"
    PREPSRC = str(KDIR / "flm_asum_prepare.cc") if HOSTNORM else NORM_SRC

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    op_ty = np.ndarray[(OPERAND,), np.dtype[np.uint8]]      # ONE operand type
    oppair_ty = np.ndarray[(2 * OPERAND,), np.dtype[np.uint8]]
    p1o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    p1opair_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    p2o_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]
    p2opair_ty = np.ndarray[(2 * GQA * HEAD,), np.dtype[bfloat16]]
    # P1's weights, then P2's q'+KV objects, on the same fifo
    w_all_ty = np.ndarray[(2 * HPC * TPH * wt,), np.dtype[np.uint8]]
    kvin_ty = bc_ty                      # q' rides the broadcast object
    q_ty = np.ndarray[(4 * OBJ,), np.dtype[bfloat16]]
    # uint8, matching the operand fifo it feeds. A fill whose buffer and fifo
    # disagree on element width counts its sizes in the wrong unit.
    cache_ty = np.ndarray[(2 * NATT * SLOT,), np.dtype[np.uint8]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS={NQ}", f"-DDIM_QKHEADS={NK}",
             f"-DDIM_GQA={GQA}", f"-DDIM_TSEQ={TSEQ}", f"-DDIM_KVPER={KVPER}",
             f"-DDIM_QSTRIDE={OBJ}", f"-DDIM_KVOBJ={OPERAND}",
             f"-DDIM_NPADOFF={32 * OBJ}", "-DQOFF_FROM_KV=1"]
    P = ", ".join(f"w{i}: In" for i in range(npairs))
    P += ", " + ", ".join(f"kvin{i}: In" for i in range(apairs))
    if HOSTKV:
        P += ", " + ", ".join(f"hostkv{i}: In" for i in range(apairs))
    P += ", " + ", ".join(f"q{i}: Out" for i in range(npairs))
    P += ", " + ", ".join(f"cache{i}: Out" for i in range(npairs))
    P += ", " + ", ".join(f"attn{i}: Out" for i in range(apairs))
    src = f'''
def _design(bc: In, {P}):
    kq = ExternalFunction("flm_gemv_qkv", source_file=QKV_SRC,
                          arg_types=[bc_ty, op_ty], compile_flags=FLAGS)
    ke = ExternalFunction("flm_p1_emit", source_file=EMIT_SRC,
                          arg_types=[op_ty, p1o_ty], compile_flags=FLAGS)
    # HOSTNORM: the plain block-sum prologue, which does NOT write the
    # broadcast. Isolates flm_norm_prepare's in-place modification of a
    # broadcast object that the SAME fifo later reuses to deliver q'.
    kn = ExternalFunction({PREP!r}, source_file={PREPSRC!r},
                          arg_types=[bc_ty], compile_flags=FLAGS)
    # q' now arrives on the broadcast fifo, so its declared type is bc_ty.
    # The kernels take `const uint8*` and cast internally, so only the memref
    # shape has to agree with the fifo the object comes from.
    kab = ExternalFunction("flm_attn_begin", source_file=BEG_SRC,
                           arg_types=[bc_ty], compile_flags=FLAGS)
    kat = ExternalFunction("flm_attn_tile", source_file=ATT_SRC,
                           arg_types=[bc_ty, op_ty], compile_flags=FLAGS)
    kaf = ExternalFunction("flm_attn_finish", source_file=FIN_SRC,
                           arg_types=[p2o_ty, bc_ty], compile_flags=FLAGS)

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
        # A broadcast object must be consumed by EVERY consumer of the fifo.
        # These cores sit out P2, but the fifo still delivers them the q'
        # object, and leaving it unreleased stalls the accounting for the cores
        # that do use it.
        eb2 = bcc.acquire(1)
        bcc.release(1)

    def core_p1p2(bcc, wc, op, ap, kqkv, kemit, kprep, kbeg, ktile, kfin):
        if not {SKIP_P1}:
            p1_body(bcc, wc, op, kqkv, kemit, kprep)
        # ---- P2: q' arrives on the BROADCAST, KV tiles on the weight fifo ----
        # This used to read q' from `wc` while the sequence delivered it on the
        # broadcast — so kbeg/ktile were handed KV-cache bytes as q'. That is
        # the whole P2 fault: a wrong input, which is why it was invariant to
        # core count, q stride and sequence length, and why it survived a
        # host-built cache AND a host-built q'.
        #
        # kbeg writes the online-softmax state and ktile reads it; an acquire
        # between two kernels sharing a global loses the handoff
        # (global_handoff_probe.py), so the first KV acquire stays hoisted above
        # kbeg. With q' on its own fifo this now matches attn_phase.py, which
        # passes precisely because its q and KV are on different fifos.
        eq = bcc.acquire(1)
        ekv = wc.acquire(1)
        kbeg(eq)
        ktile(eq, ekv)
        wc.release(1)
        for _ in range_({nobj} - 1):
            ekv = wc.acquire(1)
            ktile(eq, ekv)
            wc.release(1)
        eo = ap.acquire(1)
        kfin(eo, eq)
        ap.release(1)
        bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            c = 2 * p + j
            if p >= {npairs} - {apairs}:
                workers.append(Worker(core_p1p2,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             p2_sub[p - ({npairs} - {apairs})][j].prod(), kq, ke, kn, kab, kat, kaf],
                    stack_size=8192))
            else:
                workers.append(Worker(core_p1,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             kq, ke, kn], stack_size=8192))

    def sequence(*args):
        n, a = {npairs}, {apairs}
        bcb = args[0]
        wb = [args[1 + i] for i in range(n)]
        kvb = [args[1 + n + i] for i in range(a + (a if {HOSTKV} else 0))]
        ax = a + (a if {HOSTKV} else 0)
        qb = [args[1 + n + ax + i] for i in range(n)]
        cb = [args[1 + 2 * n + ax + i] for i in range(n)]
        ab = [args[1 + 3 * n + ax + i] for i in range(a)]
        base = 1 + 3 * n + ax + a
        bch = args[base]
        wh = [args[base + 1 + i] for i in range(n)]
        p1h = [args[base + 1 + n + i] for i in range(n)]
        p2h = [args[base + 1 + 2 * n + i] for i in range(a)]

        tg = TaskGroup()
        bch.fill(bcb, group=tg)
        for i in range(n):
            if {SKIP_P1} and i >= n - a:
                continue          # these cores skip P1, so no weights for them
            wh[i].fill(wb[i], group=tg)
        for i in range(n):
            if {SKIP_P1} and i >= n - a:
                continue
            p1h[i].drain(qb[i], wait=True, group=tg,
                         sizes=[1, 1, 1, 4 * {OBJ}], strides=[0, 0, 0, 1])
            if i < n // 2:
                p1h[i].drain(cb[i], wait=True, group=tg,
                             offset=2 * (2 * i * {SLOT} + {off}),
                             sizes=[1, 2, {HEAD}, 4],
                             strides=[0, 2 * {SLOT}, 2 * {TSEQ}, 1])
            else:
                p1h[i].drain(cb[i], wait=True, group=tg,
                             offset=2 * (2 * (i - n // 2) * {SLOT} + {KTILE}
                                         + {pos} * {HEAD}),
                             sizes=[1, 2, 1, 2 * {OBJ}],
                             strides=[0, 2 * {SLOT}, 0, 1])
        tg.finish()

        # ---- P2 ----------------------------------------------------------
        tg = TaskGroup()
        bch.fill(kvb[0], group=tg)          # the broadcast now carries q'
        for i in range(a):
            src = kvb[1 + i] if {HOSTKV} else cb[i]
            wh[n - a + i].fill(src, group=tg, offset=2 * i * {OPERAND},
                       sizes=[1, 1, 1, 2 * {OPERAND}], strides=[0, 0, 0, 1])
        for i in range(a):
            p2h[i].drain(ab[i], wait=True, group=tg)
        tg.finish()

    at = [bc_ty] + [w_all_ty] * {npairs} + [kvin_ty] * {apairs}
    at += [cache_ty] * ({apairs} if {HOSTKV} else 0)
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
              cache_ty=cache_ty, SKIP_P1=SKIP_P1, HOSTKV=HOSTKV,
              PREP=PREP, PREPSRC=PREPSRC,
              __name__="flm_p1p2")
    exec(src, ns)
    return iron.jit(ns["_design"],
                    source_files=[QKV_SRC, EMIT_SRC, NORM_SRC, ATT_SRC,
                                  BEG_SRC, FIN_SRC,
                                  str(KDIR / 'flm_asum_prepare.cc')],
                    full_elf=True), wt, KVSTRIDE


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--bench", action="store_true",
                   help="time the P1->P2 pair; the seam's cost is otherwise\n"
                        "only inferred from P1 and P2 measured apart")
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
    xd = x.astype(np.float64)
    inv = np.float32(1.0 / np.sqrt((xd * xd).mean() + EPS))
    xn = rnd(rnd(x * rnd(inv)) * nw)

    bc = np.zeros(2 * K_DIM + 2 * HEAD, np.float32)
    bc[:K_DIM] = (xn if __import__("os").environ.get("CHAIN_HOST_NORM") else x)
    bc[K_DIM:2 * K_DIM] = nw
    bc[2 * K_DIM:2 * K_DIM + HEAD] = cs_q
    bc[2 * K_DIM + HEAD:] = cs_k
    bc_t = iron.tensor(bc.astype(bfloat16), dtype=bfloat16, device="npu")
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

    # prior cache: positions 0..pos-1, laid out in OPERAND-sized head slots
    SLOT = KVSTRIDE
    Kc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    Vc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    cache = np.zeros((NATT, SLOT), np.float32)
    for g in range(NATT):
        K = cache[g, :KTILE].reshape(HEAD, TSEQ)
        V = cache[g, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
        if pos:
            K[:, :pos] = Kc[g].T
            V[:pos] = Vc[g]
    craw = cache.reshape(NATT, SLOT).astype(bfloat16).view(np.uint16)
    for g in range(NATT):
        # trailer: this core's offset into the shared q' block
        craw[g, (OPERAND - 64) // 2:(OPERAND - 64) // 2 + 2] = \
            np.array([float(g * GQA * OBJ)], np.float32).view(np.uint16)
    cache_t = iron.tensor(craw.reshape(-1).view(np.uint8),
                          dtype=np.uint8, device="npu")

    # P2's q' object per pair — the rest of its operand stream is the cache
    # ONE broadcast-shaped q' object: all 32 heads at OBJ stride, then npad as
    # an f32 bit pattern (written after the bf16 conversion, or it is destroyed).
    BCN = 2 * K_DIM + 2 * HEAD
    qall = np.zeros(BCN, np.float32)
    for h in range(NQ):
        qall[h * OBJ:h * OBJ + HEAD] = ref[h][:HEAD]
    qraw = qall.astype(bfloat16).view(np.uint16)
    qraw[NQ * OBJ:NQ * OBJ + 2] = np.array([float(npad)], np.float32).view(np.uint16)
    # one broadcast-shaped q' per attention pair — the design takes `apairs`
    # of them, and a short list silently shifts every later argument.
    q_in = [iron.tensor(qraw.view(bfloat16), dtype=bfloat16, device="npu")] * apairs

    q_ts = [iron.zeros(4 * OBJ, dtype=bfloat16, device="npu")
            for _ in range(npairs)]
    a_ts = [iron.zeros(2 * GQA * HEAD, dtype=bfloat16, device="npu")
            for _ in range(apairs)]
    import os as _oh
    if _oh.environ.get("CHAIN_HOST_KV"):
        # the same cache contents, built on the host: P1 still runs and still
        # drains, but P2 reads this instead
        hostc = cache.reshape(NATT, SLOT).copy()
        for g in range(NATT):
            K = hostc[g, :KTILE].reshape(HEAD, TSEQ)
            V = hostc[g, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
            K[:, pos] = ref[NQ + g]
            V[pos] = ref[NK + g]
        hraw = hostc.astype(bfloat16).view(np.uint16)
        for g in range(NATT):
            hraw[g, (OPERAND - 64) // 2:(OPERAND - 64) // 2 + 2] = \
                np.array([float(g * GQA * OBJ)], np.float32).view(np.uint16)
        host_t = iron.tensor(hraw.reshape(-1).view(np.uint8), dtype=np.uint8,
                             device="npu")
        design(bc_t, *w_ts, *q_in, *[host_t] * apairs, *q_ts,
               *[cache_t] * npairs, *a_ts)
    else:
        _args = (bc_t, *w_ts, *q_in, *q_ts, *[cache_t] * npairs, *a_ts)
        if o.bench:
            _b = run_iters(design, *_args, warmup=2, iters=10)
            _us = _b.npu.min_us if _b.npu else _b.e2e.min_us
        else:
            design(*_args)
            _us = None

    # first: did P1 write the cache correctly inside THIS harness?
    cv = cache_t.numpy().view(bfloat16).astype(np.float64).reshape(NATT, SLOT)
    ke = ve = 0.0
    for g in range(NATT):
        K = cv[g, :KTILE].reshape(HEAD, TSEQ)
        V = cv[g, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
        ke = max(ke, np.abs(K[:, pos] - ref[NQ + g]).max())
        ve = max(ve, np.abs(V[pos] - ref[NK + g]).max())
        if pos:
            ke = max(ke, np.abs(K[:, :pos] - Kc[g].T).max())
            ve = max(ve, np.abs(V[:pos] - Vc[g]).max())
    print(f"  P1 cache: k' col {pos} + prior cols max err {ke:.4e};  "
          f"v' row {pos} + prior rows max err {ve:.4e}")

    # ---- reference: attention over the cache INCLUDING P1's appended token --
    if o.bench and _us is not None:
        FIXED_US = 92.9
        p1_b = npairs * 2 * HPC * TPH * q4nx.tile_bytes(K_DIM, NROWS)
        kv_b = apairs * 2 * OPERAND * nobj
        mb = (p1_b + kv_b) / 1e6
        print(f"  bench: {mb:.2f} MB  {mb*1e3/_us:.1f} GB/s  {_us:.1f} us "
              f"(marginal {_us - FIXED_US:.1f}, 16-core ideal {mb*17.85:.1f})")
    print(f"P1 -> P2 in one dispatch: seq {o.seq} (P1 appends at pos {pos}), "
          f"{nobj} KV objects, npad {npad}")
    worst, scale = 0.0, 0.0
    for a in range(NATT):
        Kfull = np.zeros((o.seq, HEAD), np.float64)
        Vfull = np.zeros((o.seq, HEAD), np.float64)
        if pos:
            Kfull[:pos], Vfull[:pos] = Kc[a], Vc[a]
        Kfull[pos] = ref[NQ + a]              # k' P1 just wrote
        Vfull[pos] = ref[NK + a]              # v' P1 just wrote
        qr = np.stack([ref[GQA * a + sl] for sl in range(GQA)])
        # q' already carries the 1/sqrt(d)*log2(e) scale from cs_q
        sc = (qr @ Kfull.T) / math.log2(math.e)
        e = np.exp(sc - sc.max(1, keepdims=True))
        want = (e / e.sum(1, keepdims=True)) @ Vfull
        got = a_ts[a // 2].numpy().astype(np.float64).reshape(2, GQA, HEAD)[a % 2]
        worst = max(worst, np.abs(got - want).max())
        scale = max(scale, np.abs(want).mean())
        if a == 0:
            print(f"  DIAG head0 got[0,:4] {got[0,:4].round(4)}")
            print(f"  DIAG head0 want[0,:4] {want[0,:4].round(4)}")
            # what would attention over ONLY the appended token give?
            w1 = Vfull[pos]
            print(f"  DIAG if it saw only pos {pos}: {w1[:4].round(4)}  "
                  f"err {np.abs(got[0] - w1).max():.3e}")
            # ... and over the prior cache only?
            sc0 = (qr @ Kfull[:pos].T) / math.log2(math.e)
            e0 = np.exp(sc0 - sc0.max(1, keepdims=True))
            w0 = (e0 / e0.sum(1, keepdims=True)) @ Vfull[:pos]
            print(f"  DIAG if it missed pos {pos}: err "
                  f"{np.abs(got[0] - w0[0]).max():.3e}")
    tol = 8e-2 * scale                        # AIE2P exp2 NLF floor
    print(f"  attention out: max err {worst:.4e}   mean|ref| {scale:.5f}   "
          f"tol {tol:.4e}")
    print(f"  -> {'PASS' if worst <= tol else 'FAIL'}  (floor is the exp2 NLF)")
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
