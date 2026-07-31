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
from qkv_verify import HEAD, K_DIM, EPS, ROPE_THETA, qkv_rows, rope_ref  # noqa: E402

# **The fused layer sizes its operand for DATA memory, not for one phase.**
# Every phase's weights and P2's KV ride one fifo, so the object is the max of a
# q4_1 tile and a KV object, and two of them live in each core's 64 KB. At
# NROWS=16/KVPER=2 that is 20544 B and 41088 B per core — 63% — and P4's buffers
# fail to allocate. At NROWS=8/KVPER=1 it is 10304 B and 20608 B, 31%.
#
# Measured: NROWS 16->8 costs 4.5% of GEMV bandwidth (48.9 -> 46.7 GB/s);
# KVPER 2->1 costs nothing (medians 27.4 -> 19.9 us, ranges overlapping).
NROWS = 8
TPH = HEAD // NROWS
from ffn_verify import load_linear  # noqa: E402
from p1_route import (NQ, NK, NV, NCORES, HPC, heads_of, rnd,  # noqa: E402
                      head_layout, hpc_for, drain_plan)

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
RES_SRC = str(KDIR / "flm_gemv_residual.cc")
ASUM_SRC = str(KDIR / "flm_asum_prepare.cc")
HEMIT_SRC = str(KDIR / "flm_h_emit.cc")
GATE_SRC = str(KDIR / "flm_gemv_gate.cc")
UPS_SRC = str(KDIR / "flm_gemv_up_swiglu.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
TSEQ, GQA, KVPER = 32, 4, 1
# follows NROWS and KVPER: max(one KV tile 8192, a q4_1 tile 10304)
OPERAND = max(2 * TSEQ * HEAD * 2 * KVPER, q4nx.tile_bytes(K_DIM, NROWS))
NATT = 4                       # attention cores = KV heads


def build(pos, nobj):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = NCORES // 2
    apairs = NATT // 2
    # Partition B: the attention cores cannot also hold P1's kernels (measured —
    # five phases overflow 16 KB), so P1 spreads over the remaining cores.
    p1pairs = npairs - apairs
    p1cores = 2 * p1pairs
    hpc = hpc_for(p1cores)
    layout = head_layout(p1cores)
    qobj, kvplan = drain_plan(p1cores)
    # Where each pair's q' belongs in the broadcast. Measured, not assumed (see
    # CHAIN_QMAP): a pair's stream is [slot s][core j] and its head is
    # qbase + hpcc*j + s, so the scatter is a plain 3-level stride.
    qbase = [sum(qobj[:i]) for i in range(len(qobj))]
    hpcc = [q // 2 for q in qobj]
    # P3 runs on ALL cores — only P1 and P2 are partitioned — so o_proj's 2048
    # output rows split over every core, NROWS at a time.
    p3tiles = K_DIM // (NCORES * NROWS)
    BC = 2 * K_DIM + 2 * HEAD
    OBJ = 2 * HEAD                                  # P1 result object, bf16
    D_FF = 8192
    p4tiles = D_FF // (NCORES * NROWS)      # gate/up steps per core
    p4per = OBJ // NROWS                    # steps sharing one result object
    p4objs = p4tiles // p4per               # result objects per core
    rpp4 = D_FF // (NCORES // 2)            # rows a pair owns
    KTILE, VTILE = HEAD * TSEQ, TSEQ * HEAD
    # One slot per KV head, OPERAND bytes wide — not KVSTRIDE. P2's fill has
    # to deliver whole operand objects, and the fifo side of a transfer is
    # linear, so a tightly-packed cache would land head g+1's data at the
    # wrong offset inside the pair object. The slack inside each slot is the
    # same 4160 B attn_phase pads with, and must be zero for npad.
    SLOT = OPERAND // 2                 # bf16 elements per head slot
    KVSTRIDE = SLOT
    # WHICH object the appended token lands in. Without this the drain always
    # targeted object 0, so at pos >= TSEQ it wrote past the K tile's columns and
    # into V -- silently, because nothing verified beyond one tile.
    kv_ob = pos // TSEQ                 # object holding logical position `pos`
    kv_in = pos % TSEQ                  # its column/row within that object
    kv_obase = kv_ob * NATT             # cache is flat [obj*NATT + head][SLOT]
    off = kv_in - (kv_in & 1)
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
    # P3's o_proj is over the whole 2048-dim vector, so P2's per-pair results
    # have to land in ONE buffer before they can feed the next phase's
    # broadcast. Several pairs draining into one BO at different offsets is the
    # ffn_chain pattern, already used here for the KV cache.
    attn_all_ty = np.ndarray[(apairs * 2 * GQA * HEAD,), np.dtype[bfloat16]]
    w3_ty = np.ndarray[(2 * p3tiles * OPERAND,), np.dtype[np.uint8]]
    # h's DRAIN TARGET, not its object: the drain scatters a pair's 2*OBJ
    # object into the full broadcast-shaped buffer that P4 is filled from,
    # so the runtime argument is that buffer's shape.
    h_ty = np.ndarray[(2 * K_DIM + 2 * HEAD,), np.dtype[bfloat16]]
    w4_ty = np.ndarray[(2 * 2 * p4tiles * OPERAND,), np.dtype[np.uint8]]
    # ONE row-ordered sw buffer for all pairs. P5 slices it by K-chunk, so it
    # must be in row order — the per-pair stream is [object][core] and a
    # pair's cores are rpp4/2 rows apart, which is not ascending. A drain
    # shapes its destination, so each object goes straight to its row.
    sw_ty = np.ndarray[(D_FF,), np.dtype[bfloat16]]
    # P1's weights, then P2's q'+KV objects, on the same fifo
    w_all_ty = np.ndarray[(2 * hpc * TPH * wt,), np.dtype[np.uint8]]
    kvin_ty = bc_ty                      # q' rides the broadcast object
    # one per P1 pair: at 12 P1 cores pairs 0-3 emit 4 q objects and
    # pairs 4-5 emit 8, since the KV-carrying cores spend two slots on k/v
    # drain TARGET shape, not object shape -- the same trap h_ty hit
    q_tys = [np.ndarray[(2 * K_DIM + 2 * HEAD,), np.dtype[bfloat16]]
             for i in range(p1pairs)]
    # uint8, matching the operand fifo it feeds. A fill whose buffer and fifo
    # disagree on element width counts its sizes in the wrong unit.
    cache_ty = np.ndarray[(nobj * 2 * NATT * SLOT,), np.dtype[np.uint8]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS={NQ}", f"-DDIM_QKHEADS={NK}",
             f"-DDIM_GQA={GQA}", f"-DDIM_TSEQ={TSEQ}", f"-DDIM_KVPER={KVPER}",
             f"-DDIM_QSTRIDE={OBJ}", f"-DDIM_KVOBJ={OPERAND}",
             f"-DDIM_NPADOFF={32 * OBJ}", "-DQOFF_FROM_KV=1",
             f"-DDIM_RESN={2 * p3tiles * NROWS}",
             f"-DDIM_P3TILES={p3tiles}",
             f"-DDIM_OBJROWS={OBJ}"]
    # This list must match `at` element for element. Weights and q results are
    # per P1 PAIR (only those cores run P1); the cache, P3's weights and P3's
    # results are per pair, since every core runs P3.
    P = ", ".join(f"w{i}: In" for i in range(p1pairs))
    P += ", " + ", ".join(f"kvin{i}: In" for i in range(apairs))
    if HOSTKV:
        P += ", " + ", ".join(f"hostkv{i}: In" for i in range(apairs))
    P += ", " + ", ".join(f"q{i}: Out" for i in range(p1pairs))
    P += ", " + ", ".join(f"cache{i}: Out" for i in range(npairs))
    P += ", " + ", ".join(f"attn{i}: Out" for i in range(apairs))
    P += ", bc3: In"
    P += ", " + ", ".join(f"w3_{i}: In" for i in range(npairs))
    P += ", " + ", ".join(f"h{i}: Out" for i in range(npairs))
    P += ", bc4: In"
    P += ", " + ", ".join(f"w4_{i}: In" for i in range(npairs))
    P += ", " + ", ".join(f"sw{i}: Out" for i in range(npairs))
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
    # P3 shares P1's result fifo: a fifo of its own would need 8 more shim
    # outputs against a budget of 10 in 16. Its object is therefore P1-sized
    # (2*HEAD bf16) and the kernel fills only the first NROWS of it.
    # P3 needs its OWN activation-sum prepare. flm_q4_1_tile folds the dequant
    # as d*sum(q*a) + m*sum(a) and keeps sum(a) in the global g_asum, so a phase
    # that changes the activation without re-running a prepare computes its `m`
    # term against the PREVIOUS phase's sums. flm_asum_prepare, not
    # flm_norm_prepare — P3 must not renormalise.
    kas = ExternalFunction("flm_asum_prepare", source_file=ASUM_SRC,
                           arg_types=[bc_ty], compile_flags=FLAGS)
    khe = ExternalFunction("flm_h_emit", source_file=HEMIT_SRC,
                           arg_types=[op_ty, p1o_ty], compile_flags=FLAGS)
    kg4 = ExternalFunction("flm_gemv_gate", source_file=GATE_SRC,
                           arg_types=[bc_ty, op_ty], compile_flags=FLAGS)
    ku4 = ExternalFunction("flm_gemv_up_swiglu", source_file=UPS_SRC,
                           arg_types=[bc_ty, op_ty, p1o_ty], compile_flags=FLAGS)
    kr3 = ExternalFunction("flm_gemv_q4_1_residual", source_file=RES_SRC,
                           arg_types=[bc_ty, op_ty, p1o_ty],
                           compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name=f"bc_kvtile")
    bc_cons = [f_bc.cons() for _ in range({NCORES})]
    f_w = [ObjectFifo(oppair_ty, name=f"wp{{i}}_n{NROWS}k{KVPER}") for i in range({npairs})]
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
        for _ in range_({hpc}):
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

    def p3_body(bcc, wc, op, kres, kasum, khemit):
        """o_proj + residual. Every core runs this — P3 is on all 16, only P1
        and P2 are partitioned. The broadcast's third fill carries the gathered
        attention output in its first K_DIM and the residual stream after it."""
        eb = bcc.acquire(1)
        kasum(eb)                      # g_asum for P3's activation
        # ONE result object for the core's whole 128-row slice, not one per
        # tile. Per-tile objects are 12% dense and a drain cannot skip the
        # padding, so P4 could never be broadcast a dense h from them. kres
        # still takes an `out` and writes its NROWS there each time — those
        # writes are overwritten and unused; the values that matter go to
        # g_resid, which is where they were already going for P5.
        eo = op.acquire(1)
        for _ in range_({p3tiles} - 1):
            ew = wc.acquire(1)
            kres(eb, ew, eo)
            wc.release(1)
        # the emit reuses the LAST tile for its row_base, exactly as p1_body
        # does: an object released inside the loop does not dominate a use
        # after it, and a separate acquire would desynchronise the stream.
        ew = wc.acquire(1)
        kres(eb, ew, eo)
        khemit(ew, eo)
        wc.release(1)
        op.release(1)
        bcc.release(1)

    def p4_body(bcc, wc, op, kgate, kups, kprep):
        """gate/up + SwiGLU. One result object per p4per steps: the kernel writes
        at row_base % DIM_OBJROWS, so the object fills densely and P5 can be
        broadcast a dense sw."""
        eb = bcc.acquire(1)
        # The POST-ATTENTION RMSNorm, and g_asum for the normalised h. This used
        # to be flm_asum_prepare, which computes the sums but does NOT normalise
        # -- so the FFN ran on raw h and the layer was missing one of its two
        # norms. The host reference made the same omission, so the check passed
        # on both sides computing the wrong layer.
        #
        # Costs no program memory: flm_norm_prepare is already linked for P1.
        # Under CHAIN_HOST_NORM `kprep` is the asum kernel and the host
        # pre-normalises h, exactly as it does for x.
        kprep(eb)
        for _ in range_({p4objs}):
            eo = op.acquire(1)
            for _ in range_({p4per}):
                eg = wc.acquire(1)
                kgate(eb, eg)              # gate -> in-core stash
                wc.release(1)
                eu = wc.acquire(1)
                kups(eb, eu, eo)           # up, then SwiGLU against the stash
                wc.release(1)
            op.release(1)
        bcc.release(1)

    def core_p1(bcc, wc, op, kqkv, kemit, kprep, kres, kasum, khemit,
                kgate, kups):
        p1_body(bcc, wc, op, kqkv, kemit, kprep)
        # A broadcast object must be consumed by EVERY consumer of the fifo.
        # These cores sit out P2, but the fifo still delivers them the q'
        # object, and leaving it unreleased stalls the accounting for the cores
        # that do use it.
        eb2 = bcc.acquire(1)
        bcc.release(1)
        p3_body(bcc, wc, op, kres, kasum, khemit)
        p4_body(bcc, wc, op, kgate, kups, kprep)

    def core_p1p2(bcc, wc, op, ap, kqkv, kemit, kprep, kbeg, ktile, kfin,
                  kres, kasum, khemit, kgate, kups):
        # Partition B: these cores do NOT run P1 — measured, five phases
        # overflow 16 KB of program memory. They must still consume the
        # broadcast's first fill (the activation), because a broadcast object
        # is only recycled once every consumer has taken it; skipping it stalls
        # the cores that do use it.
        _act = bcc.acquire(1)
        bcc.release(1)
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
        p3_body(bcc, wc, op, kres, kasum, khemit)
        p4_body(bcc, wc, op, kgate, kups, kprep)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            c = 2 * p + j
            if p >= {npairs} - {apairs}:
                workers.append(Worker(core_p1p2,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             p2_sub[p - ({npairs} - {apairs})][j].prod(), kq, ke, kn, kab, kat, kaf, kr3, kas, khe, kg4, ku4],
                    stack_size=8192))
            else:
                workers.append(Worker(core_p1,
                    fn_args=[bc_cons[c], w_sub[p][j].cons(), p1_sub[p][j].prod(),
                             kq, ke, kn, kr3, kas, khe, kg4, ku4], stack_size=8192))

    def sequence(*args):
        n, a = {npairs}, {apairs}
        bcb = args[0]
        wb = [args[1 + i] for i in range({p1pairs})]
        kvb = [args[1 + {p1pairs} + i]
               for i in range(a + (a if {HOSTKV} else 0))]
        ax = a + (a if {HOSTKV} else 0)
        qb = [args[1 + {p1pairs} + ax + i] for i in range({p1pairs})]
        cb = [args[1 + 2 * {p1pairs} + ax + i] for i in range(n)]
        ab = [args[1 + 2 * {p1pairs} + ax + n + i] for i in range(a)]
        # tensor args: bc, w*p1pairs, (kvin+hostcache)*ax, q*p1pairs,
        # cache*n, attn*a, then P3's bc + w3*n + h*n. Handles follow all of them.
        base3 = 1 + 2 * {p1pairs} + ax + n + a
        base4 = base3 + 1 + 2 * n
        base = base4 + 1 + 2 * n
        bch = args[base]
        wh = [args[base + 1 + i] for i in range(n)]   # all pairs: KV rides these
        p1h = [args[base + 1 + n + i] for i in range(n)]
        p2h = [args[base + 1 + 2 * n + i] for i in range(a)]

        tg = TaskGroup()
        bch.fill(bcb, group=tg)
        for i in range({p1pairs}):     # partition B: only these pairs run P1
            wh[i].fill(wb[i], group=tg)
        QOBJ, KVPLAN = {qobj!r}, {kvplan!r}
        QBASE, HPCC = {qbase!r}, {hpcc!r}
        for i in range({p1pairs}):
            p1h[i].drain(qb[i], wait=True, group=tg,
                         offset=QBASE[i] * {OBJ},
                         sizes=[1, HPCC[i], 2, {OBJ}],
                         strides=[0, {OBJ}, HPCC[i] * {OBJ}, 1])
            for _kind, _base in KVPLAN[i]:
                if _kind == "k":
                    p1h[i].drain(cb[i], wait=True, group=tg,
                                 offset=2 * (({kv_obase} + _base) * {SLOT}
                                             + {off}),
                                 sizes=[1, 2, {HEAD}, 4],
                                 strides=[0, 2 * {SLOT}, 2 * {TSEQ}, 1])
                else:
                    p1h[i].drain(cb[i], wait=True, group=tg,
                                 offset=2 * (({kv_obase} + _base) * {SLOT}
                                             + {KTILE} + {kv_in} * {HEAD}),
                                 sizes=[1, 2, 1, 2 * {OBJ}],
                                 strides=[0, 2 * {SLOT}, 0, 1])
        tg.finish()

        # ---- P2 ----------------------------------------------------------
        tg = TaskGroup()
        bch.fill(kvb[0], group=tg)          # the broadcast now carries q'
        for i in range(a):
            # the host caches follow the `a` q' broadcast buffers, so they
            # start at kvb[a] — `kvb[1 + i]` only happened to be right when
            # apairs was 1 (NATT=2) and silently read a q' buffer as a cache
            # at NATT=4.
            src = kvb[a + i] if {HOSTKV} else cb[i]
            # One fill per KV object. A single strided fill cannot express this:
            # the 2*OPERAND run decomposes into 6 x 3424, which uses up the BD's
            # dimensions and pushes the object stride into the repeat-count slot
            # ("Do not include the highest dimension size in transfer length").
            for _j in range({nobj}):
                wh[n - a + i].fill(src, group=tg,
                       offset=2 * i * {OPERAND} + _j * {NATT} * {OPERAND},
                       sizes=[1, 1, 1, 2 * {OPERAND}], strides=[0, 0, 0, 1])
        for i in range(a):
            p2h[i].drain(ab[i], wait=True, group=tg,
                         offset=i * 2 * {GQA} * {HEAD},
                         sizes=[1, 1, 1, 2 * {GQA} * {HEAD}],
                         strides=[0, 0, 0, 1])
        tg.finish()

        # ---- P3 -----------------------------------------------------------
        # The broadcast's THIRD fill: attention output, then the residual
        # stream. Same refill mechanism as the first two (activation, q'),
        # which chain_probe.py verified across a phase boundary.
        p3b = args[base3]
        w3b = [args[base3 + 1 + i] for i in range(n)]
        hb = [args[base3 + 1 + n + i] for i in range(n)]
        tg = TaskGroup()
        bch.fill(p3b, group=tg)
        for i in range(n):
            wh[i].fill(w3b[i], group=tg)
        for i in range(n):
            # Scatter each pair's h into NATURAL row order so P4 can be filled
            # straight from it. A pair's object is [core j][tile t][row r] and
            # core (pr, j) owns rows pr*rpp3 + t*2*NROWS + j*NROWS + r, so the
            # permutation is a plain 3-level stride -- the same trick the P4
            # drain uses for sw.
            p1h[i].drain(hb[i], wait=True, group=tg,
                         offset=i * 2 * {NROWS} * {p3tiles},
                         sizes=[1, 2, {p3tiles}, {NROWS}],
                         strides=[0, {NROWS}, {2 * NROWS}, 1])
        tg.finish()

        # ---- P4 -----------------------------------------------------------
        p4b = args[base4]
        w4b = [args[base4 + 1 + i] for i in range(n)]
        swb = [args[base4 + 1 + n + i] for i in range(n)]
        tg = TaskGroup()
        bch.fill(p4b, group=tg)
        for i in range(n):
            wh[i].fill(w4b[i], group=tg)
        for i in range(n):
            p1h[i].drain(swb[i], wait=True, group=tg,
                         offset=i * {rpp4},
                         sizes=[1, {p4objs}, 2, {OBJ}],
                         strides=[0, {OBJ}, {rpp4} // 2, 1])
        tg.finish()

    at = [bc_ty] + [w_all_ty] * {p1pairs} + [kvin_ty] * {apairs}
    at += [cache_ty] * ({apairs} if {HOSTKV} else 0)
    at += list(q_tys) + [cache_ty] * {npairs} + [attn_all_ty] * {apairs}
    at += [bc_ty] + [w3_ty] * {npairs} + [h_ty] * {npairs}
    at += [bc_ty] + [w4_ty] * {npairs} + [sw_ty] * {npairs}
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
              BEG_SRC=BEG_SRC, RES_SRC=RES_SRC, ASUM_SRC=ASUM_SRC, HEMIT_SRC=HEMIT_SRC,
              GATE_SRC=GATE_SRC, UPS_SRC=UPS_SRC, FIN_SRC=FIN_SRC, FLAGS=flags, bc_ty=bc_ty,
              op_ty=op_ty, oppair_ty=oppair_ty, p1o_ty=p1o_ty,
              p1opair_ty=p1opair_ty, p2o_ty=p2o_ty, p2opair_ty=p2opair_ty, attn_all_ty=attn_all_ty, w3_ty=w3_ty, h_ty=h_ty, p3tiles=p3tiles,
              w4_ty=w4_ty, sw_ty=sw_ty, p4tiles=p4tiles,
              p4per=p4per, p4objs=p4objs, rpp4=rpp4, D_FF=D_FF,
              w_all_ty=w_all_ty, kvin_ty=kvin_ty, q_tys=q_tys,
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
    p.add_argument("--kvobj", type=int, default=0,
                   help="force N KV objects instead of deriving them from --seq.\n"
                        "The seam's cost is otherwise measured with P2 at 7.6%% of\n"
                        "the bytes, which says nothing about it at decode scale.\n"
                        "The extra positions are padding and npad masks them, so\n"
                        "correctness still holds while the KV stream is realistic.")
    p.add_argument("--layer-pass", action="store_true",
                   help="after side A, run side B (P5) on the sw buffer side A\n"
                        "just produced, and check x_out — the whole layer in two\n"
                        "dispatches, end to end")
    p.add_argument("--bench", action="store_true",
                   help="time the P1->P2 pair; the seam's cost is otherwise\n"
                        "only inferred from P1 and P2 measured apart")
    p.add_argument("--seq", type=int, default=32,
                   help="cache length INCLUDING the token P1 appends")
    o = p.parse_args()
    pos = o.seq - 1                       # P1 appends at the end
    ntiles = -(-o.seq // TSEQ)
    nobj = -(-ntiles // KVPER)
    if o.kvobj:
        if o.kvobj < nobj:
            raise SystemExit(f"--kvobj {o.kvobj} < the {nobj} needed for seq {o.seq}")
        nobj = o.kvobj
        # npad describes padding in the LAST tile pair only (see
        # flm_attn_finish.cc). Padding whole extra objects overruns that and the
        # result is silently wrong -- 1.5855e-02 against a 3.9e-03 tolerance,
        # identical at kvobj 2 and 4 because the correction saturates rather
        # than accumulating. Refuse instead of reporting a wrong PASS.
        if nobj * KVPER * TSEQ - o.seq > KVPER * TSEQ:
            raise SystemExit(
                f"--kvobj {nobj} needs npad {nobj * KVPER * TSEQ - o.seq}, but "
                f"npad covers at most one tile pair ({KVPER * TSEQ}).\n"
                f"Measuring the seam at decode-scale KV needs REAL data in the "
                f"extra objects, not padding -- i.e. multi-tile cache "
                f"verification, not this flag.")
    npad = nobj * KVPER * TSEQ - o.seq
    npairs, apairs = NCORES // 2, NATT // 2
    p1pairs = npairs - apairs          # partition B: attention cores skip P1
    hpc = hpc_for(2 * p1pairs)
    layout = head_layout(2 * p1pairs)
    qobj, _kvplan = drain_plan(2 * p1pairs)
    p3tiles = K_DIM // (NCORES * NROWS)

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
    for pr in range(p1pairs):
        per = []
        for j in range(2):
            blob = []
            for h in layout[2 * pr + j]:
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
        b = np.empty((hpc * TPH, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w_ts.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))

    # prior cache: positions 0..pos-1, laid out in OPERAND-sized head slots
    SLOT = KVSTRIDE
    Kc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    Vc = rnd(rng.standard_normal((NATT, pos, HEAD)) * 0.3) if pos else \
        np.zeros((NATT, 0, HEAD), np.float32)
    # [obj][head][SLOT]. Real KV now spans every object it needs -- position p
    # lives in object p // TSEQ at slot p % TSEQ. Previously all of it went in
    # object 0 and the rest were padding, which meant nothing past one tile was
    # ever exercised with data.
    assert KVPER == 1, "multi-tile cache layout assumes one KV tile per object"
    cache = np.zeros((nobj, NATT, SLOT), np.float32)
    for g in range(NATT):
        for ob in range(nobj):
            K = cache[ob, g, :KTILE].reshape(HEAD, TSEQ)
            V = cache[ob, g, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
            lo, hi = ob * TSEQ, min(pos, (ob + 1) * TSEQ)
            if hi > lo:
                K[:, :hi - lo] = Kc[g][lo:hi].T
                V[:hi - lo] = Vc[g][lo:hi]
    craw = cache.reshape(nobj * NATT, SLOT).astype(bfloat16).view(np.uint16)
    for g in range(NATT):
        # trailer: this core's offset into the shared q' block
        for _o in range(nobj):
            craw[_o * NATT + g, (OPERAND - 64) // 2:(OPERAND - 64) // 2 + 2] = \
            np.array([float(g * GQA * OBJ)], np.float32).view(np.uint16)
    cache_t = iron.tensor(craw.reshape(-1).view(np.uint8),
                          dtype=np.uint8, device="npu")

    # P2's q' object per pair — the rest of its operand stream is the cache
    # ONE broadcast-shaped q' object: all 32 heads at OBJ stride, then npad as
    # an f32 bit pattern (written after the bf16 conversion, or it is destroyed).
    BCN = 2 * K_DIM + 2 * HEAD
    qall = np.zeros(BCN, np.float32)
    qraw = qall.astype(bfloat16).view(np.uint16)
    qraw[NQ * OBJ:NQ * OBJ + 2] = np.array([float(npad)], np.float32).view(np.uint16)
    # ONE buffer: P1's q' drain target AND P2's broadcast source. The heads are
    # left ZERO on the host -- P1's drain scatters every one of them to h*OBJ,
    # so any host value here would be overwritten. If P1 ever failed to write a
    # head, attention would see zeros rather than a plausible host value, which
    # is the failure mode this seam is supposed to expose.
    #
    # npad rides at NQ*OBJ as an f32 bit pattern and the drain never reaches it:
    # the highest byte any pair writes is qbase[-1]*OBJ + ... = 4095 < 4096.
    q_all = iron.tensor(qraw.view(bfloat16), dtype=bfloat16, device="npu")
    q_in = [q_all] * apairs
    q_ts = [q_all] * p1pairs
    attn_out = iron.zeros(apairs * 2 * GQA * HEAD, dtype=bfloat16, device="npu")
    a_ts = [attn_out] * apairs        # every pair drains into the same buffer
    # ---- P3: o_proj + residual -------------------------------------------
    # Its activation is host-supplied for now. The device's own attn_out covers
    # only NATT of the 8 KV groups (16 of 32 q heads at NATT=4), so it is not a
    # whole 2048-vector and cannot feed an o_proj GEMV yet. Verifying P3 against
    # a full host vector separates "P3 works in the chained design" from "the
    # P2->P3 handoff carries the right bytes", which is the next step.
    od, om, oc = load_linear(c, f"model.layers.{o.layer}.self_attn.o_proj.weight",
                             K_DIM, K_DIM)
    if __import__("os").environ.get("CHAIN_P3_MARK"):
        # Each tile carries its own id: codes and d zeroed, m[.,0] = tile_id, so
        # with an all-ones activation the GEMV for every row of tile t is
        # exactly 32*t. The device output then names which tile landed where,
        # which the all-zero control cannot show.
        od = np.zeros_like(od); oc = np.zeros_like(oc)
        om = np.zeros_like(om)
        if __import__("os").environ.get("CHAIN_P3_MARK") == "d":
            # marker in d*code instead of m: separates "the activation is not
            # ones" from "m never reaches the kernel". d=1, code=1 over one
            # block, act=ones -> every row of every tile returns exactly 32.
            od[:, 0] = 1.0
            oc[:, 0, :] = 1
        else:
            for _r in range(K_DIM):
                om[_r, 0] = _r // NROWS
        attn3_override = np.ones(K_DIM, np.float32)
    else:
        attn3_override = None
    if __import__("os").environ.get("CHAIN_P3_WZERO"):
        # zero weights -> the GEMV term is 0 and h must be exactly the residual.
        # If it is not, P3 is not reading the tiles this harness packs.
        od = np.zeros_like(od); om = np.zeros_like(om); oc = np.zeros_like(oc)
    attn3 = (attn3_override if attn3_override is not None
             else np.zeros(K_DIM, np.float32)
             if __import__('os').environ.get('CHAIN_P3_ZERO')
             else rnd(rng.standard_normal(K_DIM) * 0.05))
    bc3 = np.zeros(2 * K_DIM + 2 * HEAD, np.float32)
    bc3[:K_DIM] = attn3
    bc3[K_DIM:2 * K_DIM] = x                            # the residual P3 adds
    bc3_t = iron.tensor(bc3.astype(bfloat16), dtype=bfloat16, device="npu")

    # rows so a pair's join is a contiguous global run, as resid_chain packs them
    rpp3 = K_DIM // npairs
    p3rows = lambda pr, j: [pr * rpp3 + t * 2 * NROWS + j * NROWS
                            for t in range(p3tiles)]
    nbc3 = K_DIM // 32
    w3, h_ref = [], np.zeros(K_DIM, np.float64)
    for pr in range(npairs):
        per = []
        for j in range(2):
            blob = []
            for r0 in p3rows(pr, j):
                blob.append(q4nx.pack_tile(od[r0:r0 + NROWS, :nbc3],
                                           om[r0:r0 + NROWS, :nbc3],
                                           oc[r0:r0 + NROWS, :nbc3],
                                           row_base=r0, flags=0.0))
                got = q4nx.gemv_reference_bf16(attn3, od[r0:r0 + NROWS, :nbc3],
                                               om[r0:r0 + NROWS, :nbc3],
                                               oc[r0:r0 + NROWS, :nbc3])
                h_ref[r0:r0 + NROWS] = rnd(got + x[r0:r0 + NROWS])
            per.append(np.concatenate(blob))
        b = np.empty((p3tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w3.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))
    # ONE buffer that P3 drains into and P4 is filled from -- this is what
    # actually chains the two phases -- built below, once nw2 is loaded.

    # ---- P4: gate/up + SwiGLU ---------------------------------------------
    # Its activation is P3's OWN DEVICE OUTPUT: P3 drains h into `h_all` in
    # natural row order and P4's broadcast is filled from that same buffer, so
    # this seam is genuinely composed rather than host-fed.
    D_FF = 8192
    p4tiles = D_FF // (NCORES * NROWS)
    p4per = 2 * HEAD // NROWS
    p4objs = p4tiles // p4per
    gd, gm, gc = load_linear(c, f"model.layers.{o.layer}.mlp.gate_proj.weight",
                             D_FF, K_DIM)
    ud, um, uc = load_linear(c, f"model.layers.{o.layer}.mlp.up_proj.weight",
                             D_FF, K_DIM)
    h_act = rnd(h_ref.astype(np.float32))
    # h is the FFN's input and it must be RMSNormed with the layer's SECOND
    # norm weight before gate/up. Mirrors P1's rounding exactly.
    nw2 = c.bf16(f"model.layers.{o.layer}.post_attention_layernorm.weight"
                 ).astype(np.float32)[:K_DIM]
    hd = h_act.astype(np.float64)
    inv2 = np.float32(1.0 / np.sqrt((hd * hd).mean() + EPS))
    h_n = rnd(rnd(h_act * rnd(inv2)) * nw2)
    # ONE buffer: P3's drain target AND P4's broadcast source. [0:K_DIM] is
    # written by the drain (so its initial value is irrelevant); nw2 lives at
    # [K_DIM:2*K_DIM] and the drain never touches it.
    bc4 = np.zeros(2 * K_DIM + 2 * HEAD, np.float32)
    bc4[K_DIM:2 * K_DIM] = nw2
    h_all = iron.tensor(bc4.astype(bfloat16), dtype=bfloat16, device="npu")
    h_ts = [h_all] * npairs          # every pair scatters into the same buffer
    bc4_t = h_all                    # ...which P4 is then filled from

    rpp4 = D_FF // npairs
    # A core's rows must be CONTIGUOUS, not interleaved with its partner's.
    # flm_gemv_up_swiglu writes at row_base % DIM_OBJROWS, and an interleave of
    # 2*NROWS makes that modulo collide — 8 distinct slots for 16 steps, half
    # the object unwritten and half written twice. Contiguous gives
    # t*NROWS % 128 = 0,8,...,120, tiling the object exactly.
    p4rows = lambda pr, j: [pr * rpp4 + j * (rpp4 // 2) + t * NROWS
                            for t in range(p4tiles)]
    nbc4 = K_DIM // 32
    w4, sw_ref = [], np.zeros(D_FF, np.float64)
    for pr in range(npairs):
        per = []
        for j in range(2):
            blob = []
            for r0 in p4rows(pr, j):
                sl = slice(r0, r0 + NROWS)
                blob.append(q4nx.pack_tile(gd[sl, :nbc4], gm[sl, :nbc4],
                                           gc[sl, :nbc4], row_base=r0, flags=0.0))
                blob.append(q4nx.pack_tile(ud[sl, :nbc4], um[sl, :nbc4],
                                           uc[sl, :nbc4], row_base=r0, flags=0.0))
                g = rnd(q4nx.gemv_reference_bf16(h_n, gd[sl, :nbc4],
                                                 gm[sl, :nbc4], gc[sl, :nbc4]))
                u = rnd(q4nx.gemv_reference_bf16(h_n, ud[sl, :nbc4],
                                                 um[sl, :nbc4], uc[sl, :nbc4]))
                sw_ref[sl] = rnd(g / (1.0 + np.exp(-g.astype(np.float64))) * u)
            per.append(np.concatenate(blob))
        b = np.empty((2 * p4tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w4.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))
    sw_all = iron.zeros(D_FF, dtype=bfloat16, device="npu")
    sw_ts = [sw_all] * npairs      # every pair scatters into the same buffer

    import os as _oh
    if _oh.environ.get("CHAIN_HOST_KV"):
        # the same cache contents, built on the host: P1 still runs and still
        # drains, but P2 reads this instead
        # multi-object: the appended token lives in object pos // TSEQ, and the
        # trailer belongs on EVERY object, not just the first.
        hostc = cache.reshape(nobj * NATT, SLOT).copy()
        for g in range(NATT):
            row = (pos // TSEQ) * NATT + g
            K = hostc[row, :KTILE].reshape(HEAD, TSEQ)
            V = hostc[row, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
            K[:, pos % TSEQ] = ref[NQ + g]
            V[pos % TSEQ] = ref[NK + g]
        hraw = hostc.astype(bfloat16).view(np.uint16)
        for _o in range(nobj):
            for g in range(NATT):
                hraw[_o * NATT + g, (OPERAND - 64) // 2:(OPERAND - 64) // 2 + 2] = \
                    np.array([float(g * GQA * OBJ)], np.float32).view(np.uint16)
        host_t = iron.tensor(hraw.reshape(-1).view(np.uint8), dtype=np.uint8,
                             device="npu")
        design(bc_t, *w_ts, *q_in, *[host_t] * apairs, *q_ts,
               *[cache_t] * npairs, *a_ts, bc3_t, *w3, *h_ts,
               bc4_t, *w4, *sw_ts)
    else:
        _args = (bc_t, *w_ts, *q_in, *q_ts, *[cache_t] * npairs, *a_ts,
                 bc3_t, *w3, *h_ts, bc4_t, *w4, *sw_ts)
        if o.bench:
            _b = run_iters(design, *_args, warmup=2, iters=10)
            _us = _b.npu.min_us if _b.npu else _b.e2e.min_us
        else:
            design(*_args)
            _us = None

    # first: did P1 write the cache correctly inside THIS harness?
    # object 0 carries the real KV; P1 appends there and the rest is padding
    cv = (cache_t.numpy().view(bfloat16).astype(np.float64)
          .reshape(nobj, NATT, SLOT))
    ke = ve = 0.0
    for g in range(NATT):
        for ob in range(nobj):
            K = cv[ob, g, :KTILE].reshape(HEAD, TSEQ)
            V = cv[ob, g, KTILE:2 * KTILE].reshape(TSEQ, HEAD)
            if ob == pos // TSEQ:               # the token P1 just appended
                ke = max(ke, np.abs(K[:, pos % TSEQ] - ref[NQ + g]).max())
                ve = max(ve, np.abs(V[pos % TSEQ] - ref[NK + g]).max())
            lo, hi = ob * TSEQ, min(pos, (ob + 1) * TSEQ)
            if hi > lo:                          # the prior cache in this object
                ke = max(ke, np.abs(K[:, :hi - lo] - Kc[g][lo:hi].T).max())
                ve = max(ve, np.abs(V[:hi - lo] - Vc[g][lo:hi]).max())
    print(f"  P1 cache: k' col {pos} + prior cols max err {ke:.4e};  "
          f"v' row {pos} + prior rows max err {ve:.4e}")

    # ---- reference: attention over the cache INCLUDING P1's appended token --
    if o.bench and _us is not None:
        FIXED_US = 92.9
        p1_b = p1pairs * 2 * hpc * TPH * q4nx.tile_bytes(K_DIM, NROWS)
        kv_b = apairs * 2 * OPERAND * nobj
        mb = (p1_b + kv_b) / 1e6
        print(f"  bench: {mb:.2f} MB  {mb*1e3/_us:.1f} GB/s  {_us:.1f} us "
              f"(marginal {_us - FIXED_US:.1f}, 16-core ideal {mb*17.85:.1f})")
    # dense now: each core emits its whole 128-row slice in one object
    # q' is now P1's OWN output, scattered into the broadcast P2 reads. The host
    # writes ZEROS for every head, so this check fails loudly if the scatter is
    # wrong -- attention would be running on zeros, not on a plausible host value.
    qgot = q_all.numpy().astype(np.float64)
    eq = max(np.abs(qgot[h * OBJ:h * OBJ + HEAD] - ref[h][:HEAD]).max()
             for h in range(NQ))
    print(f"  P1 q' in P2's broadcast: max err {eq:.4e} over {NQ} heads "
          f"(host wrote zeros)")

    # h_all is one shared buffer now, already in natural row order.
    got_h = h_all.numpy().astype(np.float64)[:K_DIM]
    order = np.concatenate([np.array(p3rows(pr, j)) + r
                            for pr in range(npairs) for t in range(p3tiles)
                            for j in (0, 1) for r in range(0)] or [np.zeros(0, int)])
    # object order is [pair][core], each carrying that core's rows in tile order
    h_idx = np.arange(K_DIM)      # the drain scatters, so no permutation
    e3 = np.abs(got_h - h_ref[h_idx]).max()
    if __import__("os").environ.get("CHAIN_P3_DIAG"):
        sg, sr = np.sort(got_h), np.sort(h_ref)
        print(f"    DIAG sorted-multiset maxdiff {np.abs(sg-sr).max():.4e} -> "
              f"{'PERMUTATION (my h_idx)' if np.abs(sg-sr).max()<1e-2 else 'VALUES differ (device)'}")
    print(f"  P3 h      : max err {e3:.4e}  mean|ref| {np.abs(h_ref).mean():.5f}")
    if __import__("os").environ.get("CHAIN_P3_DIAG"):
        srt_g, srt_r = np.sort(got_h), np.sort(h_ref)
        print(f"    DIAG got[:6]  {got_h[:6].round(4)}")
        print(f"    DIAG want[:6] {h_ref[h_idx][:6].round(4)}")
        print(f"    DIAG sorted-multiset maxdiff {np.abs(srt_g - srt_r).max():.4e}"
              f"  -> {'PERMUTATION (ordering bug)' if np.abs(srt_g-srt_r).max() < 1e-2 else 'different VALUES (not ordering)'}")
        print(f"    DIAG |got| mean {np.abs(got_h).mean():.5f} vs |ref| {np.abs(h_ref).mean():.5f}"
              f"  ratio {np.abs(got_h).mean()/max(np.abs(h_ref).mean(),1e-12):.3f}")
        # which broadcast fill did P3 actually read? try the other two.
        for nm, act in (("fill1 x (raw)", bc[:K_DIM].astype(np.float32)),
                        ("fill1 xn (normed)", xn.astype(np.float32)),
                        ("fill2 q' block", qall[:K_DIM].astype(np.float32))):
            alt = np.zeros(K_DIM, np.float64)
            for pr in range(npairs):
                for j in (0, 1):
                    for r0 in p3rows(pr, j):
                        g = q4nx.gemv_reference_bf16(rnd(act),
                                od[r0:r0+NROWS, :nbc3], om[r0:r0+NROWS, :nbc3],
                                oc[r0:r0+NROWS, :nbc3])
                        alt[r0:r0+NROWS] = rnd(g + x[r0:r0+NROWS])
            print(f"    DIAG vs {nm:18s}: max err {np.abs(got_h - alt[h_idx]).max():.4e}")
    got_sw = sw_all.numpy().astype(np.float64)   # already row-ordered
    # stream order is [object][core], each object carrying OBJ contiguous rows
    e4 = np.abs(got_sw - sw_ref).max()
    print(f"  P4 sw     : max err {e4:.4e}  mean|ref| {np.abs(sw_ref).mean():.5f}")
    if o.layer_pass:
        import p5_pass
        # the residual P5 adds is x, the layer's input — in the fused layer it
        # reaches P5 through g_resid, which P3 stashed and which persists across
        # the dispatch boundary (static_persist_probe). Host-supplied here.
        x_out, us5 = p5_pass.run(sw_all, x.astype(np.float32), layer=o.layer)
        # reference: down_proj over the sw side A actually produced, plus x
        swv = sw_all.numpy().astype(np.float32)
        dd, dm, dc = p5_pass.load_linear(
            c, f"model.layers.{o.layer}.mlp.down_proj.weight", 2048, 8192)
        nbc5 = 8192 // 32
        ref5 = np.zeros(2048, np.float64)
        for r0 in range(0, 2048, 8):
            sl = slice(r0, r0 + 8)
            acc = np.zeros(8, np.float64)
            for ch in range(4):
                lo = ch * (nbc5 // 4); hi = lo + nbc5 // 4
                acc += q4nx.gemv_reference_bf16(
                    rnd(swv[ch * 2048:(ch + 1) * 2048]),
                    dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi])
            ref5[r0:r0 + 8] = rnd(acc + x[r0:r0 + 8])
        e5 = np.abs(x_out - ref5).max()
        print(f"  LAYER x_out: max err {e5:.4e}  mean|ref| {np.abs(ref5).mean():.5f}"
              f"  (side B on side A's own sw)")
    print(f"P1 -> P2 in one dispatch: seq {o.seq} (P1 appends at pos {pos}), "
          f"{nobj} KV objects, npad {npad}")
    worst, scale = 0.0, 0.0
    pmax = spread = 0.0
    nres = rmag = 0.0
    vmax = vmean = 0.0
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
        got = (attn_out.numpy().astype(np.float64)
               .reshape(apairs, 2, GQA, HEAD)[a // 2, a % 2])
        worst = max(worst, np.abs(got - want).max())
        scale = max(scale, np.abs(want).mean())
        # How SHARP is this head's softmax? The exp2 NLF is a piecewise LUT, so
        # its relative error is worst where the distribution concentrates: a
        # near-one-hot softmax puts the whole output on a few v rows and the
        # LUT's error on those weights lands undiluted. Track it alongside the
        # error so a growing error can be attributed rather than guessed at.
        pmax = max(pmax, (e / e.sum(1, keepdims=True)).max())
        spread = max(spread, (sc.max(1) - sc.min(1)).max())
        # P2 is an ONLINE softmax: it walks the sequence keeping a running max
        # and, whenever a bigger score appears, rescales everything accumulated
        # so far by exp2(m_old - m_new). Each rescale is a lossy bf16 multiply
        # over the whole accumulator, so the error depends on the rescale
        # HISTORY -- how often the max moves and how far -- not on how sharp the
        # final distribution is. Sharpness is flat across layers; this need not
        # be, which makes it the candidate that survives.
        for row in sc:
            m, n, tot = -np.inf, 0, 0.0
            for v in row:
                if v > m:
                    if m > -np.inf:
                        n += 1
                        tot += v - m
                    m = v
            nres, rmag = max(nres, n), max(rmag, tot)
        # The last input-side property that can vary with depth: V's dynamic
        # range. The output is a weighted SUM of v rows, so a few large-magnitude
        # rows accumulated alongside many small ones lose the small ones' low
        # bits -- classic bf16 cancellation, and it scales with max|V| rather
        # than with mean|V| (which the tolerance is built from).
        vmax = max(vmax, np.abs(Vfull).max())
        vmean = max(vmean, np.abs(Vfull).mean())
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
    print(f"  softmax: max weight {pmax:.4f}   max logit spread {spread:.2f}   rescales {nres:.0f}  total rescale {rmag:.2f}")
    print(f"  attention floor: {worst / max(vmax, 1e-9) / 2**-8:.2f} ULP of max|V|  (envelope 2.00, empirical)")
    print(f"  V range: max|V| {vmax:.4f}  mean|V| {vmean:.4f}  ratio {vmax / max(vmean, 1e-9):.1f}")
    # The floor is ONE bf16 ULP at the scale of the largest v the accumulator
    # ever holds -- not the exp2 NLF, and not proportional to mean|ref|.
    #
    # Measured at layers 0/7/15: err/max|V| = 2.88e-3, 4.04e-3, 3.89e-3 against
    # bf16 eps = 2^-8 = 3.906e-3. Every one is within ~1 ULP. The old
    # `8e-2 * mean|ref|` tolerance tracked the wrong quantity and so failed at
    # layers 7 and 15 purely because max|V| doubles there (1.20 -> 2.39) while
    # mean|ref| moves 22%.
    #
    # Three other explanations were measured and refuted before this one: the
    # exp2 NLF's sharpness (max softmax weight is FLAT at 0.13-0.17 and flattest
    # where the error is worst), inheritance from P1 (k' and v' are bit-exact at
    # layers 7 and 15) and the online-softmax rescale history (the rescale count
    # DECREASES, 7 -> 6 -> 5, as the error grows).
    # 2 ULP is an EMPIRICAL envelope, not a derived bound. Measured err/ULP:
    #
    #   seq  9 (npad 23): 1.61      layer  0: 0.74
    #   seq 17 (npad 15): 0.95      layer  7: 1.03
    #   seq 31 (npad  1): 0.74      layer 15: 1.00
    #   seq 32 (npad  0): 1.19
    #
    # The bf16-ULP SCALE is solid -- dividing by max|V| is what collapses the
    # across-layer spread from 2.7x to ~1x. What is NOT explained is the
    # remaining 0.74-1.61 variation across sequence length: neither npad nor
    # softmax concentration orders it (seq 32 has the least padding AND the
    # flattest softmax, yet lands at 1.19). So this bound is fitted to observed
    # worst case, and a regression that pushed the true floor to 1.9 ULP would
    # pass. The printed ratio below is the number to watch, not the verdict.
    tol = 2.0 * 2**-8 * vmax
    print(f"  attention out: max err {worst:.4e}   mean|ref| {scale:.5f}   "
          f"tol {tol:.4e}")
    print(f"  -> {'PASS' if worst <= tol else 'FAIL'}  (bf16-ULP envelope at max|V| = {vmax:.3f})")
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
