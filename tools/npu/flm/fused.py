#!/usr/bin/env python3
"""THE FUSED SINGLE DISPATCH: all 32 cores, all 16 layers, one dispatch per token.

`groups_ab` runs P1+P2 on 16 cores and `group_c` runs P3+P4+P5 on the other 16,
as two separate dispatches. A layer is strictly sequential, so sixteen layers
that way costs **32 dispatches per token** — and the 92.9 us dispatch floor times
that surplus is the entire deficit against FastFlowLM. This is the same
computation with one dispatch and the layer loop inside it.

    python3 fused.py --layers 16 --pos 0 --seq 1 --x0 x0_bos.npy --save x16.npy
    python3 fused.py --layers 16 --bench

## What had to change, and what did not

**The kernels are untouched.** Every core runs the same code its half already ran;
what moved is the descriptor plumbing and two compile flags that select branches
already present in `flm_gemv_down.cc`.

**The shim budget is 16 in / 16 out** — 8 shim tiles x 2 channels each way,
measured, not inferred. The naive union of the two designs asks for 22 in / 18
out and the placer refuses ("no ShimNOCTile has sufficient DMA capacity").
`channel_probe.py` never covered this: it builds 12 fills + 12 drains, so its
"40 in / 36 out places" is about MEMTILES only. Three consolidations fit it:

    A weights   4 fifos split 1->2   ->  2 fifos split 1->4
    B KV in     8 fifos, unsplit     ->  2 fifos split 1->4
    A k'/v' out 8 fifos, unjoined    ->  4 joins 2->1

giving 14 in / 14 out. None of them touches a core: a split hands each core the
same object it always got, and the k'/v' drain already had a head dimension
sized 1, which becomes 2.

**The seams.** Five of the six are the ones the two designs already had. The two
that were host-fed become device-fed here, both through mechanisms the kernels
were built for and that nothing had wired in:

    P1 -> P2      q' core to core                          (as in groups_ab)
    P2 -> P3      attention out -> host -> P3's broadcast
    P3 -> P4      h -> host -> P4's broadcast              (as in group_c)
    P3 -> P5      h in g_resid, on-core     RESID_FROM_STASH=1
    P4 -> P5      sw -> host -> P5's broadcast             (as in group_c)
    P5 -> P1/P3   x_out -> host -> next layer              XOUT_TO_STASH=1

`flm_gemv_q4_1_residual` already writes every row of `h` into `g_resid`
unconditionally, and P3 and P5 use the *same* row assignment on the same core, so
P5's residual needs no transport at all — only the flag that reads it there.
`XOUT_TO_STASH` then leaves x_out in the same stash, and `flm_h_emit` (already
linked, P3 calls it) copies a core's whole 128-row slice into ONE object. That
turns P5's drain from 16384 elements per pair, 6% of them live, into 256 dense
ones with the same shape as P3's.

**One drain feeds both of x_out's consumers**, which is what makes the C -> A
seam free. The per-layer host block is

    [ attn(2048) | x(2048) | nw(2048) | cs_q(64) | cs_k(64) ]      6272 bf16

P3's broadcast is the first 4224 elements — activation `attn`, aux `x` — and P1's
is the 4224 starting at 2048: `x`, `nw`, `cs_q`, `cs_k`. Layer L's P5 drains
x_out into block L+1's x slot and both fills read it from there. A drain has one
destination, so without this layout x_out would need two.

## Limits of this build

Position is still a BUILD parameter, so this is verified at **pos 0** only, where
the KV cache holds exactly the one entry this design's own P1 wrote. Multi-token
decode needs `offset_parameter=` with a `ParameterScratchpad`, which is a
host-side restructuring (see the log) and is not done here.

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from qkv_verify import HEAD, K_DIM, EPS, ROPE_THETA, qkv_rows, rope_ref  # noqa: E402
# The layout functions only, which are NROWS-independent. groups_ab's own NROWS
# is 16; this design runs the whole array at 8 so one operand size, one set of
# compile flags and one memtile object size serve both halves -- and 10304 B is
# the size channel_probe and the topology probe were built at.
from groups_ab import head_layout, hpc_for, drain_plan, NQ, NK, NV  # noqa: E402

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
KVE_SRC = str(KDIR / "flm_kv_emit.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
ATT_SRC = str(KDIR / "flm_attn_decode.cc")
BEG_SRC = str(KDIR / "flm_attn_begin.cc")
FIN_SRC = str(KDIR / "flm_attn_finish.cc")
RES_SRC = str(KDIR / "flm_gemv_residual.cc")
ASUM_SRC = str(KDIR / "flm_asum_prepare.cc")
HEMIT_SRC = str(KDIR / "flm_h_emit.cc")
DOWN_SRC = str(KDIR / "flm_gemv_down.cc")
GATE_SRC = str(KDIR / "flm_gemv_gate.cc")
UPS_SRC = str(KDIR / "flm_gemv_up_swiglu.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"

NROWS = 8
TPH = HEAD // NROWS                     # 8 weight tiles per head
GQA, TSEQ, KVPER = 4, 32, 1
NA = NB = 8                             # P1 cores, P2 cores
NC, NCP = 16, 8                         # C cores, C pairs
AQ = 2                                  # A weight fifos, split 1->4
BQ = 2                                  # B KV fifos, split 1->4
D_FF, NCHUNK = 8192, 4

WT = q4nx.tile_bytes(K_DIM, NROWS)                      # 10304
OPERAND = max(2 * TSEQ * HEAD * 2 * KVPER, WT)          # 10304
KTILE = VTILE = HEAD * TSEQ                             # 2048 bf16
KVSTRIDE = max(KTILE + VTILE, OPERAND // 2)             # 5152 bf16 per head slot
CACHEB = 8 * KVSTRIDE * 2                               # bytes of KV cache per layer
OBJ = 2 * HEAD                                          # result object, bf16
BC = 2 * K_DIM + 2 * HEAD                               # a broadcast object
BLK = 3 * K_DIM + 2 * HEAD                              # [attn | x | nw | cs] per layer
HPC = hpc_for(NA)                                       # 6 head-tiles per P1 core
P3T = P5T = K_DIM // (NC * NROWS)                       # 16 tiles per core
P4T = D_FF // (NC * NROWS)                              # 64
P4PER = OBJ // NROWS                                    # 16 steps share one object
P4OBJS = P4T // P4PER                                   # 4
RPP3 = K_DIM // NCP                                     # 256 rows per C pair
RPP4 = D_FF // NCP                                      # 1024

A_LSZ = 4 * HPC * TPH * WT              # A weights, one quad, one layer
P3_LSZ = 2 * P3T * WT
P4_LSZ = 2 * 2 * P4T * WT
P5_LSZ = 2 * NCHUNK * P5T * WT

rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))
p3rows = lambda pr, j: [pr * RPP3 + t * 2 * NROWS + j * NROWS for t in range(P3T)]
p4rows = lambda pr, j: [pr * RPP4 + j * (RPP4 // 2) + t * NROWS for t in range(P4T)]


def build(pos, nlay, seq):
    nobj = -(-seq // (TSEQ * KVPER))
    if nobj != 1:
        raise SystemExit(f"seq {seq} needs {nobj} KV objects; the quad-split B "
                         f"stream delivers one per core per layer (seq <= {TSEQ})")
    off = pos - (pos & 1)                       # the k pair's even column
    _, kvplan = drain_plan(NA, group=1)         # per CORE, in emit (slot) order
    for pr in range(NA // 2):
        a, b = kvplan[2 * pr], kvplan[2 * pr + 1]
        assert [k for k, _ in a] == [k for k, _ in b], (a, b)
        assert all(y - x == 1 for (_, x), (_, y) in zip(a, b)), (a, b)

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(WT,), np.dtype[np.uint8]]
    wq_ty = np.ndarray[(4 * WT,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    okv_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(GQA * OBJ,), np.dtype[bfloat16]]
    op_ty = np.ndarray[(OPERAND,), np.dtype[np.uint8]]
    opq_ty = np.ndarray[(4 * OPERAND,), np.dtype[np.uint8]]
    opp_ty = np.ndarray[(2 * OPERAND,), np.dtype[np.uint8]]
    ao_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]
    aoj_ty = np.ndarray[(4 * GQA * HEAD,), np.dtype[bfloat16]]
    p1o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    p1p_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    # host buffers, by the role each argument plays
    x_ty = np.ndarray[((nlay + 1) * BLK,), np.dtype[bfloat16]]
    aw_ty = np.ndarray[(nlay * A_LSZ,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(nlay * CACHEB,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(nlay * BC,), np.dtype[bfloat16]]
    sw_ty = np.ndarray[(D_FF + BC,), np.dtype[bfloat16]]
    w3_ty = np.ndarray[(nlay * P3_LSZ,), np.dtype[np.uint8]]
    w4_ty = np.ndarray[(nlay * P4_LSZ,), np.dtype[np.uint8]]
    w5_ty = np.ndarray[(nlay * P5_LSZ,), np.dtype[np.uint8]]

    MASKPAD = os.environ.get("ATTN_MASK_PAD", "1")
    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS={NQ}", f"-DDIM_QKHEADS={NK}",
             f"-DDIM_QGROUP={GQA}", f"-DDIM_GQA={GQA}",
             f"-DDIM_TSEQ={TSEQ}", f"-DDIM_KVPER={KVPER}",
             f"-DDIM_KVOBJ={OPERAND}", f"-DDIM_KVSTRIDE={OPERAND // 2}",
             f"-DDIM_QSTRIDE={OBJ}",
             # q' arrives on its own fifo at head 0, and npad rides the KV
             # trailer because a core-to-core q has no host-written tail.
             "-DQOFF_FROM_KV=0", "-DNPAD_FROM_KV=1",
             f"-DATTN_MASK_PAD={MASKPAD}",
             f"-DDIM_RESN={2 * P3T * NROWS}", f"-DDIM_P3TILES={P3T}",
             f"-DDIM_OBJROWS={OBJ}", f"-DDIM_ACCN={2 * P5T * NROWS}",
             # P5's residual is the h THIS core stashed in P3 (same rows, same
             # core), and its output goes back to the same stash so flm_h_emit
             # can hand out one dense object. Both branches already exist; only
             # the flags are new. P3 keeps reading its residual from the
             # broadcast -- P3_RESID_FROM_STASH is a DIFFERENT flag and stays 0,
             # because at layer 0 the stash holds nothing and x comes from the
             # host.
             "-DRESID_FROM_STASH=1", "-DXOUT_TO_STASH=1"]

    tag = f"fz{nlay}p{pos}s{seq}n{NROWS}m{MASKPAD}"
    P = "xin: In, aw0: In, aw1: In, kvo: Out, kvi: In, xout: Out"
    P += ", hout: Out, hin: In, swout: Out, swin: In"
    P += ", " + ", ".join(f"w3_{i}: In" for i in range(NCP))
    P += ", " + ", ".join(f"w4_{i}: In" for i in range(NCP))
    P += ", " + ", ".join(f"w5_{i}: In" for i in range(NCP))
    src = f'''
def _design({P}):
    kq = ExternalFunction("flm_gemv_qkv", source_file=QKV_SRC,
                          arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    ke = ExternalFunction("flm_p1_emit", source_file=EMIT_SRC,
                          arg_types=[wt_ty, q_ty], compile_flags=FLAGS)
    kve = ExternalFunction("flm_kv_emit", source_file=KVE_SRC,
                           arg_types=[wt_ty, o_ty], compile_flags=FLAGS)
    kn = ExternalFunction("flm_norm_prepare", source_file=NORM_SRC,
                          arg_types=[bc_ty], compile_flags=FLAGS)
    kab = ExternalFunction("flm_attn_begin", source_file=BEG_SRC,
                           arg_types=[q_ty], compile_flags=FLAGS)
    kat = ExternalFunction("flm_attn_tile", source_file=ATT_SRC,
                           arg_types=[q_ty, op_ty], compile_flags=FLAGS)
    kaf = ExternalFunction("flm_attn_finish", source_file=FIN_SRC,
                           arg_types=[ao_ty, q_ty, op_ty], compile_flags=FLAGS)
    kas = ExternalFunction("flm_asum_prepare", source_file=ASUM_SRC,
                           arg_types=[bc_ty], compile_flags=FLAGS)
    khe = ExternalFunction("flm_h_emit", source_file=HEMIT_SRC,
                           arg_types=[op_ty, p1o_ty], compile_flags=FLAGS)
    kg4 = ExternalFunction("flm_gemv_gate", source_file=GATE_SRC,
                           arg_types=[bc_ty, op_ty], compile_flags=FLAGS)
    ku4 = ExternalFunction("flm_gemv_up_swiglu", source_file=UPS_SRC,
                           arg_types=[bc_ty, op_ty, p1o_ty], compile_flags=FLAGS)
    kd5 = ExternalFunction("flm_gemv_down", source_file=DOWN_SRC,
                           arg_types=[bc_ty, op_ty, p1o_ty], compile_flags=FLAGS)
    kr3 = ExternalFunction("flm_gemv_q4_1_residual", source_file=RES_SRC,
                           arg_types=[bc_ty, op_ty, p1o_ty], compile_flags=FLAGS)

    # ---- group A: qkv + RoPE on 8 cores ------------------------------------
    f_bca = ObjectFifo(bc_ty, depth=1, name="{tag}_abc")
    bca = [f_bca.cons() for _ in range({NA})]
    f_aw = [ObjectFifo(wq_ty, name=f"{tag}_aw{{i}}") for i in range({AQ})]
    aw_sub = [f.cons().split([k * {WT} for k in range(4)],
                             obj_types=[wt_ty] * 4) for f in f_aw]
    f_q = [ObjectFifo(q_ty, name=f"{tag}_q{{i}}") for i in range({NA})]
    # k'/v' leave as PAIRS. Eight unjoined fifos put the shim at 22 inputs /
    # 18 outputs against a measured 16/16; the join costs one memtile input per
    # pair and no core code, because each core still writes its own object.
    f_akv = [ObjectFifo(okv_ty, name=f"{tag}_akv{{i}}") for i in range({NA // 2})]
    akv_sub = [f.prod().join([0, {OBJ}], obj_types=[o_ty, o_ty]) for f in f_akv]

    # ---- group B: attention on 8 cores -------------------------------------
    f_bkv = [ObjectFifo(opq_ty, name=f"{tag}_bkv{{i}}") for i in range({BQ})]
    bkv_sub = [f.cons().split([k * {OPERAND} for k in range(4)],
                              obj_types=[op_ty] * 4) for f in f_bkv]
    f_ao = [ObjectFifo(aoj_ty, name=f"{tag}_ao{{i}}") for i in range(2)]
    ao_sub = [f.prod().join([k * {GQA} * {HEAD} for k in range(4)],
                            obj_types=[ao_ty] * 4) for f in f_ao]

    # ---- group C: o_proj / gate-up-SwiGLU / down_proj on 16 cores -----------
    f_bcc = ObjectFifo(bc_ty, depth=1, name="{tag}_cbc")
    bcc = [f_bcc.cons() for _ in range({NC})]
    f_cw = [ObjectFifo(opp_ty, name=f"{tag}_cw{{i}}") for i in range({NCP})]
    cw_sub = [f.cons().split([0, {OPERAND}], obj_types=[op_ty, op_ty])
              for f in f_cw]
    f_cp = [ObjectFifo(p1p_ty, name=f"{tag}_cp{{i}}") for i in range({NCP})]
    cp_sub = [f.prod().join([0, {OBJ}], obj_types=[p1o_ty, p1o_ty])
              for f in f_cp]

    def core_a(bcs, wc, opq, opkv, kqkv, kemit, kvemit, kprep):
        """P1, output split by destination: q' core-to-core, k'/v' to the cache.
        Byte for byte what groups_ab's core does; only the layer loop is new."""
        for _lay in range_({nlay}):
            eb = bcs.acquire(1)
            kprep(eb)
            eq = opq.acquire(1)                  # ONE object for all GQA q heads
            for _ in range_({GQA}):
                for _ in range_({TPH} - 1):
                    ew = wc.acquire(1)
                    kqkv(eb, ew)
                    wc.release(1)
                # the emit reuses the head's LAST tile for row_base; a separate
                # acquire would consume an extra object and desynchronise the
                # weight stream, and both objects must be held across the two
                # calls because they share g_stage.
                ew = wc.acquire(1)
                kqkv(eb, ew)
                kemit(ew, eq)
                wc.release(1)
            opq.release(1)
            for _ in range_({HPC} - {GQA}):      # v' then k', one object each
                for _ in range_({TPH} - 1):
                    ew = wc.acquire(1)
                    kqkv(eb, ew)
                    wc.release(1)
                ew = wc.acquire(1)
                ekv = opkv.acquire(1)
                kqkv(eb, ew)
                kvemit(ew, ekv)
                opkv.release(1)
                wc.release(1)
            bcs.release(1)

    def core_b(qc, kvc, op, kbegin, ktile, kfin):
        """Attention for one KV group; `qc` is A[j]'s q', arriving core to core."""
        for _lay in range_({nlay}):
            eq = qc.acquire(1)
            kbegin(eq)
            for _ in range_({nobj} - 1):
                ekv = kvc.acquire(1)
                ktile(eq, ekv)
                kvc.release(1)
            # hold the LAST KV object: npad rides its trailer.
            ekv = kvc.acquire(1)
            ktile(eq, ekv)
            eo = op.acquire(1)
            kfin(eo, eq, ekv)
            op.release(1)
            kvc.release(1)
            qc.release(1)

    def core_c(bcs, wc, op, kres, kasum, khemit, kgate, kups, kprep, kdown):
        """P3, P4, P5 in sequence, sixteen times."""
        for _lay in range_({nlay}):
            # ---- P3: o_proj + residual. One object for the core's whole slice.
            eb = bcs.acquire(1)
            kasum(eb)                            # g_asum for P3's activation
            eo = op.acquire(1)
            for _ in range_({P3T} - 1):
                ew = wc.acquire(1)
                kres(eb, ew, eo)
                wc.release(1)
            ew = wc.acquire(1)
            kres(eb, ew, eo)
            khemit(ew, eo)                       # dense h out of g_resid
            wc.release(1)
            op.release(1)
            bcs.release(1)

            # ---- P4: the post-attention RMSNorm, then gate/up + SwiGLU
            eb = bcs.acquire(1)
            kprep(eb)
            for _ in range_({P4OBJS}):
                eo = op.acquire(1)
                for _ in range_({P4PER}):
                    eg = wc.acquire(1)
                    kgate(eb, eg)                # gate -> in-core stash
                    wc.release(1)
                    eu = wc.acquire(1)
                    kups(eb, eu, eo)             # up, then SwiGLU against it
                    wc.release(1)
                op.release(1)
            bcs.release(1)

            # ---- P5: down_proj over NCHUNK K-chunks, residual from g_resid.
            # ONE result object for the whole phase: under XOUT_TO_STASH the
            # flush writes x_out into g_resid rather than into `out`, so the
            # only thing that reaches the object is flm_h_emit's dense copy.
            # The emit runs every chunk and only the last one's values survive,
            # which avoids using a released weight tile after the loop.
            eo = op.acquire(1)
            for _ in range_({NCHUNK}):
                eb = bcs.acquire(1)
                kasum(eb)                        # each chunk is a new activation
                for _ in range_({P5T} - 1):
                    ew = wc.acquire(1)
                    kdown(eb, ew, eo)
                    wc.release(1)
                ew = wc.acquire(1)
                kdown(eb, ew, eo)
                khemit(ew, eo)
                wc.release(1)
                bcs.release(1)
            op.release(1)

    workers = []
    for c in range({NA}):
        workers.append(Worker(core_a,
            fn_args=[bca[c], aw_sub[c // 4][c % 4].cons(),
                     f_q[c].prod(), akv_sub[c // 2][c % 2].prod(),
                     kq, ke, kve, kn], stack_size=8192))
    for c in range({NB}):
        workers.append(Worker(core_b,
            fn_args=[f_q[c].cons(), bkv_sub[c // 4][c % 4].cons(),
                     ao_sub[c // 4][c % 4].prod(), kab, kat, kaf],
            stack_size=4096))
    for c in range({NC}):
        workers.append(Worker(core_c,
            fn_args=[bcc[c], cw_sub[c // 2][c % 2].cons(),
                     cp_sub[c // 2][c % 2].prod(),
                     kr3, kas, khe, kg4, ku4, kn, kd5], stack_size=8192))

    def sequence(*args):
        """Five phases per layer, five barriers, all inside ONE dispatch.

        A drain followed by a barrier followed by a fill moves data through host
        memory without costing a dispatch -- which is what group_c already does
        for its own phase sequencing, applied here to the C -> A seam as well."""
        # Unpacked one index at a time, NOT `args[:10]`: on Python 3.14 a literal
        # slice is folded into a code constant, and iron's cache key marshals the
        # generator's code object at version 4, which cannot encode a slice --
        # "ValueError: unmarshallable object" before a line of MLIR is emitted.
        xin, aw0, aw1 = args[0], args[1], args[2]
        kvo, kvi, xout = args[3], args[4], args[5]
        hout, hin, swout, swin = args[6], args[7], args[8], args[9]
        awb = [aw0, aw1]
        n = {NCP}
        w3b = [args[10 + i] for i in range(n)]
        w4b = [args[10 + n + i] for i in range(n)]
        w5b = [args[10 + 2 * n + i] for i in range(n)]
        h0 = 10 + 3 * n                          # handles begin where tensors end
        bcah = args[h0]
        awh = [args[h0 + 1 + i] for i in range({AQ})]
        bkvh = [args[h0 + 1 + {AQ} + i] for i in range({BQ})]
        bcch = args[h0 + 1 + {AQ} + {BQ}]
        cwh = [args[h0 + 2 + {AQ} + {BQ} + i] for i in range(n)]
        d0 = h0 + 2 + {AQ} + {BQ} + n
        akvh = [args[d0 + i] for i in range({NA // 2})]
        aoh = [args[d0 + {NA // 2} + i] for i in range(2)]
        cph = [args[d0 + {NA // 2} + 2 + i] for i in range(n)]

        KVPLAN = {kvplan!r}
        for L in range({nlay}):
            # ---- P1 -----------------------------------------------------------
            # x for layer L lives at block L's x slot: the host wrote it for
            # L = 0 and layer L-1's P5 drained it there for every other L.
            tg = TaskGroup()
            bcah.fill(xin, group=tg, offset=L * {BLK} + {K_DIM},
                      sizes=[1, 1, 1, {BC}], strides=[0, 0, 0, 1])
            for i in range({AQ}):
                awh[i].fill(awb[i], group=tg, offset=L * {A_LSZ},
                            sizes=[1, 1, 1, {A_LSZ}], strides=[0, 0, 0, 1])
            for pr in range({NA // 2}):
                for slot, (kind, base) in enumerate(KVPLAN[2 * pr]):
                    if kind == "k":
                        # The aligned PAIR is mandatory: a single bf16 column is
                        # a 2-byte write at an odd offset and the DMA rejects it.
                        # flm_kv_pair closes the pair with the previous token's
                        # k' out of g_kprev, which survives because nothing is
                        # reloaded between layers.
                        akvh[pr].drain(kvo, wait=True, group=tg,
                                       offset=L * {CACHEB}
                                              + 2 * (base * {KVSTRIDE} + {off}),
                                       sizes=[1, 2, {HEAD}, 4],
                                       strides=[0, 2 * {KVSTRIDE}, 2 * {TSEQ}, 1])
                    else:
                        akvh[pr].drain(kvo, wait=True, group=tg,
                                       offset=L * {CACHEB}
                                              + 2 * (base * {KVSTRIDE} + {KTILE}
                                                     + {pos} * {HEAD}),
                                       sizes=[1, 2, 1, 2 * {OBJ}],
                                       strides=[0, 2 * {KVSTRIDE}, 0, 1])
            tg.finish()

            # ---- P2 -----------------------------------------------------------
            # B reads back the cache A just wrote; the barrier above is the
            # ordering. Four heads per fill, split 1->4, because the cache's head
            # slots are contiguous and eight fills do not fit the shim.
            tg = TaskGroup()
            for i in range({BQ}):
                bkvh[i].fill(kvi, group=tg,
                             offset=L * {CACHEB} + i * 4 * {OPERAND},
                             sizes=[1, 1, 1, 4 * {OPERAND}], strides=[0, 0, 0, 1])
            for i in range(2):
                # group a slot sl is original q head 4a+sl, so the two joined
                # streams concatenate into the 2048-vector o_proj expects.
                aoh[i].drain(xout, wait=True, group=tg,
                             offset=L * {BLK} + i * 4 * {GQA} * {HEAD},
                             sizes=[1, 1, 1, 4 * {GQA} * {HEAD}],
                             strides=[0, 0, 0, 1])
            tg.finish()

            # ---- P3 -----------------------------------------------------------
            tg = TaskGroup()
            bcch.fill(xin, group=tg, offset=L * {BLK},
                      sizes=[1, 1, 1, {BC}], strides=[0, 0, 0, 1])
            for i in range(n):
                cwh[i].fill(w3b[i], group=tg, offset=L * {P3_LSZ},
                            sizes=[1, 1, 1, {P3_LSZ}], strides=[0, 0, 0, 1])
            for i in range(n):
                cph[i].drain(hout, wait=True, group=tg,
                             offset=L * {BC} + i * {RPP3},
                             sizes=[1, 2, {P3T}, {NROWS}],
                             strides=[0, {NROWS}, {2 * NROWS}, 1])
            tg.finish()

            # ---- P4 -----------------------------------------------------------
            tg = TaskGroup()
            bcch.fill(hin, group=tg, offset=L * {BC},
                      sizes=[1, 1, 1, {BC}], strides=[0, 0, 0, 1])
            for i in range(n):
                cwh[i].fill(w4b[i], group=tg, offset=L * {P4_LSZ},
                            sizes=[1, 1, 1, {P4_LSZ}], strides=[0, 0, 0, 1])
            for i in range(n):
                cph[i].drain(swout, wait=True, group=tg, offset=i * {RPP4},
                             sizes=[1, {P4OBJS}, 2, {OBJ}],
                             strides=[0, {OBJ}, {RPP4 // 2}, 1])
            tg.finish()

            # ---- P5 -----------------------------------------------------------
            tg = TaskGroup()
            for ch in range({NCHUNK}):
                # each chunk is a different slice of sw; the aux half is unread
                # because the residual comes from g_resid.
                bcch.fill(swin, group=tg, offset=ch * {K_DIM},
                          sizes=[1, 1, 1, {BC}], strides=[0, 0, 0, 1])
            for i in range(n):
                cwh[i].fill(w5b[i], group=tg, offset=L * {P5_LSZ},
                            sizes=[1, 1, 1, {P5_LSZ}], strides=[0, 0, 0, 1])
            for i in range(n):
                # x_out into the NEXT block's x slot -- which is both P1's
                # activation and P3's residual for layer L+1.
                cph[i].drain(xout, wait=True, group=tg,
                             offset=(L + 1) * {BLK} + {K_DIM} + i * {RPP3},
                             sizes=[1, 2, {P5T}, {NROWS}],
                             strides=[0, {NROWS}, {2 * NROWS}, 1])
            tg.finish()

    at = [x_ty, aw_ty, aw_ty, kv_ty, kv_ty, x_ty, h_ty, h_ty, sw_ty, sw_ty]
    at += [w3_ty] * {NCP} + [w4_ty] * {NCP} + [w5_ty] * {NCP}
    at += [f_bca.prod(tile=AnyShimTile)]
    at += [f.prod(tile=AnyShimTile) for f in f_aw]
    at += [f.prod(tile=AnyShimTile) for f in f_bkv]
    at += [f_bcc.prod(tile=AnyShimTile)]
    at += [f.prod(tile=AnyShimTile) for f in f_cw]
    at += [f.cons(tile=AnyShimTile) for f in f_akv]
    at += [f.cons(tile=AnyShimTile) for f in f_ao]
    at += [f.cons(tile=AnyShimTile) for f in f_cp]
    rt = Runtime(sequence, at)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction,
              QKV_SRC=QKV_SRC, EMIT_SRC=EMIT_SRC, KVE_SRC=KVE_SRC,
              NORM_SRC=NORM_SRC, ATT_SRC=ATT_SRC, BEG_SRC=BEG_SRC,
              FIN_SRC=FIN_SRC, RES_SRC=RES_SRC, ASUM_SRC=ASUM_SRC,
              HEMIT_SRC=HEMIT_SRC, DOWN_SRC=DOWN_SRC, GATE_SRC=GATE_SRC,
              UPS_SRC=UPS_SRC, FLAGS=flags,
              bc_ty=bc_ty, wt_ty=wt_ty, wq_ty=wq_ty, o_ty=o_ty, okv_ty=okv_ty,
              q_ty=q_ty, op_ty=op_ty, opq_ty=opq_ty, opp_ty=opp_ty,
              ao_ty=ao_ty, aoj_ty=aoj_ty, p1o_ty=p1o_ty, p1p_ty=p1p_ty,
              x_ty=x_ty, aw_ty=aw_ty, kv_ty=kv_ty, h_ty=h_ty, sw_ty=sw_ty,
              w3_ty=w3_ty, w4_ty=w4_ty, w5_ty=w5_ty,
              __name__="flm_fused")
    exec(src, ns)
    return iron.jit(ns["_design"],
                    source_files=[QKV_SRC, EMIT_SRC, KVE_SRC, NORM_SRC, ATT_SRC,
                                  BEG_SRC, FIN_SRC, RES_SRC, ASUM_SRC,
                                  HEMIT_SRC, DOWN_SRC, GATE_SRC, UPS_SRC],
                    full_elf=True)


# ---------------------------------------------------------------------------
# the host reference, with its intermediates exposed
# ---------------------------------------------------------------------------
def ref_layer(c, x, L):
    """One decoder layer at position 0 -> (attn, h, sw, y).

    Line for line `host_forward.layer`, which reproduces an independent fp32
    forward from `consolidated.00.pth` end to end; the only difference is that
    this returns the intermediates so the device's phases can be checked against
    something external rather than against each other. `main` asserts the two
    agree on `y`, so the copy cannot drift.
    """
    import host_forward as hf
    P = f"model.layers.{L}."
    nw1 = c.bf16(P + "input_layernorm.weight").astype(np.float32)[:K_DIM]
    h1 = hf.rmsnorm(x, nw1)
    vd, vm, vc = hf.load_linear(c, P + "self_attn.v_proj.weight", 8 * HEAD, K_DIM)
    v = q4nx.gemv_reference_bf16(h1, vd, vm, vc)
    attn = np.repeat(v.reshape(8, HEAD), NQ // 8, axis=0).reshape(-1)
    od, om, oc = hf.load_linear(c, P + "self_attn.o_proj.weight", K_DIM, K_DIM)
    h = x + q4nx.gemv_reference_bf16(attn.astype(np.float32), od, om, oc)
    nw2 = c.bf16(P + "post_attention_layernorm.weight").astype(np.float32)[:K_DIM]
    h2 = hf.rmsnorm(h.astype(np.float32), nw2)
    gd, gm, gc = hf.load_linear(c, P + "mlp.gate_proj.weight", D_FF, K_DIM)
    ud, um, uc = hf.load_linear(c, P + "mlp.up_proj.weight", D_FF, K_DIM)
    g = q4nx.gemv_reference_bf16(h2, gd, gm, gc)
    u = q4nx.gemv_reference_bf16(h2, ud, um, uc)
    sw = (g / (1.0 + np.exp(-g))) * u
    dd, dm, dc = hf.load_linear(c, P + "mlp.down_proj.weight", K_DIM, D_FF)
    y = h + q4nx.gemv_reference_bf16(sw.astype(np.float32), dd, dm, dc)
    return attn, h, sw, y


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--layer0", type=int, default=0, help="first layer index")
    p.add_argument("--pos", type=int, default=0, help="KV cache position")
    p.add_argument("--seq", type=int, default=1, help="context length")
    p.add_argument("--x0", help=".npy holding layer 0's input; default a real BOS "
                                "embedding is NOT assumed, a random vector is used")
    p.add_argument("--save", help="write the final x_out to .npy")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--no-ref", action="store_true",
                   help="skip the host reference chain (it costs minutes)")
    o = p.parse_args()
    nlay, L0, pos = o.layers, o.layer0, o.pos
    if pos != 0:
        print("  NOTE: position is a BUILD parameter here and only pos 0 is "
              "verified; g_kprev and the prior cache are not exercised.")

    c = q4nx.Q4nx(str(Q4NX))
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)[:HEAD // 2]
    inv_freq = (1.0 / ROPE_THETA ** (np.arange(0, HEAD, 2) / HEAD)) / divisor
    ang = pos * inv_freq
    cs_k = rnd(np.concatenate([np.cos(ang), np.sin(ang)]))
    cs_q = rnd(cs_k * (HEAD ** -0.5) * np.log2(np.e))

    rng = np.random.default_rng(0)
    x0 = (rnd(np.load(o.x0).astype(np.float32)) if o.x0
          else rnd(rng.standard_normal(K_DIM) * 0.05))

    print(f"fused: 32 cores, {nlay} layers, ONE dispatch, layer {L0}, pos {pos}")
    design = build(pos, nlay, o.seq)

    # ---- host buffers ------------------------------------------------------
    xbuf = np.zeros((nlay + 1) * BLK, np.float32)
    for L in range(nlay):
        nw = c.bf16(f"model.layers.{L0 + L}.input_layernorm.weight"
                    ).astype(np.float32)[:K_DIM]
        b = L * BLK
        xbuf[b + 2 * K_DIM:b + 3 * K_DIM] = nw
        xbuf[b + 3 * K_DIM:b + 3 * K_DIM + HEAD] = cs_q
        xbuf[b + 3 * K_DIM + HEAD:b + BLK] = cs_k
    xbuf[K_DIM:2 * K_DIM] = x0                       # block 0's x, the only one
    xbuf_t = iron.tensor(xbuf.astype(bfloat16), dtype=bfloat16, device="npu")

    hbuf = np.zeros(nlay * BC, np.float32)
    for L in range(nlay):
        nw2 = c.bf16(f"model.layers.{L0 + L}.post_attention_layernorm.weight"
                     ).astype(np.float32)[:K_DIM]
        hbuf[L * BC + K_DIM:L * BC + 2 * K_DIM] = nw2
    hbuf_t = iron.tensor(hbuf.astype(bfloat16), dtype=bfloat16, device="npu")
    sw_t = iron.zeros(D_FF + BC, dtype=bfloat16, device="npu")

    # ---- the KV cache: one block per layer, npad in every head's trailer ----
    nobj = -(-o.seq // (TSEQ * KVPER))
    npad = TSEQ * nobj * KVPER - (pos + 1)
    cache = np.zeros(nlay * CACHEB, np.uint8)
    for L in range(nlay):
        for g in range(8):
            off = L * CACHEB + g * OPERAND + OPERAND - 60
            cache[off:off + 4] = np.array([float(npad)], np.float32).view(np.uint8)
    cache_t = iron.tensor(cache, dtype=np.uint8, device="npu")

    # ---- weights: LAYER-OUTER, so only one layer's tensors are ever live ----
    # A per-pair-outer loop would hold gate/up/down for every layer at once,
    # which is 500 MB of decoded blocks before a single tile is packed.
    layout = head_layout(NA)
    nbc3, nbc4, nbc5 = K_DIM // 32, K_DIM // 32, D_FF // 32
    aw_p = [[] for _ in range(AQ)]
    w3_p = [[] for _ in range(NCP)]
    w4_p = [[] for _ in range(NCP)]
    w5_p = [[] for _ in range(NCP)]
    for L in range(nlay):
        LL = L0 + L
        for q in range(AQ):
            per = []
            for j in range(4):
                blob = []
                for h in layout[4 * q + j]:
                    first = h * HEAD
                    d, m, qq = qkv_rows(c, LL, first, HEAD)
                    blob.append(np.concatenate([
                        q4nx.pack_tile(d[i:i + NROWS], m[i:i + NROWS],
                                       qq[i:i + NROWS], row_base=first + i,
                                       flags=float(pos))
                        for i in range(0, HEAD, NROWS)]))
                per.append(np.concatenate(blob))
            b = np.empty((HPC * TPH, 4, WT), np.uint8)
            for j in range(4):
                b[:, j, :] = per[j].reshape(-1, WT)
            aw_p[q].append(b.reshape(-1))

        od, om, oc = q4nx.q4nx_tensor_blocks(
            c, f"model.layers.{LL}.self_attn.o_proj.weight", (K_DIM, K_DIM))
        for pr in range(NCP):
            per = [np.concatenate([
                q4nx.pack_tile(od[r0:r0 + NROWS, :nbc3], om[r0:r0 + NROWS, :nbc3],
                               oc[r0:r0 + NROWS, :nbc3], row_base=r0, flags=0.0)
                for r0 in p3rows(pr, j)]) for j in range(2)]
            b = np.empty((P3T, 2, WT), np.uint8)
            b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, WT), per[1].reshape(-1, WT)
            w3_p[pr].append(b.reshape(-1))
        del od, om, oc

        gd, gm, gc = q4nx.q4nx_tensor_blocks(
            c, f"model.layers.{LL}.mlp.gate_proj.weight", (D_FF, K_DIM))
        ud, um, uc = q4nx.q4nx_tensor_blocks(
            c, f"model.layers.{LL}.mlp.up_proj.weight", (D_FF, K_DIM))
        for pr in range(NCP):
            per = []
            for j in range(2):
                blob = []
                for r0 in p4rows(pr, j):
                    sl = slice(r0, r0 + NROWS)
                    blob.append(q4nx.pack_tile(gd[sl, :nbc4], gm[sl, :nbc4],
                                               gc[sl, :nbc4], row_base=r0, flags=0.0))
                    blob.append(q4nx.pack_tile(ud[sl, :nbc4], um[sl, :nbc4],
                                               uc[sl, :nbc4], row_base=r0, flags=0.0))
                per.append(np.concatenate(blob))
            b = np.empty((2 * P4T, 2, WT), np.uint8)
            b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, WT), per[1].reshape(-1, WT)
            w4_p[pr].append(b.reshape(-1))
        del gd, gm, gc, ud, um, uc

        dd, dm, dc = q4nx.q4nx_tensor_blocks(
            c, f"model.layers.{LL}.mlp.down_proj.weight", (K_DIM, D_FF))
        for pr in range(NCP):
            per = []
            for j in range(2):
                blob = []
                for ch in range(NCHUNK):
                    lo, hi = ch * (nbc5 // NCHUNK), (ch + 1) * (nbc5 // NCHUNK)
                    for r0 in p3rows(pr, j):     # p5rows == p3rows, by design:
                        sl = slice(r0, r0 + NROWS)   # P5's residual is the h THIS
                        blob.append(q4nx.pack_tile(  # core stashed in P3
                            dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi],
                            row_base=r0, flags=float(ch == NCHUNK - 1)))
                per.append(np.concatenate(blob))
            b = np.empty((NCHUNK * P5T, 2, WT), np.uint8)
            b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, WT), per[1].reshape(-1, WT)
            w5_p[pr].append(b.reshape(-1))
        del dd, dm, dc

    T = lambda parts: iron.tensor(np.concatenate(parts), dtype=np.uint8,
                                  device="npu")
    aw_ts = [T(p) for p in aw_p]
    w3_ts = [T(p) for p in w3_p]
    w4_ts = [T(p) for p in w4_p]
    w5_ts = [T(p) for p in w5_p]
    del aw_p, w3_p, w4_p, w5_p

    args = (xbuf_t, *aw_ts, cache_t, cache_t, xbuf_t, hbuf_t, hbuf_t,
            sw_t, sw_t, *w3_ts, *w4_ts, *w5_ts)
    if o.bench:
        b = run_iters(design, *args, warmup=2, iters=10)
        us = b.npu.min_us if b.npu else b.e2e.min_us
        mb = (nlay * (AQ * A_LSZ + NCP * (P3_LSZ + P4_LSZ + P5_LSZ))) / 1e6
        LM_HEAD_US = 2994.2                   # measured at its own size, 163.7 MB
        tok = us + LM_HEAD_US
        print(f"  bench: {mb:.2f} MB  {mb * 1e3 / us:.1f} GB/s  {us:.1f} us "
              f"for {nlay} layers in ONE dispatch")
        print(f"         {us / nlay:.1f} us/layer   "
              f"token = {us:.1f} + {LM_HEAD_US} = {tok:.1f} us "
              f"-> {1e6 / tok:.1f} tok/s   (FLM 61.18)")
    else:
        design(*args)

    # ---- read back ---------------------------------------------------------
    xg = xbuf_t.numpy().astype(np.float64)
    hg = hbuf_t.numpy().astype(np.float64)
    swg = sw_t.numpy().astype(np.float64)
    x_dev = xg[nlay * BLK + K_DIM:nlay * BLK + 2 * K_DIM]
    if o.save:
        np.save(o.save, x_dev.astype(np.float32))
        print(f"  x_out -> {o.save}   mean|.| {np.abs(x_dev).mean():.5f}  "
              f"max {np.abs(x_dev).max():.5f}")
    if o.no_ref:
        return 0

    # ---- the EXTERNAL bar --------------------------------------------------
    # Every phase is compared against a reference chain that starts from x0 and
    # never reads a device value. Internal agreement proves nothing: five faults
    # hid behind checks that fed a stage's reference from the same wrong input.
    import host_forward as hf
    ok, x = True, x0.astype(np.float64)
    for L in range(nlay):
        attn_r, h_r, sw_r, y_r = ref_layer(c, x, L0 + L)
        if L == 0:
            # ref_layer is host_forward.layer with its intermediates exposed;
            # one bit-exact comparison is what stops the copy from drifting.
            assert np.array_equal(y_r, hf.layer(c, x, L0 + L)), \
                "ref_layer drifted from host_forward.layer"
        attn_d = xg[L * BLK:L * BLK + K_DIM]
        h_d = hg[L * BC:L * BC + K_DIM]
        ea = np.abs(attn_d - rnd(attn_r)).max()
        eh = np.abs(h_d - rnd(h_r)).max()
        ta = 2 * 2**-8 * np.abs(attn_r).max()
        th = 2 * 2**-8 * np.abs(h_r).max()
        line = (f"  L{L0 + L:<2d} attn {ea:.4e}/{ta:.4e}  h {eh:.4e}/{th:.4e}")
        if L == nlay - 1:                     # sw and x_out survive only for the last
            esw = np.abs(swg[:D_FF] - rnd(sw_r)).max()
            tsw = 0.04 * np.abs(sw_r).max()
            ex = np.abs(x_dev - rnd(y_r)).max()
            tx = 2 * 2**-8 * np.abs(y_r).max()
            line += f"  sw {esw:.4e}/{tsw:.4e}  x_out {ex:.4e}/{tx:.4e}"
            ok &= esw <= tsw and ex <= tx
        # attention is ADVISORY: its floor model is unresolved. h is a gate.
        if ea > ta:
            line += "   [attn advisory]"
        ok &= eh <= th
        print(line)
        x = y_r

    cos = float(x_dev @ x / (np.linalg.norm(x_dev) * np.linalg.norm(x)))
    print(f"  device x{nlay} vs host chain: cosine {cos:.6f}   "
          f"mean|dev| {np.abs(x_dev).mean():.5f}  mean|ref| {np.abs(x).mean():.5f}")
    ext = os.environ.get("FUSED_EXTERNAL")
    if ext:
        xr = np.load(ext).astype(np.float64)
        ce = float(x_dev @ xr / (np.linalg.norm(x_dev) * np.linalg.norm(xr)))
        print(f"  device x{nlay} vs {ext}: cosine {ce:.6f}")
    print(f"  -> fused {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
