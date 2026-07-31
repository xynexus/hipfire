#!/usr/bin/env python3
"""Phase P1 complete: norm + qkv + RoPE, routed to q′, the K cache and the V cache.

**Verified** at even and odd cache positions: q′ 9.5e-07, v′ exact, k′ at one
bf16 ulp. Phase P1 now produces everything P2 needs, in the layout it needs.

Two things had to be right, and the second is a DMA semantics point worth
knowing:

1. The emit reuses the head's **last weight tile** rather than acquiring its
   own. It needs a tile only for `row_base`/`flags`, and a separate acquire
   consumes a fifth object per head against the four packed, desynchronising the
   weight stream after the first head.
2. **A drain consumes its source LINEARLY.** `sizes`/`strides` shape the
   *destination* walk only — there is no way to skip source elements. So the
   unused half of a q′/v′ object cannot be dropped on the way out; the drain
   takes whole objects and the host indexes into them. `flm_p1_emit` zeroes that
   half, which for v′ lands on cache row pos+1 — a future position, where zero
   is exactly what attention's `npad` correction wants.

    python3 p1_route.py
    python3 p1_route.py --pos 1        # odd position: k' closes a column pair

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from qkv_verify import HEAD, K_DIM, NROWS, TPH, EPS, ROPE_THETA, qkv_rows, rope_ref  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
QKV_SRC = str(KDIR / "flm_gemv_qkv.cc")
EMIT_SRC = str(KDIR / "flm_p1_emit.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
TSEQ = 32
NCORES, HPC = 16, 3
NQ, NK, NV = 32, 40, 48          # head-index boundaries
rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


NHEADS = NV                      # 48 head-tiles: 32 q + 8 k + 8 v


def hpc_for(ncores):
    """Head-tiles per core. Raises rather than truncating on a ragged split."""
    if NHEADS % ncores:
        raise ValueError(f"{NHEADS} head-tiles do not divide over {ncores} cores")
    return NHEADS // ncores


def heads_of(core, ncores=NCORES):
    """Core c owns {c, c+n, c+2n, ...} over `ncores` P1 cores.

    The stride is the CORE COUNT, not a constant 16. At 16 cores that is the
    original {c, c+16, c+32}; at 12 (partition B, where the four attention cores
    sit P1 out) it becomes {c, c+12, c+24, c+36}, four tiles each, and 48/12
    divides exactly.

    **The KV placement is not uniform at 12 cores**, and that is the part that
    costs work. At 16 every k head sits in slot 2 on cores 0-7 and every v head
    in slot 2 on cores 8-15, so one drain formula covers both. At 12 the k heads
    split across slot 2 (cores 8-11) and slot 3 (cores 0-3), and v lands in slot
    3 on cores 4-11. Any drain that hardcodes "K from core g, V from core g+8"
    is wrong off 16 cores; use `kv_placement()`.
    """
    return head_layout(ncores)[core]


def head_layout(ncores=NCORES):
    """-> [heads owned by core 0, core 1, ...], chosen so the KV drains stay
    uniform across pairs.

    The assignment is **free** — any bijection over the 48 head-tiles works, and
    the host packs the weight stream to match. So it should be chosen to make
    the drains simple, not derived from a stride and then coped with. A pure
    stride-`ncores` rule is the obvious thing and it is the wrong thing: at 12
    cores it puts a k head and a v head on the SAME core in different slots
    (pairs 4-5), while pairs 0-1 carry only k. Every pair then needs a different
    drain shape.

    At 16 cores the original rule already happens to be uniform — k in slot 2 on
    cores 0-7, v in slot 2 on cores 8-15 — so it is kept exactly, and
    `p1_route` at 16 cores is unaffected.

    At 12 cores this puts **both** a k and a v head on each of cores 0-7, in
    fixed slots, so pairs 0-3 all have the identical two-drain shape and pairs
    4-5 are pure q with none. Uniform in both groups, which is what the drain
    code can express.
    """
    hpc = hpc_for(ncores)
    if ncores == NCORES:                      # the original stride-16 rule
        return [[c + NCORES * h for h in range(hpc)] for c in range(ncores)]

    nkv = NV - NQ                             # 16 kv head-tiles
    kv_cores = nkv // 2                       # 8 cores carry one k and one v
    if kv_cores > ncores or hpc < 2:
        raise ValueError(f"cannot place {nkv} kv tiles over {ncores} cores")
    layout, q = [], iter(range(NQ))
    for c in range(ncores):
        if c < kv_cores:                      # ... q ..., v[c], k[c]
            row = [next(q) for _ in range(hpc - 2)] + [NK + c, NQ + c]
        else:
            row = [next(q) for _ in range(hpc)]
        layout.append(row)
    if next(q, None) is not None:
        raise ValueError("q head-tiles left over: layout is not a bijection")
    return layout


def kv_placement(ncores=NCORES):
    """-> (k_at, v_at), each {kv_head_index: (core, slot)}.

    Derived from `heads_of` rather than assumed, so a drain built on it follows
    the assignment automatically when the core count changes.
    """
    k_at, v_at = {}, {}
    for c, row in enumerate(head_layout(ncores)):
        for slot, h in enumerate(row):
            if NQ <= h < NK:
                k_at[h - NQ] = (c, slot)
            elif h >= NK:
                v_at[h - NK] = (c, slot)
    return k_at, v_at


def drain_plan(ncores=NCORES):
    """-> (qobj, kvplan): q objects per pair, and its KV drains in SLOT order.

    Derived from `head_layout` so the drains follow the assignment instead of
    encoding one. The old code branched `elif i < n // 2`, which states the
    16-core placement as a rule; at 12 cores pairs 0-3 carry a v *and* a k and
    pairs 4-5 carry neither, which that branch cannot say.

    Slot order matters: a pair's stream is consumed linearly, so the drains must
    be emitted in the order the core wrote them.
    """
    layout = head_layout(ncores)
    qobj, kvplan = [], []
    for pr in range(ncores // 2):
        row = layout[2 * pr]
        qslots = [s for s, h in enumerate(row) if h < NQ]
        if qslots != list(range(len(qslots))):
            raise ValueError(f"pair {pr}: q slots {qslots} are not a prefix; "
                             "the q drain takes the head of the stream")
        qobj.append(2 * len(qslots))
        kvplan.append([("k", h - NQ) if h < NK else ("v", h - NK)
                       for h in row if h >= NQ])
    return qobj, kvplan


def build(pos, ncores=NCORES):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = ncores // 2
    hpc = hpc_for(ncores)
    qobj, kvplan = drain_plan(ncores)
    BC = 2 * K_DIM + 2 * HEAD
    OBJ = 2 * HEAD                              # 2*HEAD for every head
    KTILE, VTILE = HEAD * TSEQ, TSEQ * HEAD
    off = pos - (pos & 1)                       # even column of this pair

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * hpc * TPH * wt,), np.dtype[np.uint8]]
    import os as _os
    SINGLE = 1 if _os.environ.get("P1_SINGLE") else 0
    # whole pair stream when bisecting: HPC steps x 2 cores x OBJ
    q_ty = (np.ndarray[(hpc * 2 * OBJ,), np.dtype[bfloat16]] if SINGLE
            else np.ndarray[(4 * OBJ,), np.dtype[bfloat16]])
    # ONE cache buffer, 8 KV heads of [K tile][V tile] — the shape P2's operand
    # objects are cut from. K for KV head g comes from core g (pairs 0-3) and V
    # from core g+8 (pairs 4-7), so the two halves of a head are written by
    # different pairs into the same buffer at different offsets. Several pairs
    # draining into one BO is the ffn_chain pattern.
    KVSTRIDE = KTILE + VTILE
    kv_ty = np.ndarray[(8 * KVSTRIDE,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS={NQ}", f"-DDIM_QKHEADS={NK}"]
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"q{i}: Out" for i in range(npairs))
    params += ", " + ", ".join(f"kv{i}: Out" for i in range(npairs))
    src = f'''
def _design(bc: In, {params}):
    kq = ExternalFunction("flm_gemv_qkv", source_file=QKV_SRC,
                          arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    ke = ExternalFunction("flm_p1_emit", source_file=EMIT_SRC,
                          arg_types=[wt_ty, o_ty], compile_flags=FLAGS)
    kn = ExternalFunction("flm_norm_prepare", source_file=NORM_SRC,
                          arg_types=[bc_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name="bc")
    bc_cons = [f_bc.cons() for _ in range({ncores})]
    f_w = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_w]
    f_o = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {OBJ}], obj_types=[o_ty, o_ty]) for f in f_o]

    def core(bcc, wc, op, kqkv, kemit, kprep):
        eb = bcc.acquire(1)
        kprep(eb)
        for _ in range_({hpc}):
            for _ in range_({TPH} - 1):
                ew = wc.acquire(1)
                kqkv(eb, ew)
                wc.release(1)
            # The emit reuses the head's LAST tile rather than acquiring its
            # own: it needs a tile only for row_base/flags, and a separate
            # acquire would consume a fifth object per head, desynchronising the
            # weight stream after the first head. Its row_base is h*HEAD+48, so
            # row_base/HEAD is still h.
            #
            # Both objects are acquired before either call — an acquire between
            # two kernels sharing a global loses the handoff
            # (global_handoff_probe.py), and these share g_stage.
            ew = wc.acquire(1)
            eo = op.acquire(1)
            kqkv(eb, ew)
            kemit(ew, eo)
            op.release(1)
            wc.release(1)
        bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[bc_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kq, ke, kn], stack_size=8192))

    def sequence(*args):
        n = {npairs}
        QOBJ, KVPLAN = {qobj!r}, {kvplan!r}
        bcb = args[0]
        wb = [args[1 + i] for i in range(n)]
        qb = [args[1 + n + i] for i in range(n)]
        kvb = [args[1 + 2 * n + i] for i in range(n)]
        bch = args[1 + 3 * n]
        wh = [args[2 + 3 * n + i] for i in range(n)]
        oh = [args[2 + 4 * n + i] for i in range(n)]
        tg = TaskGroup()
        bch.fill(bcb, group=tg)
        for i in range(n):
            wh[i].fill(wb[i], group=tg)
        for i in range(n):
            if {SINGLE}:
                # bisect: ONE plain drain of the whole pair stream, the shape
                # qkv_verify.py uses. Isolates the 3-way split from everything
                # else that differs here.
                oh[i].drain(qb[i], wait=True, group=tg)
                continue
            # steps 0-1: 4 q objects, drained WHOLE. The source streams
            # linearly and sizes/strides shape only the destination, so the
            # unused half of each object cannot be skipped on the way out — the
            # host reads the first HEAD of each OBJ instead.
            oh[i].drain(qb[i], wait=True, group=tg,
                        sizes=[1, 1, 1, QOBJ[i] * {OBJ}], strides=[0, 0, 0, 1])
            for _kind, _base in KVPLAN[i]:
                if _kind == "k":
                    # 2 k heads -> the K halves of KV heads base, base+1
                    oh[i].drain(kvb[i], wait=True, group=tg,
                                offset=_base * {KVSTRIDE} + {off},
                                sizes=[1, 2, {HEAD}, 2],
                                strides=[0, {KVSTRIDE}, {TSEQ}, 1])
                else:
                    # 2 v heads, each contiguous in its own V tile. Whole
                    # objects: OBJ per head, so row pos gets v' and row pos+1
                    # gets the emit's zeros — what a padded position must hold,
                    # and the next token overwrites it.
                    oh[i].drain(kvb[i], wait=True, group=tg,
                                offset=_base * {KVSTRIDE} + {KTILE}
                                       + {pos} * {HEAD},
                                sizes=[1, 2, 1, {OBJ}],
                                strides=[0, {KVSTRIDE}, 0, 1])
        tg.finish()

    arg_types = [bc_ty] + [w_all_ty] * {npairs}
    arg_types += [q_ty] * {npairs} + [kv_ty] * {npairs}
    arg_types += [f_bc.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_w]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_o]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, QKV_SRC=QKV_SRC,
              EMIT_SRC=EMIT_SRC, NORM_SRC=NORM_SRC, FLAGS=flags, bc_ty=bc_ty,
              wt_ty=wt_ty, o_ty=o_ty, wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_all_ty=w_all_ty, q_ty=q_ty, kv_ty=kv_ty, SINGLE=SINGLE,
              __name__="flm_p1_route")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[QKV_SRC, EMIT_SRC, NORM_SRC],
                    full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--pos", type=int, default=0, help="KV cache position")
    p.add_argument("--bench", action="store_true",
                   help="time P1 alone, so P2's in-chain marginal can be\n"
                        "isolated against p1p2_chain --bench")
    o = p.parse_args()
    npairs = NCORES // 2
    KTILE = HEAD * TSEQ

    c = q4nx.Q4nx(str(Q4NX))
    nw = c.bf16(f"model.layers.{o.layer}.input_layernorm.weight").astype(np.float32)[:K_DIM]
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)[:HEAD // 2]
    inv_freq = (1.0 / ROPE_THETA ** (np.arange(0, HEAD, 2) / HEAD)) / divisor
    ang = o.pos * inv_freq
    cs_k = rnd(np.concatenate([np.cos(ang), np.sin(ang)]))
    cs_q = rnd(cs_k * (HEAD ** -0.5) * np.log2(np.e))

    design, wt = build(o.pos)

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

    # weights: core c owns heads {c, c+16, c+32}, in emit order
    w_ts, ref = [], {}
    for pr in range(npairs):
        per = []
        for j in range(2):
            cix = 2 * pr + j
            blob = []
            for h in heads_of(cix):
                first = h * HEAD
                d, m, q = qkv_rows(c, o.layer, first, HEAD)
                blob.append(np.concatenate([
                    q4nx.pack_tile(d[i:i+NROWS], m[i:i+NROWS], q[i:i+NROWS],
                                   row_base=first + i, flags=float(o.pos))
                    for i in range(0, HEAD, NROWS)]))
                v = np.concatenate([
                    q4nx.gemv_reference_bf16(xn, d[i:i+NROWS], m[i:i+NROWS],
                                             q[i:i+NROWS])
                    for i in range(0, HEAD, NROWS)])
                v = rnd(v)
                if h < NQ:
                    v = rope_ref(v, cs_q)
                elif h < NK:
                    v = rope_ref(v, cs_k)
                ref[h] = rnd(v).astype(np.float64)
            per.append(np.concatenate(blob))
        b = np.empty((HPC * TPH, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w_ts.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))

    bc_t = iron.tensor(bc.astype(bfloat16), dtype=bfloat16, device="npu")
    import os as _os2
    SINGLE = bool(_os2.environ.get("P1_SINGLE"))
    qsz = HPC * 2 * 2 * HEAD if SINGLE else 4 * 2 * HEAD
    q_ts = [iron.zeros(qsz, dtype=bfloat16, device="npu") for _ in range(npairs)]
    KVSTRIDE = KTILE + TSEQ * HEAD
    cache = iron.zeros(8 * KVSTRIDE, dtype=bfloat16, device="npu")
    kv_ts = [cache] * npairs          # every pair drains into the same buffer
    if o.bench:
        _b = run_iters(design, bc_t, *w_ts, *q_ts, *kv_ts, warmup=2, iters=10)
        _us = _b.npu.min_us if _b.npu else _b.e2e.min_us
        _mb = NCORES // 2 * 2 * HPC * TPH * q4nx.tile_bytes(K_DIM, NROWS) / 1e6
        print(f"  bench: {_mb:.2f} MB  {_mb*1e3/_us:.1f} GB/s  {_us:.1f} us "
              f"(marginal {_us - 92.9:.1f})")
    else:
        design(bc_t, *w_ts, *q_ts, *kv_ts)
    cv = cache.numpy().astype(np.float64).reshape(8, KVSTRIDE)

    print(f"P1 routed to three destinations: {NCORES} cores, layer {o.layer}, "
          f"pos {o.pos}")
    ok, worst, scale = True, 0.0, 0.0
    if SINGLE:
        # the pair stream is [step][core][OBJ]; head h of core j at step t
        print("  BISECT: one plain drain of the whole pair stream")
        for pr in range(1):
            v = q_ts[pr].numpy().astype(np.float64).reshape(HPC, 2, 2 * HEAD)
            for t in range(HPC):
                for j in range(2):
                    h = heads_of(2 * pr + j)[t]
                    e = np.abs(v[t, j, :HEAD] - ref[h]).max()
                    print(f"    step {t} core {j}: head {h} err {e:.4e}")
        return 0
    # q': pair pr emits heads 2pr, 2pr+1 (step 0) then 2pr+16, 2pr+17 (step 1)
    for pr in range(npairs):
        got = q_ts[pr].numpy().astype(np.float64).reshape(4, 2 * HEAD)[:, :HEAD]
        want = [2 * pr, 2 * pr + 1, 2 * pr + 16, 2 * pr + 17]
        for slot, h in enumerate(want):
            e = np.abs(got[slot] - ref[h]).max()
            worst = max(worst, e); scale = max(scale, np.abs(ref[h]).mean())
    # Gate each group against ITS OWN scale. The device emits bf16 and the
    # reference rounds to it, so ~one ulp is the floor rather than a defect —
    # and k' heads are larger than q' heads, so a tolerance derived from q
    # under-measures k by the ratio of their magnitudes.
    def gate(err, heads, name):
        sc = np.mean([np.abs(ref[h]).mean() for h in heads])
        t = 1e-2 * sc
        print(f"  {name}  max err {err:.4e}   mean|ref| {sc:.5f}   tol {t:.4e}")
        return err <= t
    ok &= gate(worst, range(NQ), "q' : 32 heads               ")
    if __import__("os").environ.get("P1_DIAG"):
        import sys as _s
        got = q_ts[0].numpy().astype(np.float64).reshape(4, 2 * HEAD)[:, :HEAD]
        for slot in range(4):
            best = min(ref, key=lambda h: np.abs(got[slot] - ref[h]).max())
            e = np.abs(got[slot] - ref[best]).max()
            want = [0, 1, 16, 17][slot]
            print(f"  DIAG pair0 q slot {slot}: best match head {best} "
                  f"(err {e:.3e}), wanted head {want}", file=_s.stderr)
        K = kv_ts[0].numpy().astype(np.float64).reshape(2, HEAD, TSEQ)
        for j in range(2):
            best = min(ref, key=lambda h: np.abs(K[j, :, 0] - ref[h]).max())
            print(f"  DIAG pair0 K tile {j} col0: best match head {best} "
                  f"(err {np.abs(K[j,:,0]-ref[best]).max():.3e}), "
                  f"wanted {32 + j}", file=_s.stderr)

    we = 0.0
    for g in range(8):                                 # K half of KV head g
        K = cv[g, :KTILE].reshape(HEAD, TSEQ)
        e = np.abs(K[:, o.pos] - ref[NQ + g]).max()
        we = max(we, e)
    ok &= gate(we, range(NQ, NK), f"k' : 8 heads -> K col {o.pos:<2d}      ")

    ve = 0.0
    for g in range(8):                                 # V half of KV head g
        V = cv[g, KTILE:].reshape(TSEQ, HEAD)
        e = np.abs(V[o.pos] - ref[NK + g]).max()
        ve = max(ve, e)
    ok &= gate(ve, range(NK, NV), f"v' : 8 heads -> V row {o.pos:<2d}      ")
    print(f"  -> {'PASS' if ok else 'FAIL'}  (floor is one bf16 ulp)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
