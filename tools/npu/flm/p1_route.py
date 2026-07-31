#!/usr/bin/env python3
"""Phase P1 complete: norm + qkv + RoPE, routed to q′, the K cache and the V cache.

**STATUS: does not verify yet.** It builds, places and runs; the arithmetic is
`qkv_verify.py`'s, which is exact at these shapes. The symptom is specific:
**the first emit of the whole design is exact and every later one is garbage.**

    pair0 q slot 0 (core 0, emit 1)  head 0   err 0.000e+00   <- right
    pair0 q slot 1 (core 1, emit 1)  garbage
    pair0 q slot 2 (core 0, emit 2)  garbage
    pair0 q slot 3 (core 1, emit 2)  garbage

So it is not per-core and not per-head — it is the result stream after its first
object. Fixed already and not the cause: the emit used to acquire its own weight
tile, a fifth per head against the four packed, which desynchronised the weight
stream; it now reuses the head's last tile (whose `row_base/HEAD` is still the
head index). That changed nothing observable, so the fault is downstream of it.

Not yet excluded: the 3-way drain split against a `join`ed pair fifo (every
earlier probe drained a fifo whose producer was a single core), and the
result-object size being `2*HEAD` where `qkv_verify` uses `HEAD`.

`qkv_verify.py` proves P1's arithmetic but drains all 48 heads into one buffer.
This routes them where the layer actually needs them, from **one** result fifo:

    q′  contiguous, into the block P2 reads as its query
    k′  a stride-TSEQ column-pair scatter into the channel-major K cache
    v′  contiguous, into the position-major V cache

using `flm_p1_emit`, which puts each head in the right *form* (branching on the
head index in the tile's `row_base`), and one drain per destination taking
successive parts of the stream (`qkv_route_probe.py`).

**Head assignment is what makes the routing expressible.** Core `c` takes heads
`{c, c+16, c+32}`, so every core holds two q heads and one k-or-v head, and each
emit step is type-homogeneous across all 16 cores:

    step 0, 1   all q            -> pairs 0..7 each contribute 4 q heads
    step 2      k for cores 0-7  -> pairs 0..3, K cache
                v for cores 8-15 -> pairs 4..7, V cache

The natural assignment (core `c` takes heads `3c, 3c+1, 3c+2`) puts q and k in
the same emit step for the core straddling head 32, and a drain cannot split a
step.

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


def heads_of(core):
    """Core c owns heads {c, c+16, c+32} — see the module docstring."""
    return [core + 16 * h for h in range(HPC)]


def build(pos):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = NCORES // 2
    BC = 2 * K_DIM + 2 * HEAD
    OBJ = 2 * HEAD                              # 2*HEAD for every head
    KTILE, VTILE = HEAD * TSEQ, TSEQ * HEAD
    off = pos - (pos & 1)                       # even column of this pair

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(OBJ,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * OBJ,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * HPC * TPH * wt,), np.dtype[np.uint8]]
    q_ty = np.ndarray[(4 * HEAD,), np.dtype[bfloat16]]        # 4 q heads/pair
    kv_ty = np.ndarray[(2 * KTILE,), np.dtype[bfloat16]]      # 2 tiles/pair

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
    bc_cons = [f_bc.cons() for _ in range({NCORES})]
    f_w = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_w]
    f_o = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {OBJ}], obj_types=[o_ty, o_ty]) for f in f_o]

    def core(bcc, wc, op, kqkv, kemit, kprep):
        eb = bcc.acquire(1)
        kprep(eb)
        for _ in range_({HPC}):
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
            # steps 0-1: 4 q heads, first HEAD of each object
            oh[i].drain(qb[i], wait=True, group=tg,
                        sizes=[1, 4, 1, {HEAD}], strides=[0, {OBJ}, 0, 1])
            if i < n // 2:
                # step 2: 2 k heads, each a column pair in its own K tile
                oh[i].drain(kvb[i], wait=True, group=tg, offset={off},
                            sizes=[1, 2, {HEAD}, 2],
                            strides=[0, {KTILE}, {TSEQ}, 1])
            else:
                # step 2: 2 v heads, each contiguous in its own V tile
                oh[i].drain(kvb[i], wait=True, group=tg, offset={pos} * {HEAD},
                            sizes=[1, 2, 1, {HEAD}], strides=[0, {VTILE}, 0, 1])
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
              w_all_ty=w_all_ty, q_ty=q_ty, kv_ty=kv_ty, __name__="flm_p1_route")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[QKV_SRC, EMIT_SRC, NORM_SRC],
                    full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--pos", type=int, default=0, help="KV cache position")
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
    q_ts = [iron.zeros(4 * HEAD, dtype=bfloat16, device="npu") for _ in range(npairs)]
    kv_ts = [iron.zeros(2 * KTILE, dtype=bfloat16, device="npu") for _ in range(npairs)]
    design(bc_t, *w_ts, *q_ts, *kv_ts)

    print(f"P1 routed to three destinations: {NCORES} cores, layer {o.layer}, "
          f"pos {o.pos}")
    ok, worst = True, 0.0
    # q': pair pr emits heads 2pr, 2pr+1 (step 0) then 2pr+16, 2pr+17 (step 1)
    for pr in range(npairs):
        got = q_ts[pr].numpy().astype(np.float64).reshape(4, HEAD)
        want = [2 * pr, 2 * pr + 1, 2 * pr + 16, 2 * pr + 17]
        for slot, h in enumerate(want):
            e = np.abs(got[slot] - ref[h]).max()
            worst = max(worst, e); ok &= e == 0
    print(f"  q' : 32 heads over {npairs} pairs      max err {worst:.4e}")
    if __import__("os").environ.get("P1_DIAG"):
        import sys as _s
        got = q_ts[0].numpy().astype(np.float64).reshape(4, HEAD)
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
    for pr in range(npairs // 2):                      # K cache
        K = kv_ts[pr].numpy().astype(np.float64).reshape(2, HEAD, TSEQ)
        for j in range(2):
            e = np.abs(K[j, :, o.pos] - ref[2 * pr + j + NQ]).max()
            we = max(we, e); ok &= e == 0
    print(f"  k' : 8 heads into the K cache col {o.pos}  max err {we:.4e}")

    ve = 0.0
    for pr in range(npairs // 2, npairs):              # V cache
        V = kv_ts[pr].numpy().astype(np.float64).reshape(2, TSEQ, HEAD)
        for j in range(2):
            e = np.abs(V[j, o.pos] - ref[2 * pr + j + NQ]).max()
            ve = max(ve, e); ok &= e == 0
    print(f"  v' : 8 heads into the V cache row {o.pos}  max err {ve:.4e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
