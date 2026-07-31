#!/usr/bin/env python3
"""Fused q/k/v projection + RoPE epilogue in ONE dispatch — plan Task 5, phase P1.

q, k and v read the same activation, so they are one N=3072 projection, not
three. At 16 cores that is 192 rows/core, and 192 = 3 x 64, so **every core owns
exactly three whole heads and RoPE never straddles a core** — which is why this
phase runs on 16 cores rather than 32.

The whole of P1 is here: RMSNorm (fused into the GEMV prologue by
`flm_norm_prepare`, normalising the broadcast in place), the N=3072 GEMV, and
RoPE applied per completed head from the weight tile's `row_base` trailer —

    head = row_base / 64;  q if head < 32,  k if head < 40,  else v (no RoPE)

`cs_q` and `cs_k` ride the tail of the broadcast object, so no third DMA input
is needed (a core tile has two). `cs_q` carries the `head_dim^-0.5 * log2(e)`
pre-scale that attention's `exp2` would otherwise have to apply.

**`--interleaved` is a PACK-TIME row permutation, not a second kernel.** The
interleaved pairing (2i, 2i+1) is the half-split pairing (i, i+32) applied to a
permuted row order — which is exactly why llama.cpp's converter permutes q/k
weights instead of shipping a second RoPE. So the plan's `-DROPE_INTERLEAVED`
flag is unnecessary: reorder the tile's rows within a head and one kernel serves
both. It is safe because q.k is a dot product over the head dimension, so any
permutation shared by q and k leaves attention unchanged, and v is never rotated.

Which convention the container wants is still open — it cannot be settled from
the weights (`ground_truth.py`). Each run here is self-consistent against a
numpy reference using the *same* convention, which checks that the kernel
implements it, not that it is the right one.

    python3 qkv_verify.py                       # 16 cores, half-split
    python3 qkv_verify.py --interleaved
    python3 qkv_verify.py --head-base 30        # straddle the q->k->v boundaries
    python3 qkv_verify.py --bench

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from ffn_verify import load_linear  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
QKV_SRC = str(KDIR / "flm_gemv_qkv.cc")
EMIT_SRC = str(KDIR / "flm_qkv_emit.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
K_DIM, HEAD, NROWS = 2048, 64, 16
TPH = HEAD // NROWS            # 4 tiles per head
EPS = 1e-5
ROPE_THETA = 500000.0
FIXED_US = 92.9

rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


def build(ncores, heads_per_core):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = ncores // 2
    tiles = heads_per_core * TPH
    BC = 2 * K_DIM + 2 * HEAD          # [act][aux][cs_q][cs_k] = 4224 bf16

    bc_ty = np.ndarray[(BC,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(HEAD,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * HEAD,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * tiles * wt,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(2 * heads_per_core * HEAD,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
             f"-DDIM_ACT={K_DIM}"]
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(bc: In, {params}):
    kq = ExternalFunction("flm_gemv_qkv", source_file=QKV_SRC,
                          arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    ke = ExternalFunction("flm_qkv_emit", source_file=EMIT_SRC,
                          arg_types=[o_ty], compile_flags=FLAGS)
    kn = ExternalFunction("flm_norm_prepare", source_file=NORM_SRC,
                          arg_types=[bc_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name="bc")
    bc_cons = [f_bc.cons() for _ in range({ncores})]
    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_wpair]
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {HEAD}], obj_types=[o_ty, o_ty]) for f in f_opair]

    def core(bcc, wc, op, kqkv, kemit, kprep):
        eb = bcc.acquire(1)
        kprep(eb)                                  # RMSNorm in place + block sums
        for _ in range_({heads_per_core}):
            for _ in range_({TPH}):                # 4 tiles complete one head
                ew = wc.acquire(1)
                kqkv(eb, ew)
                wc.release(1)
            eo = op.acquire(1)                     # the head is rotated and done
            kemit(eo)
            op.release(1)
        bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[bc_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kq, ke, kn], stack_size=4096))

    def sequence(*args):
        b = args[0]
        wb = [args[1 + i] for i in range({npairs})]
        ob = [args[1 + {npairs} + i] for i in range({npairs})]
        bh = args[1 + 2 * {npairs}]
        wh = [args[2 + 2 * {npairs} + i] for i in range({npairs})]
        oh = [args[2 + 3 * {npairs} + i] for i in range({npairs})]
        bh.fill(b)
        for i in range({npairs}):
            wh[i].fill(wb[i])
        for i in range({npairs}):
            oh[i].drain(ob[i], wait=True)

    arg_types = [bc_ty] + [w_all_ty] * {npairs} + [o_all_ty] * {npairs}
    arg_types += [f_bc.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_wpair]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_opair]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, QKV_SRC=QKV_SRC,
              EMIT_SRC=EMIT_SRC, NORM_SRC=NORM_SRC, FLAGS=flags, bc_ty=bc_ty,
              wt_ty=wt_ty, o_ty=o_ty, wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_all_ty=w_all_ty, o_all_ty=o_all_ty, __name__="flm_qkv_verify")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[QKV_SRC, EMIT_SRC, NORM_SRC],
                    full_elf=True), wt, BC


def qkv_rows(c, layer, first, n):
    """Rows [first, first+n) of the concatenated [W_q; W_k; W_v], K=2048."""
    pre = f"model.layers.{layer}.self_attn."
    segs = [(pre + "q_proj.weight", 0, 2048), (pre + "k_proj.weight", 2048, 512),
            (pre + "v_proj.weight", 2560, 512)]
    d = np.empty((n, K_DIM // 32), np.float32)
    m = np.empty_like(d)
    q = np.empty((n, K_DIM // 32, 32), np.uint8)
    for name, base, cnt in segs:
        lo, hi = max(first, base), min(first + n, base + cnt)
        if lo >= hi:
            continue
        dd, mm, qq = load_linear(c, name, base + cnt - base, K_DIM)
        d[lo - first:hi - first] = dd[lo - base:hi - base]
        m[lo - first:hi - first] = mm[lo - base:hi - base]
        q[lo - first:hi - first] = qq[lo - base:hi - base]
    return d, m, q


def rope_ref(v, cs):
    """cs = [cos(32)][sin(32)], v = one head (64). Same math as the kernel.

    Half-split only, matching the kernel. Interleaved is this applied to the
    permuted row order, which the caller has already packed.

    `v` must already be rounded to bf16: the kernel stages the GEMV result in a
    bf16 buffer and rotates *that*, so a reference that rotates full-precision
    values is measuring its own extra precision. It costs 1.25% here, because
    `x*cos - y*sin` is a difference of similar magnitudes and amplifies the
    0.2% bf16 step. Same reason `gemv_reference_bf16` is the gate and not
    `gemv_reference`.
    """
    cv, sv = cs[:HEAD // 2].astype(np.float64), cs[HEAD // 2:].astype(np.float64)
    x, y = v[:HEAD // 2], v[HEAD // 2:]
    return np.concatenate([x * cv - y * sv, y * cv + x * sv])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--heads-per-core", type=int, default=3)
    p.add_argument("--head-base", type=int, default=0,
                   help="first head index; 30 straddles the q->k->v boundaries")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--pos", type=int, default=17)
    p.add_argument("--interleaved", action="store_true",
                   help="pack q/k rows in the (2i,2i+1) order; one kernel serves both")
    p.add_argument("--bench", action="store_true")
    o = p.parse_args()

    ncores, hpc = o.cores, o.heads_per_core
    rows_pc = hpc * HEAD
    c = q4nx.Q4nx(str(Q4NX))
    nw = c.bf16(f"model.layers.{o.layer}.input_layernorm.weight").astype(np.float32)[:K_DIM]

    # cs tables: rope_freqs.weight is the stored llama3 per-frequency divisor.
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)[:HEAD // 2]
    inv_freq = (1.0 / ROPE_THETA ** (np.arange(0, HEAD, 2) / HEAD)) / divisor
    ang = o.pos * inv_freq
    cs_k = rnd(np.concatenate([np.cos(ang), np.sin(ang)]))
    cs_q = rnd(cs_k * (HEAD ** -0.5) * np.log2(np.e))   # attention's pre-scale

    design, wt, BC = build(ncores, hpc)

    rng = np.random.default_rng(0)
    x = rnd(rng.standard_normal(K_DIM) * 0.05)
    bc = np.zeros(BC, np.float32)
    bc[:K_DIM] = x
    bc[K_DIM:2 * K_DIM] = nw
    bc[2 * K_DIM:2 * K_DIM + HEAD] = cs_q
    bc[2 * K_DIM + HEAD:] = cs_k

    # RMSNorm exactly as flm_norm_prepare does it (bf16 roundings included)
    xd = x.astype(np.float64)
    inv = np.float32(1.0 / np.sqrt((xd * xd).mean() + EPS))
    xn = rnd(rnd(x * rnd(inv)) * nw)

    # Row order within a head. Half-split pairs (i, i+32); feeding it
    # [v0,v2,...,v62, v1,v3,...,v63] makes it pair (2i, 2i+1) instead, which is
    # the interleaved convention — same kernel, different pack.
    perm = (np.concatenate([np.arange(0, HEAD, 2), np.arange(1, HEAD, 2)])
            if o.interleaved else np.arange(HEAD))

    per_core, refs = [], []
    for core in range(ncores):
        first = (o.head_base + core * hpc) * HEAD
        d, m, q = qkv_rows(c, o.layer, first, rows_pc)
        for h in range(hpc):                      # permute rows within each head
            sl = slice(h * HEAD, (h + 1) * HEAD)
            d[sl], m[sl], q[sl] = d[sl][perm], m[sl][perm], q[sl][perm]
        per_core.append(np.concatenate([
            q4nx.pack_tile(d[i:i + NROWS], m[i:i + NROWS], q[i:i + NROWS],
                           row_base=first + i)
            for i in range(0, rows_pc, NROWS)]))
        r = []
        for h in range(hpc):
            head = first // HEAD + h
            v = np.concatenate([
                q4nx.gemv_reference_bf16(xn, d[i:i + NROWS], m[i:i + NROWS],
                                         q[i:i + NROWS])
                for i in range(h * HEAD, (h + 1) * HEAD, NROWS)])
            # the kernel stages the GEMV result in bf16, then rotates, then
            # stores bf16 — model both roundings or the reference is measuring
            # its own precision.
            # v is already in the packed (permuted) row order, and half-split
            # rotation of that order IS the interleaved rotation — so the
            # reference is the half-split formula either way, and the result
            # stays in the packed order the kernel emits.
            v = rnd(v)
            if head < 32:
                v = rope_ref(v, cs_q)
            elif head < 40:
                v = rope_ref(v, cs_k)
            r.append(rnd(v))
        refs.append(np.concatenate(r))

    b_t = iron.tensor(bc.astype(bfloat16), dtype=bfloat16, device="npu")
    w_ts, o_ts = [], []
    for pr in range(ncores // 2):
        buf = np.empty(2 * per_core[0].size, np.uint8).reshape(-1, 2, wt)
        buf[:, 0, :] = per_core[2 * pr].reshape(-1, wt)
        buf[:, 1, :] = per_core[2 * pr + 1].reshape(-1, wt)
        w_ts.append(iron.tensor(buf.reshape(-1), dtype=np.uint8, device="npu"))
        o_ts.append(iron.zeros(2 * rows_pc, dtype=bfloat16, device="npu"))

    if o.bench:
        bench = run_iters(design, b_t, *w_ts, *o_ts, warmup=2, iters=10)
        us = bench.npu.min_us if bench.npu else bench.e2e.min_us
    else:
        design(b_t, *w_ts, *o_ts)
        us = None

    conv = ("interleaved (2i,2i+1), packed row order"
            if o.interleaved else "half-split (i,i+32), model row order")
    print(f"qkv+RoPE fused: K={K_DIM} N={ncores*rows_pc} {ncores} cores, "
          f"{hpc} heads/core, head_base {o.head_base}")
    print(f"  RoPE convention: {conv}  [OPEN — see ground_truth.py]")

    worst, scale = 0.0, 0.0
    for pr in range(ncores // 2):
        got = o_ts[pr].numpy().astype(np.float64).reshape(hpc, 2, HEAD)
        for j in range(2):
            ref = refs[2 * pr + j]
            e = np.abs(got[:, j, :].reshape(-1) - ref)
            worst = max(worst, e.max())
            scale = max(scale, np.abs(ref).mean())
    kinds = [("q", 32), ("k", 40), ("v", 48)]
    hb, he = o.head_base, o.head_base + ncores * hpc
    covered = ",".join(n for n, lim in kinds
                       if hb < lim and he > (0 if n == "q" else 32 if n == "k" else 40))
    print(f"  heads {hb}..{he-1} covering {covered}")
    print(f"  max err vs numpy (same convention) {worst:.4e}   mean|ref| {scale:.5f}")
    if us:
        total = ncores * hpc * TPH * wt
        print(f"  {total/1e6:.2f} MB  {total/(us*1e-6)/1e9:.1f} GB/s  {us:.1f} us "
              f"(marginal {us-FIXED_US:.1f})")
    ok = worst <= 1e-2 * scale
    print(f"  tolerance {1e-2*scale:.4e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
