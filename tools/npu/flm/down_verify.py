#!/usr/bin/env python3
"""down_proj as 4 K-chunks in ONE dispatch — the fused layer's shape for it.

`down_proj` has K=8192, so a monolithic tile needs a 16384 B activation (32768
double-buffered) and the 64 KB tile memory then allows only **2 rows per weight
tile**, against 16 for every other projection. Measured, that geometry costs
~54 us/layer.

Chunking K into 4x2048 makes it the same shape as everything else: a 4096 B
activation and 16-row tiles. It is exact because the GEMV identity is linear in
blocks, and the container's planar 5120 B row splits on a chunk boundary with
**no repacking of codes** — chunk c is `d[64c:64c+64]`, `m[64c:64c+64]`,
`codes[1024c:1024c+1024]`, which is precisely a K=2048 tile.

Chunks 0..2 run `flm_gemv_acc` (accumulate into a per-core slot); chunk 3 runs
`flm_gemv_flush` (add, apply the residual from the broadcast aux half, emit,
clear).

    python3 down_verify.py
    python3 down_verify.py --cores 16 --bench

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      TaskGroup, Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
ACC_SRC = str(KDIR / "flm_gemv_acc.cc")
FLUSH_SRC = str(KDIR / "flm_gemv_flush.cc")
PREP_SRC = str(KDIR / "flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
CHUNK_K = 2048
NCHUNK = 4
BLK = 32


def chunk_blocks(d, m, codes, c):
    """Chunk c of a K=8192 row set: a pure slice, no code repacking."""
    nb = CHUNK_K // BLK
    return (d[:, c * nb:(c + 1) * nb],
            m[:, c * nb:(c + 1) * nb],
            codes[:, c * nb:(c + 1) * nb, :])


def build(NROWS, ncores, tiles_per_core, ACCN):
    wt = q4nx.tile_bytes(CHUNK_K, NROWS)
    npairs = ncores // 2
    rows = tiles_per_core * NROWS

    act_ty = np.ndarray[(2 * CHUNK_K,), np.dtype[bfloat16]]   # [act][aux]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[np.float32]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[np.float32]]
    w_all_ty = np.ndarray[(2 * NCHUNK * tiles_per_core * wt,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(2 * rows,), np.dtype[np.float32]]
    a_all_ty = np.ndarray[(NCHUNK * 2 * CHUNK_K,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={CHUNK_K}", f"-DDIM_NROWS={NROWS}", f"-DDIM_ACCN={ACCN}"]
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(act: In, {params}):
    kacc = ExternalFunction("flm_gemv_acc", source_file=ACC_SRC,
                            arg_types=[act_ty, wt_ty], compile_flags=FLAGS)
    kfl = ExternalFunction("flm_gemv_flush", source_file=FLUSH_SRC,
                           arg_types=[act_ty, wt_ty, o_ty], compile_flags=FLAGS)
    prep = ExternalFunction("flm_asum_prepare", source_file=PREP_SRC,
                            arg_types=[act_ty], compile_flags=FLAGS)

    f_act = ObjectFifo(act_ty, depth=1, name="act")
    act_cons = [f_act.cons() for _ in range({ncores})]
    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_wpair]
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty]) for f in f_opair]

    def core(ac, wc, op, ka, kf, kp):
        # chunks 0..NCHUNK-2 accumulate; the last one flushes.
        for _ in range_({NCHUNK - 1}):
            ea = ac.acquire(1)
            kp(ea)
            for _ in range_({tiles_per_core}):
                ew = wc.acquire(1)
                ka(ea, ew)
                wc.release(1)
            ac.release(1)
        ea = ac.acquire(1)
        kp(ea)
        for _ in range_({tiles_per_core}):
            ew = wc.acquire(1)
            eo = op.acquire(1)
            kf(ea, ew, eo)
            op.release(1)
            wc.release(1)
        ac.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[act_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kacc, kfl, prep], stack_size=4096))

    def sequence(*args):
        a = args[0]
        wb = [args[1 + i] for i in range({npairs})]
        ob = [args[1 + {npairs} + i] for i in range({npairs})]
        ah = args[1 + 2 * {npairs}]
        wh = [args[2 + 2 * {npairs} + i] for i in range({npairs})]
        oh = [args[2 + 3 * {npairs} + i] for i in range({npairs})]
        # One TaskGroup per chunk so BDs are freed between them: 4 chunks x
        # npairs fills would otherwise exceed the 16 active BDs a shim supports.
        tg = TaskGroup()
        for i in range({npairs}):
            wh[i].fill(wb[i], group=tg)
        ah.fill(a, group=tg)
        for i in range({npairs}):
            oh[i].drain(ob[i], wait=True, group=tg)
        tg.finish()

    arg_types = [a_all_ty] + [w_all_ty] * {npairs} + [o_all_ty] * {npairs}
    arg_types += [f_act.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_wpair]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_opair]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              TaskGroup=TaskGroup, Worker=Worker, AnyShimTile=AnyShimTile,
              range_=range_, ExternalFunction=ExternalFunction,
              ACC_SRC=ACC_SRC, FLUSH_SRC=FLUSH_SRC, PREP_SRC=PREP_SRC, FLAGS=flags,
              act_ty=act_ty, wt_ty=wt_ty, o_ty=o_ty, wpair_ty=wpair_ty,
              opair_ty=opair_ty, w_all_ty=w_all_ty, o_all_ty=o_all_ty,
              a_all_ty=a_all_ty, __name__="flm_down_verify")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[ACC_SRC, FLUSH_SRC, PREP_SRC],
                    full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--nrows", type=int, default=16)
    p.add_argument("--tiles", type=int, default=2, help="tiles per core per chunk")
    p.add_argument("--layer", type=int, default=0)
    o = p.parse_args()

    NROWS, ncores, tiles = o.nrows, o.cores, o.tiles
    rows = tiles * NROWS            # rows per core
    N = rows * ncores // 2 * 2      # pairs share; total distinct rows per pair
    ACCN = rows

    c = q4nx.Q4nx(str(Q4NX))
    d_all, m_all, codes_all = c.blocks(f"model.layers.{o.layer}.mlp.down_proj.weight")
    nb8 = 8192 // BLK
    d = d_all[:rows, :nb8].astype(np.float32)
    m = m_all[:rows, :nb8].astype(np.float32)
    codes = codes_all[:rows, :nb8, :]

    rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))
    rng = np.random.default_rng(0)
    act8 = rnd(rng.standard_normal(8192) * 0.05)
    resid = rnd(rng.standard_normal(rows) * 0.1)

    # host buffers: activation per chunk = [act_chunk][aux], aux carries the
    # residual (only the flush chunk reads it)
    abuf = np.concatenate([
        np.concatenate([act8[ch * CHUNK_K:(ch + 1) * CHUNK_K],
                        np.pad(resid, (0, CHUNK_K - rows))])
        for ch in range(NCHUNK)])

    design, wt = build(NROWS, ncores, tiles, ACCN)
    per = np.concatenate([
        q4nx.pack_tile(*chunk_blocks(d[i:i + NROWS], m[i:i + NROWS],
                                     codes[i:i + NROWS], ch), row_base=i)
        for ch in range(NCHUNK) for i in range(0, rows, NROWS)])
    assert per.size == NCHUNK * tiles * wt, (per.size, NCHUNK * tiles * wt)
    wpair = np.empty(2 * per.size, np.uint8)
    v = wpair.reshape(NCHUNK * tiles, 2, wt)
    pv = per.reshape(NCHUNK * tiles, wt)
    v[:, 0, :] = pv
    v[:, 1, :] = pv

    a_t = iron.tensor(abuf.astype(bfloat16), dtype=bfloat16, device="npu")
    w_ts = [iron.tensor(wpair, dtype=np.uint8, device="npu")
            for _ in range(ncores // 2)]
    o_ts = [iron.zeros(2 * rows, dtype=np.float32, device="npu")
            for _ in range(ncores // 2)]
    bench = run_iters(design, a_t, *w_ts, *o_ts, warmup=2, iters=10)
    npu = bench.npu
    us = npu.min_us if npu else bench.e2e.min_us
    total = ncores * NCHUNK * tiles * wt
    got = o_ts[0].numpy().astype(np.float64).reshape(tiles, 2, NROWS)[:, 0, :].reshape(-1)

    ref = np.concatenate([
        q4nx.gemv_reference_bf16(act8, d[i:i + NROWS], m[i:i + NROWS],
                                 codes[i:i + NROWS])
        for i in range(0, rows, NROWS)]) + resid.astype(np.float64)

    err = np.abs(got - ref)
    scale = np.abs(ref).mean()
    print(f"down_proj K-chunked: {NCHUNK} x K={CHUNK_K}, {NROWS} rows/tile, "
          f"{ncores} cores, {tiles} tiles/core/chunk")
    print(f"  {total/1e6:.2f} MB  {total/(us*1e-6)/1e9:.1f} GB/s  {us:.1f} us")
    print(f"  vs monolithic K=8192 reference: max {err.max():.4e}  "
          f"mean {err.mean():.4e}  (mean|ref| {scale:.5f})")
    ok = err.max() <= 1e-2 * scale
    print(f"  tolerance {1e-2*scale:.4e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
