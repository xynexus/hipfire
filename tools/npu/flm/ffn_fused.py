#!/usr/bin/env python3
"""Fused gate/up + SwiGLU across 16 cores in ONE dispatch, verified and timed.

`ffn_verify.py` proved the FFN arithmetic but ran four dispatches with host glue
between them. This fuses the FFN's first half — gate GEMV, up GEMV and SwiGLU —
into a single kernel and a single dispatch across the paired 16-core array.

It fuses cleanly because that half is entirely local: gate and up read the same
activation and produce the same output rows, so **the 8192-wide intermediate is
never materialised in memory**. Each core owns a slice and nothing crosses cores.
(down_proj is the opposite — its activation is the whole SwiGLU output — so it
stays a separate phase.)

    python3 ffn_fused.py                    # verify + time, 16 cores
    python3 ffn_fused.py --sweep-cores 8,16

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
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
FFN_SRC = str(KDIR / "flm_ffn_gate_up.cc")
PREP_SRC = str(KDIR / "flm_asum_prepare.cc")
BLK = 32
FLM_DECODE_GBS = 46.2


def tile_bytes(K, NROWS):
    return 2 * NROWS * (K // BLK) * 2 + NROWS * (K // 2)


def build(K, NROWS, ncores, tiles_per_core):
    """Cores in pairs (shim budget), one fused weight stream per pair.

    NROWS is capped harder than the plain GEMV: the fused tile carries BOTH the
    gate and the up weights, so it is 2x as large. At K=2048, NROWS=8 gives
    2x10240 = 20480 B, 40960 double-buffered, +8192 activation = 49152 of 65536.
    NROWS=16 would need 81920 and does not fit.
    """
    if ncores % 2:
        raise ValueError("--cores must be even (cores are wired in pairs)")
    wt = 2 * tile_bytes(K, NROWS)          # gate tile + up tile
    npairs = ncores // 2
    rows_per_core = tiles_per_core * NROWS

    act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * tiles_per_core * wt,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(2 * rows_per_core,), np.dtype[bfloat16]]

    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(act: In, {params}):
    kern = ExternalFunction(
        "flm_ffn_gate_up", source_file=FFN_SRC,
        arg_types=[act_ty, wt_ty, o_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"])
    prep = ExternalFunction(
        "flm_asum_prepare", source_file=PREP_SRC, arg_types=[act_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"])

    f_act = ObjectFifo(act_ty, name="act")
    act_cons = [f_act.cons() for _ in range({ncores})]
    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_wpair]
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty]) for f in f_opair]

    def core(a_cons, w_cons, o_prod, k, kprep):
        ea = a_cons.acquire(1)
        kprep(ea)
        for _ in range_({tiles_per_core}):
            ew = w_cons.acquire(1)
            eo = o_prod.acquire(1)
            k(ea, ew, eo)
            o_prod.release(1)
            w_cons.release(1)
        a_cons.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(
                core,
                fn_args=[act_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kern, prep],
                stack_size=8192))

    def sequence(*args):
        a = args[0]
        wb = [args[1 + i] for i in range({npairs})]
        ob = [args[1 + {npairs} + i] for i in range({npairs})]
        ah = args[1 + 2 * {npairs}]
        wh = [args[2 + 2 * {npairs} + i] for i in range({npairs})]
        oh = [args[2 + 3 * {npairs} + i] for i in range({npairs})]
        ah.fill(a)
        for i in range({npairs}):
            wh[i].fill(wb[i])
        for i in range({npairs}):
            oh[i].drain(ob[i], wait=True)

    arg_types = [act_ty] + [w_all_ty] * {npairs} + [o_all_ty] * {npairs}
    arg_types += [f_act.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_wpair]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_opair]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, FFN_SRC=FFN_SRC, PREP_SRC=PREP_SRC,
              act_ty=act_ty, wt_ty=wt_ty, o_ty=o_ty, wpair_ty=wpair_ty,
              opair_ty=opair_ty, w_all_ty=w_all_ty, o_all_ty=o_all_ty,
              __name__="flm_ffn_fused")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[FFN_SRC, PREP_SRC],
                    full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--nrows", type=int, default=8)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--tiles", type=int, default=64,
                   help="fused tiles per core (64 x 8 rows x 16 cores = 8192 "
                        "intermediate rows, one llama FFN first half)")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sweep-cores", default=None)
    o = p.parse_args()

    K, NROWS = o.k, o.nrows
    c = q4nx.Q4nx(str(Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"))
    pre = f"model.layers.{o.layer}."

    points = ([int(x) for x in o.sweep_cores.split(",")]
              if o.sweep_cores else [o.cores])
    print(f"fused gate/up + SwiGLU: K={K} {NROWS} rows/tile {o.tiles} tiles/core")
    print(f"{'cores':>5s} {'MB':>7s} {'GB/s':>7s} {'us':>9s} {'vs FLM':>7s} {'max err':>10s}")
    print("-" * 52)

    for ncores in points:
        design, wt = build(K, NROWS, ncores, o.tiles)
        rows = o.tiles * NROWS
        gd, gm, gc_ = load_linear(c, pre + "mlp.gate_proj.weight", rows, K)
        ud, um, uc_ = load_linear(c, pre + "mlp.up_proj.weight", rows, K)

        # one fused tile = [gate tile][up tile]; the pair stream interleaves the
        # two cores' fused tiles, since the memtile split hands [0,wt) to one.
        per = np.concatenate([
            np.concatenate([
                q4nx.pack_tile(gd[i:i + NROWS], gm[i:i + NROWS], gc_[i:i + NROWS]),
                q4nx.pack_tile(ud[i:i + NROWS], um[i:i + NROWS], uc_[i:i + NROWS]),
            ]) for i in range(0, rows, NROWS)
        ])
        assert per.size == o.tiles * wt
        wpair = np.empty(2 * per.size, np.uint8)
        v = wpair.reshape(o.tiles, 2, wt)
        v[:, 0, :] = per.reshape(o.tiles, wt)
        v[:, 1, :] = per.reshape(o.tiles, wt)

        rng = np.random.default_rng(0)
        h = q4nx.bf16_to_f32(q4nx.f32_to_bf16(
            (rng.standard_normal(K) * 0.05).astype(np.float32)))

        a_t = iron.tensor(h.astype(bfloat16), dtype=bfloat16, device="npu")
        w_ts = [iron.tensor(wpair, dtype=np.uint8, device="npu")
                for _ in range(ncores // 2)]
        o_ts = [iron.zeros(2 * rows, dtype=bfloat16, device="npu")
                for _ in range(ncores // 2)]

        bench = run_iters(design, a_t, *w_ts, *o_ts, warmup=2, iters=10)
        npu = bench.npu
        us = npu.min_us if npu else bench.e2e.min_us
        total = ncores * o.tiles * wt
        gbs = total / (us * 1e-6) / 1e9

        # reference: the same fused math in float64
        g = np.concatenate([q4nx.gemv_reference_bf16(h, gd[i:i+NROWS], gm[i:i+NROWS],
                                                     gc_[i:i+NROWS])
                            for i in range(0, rows, NROWS)])
        u = np.concatenate([q4nx.gemv_reference_bf16(h, ud[i:i+NROWS], um[i:i+NROWS],
                                                     uc_[i:i+NROWS])
                            for i in range(0, rows, NROWS)])
        ref = (g / (1.0 + np.exp(-g))) * u
        got = o_ts[0].numpy().astype(np.float64).reshape(o.tiles, 2, NROWS)[:, 0, :].reshape(-1)
        err = np.abs(got - ref).max()

        print(f"{ncores:5d} {total/1e6:7.1f} {gbs:7.1f} {us:9.1f} "
              f"{gbs/FLM_DECODE_GBS:6.2f}x {err:10.2e}")

    # SwiGLU rides the hardware exp2, measured at 3.5% mean / 5.9% max relative
    # error, so the tolerance is set by the NLF unit and not by bf16.
    print("\n(err is vs a float64 reference; the floor is AIE2P's hardware exp2,")
    print(" measured at 3.54% mean / 5.86% max relative — see the refe log.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
