#!/usr/bin/env python3
"""gate/up + SwiGLU by ALTERNATING acquires at 16 rows/tile — plan Task 4.

`ffn_fused.py` takes ONE tile carrying both gate and up weights, so the tile is
2x the size and the 64 KB tile memory allows only **8 rows**, against 16
everywhere else. 8 rows measured ~25% worse than 16 on the plain GEMV, which is
the hypothesis: that the fused stage's 41.4 GB/s is the *row count*, not the
kernel's arithmetic.

This streams gate and up as alternating acquires of single 16-row tiles — the
weight stream reordered offline to `[gate t0][up t0][gate t1][up t1]...`, same
bytes, no extra DMA channel, no extra dispatch. The 8192-wide intermediate is
still never materialised: the gate result is stashed in 64 B in-core and SwiGLU
runs on the up tile.

**This is the plan's most falsifiable claim.** Gate: marginal time (wall minus
the 92.9 us fixed cost) <= 400 us on 21.0 MB, against ~414 us today and an
ideal of 369 us. If it lands at ~414, the row count was NOT the cause and this
should be reverted in favour of the simpler single-tile kernel.

    python3 ffn_alt.py --cores 16 --tiles 32

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
GATE_SRC = str(KDIR / "flm_gemv_gate.cc")
UP_SRC = str(KDIR / "flm_gemv_up_swiglu.cc")
PREP_SRC = str(KDIR / "flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
FIXED_US = 92.9


def build(K, NROWS, ncores, tiles_per_core):
    """tiles_per_core counts gate/up PAIRS; each pair is two acquires."""
    wt = q4nx.tile_bytes(K, NROWS)
    npairs = ncores // 2
    rows = tiles_per_core * NROWS

    act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * 2 * tiles_per_core * wt,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(2 * rows,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K}", f"-DDIM_NROWS={NROWS}"]
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(act: In, {params}):
    kg = ExternalFunction("flm_gemv_gate", source_file=GATE_SRC,
                          arg_types=[act_ty, wt_ty], compile_flags=FLAGS)
    ku = ExternalFunction("flm_gemv_up_swiglu", source_file=UP_SRC,
                          arg_types=[act_ty, wt_ty, o_ty], compile_flags=FLAGS)
    prep = ExternalFunction("flm_asum_prepare", source_file=PREP_SRC,
                            arg_types=[act_ty], compile_flags=FLAGS)

    f_act = ObjectFifo(act_ty, depth=1, name="act")
    act_cons = [f_act.cons() for _ in range({ncores})]
    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_wpair]
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty]) for f in f_opair]

    def core(ac, wc, op, kgate, kup, kp):
        ea = ac.acquire(1)
        kp(ea)
        for _ in range_({tiles_per_core}):
            eg = wc.acquire(1)      # gate tile
            kgate(ea, eg)
            wc.release(1)
            eu = wc.acquire(1)      # up tile, then SwiGLU in-core
            eo = op.acquire(1)
            kup(ea, eu, eo)
            op.release(1)
            wc.release(1)
        ac.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[act_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kg, ku, prep], stack_size=4096))

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
              ExternalFunction=ExternalFunction, GATE_SRC=GATE_SRC, UP_SRC=UP_SRC,
              PREP_SRC=PREP_SRC, FLAGS=flags, act_ty=act_ty, wt_ty=wt_ty,
              o_ty=o_ty, wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_all_ty=w_all_ty, o_all_ty=o_all_ty, __name__="flm_ffn_alt")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[GATE_SRC, UP_SRC, PREP_SRC],
                    full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--nrows", type=int, default=16)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--tiles", type=int, default=32, help="gate/up PAIRS per core")
    p.add_argument("--layer", type=int, default=0)
    o = p.parse_args()

    K, NROWS, ncores, tiles = o.k, o.nrows, o.cores, o.tiles
    rows = tiles * NROWS
    c = q4nx.Q4nx(str(Q4NX))
    pre = f"model.layers.{o.layer}."
    gd, gm, gc_ = load_linear(c, pre + "mlp.gate_proj.weight", rows, K)
    ud, um, uc_ = load_linear(c, pre + "mlp.up_proj.weight", rows, K)

    design, wt = build(K, NROWS, ncores, tiles)
    # [gate t0][up t0][gate t1][up t1]... — reordered offline, same bytes
    per = np.concatenate([
        np.concatenate([
            q4nx.pack_tile(gd[i:i+NROWS], gm[i:i+NROWS], gc_[i:i+NROWS], row_base=i),
            q4nx.pack_tile(ud[i:i+NROWS], um[i:i+NROWS], uc_[i:i+NROWS], row_base=i),
        ]) for i in range(0, rows, NROWS)])
    assert per.size == 2 * tiles * wt
    wpair = np.empty(2 * per.size, np.uint8)
    v = wpair.reshape(2 * tiles, 2, wt)
    pv = per.reshape(2 * tiles, wt)
    v[:, 0, :] = pv
    v[:, 1, :] = pv

    rnd = lambda x: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(x, np.float32)))
    rng = np.random.default_rng(0)
    h = rnd(rng.standard_normal(K) * 0.05)

    a_t = iron.tensor(h.astype(bfloat16), dtype=bfloat16, device="npu")
    w_ts = [iron.tensor(wpair, dtype=np.uint8, device="npu") for _ in range(ncores // 2)]
    o_ts = [iron.zeros(2 * rows, dtype=bfloat16, device="npu") for _ in range(ncores // 2)]
    bench = run_iters(design, a_t, *w_ts, *o_ts, warmup=2, iters=10)
    npu = bench.npu
    us = npu.min_us if npu else bench.e2e.min_us
    total = ncores * 2 * tiles * wt

    g = np.concatenate([q4nx.gemv_reference_bf16(h, gd[i:i+NROWS], gm[i:i+NROWS],
                                                 gc_[i:i+NROWS])
                        for i in range(0, rows, NROWS)])
    u = np.concatenate([q4nx.gemv_reference_bf16(h, ud[i:i+NROWS], um[i:i+NROWS],
                                                 uc_[i:i+NROWS])
                        for i in range(0, rows, NROWS)])
    ref = (g / (1.0 + np.exp(-g))) * u
    got = o_ts[0].numpy().astype(np.float64).reshape(tiles, 2, NROWS)[:, 0, :].reshape(-1)
    err = np.abs(got - ref).max()

    marg = us - FIXED_US
    ideal = total / 1e6 * 17.547
    print(f"gate/up alternating acquires: K={K} {NROWS} rows/tile, {ncores} cores, "
          f"{tiles} pairs/core")
    print(f"  {total/1e6:.2f} MB  {total/(us*1e-6)/1e9:.1f} GB/s  wall {us:.1f} us")
    print(f"  marginal {marg:.1f} us   (ideal {ideal:.1f}, single-tile 8-row ~414)")
    print(f"  max err vs float64 {err:.3e}  (exp2 NLF floor)")
    print(f"  gate: marginal <= 400 us -> {'PASS' if marg <= 400 else 'FAIL'}")
    return 0 if marg <= 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
