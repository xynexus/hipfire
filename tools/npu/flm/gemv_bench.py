#!/usr/bin/env python3
"""Decode GEMV across N cores in ONE dispatch — the milestone-4 throughput bar.

Puts `kernels/npu/flm_gemv_q4_1.cc` under the dataflow `layer.xclbin` uses at
decode: **one activation vector broadcast to every GEMM core**, a **private
weight stream per core**, one command for the whole thing. FLM runs 16 GEMM
cores fed from 4 memtile columns and achieves 46.2 GB/s of q4_1 weights
(`docs/npu/flm-layer-dataflow.md`).

Weight traffic is what decode is bound by: llama streams 38.0 MB per layer at
5.00 bpw, and `dispatch_bw_probe.py` established that the fabric delivers
48.6 GB/s through 4 feed streams and 56.2 GB/s through 8. This measures what is
left of that once real GEMV arithmetic runs against the same bytes.

    python3 gemv_bench.py                     # 16 cores, one layer's weights
    python3 gemv_bench.py --sweep-cores 4,8,16

Result: **48.1 GB/s = 1.04x FLM decode** on 38.0 MB in one dispatch.

Reference points:
  46.2 GB/s   FLM decode, measured end to end
  48.6 GB/s   fabric through 4 feed streams, no compute (dispatch_bw_probe)
  56.5 GB/s   fabric roof
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KERNEL_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_gemv_q4_1.cc")
PREP_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_asum_prepare.cc")
BLK = 32
FLM_DECODE_GBS = 46.2
FABRIC_ROOF_GBS = 56.5
FOUR_STREAM_GBS = 48.6


def tile_bytes(K, NROWS):
    return 2 * NROWS * (K // BLK) * 2 + NROWS * (K // 2)


def build(K, NROWS, ncores, tiles_per_core):
    """Cores in PAIRS, the way `layer.xclbin` is wired.

    One shim stream feeds each pair and a memtile splits it in two; each pair's
    two output streams are joined back into one before the shim. That is FLM's
    structure — 16 weight streams out of 4 memtiles, and a GEMM pair whose two
    N-groups concatenate into one result stream — and it is not cosmetic:
    16 private weight streams plus an activation is 17 shim inputs against
    8 columns x 2 channels, and the placer rejects it outright with
    `no ShimNOCTile has sufficient DMA capacity`. Pairing halves both counts.
    """
    if ncores % 2:
        raise ValueError("--cores must be even (cores are wired in pairs)")
    wtile = tile_bytes(K, NROWS)
    rows_per_core = tiles_per_core * NROWS
    npairs = ncores // 2

    act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wtile,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[np.float32]]
    wpair_ty = np.ndarray[(2 * wtile,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[np.float32]]
    w_pair_all_ty = np.ndarray[(2 * tiles_per_core * wtile,), np.dtype[np.uint8]]
    o_pair_all_ty = np.ndarray[(2 * rows_per_core,), np.dtype[np.float32]]

    # Generated, because the buffer count is the point -- the whole array is
    # bound to ONE dispatch. Indexed, never sliced: a constant slice folds into
    # co_consts and mlir-aie's jit cache hashes the generator with
    # marshal.dumps(code, 4), which cannot serialize a slice.
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(act: In, {params}):
    kern = ExternalFunction(
        "flm_gemv_q4_1", source_file=KERNEL_SRC,
        arg_types=[act_ty, wt_ty, o_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"])
    # The activation block-sums depend only on the activation, so they are
    # computed once per acquire instead of once per weight tile.
    prep = ExternalFunction(
        "flm_asum_prepare", source_file=PREP_SRC, arg_types=[act_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"])

    # One activation fifo per consumer. FLM broadcasts a single memtile channel
    # to 17 consumers; IRON expresses the same sharing as one producer handle
    # with many consumer handles.
    f_act = ObjectFifo(act_ty, name="act")
    act_cons = [f_act.cons() for _ in range({ncores})]

    # One shim stream per pair, split in a memtile into the pair's two cores.
    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wtile}], obj_types=[wt_ty, wt_ty])
             for f in f_wpair]
    # The pair's two result streams concatenate before the shim. Offsets are in
    # ELEMENTS, not bytes -- a byte offset here overshoots the fifo and emits a
    # BD with a negative length (`XAie_DmaSetAddrLen ... static_len = -16`).
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty])
             for f in f_opair]

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
                stack_size=4096))

    def sequence(*args):
        a = args[0]
        wbufs = [args[1 + i] for i in range({npairs})]
        obufs = [args[1 + {npairs} + i] for i in range({npairs})]
        ah = args[1 + 2 * {npairs}]
        wh = [args[2 + 2 * {npairs} + i] for i in range({npairs})]
        oh = [args[2 + 3 * {npairs} + i] for i in range({npairs})]
        ah.fill(a)
        for i in range({npairs}):
            wh[i].fill(wbufs[i])
        for i in range({npairs}):
            oh[i].drain(obufs[i], wait=True)

    arg_types = [act_ty]
    arg_types += [w_pair_all_ty] * {npairs}
    arg_types += [o_pair_all_ty] * {npairs}
    arg_types += [f_act.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_wpair]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_opair]

    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, KERNEL_SRC=KERNEL_SRC, PREP_SRC=PREP_SRC,
              act_ty=act_ty, wt_ty=wt_ty, o_ty=o_ty,
              wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_pair_all_ty=w_pair_all_ty, o_pair_all_ty=o_pair_all_ty,
              # exec()'d functions get __module__ = None, which mlir-aie's jit
              # cache hashes with .encode() and dies on.
              __name__="flm_gemv_bench")
    exec(src, ns)
    # full_elf: the vararg dispatch path caps at ~20 host buffers, fails as a
    # firmware hang rather than an error, and is ~34% slower where it works.
    return iron.jit(ns["_design"], source_files=[KERNEL_SRC, PREP_SRC], full_elf=True), wtile


def run(K, NROWS, ncores, tiles_per_core, warmup, iters, tensor, check):
    design, wtile = build(K, NROWS, ncores, tiles_per_core)
    rows_per_core = tiles_per_core * NROWS
    nb = K // BLK

    c = q4nx.Q4nx(str(Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"))
    d_all, m_all, codes_all = c.blocks(tensor)
    need = rows_per_core * nb
    avail = d_all.size
    reps = -(-need // avail)
    d = np.tile(d_all.ravel(), reps)[:need].reshape(rows_per_core, nb).astype(np.float32)
    m = np.tile(m_all.ravel(), reps)[:need].reshape(rows_per_core, nb).astype(np.float32)
    codes = np.tile(codes_all.reshape(-1, BLK), (reps, 1))[:need].reshape(rows_per_core, nb, BLK)

    wbuf = np.concatenate([
        q4nx.pack_tile(d[i:i + NROWS], m[i:i + NROWS], codes[i:i + NROWS])
        for i in range(0, rows_per_core, NROWS)
    ])
    assert wbuf.size == tiles_per_core * wtile

    rng = np.random.default_rng(0)
    act = q4nx.bf16_to_f32(q4nx.f32_to_bf16(rng.standard_normal(K).astype(np.float32)))

    # The pair's shim stream carries both cores' tiles interleaved, since the
    # memtile split hands byte range [0, wtile) to one core and [wtile, 2*wtile)
    # to the other on every acquire.
    npairs = ncores // 2
    wpair = np.empty(2 * wbuf.size, np.uint8)
    per = wbuf.reshape(tiles_per_core, wtile)
    wpair.reshape(tiles_per_core, 2, wtile)[:, 0, :] = per
    wpair.reshape(tiles_per_core, 2, wtile)[:, 1, :] = per

    a_t = iron.tensor(act.astype(bfloat16), dtype=bfloat16, device="npu")
    # Every core gets the same bytes: this measures delivery and arithmetic,
    # and identical content costs nothing since nothing is cached across cores.
    w_ts = [iron.tensor(wpair, dtype=np.uint8, device="npu") for _ in range(npairs)]
    o_ts = [iron.zeros(2 * rows_per_core, dtype=np.float32, device="npu")
            for _ in range(npairs)]

    bench = run_iters(design, a_t, *w_ts, *o_ts, warmup=warmup, iters=iters)
    npu = bench.npu
    us = npu.min_us if npu else bench.e2e.min_us
    total = ncores * tiles_per_core * wtile
    gbs = total / (us * 1e-6) / 1e9

    ok = None
    if check:
        # The same correctness gate as gemv_verify, on the bytes this run
        # actually streamed -- a bandwidth figure from a kernel that computed
        # nothing would otherwise look excellent.
        exp = np.concatenate([
            q4nx.gemv_reference_bf16(act, d[i:i + NROWS], m[i:i + NROWS],
                                     codes[i:i + NROWS])
            for i in range(0, rows_per_core, NROWS)
        ])
        # Each pair's output buffer is (tiles, 2, NROWS): the join writes the
        # two cores' row blocks side by side per tile.
        pair_exp = np.empty(2 * rows_per_core)
        pv = pair_exp.reshape(tiles_per_core, 2, NROWS)
        ev = exp.reshape(tiles_per_core, NROWS)
        pv[:, 0, :] = ev
        pv[:, 1, :] = ev
        errs = [np.abs(o.numpy().astype(np.float64) - pair_exp).max() for o in o_ts]
        ok = max(errs)
    return gbs, us, total, ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--nrows", type=int, default=16,
                   help="output rows per weight tile; 16 is the widest that fits "
                        "L1 with double buffering (49 KB of 64)")
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--tiles", type=int, default=116,
                   help="weight tiles per core (116 x 16 rows x 1280 B x 16 cores "
                        "= 38.0 MB, one llama decoder layer)")
    p.add_argument("--tensor", default="model.layers.0.mlp.down_proj.weight")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--sweep-cores", default=None)
    o = p.parse_args()

    points = ([int(x) for x in o.sweep_cores.split(",")]
              if o.sweep_cores else [o.cores])
    print(f"K={o.k}  {o.nrows} rows/tile  {o.tiles} tiles/core\n")
    print(f"{'cores':>5s} {'MB':>8s} {'GB/s':>8s} {'us':>9s} {'vs FLM':>7s} "
          f"{'max err':>10s}")
    print("-" * 54)
    best = 0.0
    for n in points:
        try:
            gbs, us, total, err = run(o.k, o.nrows, n, o.tiles, o.warmup,
                                      o.iters, o.tensor, not o.no_check)
        except Exception as e:
            first = str(e).strip().splitlines()[0][:60] if str(e).strip() else type(e).__name__
            print(f"{n:5d} {'FAIL':>8s} {'':>8s} {'':>9s} {'':>7s}  {first}")
            continue
        es = "skipped" if err is None else f"{err:.2e}"
        print(f"{n:5d} {total/1e6:8.1f} {gbs:8.1f} {us:9.1f} "
              f"{gbs/FLM_DECODE_GBS:6.2f}x {es:>10s}")
        best = max(best, gbs)

    if best:
        print(f"\nbest {best:.1f} GB/s = {best/FLM_DECODE_GBS:.2f}x FLM decode, "
              f"{100*best/FABRIC_ROOF_GBS:.0f}% of the {FABRIC_ROOF_GBS} GB/s roof")
        print(f"  ({100*best/FOUR_STREAM_GBS:.0f}% of the {FOUR_STREAM_GBS} GB/s "
              f"that 4 feed streams deliver with no compute)")
    return 0 if best >= 0.9 * FLM_DECODE_GBS else 1


if __name__ == "__main__":
    raise SystemExit(main())
