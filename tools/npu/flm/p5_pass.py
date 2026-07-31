#!/usr/bin/env python3
"""P5 (down_proj) as a dispatch of its own — side B of the two-dispatch layer.

The fused layer does not fit one dispatch: a core holds four phase bodies
(2848 B fixed plus ~3950 B each) and a layer has five, so P1..P4 run in one
dispatch and P5 in another. `p1p2_chain.py` is side A; this is side B.

The split falls on a boundary the data already has. P4 emits `sw` to DDR and P5
reads it back as its broadcast — a round trip the phases already make between
chunks — and P5's residual comes from `g_resid`, which P3 wrote on the same core
and which survives across dispatches (`static_persist_probe.py`; the same
mechanism carries `g_kprev` for the k' column pairs).

Here the residual is host-supplied instead (`RESID_FROM_STASH=0`), so P5's
arithmetic can be verified without needing side A to have run. Wiring the two
together is what turns the projection into a measurement.

    python3 p5_pass.py            # verify
    python3 p5_pass.py --bench    # and time it

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

from ffn_verify import load_linear  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
DOWN_SRC = str(KDIR / "flm_gemv_down.cc")
ASUM_SRC = str(KDIR / "flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"

K_DIM, D_MODEL, D_FF = 2048, 2048, 8192
NROWS, BLK, NCHUNK = 8, 32, 4
FIXED_US = 92.9
rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


def build(ncores):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = ncores // 2
    tiles = D_MODEL // (ncores * NROWS)      # output tiles per core per chunk
    accn = 2 * tiles * NROWS                 # g_acc_down spans a PAIR's rows

    bc_ty = np.ndarray[(2 * K_DIM,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[bfloat16]]
    w_all_ty = np.ndarray[(2 * NCHUNK * tiles * wt,), np.dtype[np.uint8]]
    # NCHUNK times the live data: the accumulating chunks still acquire and
    # release a result object (the loop is uniform, which is what makes P5 one
    # body instead of two), so the stream carries NCHUNK objects per tile and
    # only the last chunk's are written. The host takes that last quarter.
    o_all_ty = np.ndarray[(NCHUNK * 2 * tiles * NROWS,), np.dtype[bfloat16]]
    bc_all_ty = np.ndarray[(NCHUNK * 2 * K_DIM,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_ACCN={accn}"]
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    src = f'''
def _design(bc: In, {params}):
    kd = ExternalFunction("flm_gemv_down", source_file=DOWN_SRC,
                          arg_types=[bc_ty, wt_ty, o_ty], compile_flags=FLAGS)
    kas = ExternalFunction("flm_asum_prepare", source_file=ASUM_SRC,
                           arg_types=[bc_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name=f"p5bc_c{NCHUNK}")
    bc_cons = [f_bc.cons() for _ in range({ncores})]
    f_w = [ObjectFifo(wpair_ty, name=f"p5w{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_w]
    f_o = [ObjectFifo(opair_ty, name=f"p5o{{i}}_c{NCHUNK}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty]) for f in f_o]

    def core(bcc, wc, op, kdown, kasum):
        # One body for all NCHUNK chunks: flm_gemv_down picks accumulate vs
        # flush from the tile flag, so the accumulating chunks and the flushing
        # one share a loop. Two bodies cost ~1700 B more, measured.
        #
        # Every chunk is a different activation, so every chunk needs its own
        # asum prepare — g_asum is what the tile body's `m` term reads, and a
        # phase that skips it computes that term against the previous
        # activation's sums.
        for _ in range_({NCHUNK}):
            eb = bcc.acquire(1)
            kasum(eb)
            for _ in range_({tiles}):
                ew = wc.acquire(1)
                eo = op.acquire(1)
                kdown(eb, ew, eo)
                op.release(1)
                wc.release(1)
            bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[bc_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kd, kas],
                stack_size=4096))

    def sequence(*args):
        n = {npairs}
        bcb = args[0]
        wb = [args[1 + i] for i in range(n)]
        ob = [args[1 + n + i] for i in range(n)]
        bch = args[1 + 2 * n]
        wh = [args[2 + 2 * n + i] for i in range(n)]
        oh = [args[2 + 3 * n + i] for i in range(n)]
        tg = TaskGroup()
        for ch in range({NCHUNK}):
            bch.fill(bcb, group=tg, offset=ch * 2 * {K_DIM},
                     sizes=[1, 1, 1, 2 * {K_DIM}], strides=[0, 0, 0, 1])
        for i in range(n):
            wh[i].fill(wb[i], group=tg)
        for i in range(n):
            oh[i].drain(ob[i], wait=True, group=tg)
        tg.finish()

    at = [bc_all_ty] + [w_all_ty] * {npairs} + [o_all_ty] * {npairs}
    at += [f_bc.prod(tile=AnyShimTile)]
    at += [f.prod(tile=AnyShimTile) for f in f_w]
    at += [f.cons(tile=AnyShimTile) for f in f_o]
    rt = Runtime(sequence, at)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, DOWN_SRC=DOWN_SRC,
              ASUM_SRC=ASUM_SRC, FLAGS=flags, bc_ty=bc_ty, wt_ty=wt_ty,
              o_ty=o_ty, wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_all_ty=w_all_ty, o_all_ty=o_all_ty, bc_all_ty=bc_all_ty,
              __name__="flm_p5_pass")
    exec(src, ns)
    return (iron.jit(ns["_design"], source_files=[DOWN_SRC, ASUM_SRC],
                     full_elf=True), wt, tiles)


def run(sw_t, x, layer=0, ncores=16, bench=False):
    """Side B, driven by side A: `sw_t` is the row-ordered D_FF buffer side A's
    P4 drain produced, used directly as the broadcast source. `x` is the
    residual, host-supplied here — in the fused layer P5 reads it from g_resid,
    which P3 stashed in the previous dispatch and which survives (measured:
    static_persist_probe).

    -> (x_out in row order, wall_us or None)
    """
    npairs = ncores // 2
    design, wt, tiles = build(ncores)
    c = q4nx.Q4nx(str(Q4NX))
    dd, dm, dc = load_linear(c, f"model.layers.{layer}.mlp.down_proj.weight",
                             D_MODEL, D_FF)
    sw = sw_t.numpy().astype(np.float32)
    bc = np.zeros((NCHUNK, 2 * K_DIM), np.float32)
    for ch in range(NCHUNK):
        bc[ch, :K_DIM] = sw[ch * K_DIM:(ch + 1) * K_DIM]
        bc[ch, K_DIM:] = x
    bc_t = iron.tensor(bc.reshape(-1).astype(bfloat16), dtype=bfloat16,
                       device="npu")
    rpp = D_MODEL // npairs
    rows = lambda pr, j: [pr * rpp + t * 2 * NROWS + j * NROWS
                          for t in range(tiles)]
    nbc = D_FF // BLK
    w5 = []
    for pr in range(npairs):
        per = []
        for j in range(2):
            blob = []
            for ch in range(NCHUNK):
                lo = ch * (nbc // NCHUNK)
                hi = lo + nbc // NCHUNK
                for r0 in rows(pr, j):
                    sl = slice(r0, r0 + NROWS)
                    blob.append(q4nx.pack_tile(
                        dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi],
                        row_base=r0, flags=float(ch == NCHUNK - 1)))
            per.append(np.concatenate(blob))
        b = np.empty((NCHUNK * tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w5.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))
    o_ts = [iron.zeros(NCHUNK * 2 * tiles * NROWS, dtype=bfloat16, device="npu")
            for _ in range(npairs)]
    us = None
    if bench:
        b2 = run_iters(design, bc_t, *w5, *o_ts, warmup=2, iters=10)
        us = b2.npu.min_us if b2.npu else b2.e2e.min_us
    else:
        design(bc_t, *w5, *o_ts)
    got = np.concatenate([t.numpy().astype(np.float64)
                          .reshape(NCHUNK, -1)[NCHUNK - 1] for t in o_ts])
    idx = np.concatenate([np.arange(rows(pr, j)[t], rows(pr, j)[t] + NROWS)
                          for pr in range(npairs)
                          for t in range(tiles)
                          for j in (0, 1)])
    out = np.zeros(D_MODEL, np.float64)
    out[idx] = got
    return out, us


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--bench", action="store_true")
    o = p.parse_args()
    ncores, npairs = o.cores, o.cores // 2
    design, wt, tiles = build(ncores)

    c = q4nx.Q4nx(str(Q4NX))
    dd, dm, dc = load_linear(c, f"model.layers.{o.layer}.mlp.down_proj.weight",
                             D_MODEL, D_FF)
    rng = np.random.default_rng(0)
    sw = rnd(rng.standard_normal(D_FF) * 0.05)      # P4's output
    x = rnd(rng.standard_normal(D_MODEL) * 0.05)    # the residual

    # a chunk's broadcast: [this chunk of sw][the residual]
    bc = np.zeros((NCHUNK, 2 * K_DIM), np.float32)
    for ch in range(NCHUNK):
        bc[ch, :K_DIM] = sw[ch * K_DIM:(ch + 1) * K_DIM]
        bc[ch, K_DIM:] = x
    bc_t = iron.tensor(bc.reshape(-1).astype(bfloat16), dtype=bfloat16,
                       device="npu")

    rpp = D_MODEL // npairs
    rows = lambda pr, j: [pr * rpp + t * 2 * NROWS + j * NROWS
                          for t in range(tiles)]
    nbc = D_FF // BLK
    w5, ref = [], np.zeros(D_MODEL, np.float64)
    for pr in range(npairs):
        per = []
        for j in range(2):
            blob = []
            for ch in range(NCHUNK):
                lo = ch * (nbc // NCHUNK)
                hi = lo + nbc // NCHUNK
                for r0 in rows(pr, j):
                    sl = slice(r0, r0 + NROWS)
                    # the last chunk carries the flush flag
                    blob.append(q4nx.pack_tile(
                        dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi],
                        row_base=r0, flags=float(ch == NCHUNK - 1)))
            per.append(np.concatenate(blob))
        b = np.empty((NCHUNK * tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w5.append(iron.tensor(b.reshape(-1), dtype=np.uint8, device="npu"))
    # reference: the whole K, then the residual
    for pr in range(npairs):
        for j in range(2):
            for r0 in rows(pr, j):
                sl = slice(r0, r0 + NROWS)
                acc = np.zeros(NROWS, np.float64)
                for ch in range(NCHUNK):
                    lo = ch * (nbc // NCHUNK)
                    hi = lo + nbc // NCHUNK
                    acc += q4nx.gemv_reference_bf16(
                        rnd(sw[ch * K_DIM:(ch + 1) * K_DIM]),
                        dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi])
                ref[r0:r0 + NROWS] = rnd(acc + x[r0:r0 + NROWS])

    o_ts = [iron.zeros(NCHUNK * 2 * tiles * NROWS, dtype=bfloat16, device="npu")
            for _ in range(npairs)]
    if o.bench:
        b = run_iters(design, bc_t, *w5, *o_ts, warmup=2, iters=10)
        us = b.npu.min_us if b.npu else b.e2e.min_us
    else:
        design(bc_t, *w5, *o_ts)
        us = None

    # stream is [chunk][tile][core]; only the last chunk was written
    got = np.concatenate([t.numpy().astype(np.float64)
                          .reshape(NCHUNK, -1)[NCHUNK - 1] for t in o_ts])
    # the join interleaves the pair's two cores per object, so the stream is
    # [tile][core], not [core][tile]
    idx = np.concatenate([np.arange(rows(pr, j)[t], rows(pr, j)[t] + NROWS)
                          for pr in range(npairs)
                          for t in range(tiles)
                          for j in (0, 1)])
    err = np.abs(got - ref[idx]).max()
    if __import__("os").environ.get("P5_DIAG"):
        scale0 = np.abs(ref).mean()
        lo = (NCHUNK - 1) * (nbc // NCHUNK)
        hi = lo + nbc // NCHUNK
        last = np.zeros(D_MODEL, np.float64)
        for pr in range(npairs):
            for j2 in (0, 1):
                for r0 in rows(pr, j2):
                    sl = slice(r0, r0 + NROWS)
                    g = q4nx.gemv_reference_bf16(
                        rnd(sw[(NCHUNK - 1) * K_DIM:NCHUNK * K_DIM]),
                        dd[sl, lo:hi], dm[sl, lo:hi], dc[sl, lo:hi])
                    last[r0:r0 + NROWS] = rnd(g + x[r0:r0 + NROWS])
        print(f"    DIAG vs last-chunk-only {np.abs(got - last[idx]).max():.4e}")
        print(f"    DIAG vs residual alone  {np.abs(got - x[idx]).max():.4e}")
        print(f"    DIAG |got| {np.abs(got).mean():.5f} vs |ref| {scale0:.5f}")
    scale = np.abs(ref).mean()
    mb = npairs * 2 * NCHUNK * tiles * wt / 1e6
    print(f"P5 as its own dispatch: {ncores} cores, layer {o.layer}, "
          f"{NCHUNK} K-chunks of {tiles} tiles")
    if us:
        print(f"  {mb:.2f} MB  {mb * 1e3 / us:.1f} GB/s  {us:.1f} us "
              f"(marginal {us - FIXED_US:.1f})")
    tol = 1e-2 * scale
    print(f"  x_out: max err {err:.4e}  mean|ref| {scale:.5f}  tol {tol:.4e}")
    ok = err <= tol
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
