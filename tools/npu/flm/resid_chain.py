#!/usr/bin/env python3
"""Does the residual survive a phase boundary in core memory? (layer prerequisite)

In the fused layer, phase P3 (`o_proj` + residual) produces `h`, and `h` is
needed in **five** places: P4's activation half and the aux half of each of P5's
four down-chunk objects. A drain consumes its data once, so no single phase can
write it to five destinations.

It does not have to travel. P5's flush needs only the residual for the rows *it*
outputs, and P3 and P5 have the same shape — N=2048 over 16 cores, 8 tiles each
— so with the same row assignment, the core that needs a residual is the core
that just computed it. `flm_gemv_residual` stashes its rows in 512 B of core
memory and `flm_gemv_flush` reads them under `-DRESID_FROM_STASH=1`, removing
the copy and 16 KB per layer of broadcast traffic.

**That is a value crossing a phase boundary in registers rather than memory, so
it needs its own test.** This runs exactly those two phases back to back in one
dispatch, skipping P4 (its SwiGLU output is supplied by the host, since what is
under test is the residual path, not the FFN):

    P3  o_proj + residual, K=2048 N=2048   -> h, stashed in-core
    P5  down,  4 x K=2048  -> x_out = W_down.swiglu + h   [h from the stash]

If the stash is wrong, `x_out` is wrong by exactly `h`, which is the largest
term in it — so this fails loudly rather than subtly. `--aux-residual` runs the
same design with `-DRESID_FROM_STASH=0` as a control: it must also pass, and
agree.

    python3 resid_chain.py
    python3 resid_chain.py --aux-residual

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from ffn_verify import load_linear  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      TaskGroup, Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
RES_SRC = str(KDIR / "flm_gemv_residual.cc")
ACC_SRC = str(KDIR / "flm_gemv_acc.cc")
FLUSH_SRC = str(KDIR / "flm_gemv_flush.cc")
ASUM_SRC = str(KDIR / "flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
K_DIM, NROWS, BLK = 2048, 16, 32
D_MODEL, D_FF, NCHUNK = 2048, 8192, 4

rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


def build(ncores, from_stash):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = ncores // 2
    tiles = D_MODEL // (ncores * NROWS)          # 8 per core, both phases
    accn = 2 * tiles * NROWS                     # a PAIR's row span
    resn = 2 * tiles * NROWS

    bc_ty = np.ndarray[(2 * K_DIM,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[np.float32]]
    ob_ty = np.ndarray[(NROWS,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    o3pair_ty = np.ndarray[(2 * NROWS,), np.dtype[np.float32]]
    o5pair_ty = np.ndarray[(2 * NROWS,), np.dtype[bfloat16]]
    w3_ty = np.ndarray[(2 * tiles * wt,), np.dtype[np.uint8]]
    w5_ty = np.ndarray[(2 * NCHUNK * tiles * wt,), np.dtype[np.uint8]]
    o3_ty = np.ndarray[(2 * tiles * NROWS,), np.dtype[bfloat16]]
    o5_ty = np.ndarray[(2 * tiles * NROWS,), np.dtype[bfloat16]]
    bc_all_ty = np.ndarray[((1 + NCHUNK) * 2 * K_DIM,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_ACCN={accn}",
             f"-DDIM_RESN={resn}", f"-DRESID_FROM_STASH={1 if from_stash else 0}"]
    params = ", ".join(f"w3_{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"w5_{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o3_{i}: Out" for i in range(npairs))
    params += ", " + ", ".join(f"o5_{i}: Out" for i in range(npairs))
    src = f'''
def _design(bc: In, {params}):
    kres = ExternalFunction("flm_gemv_q4_1_residual", source_file=RES_SRC,
                            arg_types=[bc_ty, wt_ty, ob_ty], compile_flags=FLAGS)
    kacc = ExternalFunction("flm_gemv_acc", source_file=ACC_SRC,
                            arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    kfl = ExternalFunction("flm_gemv_flush", source_file=FLUSH_SRC,
                           arg_types=[bc_ty, wt_ty, ob_ty], compile_flags=FLAGS)
    kas = ExternalFunction("flm_asum_prepare", source_file=ASUM_SRC,
                           arg_types=[bc_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name="bc")
    bc_cons = [f_bc.cons() for _ in range({ncores})]
    f_w = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_w]
    # P3 emits f32, P5 emits bf16, so they cannot share one result fifo. In the
    # real layer every phase emits bf16 and they do; here P3's f32 output is
    # kept so the reference can check `h` itself, which is the point.
    # ONE result fifo, reused by both phases — every phase emits bf16, which is
    # what keeps the shim inside 8 outputs and the router able to place it.
    f_o = [ObjectFifo(o5pair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[ob_ty, ob_ty]) for f in f_o]

    def core(bcc, wc, op, kr, ka, kf, kprep):
        # ---- P3: o_proj + residual, stashing h in core memory ----
        eb = bcc.acquire(1)
        kprep(eb)
        for _ in range_({tiles}):
            ew = wc.acquire(1)
            eo = op.acquire(1)
            kr(eb, ew, eo)
            op.release(1)
            wc.release(1)
        bcc.release(1)
        # ---- P5: down, 4 chunks; the last flushes with the residual ----
        for _ in range_({NCHUNK - 1}):
            eb = bcc.acquire(1)
            kprep(eb)
            for _ in range_({tiles}):
                ew = wc.acquire(1)
                ka(eb, ew)
                wc.release(1)
            bcc.release(1)
        eb = bcc.acquire(1)
        kprep(eb)
        for _ in range_({tiles}):
            ew = wc.acquire(1)
            eo = op.acquire(1)
            kf(eb, ew, eo)
            op.release(1)
            wc.release(1)
        bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[bc_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kres, kacc, kfl, kas],
                stack_size=4096))

    def sequence(*args):
        n = {npairs}
        bcb = args[0]
        w3b = [args[1 + i] for i in range(n)]
        w5b = [args[1 + n + i] for i in range(n)]
        o3b = [args[1 + 2 * n + i] for i in range(n)]
        o5b = [args[1 + 3 * n + i] for i in range(n)]
        bch = args[1 + 4 * n]
        wh = [args[2 + 4 * n + i] for i in range(n)]
        oh = [args[2 + 5 * n + i] for i in range(n)]
        tg = TaskGroup()
        bch.fill(bcb, group=tg, offset=0,
                 sizes=[1, 1, 1, {2 * K_DIM}], strides=[0, 0, 0, 1])
        for i in range(n):
            wh[i].fill(w3b[i], group=tg)
        for i in range(n):
            oh[i].drain(o3b[i], wait=True, group=tg)
        tg.finish()
        tg = TaskGroup()
        for ch in range({NCHUNK}):
            bch.fill(bcb, group=tg, offset=(1 + ch) * {2 * K_DIM},
                     sizes=[1, 1, 1, {2 * K_DIM}], strides=[0, 0, 0, 1])
        for i in range(n):
            wh[i].fill(w5b[i], group=tg)
        for i in range(n):
            oh[i].drain(o5b[i], wait=True, group=tg)
        tg.finish()

    arg_types = [bc_all_ty] + [w3_ty] * {npairs} + [w5_ty] * {npairs}
    arg_types += [o3_ty] * {npairs} + [o5_ty] * {npairs}
    arg_types += [f_bc.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_w]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_o]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              TaskGroup=TaskGroup, Worker=Worker, AnyShimTile=AnyShimTile,
              range_=range_, ExternalFunction=ExternalFunction, RES_SRC=RES_SRC,
              ACC_SRC=ACC_SRC, FLUSH_SRC=FLUSH_SRC, ASUM_SRC=ASUM_SRC,
              FLAGS=flags, bc_ty=bc_ty, wt_ty=wt_ty, o_ty=o_ty, ob_ty=ob_ty,
              wpair_ty=wpair_ty, o3pair_ty=o3pair_ty, o5pair_ty=o5pair_ty,
              w3_ty=w3_ty, w5_ty=w5_ty, o3_ty=o3_ty, o5_ty=o5_ty,
              bc_all_ty=bc_all_ty, __name__="flm_resid_chain")
    exec(src, ns)
    return iron.jit(ns["_design"],
                    source_files=[RES_SRC, ACC_SRC, FLUSH_SRC, ASUM_SRC],
                    full_elf=True), wt, tiles


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--aux-residual", action="store_true",
                   help="control: -DRESID_FROM_STASH=0, residual via the "
                        "broadcast aux half. Must pass and agree.")
    p.add_argument("--keep-cache", action="store_true",
                   help="skip the cache clear below. Only safe if the previous "
                        "run used the same residual path.")
    o = p.parse_args()
    ncores, npairs = o.cores, o.cores // 2
    from_stash = not o.aux_residual

    # **iron.jit does not hash `compile_flags`.** The two residual paths differ
    # ONLY by -DRESID_FROM_STASH, with byte-identical sources, so the second run
    # silently reuses the first run's kernel objects and reports the first
    # path's behaviour under the second path's name. That produced a confident
    # wrong diagnosis once already (docs/npu/flm-refe-log.md, 2026-07-31) — the
    # stash looked like it read zeros when in fact the flush had been compiled
    # for the aux path. Clearing is cheap next to being wrong.
    cache = Path.home() / ".npu" / "cache"
    if not o.keep_cache and cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)

    c = q4nx.Q4nx(str(Q4NX))
    pre = f"model.layers.{o.layer}."
    od, om, oc = load_linear(c, pre + "self_attn.o_proj.weight", D_MODEL, K_DIM)
    dd_, dm_, dc_ = c.blocks(pre + "mlp.down_proj.weight")
    nb8 = D_FF // BLK
    dd, dm, dc = (dd_[:D_MODEL, :nb8].astype(np.float32),
                  dm_[:D_MODEL, :nb8].astype(np.float32), dc_[:D_MODEL, :nb8])

    design, wt, tiles = build(ncores, from_stash)
    rpp = D_MODEL // npairs

    rng = np.random.default_rng(0)
    attn_out = rnd(rng.standard_normal(K_DIM) * 0.05)   # P3's activation
    x = rnd(rng.standard_normal(D_MODEL) * 0.05)        # the residual P3 adds
    sw = rnd(rng.standard_normal(D_FF) * 0.05)          # P4's output, host-supplied

    # rows so that a pair's join is a contiguous global run
    rows = lambda pr, j: [pr * rpp + t * 2 * NROWS + j * NROWS for t in range(tiles)]
    w3, w5 = [], []
    nbc = K_DIM // BLK
    for pr in range(npairs):
        per = [np.concatenate([q4nx.pack_tile(od[r:r+NROWS], om[r:r+NROWS],
                                              oc[r:r+NROWS], row_base=r)
                               for r in rows(pr, j)]) for j in range(2)]
        b = np.empty((tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w3.append(b.reshape(-1))
        per = [np.concatenate([
            q4nx.pack_tile(dd[r:r+NROWS, ch*nbc:(ch+1)*nbc],
                           dm[r:r+NROWS, ch*nbc:(ch+1)*nbc],
                           dc[r:r+NROWS, ch*nbc:(ch+1)*nbc], row_base=r)
            for ch in range(NCHUNK) for r in rows(pr, j)]) for j in range(2)]
        b = np.empty((NCHUNK * tiles, 2, wt), np.uint8)
        b[:, 0, :], b[:, 1, :] = per[0].reshape(-1, wt), per[1].reshape(-1, wt)
        w5.append(b.reshape(-1))

    bc = np.zeros((1 + NCHUNK, 2 * K_DIM), np.float32)
    bc[0, :K_DIM] = attn_out
    bc[0, K_DIM:K_DIM + D_MODEL] = x                    # P3's residual
    for ch in range(NCHUNK):
        bc[1 + ch, :K_DIM] = sw[ch * K_DIM:(ch + 1) * K_DIM]
        # Poison the aux half in stash mode: if the flush is really reading
        # the stash, this is never touched. If it reads aux, the output
        # moves by exactly this, which names the bug instead of hiding it.
        bc[1 + ch, K_DIM:K_DIM + D_MODEL] = (
            -7.0 if from_stash else 0.0)
    bc_t = iron.tensor(bc.reshape(-1).astype(bfloat16), dtype=bfloat16, device="npu")

    # The row assignment makes each pair's drain a contiguous global run
    # (pair pr covers rows [pr*rpp, (pr+1)*rpp) in order), so concatenating the
    # pair buffers gives natural row order and the reference needs no
    # permutation at all.
    # h at full precision, and its bf16 image. P3 EMITS bf16, so the emitted
    # value must be compared against the rounded one — but the in-core stash
    # keeps the float, so P5's residual add is more accurate than the aux route,
    # which round-trips h through the broadcast in bf16. Which reference is
    # right therefore depends on the path under test.
    h_exact = np.concatenate([
        q4nx.gemv_reference_bf16(attn_out, od[r:r+NROWS], om[r:r+NROWS],
                                 oc[r:r+NROWS])
        for r in range(0, D_MODEL, NROWS)]) + x.astype(np.float64)
    h_global = rnd(h_exact).astype(np.float64)
    if not from_stash:
        # The control has to route the residual through the aux half, which
        # means the HOST must already know h — exactly the copy the stash
        # removes, and why the control is only a check and not a design.
        bcx = bc.copy()
        for ch in range(NCHUNK):
            bcx[1 + ch, K_DIM:K_DIM + D_MODEL] = h_global
        bc_t = iron.tensor(bcx.reshape(-1).astype(bfloat16), dtype=bfloat16,
                           device="npu")

    w3_ts = [iron.tensor(v, dtype=np.uint8, device="npu") for v in w3]
    w5_ts = [iron.tensor(v, dtype=np.uint8, device="npu") for v in w5]
    o3_ts = [iron.zeros(2 * tiles * NROWS, dtype=bfloat16, device="npu")
             for _ in range(npairs)]
    o5_ts = [iron.zeros(2 * tiles * NROWS, dtype=bfloat16, device="npu")
             for _ in range(npairs)]
    design(bc_t, *w3_ts, *w5_ts, *o3_ts, *o5_ts)

    got_h = np.concatenate([t.numpy().astype(np.float64) for t in o3_ts])
    got_x = np.concatenate([t.numpy().astype(np.float64) for t in o5_ts])
    ref_h = h_global
    down = np.concatenate([
        q4nx.gemv_reference_bf16(sw, dd[r:r+NROWS], dm[r:r+NROWS], dc[r:r+NROWS])
        for r in range(0, D_MODEL, NROWS)])
    # x_out is emitted as bf16 too — round the reference to match.
    # the stash carries float h; the aux route carries bf16 h
    ref_x = rnd(down + (h_exact if from_stash else ref_h)).astype(np.float64)

    if __import__("os").environ.get("RC_DIAG"):
        import sys as _s
        print(f"  DIAG sorted-multiset |diff| max "
              f"{np.abs(np.sort(got_h)-np.sort(ref_h)).max():.4e}", file=_s.stderr)
        print("  DIAG  idx    got_h     ref_h   gemv-only        x", file=_s.stderr)
        gonly = ref_h - x.astype(np.float64)
        for i in (0, 1, 15, 16, 17, 31, 32, 256):
            print(f"  DIAG {i:5d} {got_h[i]:9.5f} {ref_h[i]:9.5f} "
                  f"{gonly[i]:11.5f} {x[i]:9.5f}", file=_s.stderr)
        for lab, cand in (("down + h  (correct)", down + ref_h),
                          ("down + 0  (stash empty)", down),
                          ("down + h/2", down + ref_h / 2),
                          ("down + h shifted by 16", down + np.roll(ref_h, 16))):
            e = np.abs(got_x - cand)
            print(f"  DIAG x_out vs {lab:26s} max {e.max():.3e} "
                  f"exact {np.mean(e < 1e-3):.1%}", file=_s.stderr)
    mode = "in-core stash" if from_stash else "broadcast aux (control)"
    print(f"P3 -> P5 residual across a phase boundary: {ncores} cores, "
          f"layer {o.layer}")
    print(f"  residual path: {mode}")
    e_h, e_x = np.abs(got_h - ref_h), np.abs(got_x - ref_x)
    print(f"  P3 h    : max err {e_h.max():.4e}  mean|ref| {np.abs(ref_h).mean():.5f}")
    print(f"  P5 x_out: max err {e_x.max():.4e}  mean|ref| {np.abs(ref_x).mean():.5f}")
    print(f"  (a lost residual would move x_out by ~{np.abs(ref_h).mean():.5f}, "
          f"{np.abs(ref_h).mean()/np.abs(ref_x).mean():.0%} of it)")
    ok = e_h.max() <= 1e-2 * np.abs(ref_h).mean() and \
        e_x.max() <= 1e-2 * np.abs(ref_x).mean()
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
