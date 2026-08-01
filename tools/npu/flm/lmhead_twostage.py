#!/usr/bin/env python3
"""Two-pass lm_head: the coarse tier on the NPU, the fine rescore on the host.

    coarse GEMV over the WHOLE vocabulary   ->  device, 131.9 MB, one dispatch
    top-K of those logits                   ->  host, argpartition
    exact q4nx rescore of those K rows      ->  host, 41 KB at K=32
    argmax of the rescored set              ->  the token

lm_head is bandwidth bound — 163.7 MB at 54.7 GB/s, 97% of the 56.5 GB/s
fabric roof — so the ONLY lever is streaming fewer bytes. The coarse tier is
4 bits flat plus 4 bytes a row against q4_1's 5.00 bits, which is ~20% fewer
bytes and therefore ~20% less time. The kernel is not the lever and no amount
of arithmetic cleverness would have been.

The dataflow is `gemv_bench.py`'s, which is `layer.xclbin`'s: cores in PAIRS,
one shim stream per pair split in a memtile, the pair's two result streams
joined back before the shim. 16 private weight streams plus an activation is
17 shim inputs against 8 columns x 2 channels and the placer rejects it
outright; pairing halves both counts.

Row order is chosen so the host never has to permute: pair p owns a contiguous
block of the vocabulary, and within it tile t core j is global tile
`2*t + j`, which is what the memtile split and the output join already do. So
concatenating the eight pair outputs in order gives the 128256 logits in row
order with no gather.

    python3 lmhead_twostage.py --check          # device vs host coarse, one probe
    python3 lmhead_twostage.py --bench --iters 8
    python3 lmhead_twostage.py --gate           # two-pass argmax on every probe

Needs the NPU and the coarse tier from `lmhead_coarse.py --build`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lmhead_coarse as lc  # noqa: E402
import q4nx  # noqa: E402
from head_verify import VOCAB  # noqa: E402
from qkv_verify import K_DIM  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KERNEL_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_gemv_coarse_q4row.cc")
FLM_TOK_S = 61.18            # FLM's own wall-clock server figure
LAYERS_US = 12874.0          # fused.py, 16 layers, one dispatch, run HELD
HOST_US = 10.5               # embedding row 0.38 + final RMSNorm 5.25 + argmax 4.89
# The two-pass decode keeps the first two and REPLACES the 4.89 us full argmax
# with shortlist + rescore + argmax-of-K, which is measured here.
HOST_BASE_US = 5.63
EXACT_LMHEAD_US = 3010.6     # the q4_1 lm_head dispatch, run HELD


def tile_bytes(K, NROWS):
    """[NROWS f32 scales, padded to 64][NROWS*K/2 nibbles].

    No 64-byte trailer. Every other tile in this tree carries one for
    `row_base`, which replaces per-core indexing; lm_head's output is NROWS
    floats that the fifo already places, so there is nothing for it to carry.
    """
    return (((NROWS * 4) + 63) & ~63) + NROWS * (K // 2)


def build(K, NROWS, ncores, tiles_per_core):
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

    # Generated because the buffer count is the point -- the whole array binds
    # to ONE dispatch. Indexed, never sliced: a constant slice folds into
    # co_consts and mlir-aie's jit cache hashes the generator with
    # marshal.dumps(code, 4), which cannot serialize a slice.
    params = ", ".join(f"w{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o{i}: Out" for i in range(npairs))
    # iron.jit hashes the design's AST, and `compile_flags` are NOT in it -- two
    # variants with byte-identical source silently share one build. Stamp the
    # shape into a fifo NAME so the cache key moves when the flags do.
    tag = f"c{K}x{NROWS}"
    src = f'''
def _design(act: In, {params}):
    kern = ExternalFunction(
        "flm_gemv_coarse_q4row", source_file=KERNEL_SRC,
        arg_types=[act_ty, wt_ty, o_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"])

    # depth=1: the activation is acquired ONCE and held for the whole tile
    # loop, so a second buffer has nothing to overlap with and is dead L1.
    f_act = ObjectFifo(act_ty, depth=1, name="act_{tag}")
    act_cons = [f_act.cons() for _ in range({ncores})]

    f_wpair = [ObjectFifo(wpair_ty, name=f"wp{{i}}_{tag}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wtile}], obj_types=[wt_ty, wt_ty])
             for f in f_wpair]
    # Join offsets are in ELEMENTS, not bytes -- a byte offset overshoots the
    # fifo and emits a BD with a negative length.
    f_opair = [ObjectFifo(opair_ty, name=f"op{{i}}_{tag}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty])
             for f in f_opair]

    def core(a_cons, w_cons, o_prod, k):
        ea = a_cons.acquire(1)
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
                         o_sub[p][j].prod(), kern],
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
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, Worker=Worker,
              AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, KERNEL_SRC=KERNEL_SRC,
              act_ty=act_ty, wt_ty=wt_ty, o_ty=o_ty,
              wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_pair_all_ty=w_pair_all_ty, o_pair_all_ty=o_pair_all_ty,
              # exec()'d functions get __module__ = None, which mlir-aie's jit
              # cache hashes with .encode() and dies on.
              __name__="flm_lmhead_coarse")
    exec(src, ns)
    # full_elf: the vararg dispatch path caps at ~20 host buffers, fails as a
    # firmware hang rather than an error, and is ~34% slower where it works.
    return iron.jit(ns["_design"], source_files=[KERNEL_SRC], full_elf=True), wtile


def pack_tiles(nib, scale, NROWS, wtile):
    """The coarse tier -> one (ntiles, wtile) uint8 array, in row order."""
    rows, half = nib.shape
    assert rows % NROWS == 0, f"{rows} rows is not a multiple of NROWS={NROWS}"
    ntiles = rows // NROWS
    sb = wtile - NROWS * half
    out = np.zeros((ntiles, wtile), np.uint8)
    # A column slice of a 2-D array is not contiguous, so the f32 scales are
    # laid out contiguously first and copied in as bytes.
    out[:, :NROWS * 4] = (np.ascontiguousarray(scale, np.float32)
                          .reshape(ntiles, NROWS).view(np.uint8))
    out[:, sb:] = nib.reshape(ntiles, NROWS * half)
    return out


def device_run(design, wtile, NROWS, ncores, tiles_per_core, tiles, act, iters):
    """Bind once, dispatch `iters` times, HOLD the run object.

    Rebinding buffers per call costs 1363 us on an lm_head-sized dispatch --
    larger than everything this design is trying to save. Timing through the
    iron.jit callable measures that overhead, not the dispatch.
    """
    from fused_pyxrt import PyxrtDesign

    npairs = ncores // 2
    per_pair = 2 * tiles_per_core
    a_t = iron.tensor(np.asarray(act).astype(bfloat16), dtype=bfloat16, device="npu")
    w_ts = [iron.tensor(np.ascontiguousarray(tiles[p * per_pair:(p + 1) * per_pair]).ravel(),
                        dtype=np.uint8, device="npu") for p in range(npairs)]
    o_ts = [iron.zeros(2 * tiles_per_core * NROWS, dtype=np.float32, device="npu")
            for _ in range(npairs)]

    drv = PyxrtDesign(design, iters=iters)
    drv(a_t, *w_ts, *o_ts)
    logits = np.concatenate([o.numpy() for o in o_ts])

    def again(x):
        """Re-dispatch for a different activation, weights already resident.

        This REBINDS (a fresh pyxrt.run per call), which costs ~1.4 ms on a
        dispatch this size. That is deliberate: it is used only for the
        correctness gate, where the answer matters and the clock does not, and
        the held path above is what the timing number comes from.
        """
        drv.iters = 1
        drv(iron.tensor(np.asarray(x).astype(bfloat16), dtype=bfloat16, device="npu"),
            *w_ts, *o_ts)
        return np.concatenate([o.numpy() for o in o_ts])

    def redo(n):
        drv.iters = n
        drv(a_t, *w_ts, *o_ts)
        return np.array(drv.times_us)

    return logits, np.array(drv.times_us), ncores * tiles_per_core * wtile, again, redo


def bench_exact(NROWS, ncores, tiles_per_core, act, iters):
    """The EXACT q4_1 lm_head, same row count, same driver, same session.

    The 3010.6 us in the log was measured in another session on another day, and
    same-build run-to-run spread here is 2.6%. The claim being made is a RATIO
    between two formats, so both sides are measured back to back on the machine
    as it is right now, with only the format different. Row count is 128256 in
    both, not gemv_bench's 127488, so the comparison is per-vocabulary and not
    per-byte-with-a-fudge.

    The weight CONTENT is tiled real q4_1 blocks rather than the real lm_head;
    this measures delivery of a byte count, and gemv_verify already establishes
    that this kernel computes the right thing. `--check` on the coarse side is
    what guards against timing a kernel that computes nothing.
    """
    import gemv_bench
    from fused_pyxrt import PyxrtDesign

    design, wtile = gemv_bench.build(K_DIM, NROWS, ncores, tiles_per_core)
    nb = K_DIM // q4nx.BLK
    rows_per_core = tiles_per_core * NROWS
    c = lc.q4nx_container()
    d_all, m_all, codes_all = c.blocks("model.layers.0.mlp.down_proj.weight")
    need = rows_per_core * nb
    reps = -(-need // d_all.size)
    d = np.tile(d_all.ravel(), reps)[:need].reshape(rows_per_core, nb).astype(np.float32)
    m = np.tile(m_all.ravel(), reps)[:need].reshape(rows_per_core, nb).astype(np.float32)
    codes = np.tile(codes_all.reshape(-1, q4nx.BLK), (reps, 1))[:need] \
        .reshape(rows_per_core, nb, q4nx.BLK)
    wbuf = np.concatenate([q4nx.pack_tile(d[i:i + NROWS], m[i:i + NROWS], codes[i:i + NROWS])
                           for i in range(0, rows_per_core, NROWS)])
    npairs = ncores // 2
    wpair = np.empty(2 * wbuf.size, np.uint8)
    v = wpair.reshape(tiles_per_core, 2, wtile)
    v[:, 0, :] = wbuf.reshape(tiles_per_core, wtile)
    v[:, 1, :] = wbuf.reshape(tiles_per_core, wtile)

    a_t = iron.tensor(np.asarray(act).astype(bfloat16), dtype=bfloat16, device="npu")
    w_ts = [iron.tensor(wpair, dtype=np.uint8, device="npu") for _ in range(npairs)]
    o_ts = [iron.zeros(2 * rows_per_core, dtype=np.float32, device="npu")
            for _ in range(npairs)]
    drv = PyxrtDesign(design, iters=iters)

    def redo(n):
        drv.iters = n
        drv(a_t, *w_ts, *o_ts)
        return np.array(drv.times_us)

    return redo, ncores * tiles_per_core * wtile


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrows", type=int, default=16)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--probe", type=int, default=0, help="index into the saved probe set")
    p.add_argument("--gate", action="store_true", help="two-pass argmax on EVERY probe")
    p.add_argument("--vs-exact", action="store_true",
                   help="also time the exact q4_1 lm_head, same session, same driver")
    p.add_argument("--rounds", type=int, default=5,
                   help="interleaved A/B rounds when --vs-exact")
    o = p.parse_args()

    NROWS, ncores = o.nrows, o.cores
    assert VOCAB % (NROWS * ncores) == 0, (
        f"vocab {VOCAB} does not divide into {ncores} cores x {NROWS} rows")
    tiles_per_core = VOCAB // (NROWS * ncores)

    nib, scale = lc.coarse_tier()
    probes = lc.load_probes()
    xn = probes[o.probe][1]

    design, wtile = build(K_DIM, NROWS, ncores, tiles_per_core)
    tiles = pack_tiles(nib, scale, NROWS, wtile)
    total = tiles.nbytes
    print(f"coarse lm_head: {VOCAB} rows, {NROWS}/tile, {tiles.shape[0]} tiles, "
          f"{ncores} cores x {tiles_per_core}, {wtile} B/tile, {total/1e6:.1f} MB")

    t0 = time.time()
    logits, times, streamed, again, redo = device_run(design, wtile, NROWS, ncores,
                                                      tiles_per_core, tiles, xn, o.iters)
    print(f"  dispatch set took {time.time()-t0:.1f} s wall")
    assert streamed == total, (streamed, total)

    # --- the kernel is right, or the timing is meaningless -------------------
    ref = lc.coarse_logits(nib, scale, xn)
    # ORDER IS LOAD-BEARING. `logits` is a local that owns its data, so on
    # numpy 2.1.3 + python 3.14 the expression `logits - ref` ELIDES INTO IT
    # (LOAD_FAST_BORROW keeps the refcount at 1, numpy reads that as a
    # throwaway temp) and every later read of `logits` sees the residual. This
    # printed "device coarse argmax 45" against a host argmax of 16309 once,
    # with the two logit arrays agreeing to 1.4e-6. Take the argmaxes first.
    dev_argmax, ref_argmax = int(np.argmax(logits)), int(np.argmax(ref))
    err = float(np.abs(np.subtract(logits, ref)).max())
    print(f"  device vs host coarse: max abs {err:.4e}, "
          f"rel {err/np.abs(ref).max():.3e} (peak |logit| {np.abs(ref).max():.4f})")
    print(f"  device coarse argmax {dev_argmax}, host coarse argmax {ref_argmax}")
    assert dev_argmax == ref_argmax, "device and host coarse disagree on the argmax"

    # --- timing --------------------------------------------------------------
    w = times[1:] if len(times) > 1 else times
    us = float(np.median(w))
    gbs = total / (us * 1e-6) / 1e9
    print(f"  pyxrt HELD: first {times[0]:.1f} us | warm n={len(w)} "
          f"min {w.min():.1f} median {us:.1f} max {w.max():.1f} "
          f"spread {(w.max()/w.min()-1)*100:.1f}%")
    print(f"  {gbs:.1f} GB/s")

    exact_us = EXACT_LMHEAD_US
    if o.vs_exact:
        exact_redo, ebytes = bench_exact(NROWS, ncores, tiles_per_core, xn, o.iters)
        # INTERLEAVED. A user-owned `flm serve` contends for this NPU, and a
        # single back-to-back A then B measured 161% spread on one side and 2%
        # on the other -- whichever side the contention lands on loses, and the
        # ratio is then an artefact of scheduling. Alternating rounds with both
        # designs' buffers resident puts the same drift through both.
        ca, ea = [], []
        for _ in range(o.rounds):
            ca.append(redo(o.iters)[1:])
            ea.append(exact_redo(o.iters)[1:])
        ct, et = np.concatenate(ca), np.concatenate(ea)
        # The MIN is the statistic here, not the median. Under an external
        # contender every sample is the dispatch plus a non-negative delay, so
        # the minimum is the least contaminated estimate of the dispatch itself
        # -- and it reproduces across sessions (coarse 2378.0 / 2403.6 / 2386.2)
        # where the median does not. Both are reported.
        us, exact_us = float(ct.min()), float(et.min())
        print(f"\n  interleaved A/B, {o.rounds} rounds x {o.iters-1} warm dispatches:")
        print(f"    coarse  min {ct.min():7.1f}  median {np.median(ct):7.1f}  "
              f"{total/1e6:6.1f} MB  {total/(ct.min()*1e-6)/1e9:.1f} GB/s")
        print(f"    exact   min {et.min():7.1f}  median {np.median(et):7.1f}  "
              f"{ebytes/1e6:6.1f} MB  {ebytes/(et.min()*1e-6)/1e9:.1f} GB/s")
        print(f"    coarse/exact: bytes {total/ebytes:.4f}, time (min) {us/exact_us:.4f} "
              f"-> {100*(1-us/exact_us):.1f}% faster, saving {exact_us-us:.1f} us")

    tok_now = LAYERS_US + exact_us + HOST_US
    tok_new = LAYERS_US + us + HOST_US
    print(f"\n  token now  {tok_now:.1f} us -> {1e6/tok_now:.1f} tok/s "
          f"({100*(1e6/tok_now)/FLM_TOK_S - 100:+.1f}% vs FLM {FLM_TOK_S})")
    print(f"  token two-pass {tok_new:.1f} us -> {1e6/tok_new:.1f} tok/s "
          f"({100*(1e6/tok_new)/FLM_TOK_S - 100:+.1f}% vs FLM) "
          f"[+ the fine pass, measured below]")

    # --- the two-pass answer, against the exact one --------------------------
    D, M, C = lc.lmhead_blocks()
    probe_list = probes if o.gate else [probes[o.probe]]
    fine_us = []
    bad = []
    ranks = []
    for i, (tok, x, amax) in enumerate(probe_list):
        # Every probe goes through the DEVICE. The host coarse model agrees to
        # 1.7e-7 relative, but "the shortlist the device actually produced" is
        # the thing under test and modelling it would be a weaker claim.
        cl = logits if (not o.gate and i == 0) else again(x)
        order = np.argsort(cl)[::-1][:64]
        ranks.append(int(np.where(order == amax)[0][0]) + 1 if amax in order else 999)
        # The whole host half of the two-pass decode: shortlist, exact rescore,
        # argmax of the rescored set. `gemv_bf16_fast` is used and not
        # `q4nx.gemv_reference_bf16` -- the two are BIT-IDENTICAL (measured,
        # max abs diff 0.0 over 256 real rows) but the latter is a python loop
        # over (row, block) and takes 12.5 ms for 32 rows. Timing that would
        # report the reference's shape as the algorithm's cost.
        t0 = time.perf_counter_ns()
        # `np.partition` + `flatnonzero`, not `np.argpartition`: argpartition
        # allocates and permutes a 128256-entry int64 index array (1 MB) to
        # deliver 32 numbers. Partitioning the VALUES and then finding the ones
        # at or above the K-th is the same shortlist -- asserted equal below --
        # for 78 us against 130. Both are numpy overhead rather than work; the
        # same numpy does a 513 KB argmax in 4.9 us.
        # A THRESHOLD SET IS THE TOP-N FOR SOME N, so `cl > m - d` with at least
        # K survivors provably contains the true top-K — the guarantee the recall
        # curve was measured against is preserved exactly, not approximated. One
        # max (7.5 us) plus one threshold pass beats partitioning 128256 values:
        # 19.3 us against 76-110, for the identical set.
        #
        # d = 2.0 lands ~39 candidates on real coarse logits (the 32nd is 1.87
        # below the max on both probes measured); the widening ladder and the
        # argpartition fallback cover a hidden state whose logits are flatter.
        # The trim is an argpartition over ~39 elements, not 128256, so it costs
        # nothing and keeps the rescore at exactly K rows — which matters because
        # the rescore has a cliff above K=16 (90.6 us at 16, 302.6 at 32).
        m = cl.max()
        for _d in (2.0, 4.0, 8.0, 16.0):
            idx = np.flatnonzero(cl > m - _d)
            if idx.size >= o.topk:
                if idx.size > o.topk:
                    idx = idx[np.argpartition(cl[idx], -o.topk)[-o.topk:]]
                break
        else:
            idx = np.argpartition(cl, -o.topk)[-o.topk:]
        fine = lc.gemv_bf16_fast(x, D[idx], M[idx], C[idx])
        got = int(idx[int(np.argmax(fine))])
        fine_us.append((time.perf_counter_ns() - t0) / 1e3)
        assert set(idx.tolist()) == set(np.argpartition(cl, -o.topk)[-o.topk:].tolist()), \
            "the threshold shortlist is not the top-K"
        if got != amax:
            bad.append((tok, amax, got))
    print(f"\n  two-pass argmax at K={o.topk}: "
          f"{len(probe_list)-len(bad)}/{len(probe_list)} match the exact argmax")
    for tok, amax, got in bad[:8]:
        print(f"    MISS token {tok}: exact {amax}, two-pass {got}")
    print(f"  device coarse rank of the true argmax: max {max(ranks)}, "
          f"median {int(np.median(ranks))}, outside top-64: {sum(r > 64 for r in ranks)}")
    print(f"  host fine pass: median {np.median(fine_us):.1f} us "
          f"({o.topk} rows x {K_DIM} q4nx weights = {o.topk*1280/1e3:.1f} KB)")

    host_two = HOST_BASE_US + float(np.median(fine_us))
    tok_full = LAYERS_US + us + host_two
    print(f"\n  token, every term measured on the wall clock:")
    print(f"    layers   {LAYERS_US:8.1f} us   fused.py, run HELD")
    print(f"    lm_head  {us:8.1f} us   coarse, run HELD, this session")
    print(f"    host     {host_two:8.1f} us   {HOST_BASE_US} base + "
          f"{np.median(fine_us):.1f} shortlist/rescore")
    print(f"    token    {tok_full:8.1f} us   -> {1e6/tok_full:.1f} tok/s, "
          f"{100*(1e6/tok_full)/FLM_TOK_S - 100:+.1f}% vs FLM {FLM_TOK_S}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
