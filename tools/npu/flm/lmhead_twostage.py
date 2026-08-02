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

    python3 lmhead_twostage.py                        # device vs host coarse, one probe
    python3 lmhead_twostage.py --vs-exact --rounds 5  # timing, INTERLEAVED against exact
    python3 lmhead_twostage.py --gate                 # two-pass argmax on every probe

`--check` and `--bench` were documented here and never existed in the parser; the
default run is the one-probe check and `--vs-exact` is the timing path. Timing is
interleaved by construction because a sequential A-then-B on this machine measured
chunk=16 as a win TWICE and it was a loss both times.

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
# fused.py, 16 layers, one dispatch, run HELD. This TRACKS fused.py's TSEQ default
# and is not a free constant: at TSEQ=32 it was 12874.0, and 40 -- now the default,
# because it is what clears FLM's 36-token prompt -- costs ~5% in pure compute.
# Composing a fresh lm_head number with the OLD layers figure overstates every
# tok/s this file prints by ~1.1. Re-measure with `decode.py` if TSEQ moves again;
# it is not imported from fused.py because that import builds the design.
LAYERS_US = 13163.3          # TSEQ=40, decode.py median of 4 held dispatches
HOST_US = 10.5               # embedding row 0.38 + final RMSNorm 5.25 + argmax 4.89
# The two-pass decode keeps the first two and REPLACES the 4.89 us full argmax
# with shortlist + rescore + argmax-of-K, which is measured here.
HOST_BASE_US = 5.63
EXACT_LMHEAD_US = 3010.6     # the q4_1 lm_head dispatch, run HELD


# The oq4 probe rides this same harness rather than getting its own file, so the
# A/B is INTERLEAVED in one process by construction -- the rule this tree learned
# the hard way when a sequential A/B called chunk=16 a 40% win twice and it was a
# loss both times.
OQ4_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_gemv_oq4g256.cc")
OQ4_GROUP = 256              # Oq4G256 -- the G in the format name

OQ3_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_gemv_oq3g256.cc")

# (kernel symbol, source, scale-plane bytes, CODE bytes per row) per tier. Both
# planes vary: the coarse tier carries one f32 per ROW and 4 bits a weight; oq4
# one bf16 per 256-GROUP and 4 bits; oq3 the same scales but 3 bits, as
# bit-planes. Carrying the code width here is what keeps tile_bytes honest --
# assuming K/2 for every tier would have silently over-sized the oq3 tile by a
# third and turned a bandwidth win into a bandwidth loss on paper.
_SC_ROW = lambda K, N: N * 4
_SC_GRP = lambda K, N: N * (K // OQ4_GROUP) * 2
_CD_4BIT = lambda K, N: N * (K // 2)
_CD_3BIT = lambda K, N: N * (K // 8 * 3)
TIERS = {
    "coarse": ("flm_gemv_coarse_q4row", KERNEL_SRC, _SC_ROW, _CD_4BIT),
    "oq4":    ("flm_gemv_oq4g256", OQ4_SRC, _SC_GRP, _CD_4BIT),
    # Identical tile and identical loads to their real tier, trivial arithmetic.
    # The DMA control that separates layout from arithmetic; numerically wrong by
    # construction, so a timing probe only and never a correctness one.
    "oq4_1s": ("flm_gemv_oq4g256", OQ4_SRC, _SC_GRP, _CD_4BIT),
    "oq3":    ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
    "oq3_1s": ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
    "oq3_sumq": ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
    "oq3_acc":  ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
    "oq3_dotq": ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
    "oq3_sums": ("flm_gemv_oq3g256", OQ3_SRC, _SC_GRP, _CD_3BIT),
}
# Extra -D flags per tier. These do NOT enter the iron.jit AST hash, so the tier
# name must differ too -- it is in the fifo tag, which does.
TIER_FLAGS = {"oq4_1s": ["-DOQ4_ONE_SCALE=1"], "oq3_1s": ["-DOQ3_ONE_SCALE=1"],
              "oq3_sumq": ["-DOQ3_SUMQ=1"], "oq3_acc": ["-DOQ3_ACCUM_SCALE=1"],
              "oq3_dotq": ["-DOQ3_DOTQ=1"], "oq3_sums": ["-DOQ3_SUMS=1"]}


def tile_bytes(K, NROWS, tier="coarse"):
    """[scale plane, padded to 64][NROWS*K/2 nibbles].

    No 64-byte trailer. Every other tile in this tree carries one for
    `row_base`, which replaces per-core indexing; lm_head's output is NROWS
    floats that the fifo already places, so there is nothing for it to carry.

    The codes are K/2 bytes a row in BOTH tiers -- 4 bits flat -- so the tiers
    differ only in the scale plane, and at K=2048, NROWS=16 that is 64 B for
    coarse against 256 B for oq4. 16448 vs 16640 bytes, 1.2% apart, which is
    what makes this a controlled comparison of tile SHAPE rather than of size.
    """
    return ((TIERS[tier][2](K, NROWS) + 63) & ~63) + TIERS[tier][3](K, NROWS)


def build(K, NROWS, ncores, tiles_per_core, tier="coarse"):
    if ncores % 2:
        raise ValueError("--cores must be even (cores are wired in pairs)")
    kern_name, kern_src, _, _ = TIERS[tier]
    wtile = tile_bytes(K, NROWS, tier)
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
    # The tier MUST be in the tag: iron.jit hashes the AST and the two tiers
    # differ only in a string and a byte count, so without this they would
    # silently share one cached ELF and the "comparison" would be one design
    # measured twice.
    tag = f"{tier}{K}x{NROWS}"
    extra_flags = TIER_FLAGS.get(tier, [])
    src = f'''
def _design(act: In, {params}):
    kern = ExternalFunction(
        "{kern_name}", source_file=KERN_SRC,
        arg_types=[act_ty, wt_ty, o_ty],
        compile_flags=["-DDIM_K={K}", "-DDIM_NROWS={NROWS}"] + {extra_flags})

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
              ExternalFunction=ExternalFunction, KERN_SRC=kern_src,
              act_ty=act_ty, wt_ty=wt_ty, o_ty=o_ty,
              wpair_ty=wpair_ty, opair_ty=opair_ty,
              w_pair_all_ty=w_pair_all_ty, o_pair_all_ty=o_pair_all_ty,
              # exec()'d functions get __module__ = None, which mlir-aie's jit
              # cache hashes with .encode() and dies on.
              __name__="flm_lmhead_coarse")
    exec(src, ns)
    # full_elf: the vararg dispatch path caps at ~20 host buffers, fails as a
    # firmware hang rather than an error, and is ~34% slower where it works.
    return iron.jit(ns["_design"], source_files=[kern_src], full_elf=True), wtile


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


def pack_tiles_oq4(nib, gscale, NROWS, wtile):
    """The oq4 tier -> one (ntiles, wtile) uint8 array, in row order.

    Same shape of job as `pack_tiles`, and deliberately the same row order, so
    the two tiers stream identical amounts of identical structure and the only
    difference the device sees is the scale plane: NROWS*NG f16 here against
    NROWS f32 there.
    """
    rows, half = nib.shape
    assert rows % NROWS == 0, f"{rows} rows is not a multiple of NROWS={NROWS}"
    ntiles = rows // NROWS
    ng = gscale.shape[1]
    sb = wtile - NROWS * half
    out = np.zeros((ntiles, wtile), np.uint8)
    # Contiguous first, then viewed as bytes -- a column slice of a 2-D array is
    # not contiguous and .view(uint8) on it raises.
    # bfloat16, NOT np.float16 -- see build_oq4. The widths match either way,
    # which is why this failed silently rather than loudly.
    out[:, :NROWS * ng * 2] = (np.ascontiguousarray(gscale, np.float32)
                               .astype(bfloat16)
                               .reshape(ntiles, NROWS * ng).view(np.uint8))
    out[:, sb:] = nib.reshape(ntiles, NROWS * half)
    return out


def pack_tiles_oq3(planes, gscale, NROWS, wtile):
    """The oq3 tier -> one (ntiles, wtile) uint8 array, in row order.

    `build_oq3` already emits plane-major order, so this is a reshape and the
    layout is defined in exactly one place rather than two that can drift.
    """
    rows = planes.shape[0]
    assert rows % NROWS == 0, f"{rows} rows is not a multiple of NROWS={NROWS}"
    ntiles = rows // NROWS
    ng = gscale.shape[1]
    pb = planes.shape[1] * 4                      # plane bytes per row
    sb = wtile - NROWS * pb
    out = np.zeros((ntiles, wtile), np.uint8)
    out[:, :NROWS * ng * 2] = (np.ascontiguousarray(gscale, np.float32)
                               .astype(bfloat16)
                               .reshape(ntiles, NROWS * ng).view(np.uint8))
    out[:, sb:] = (np.ascontiguousarray(planes, np.uint32)
                   .view(np.uint8).reshape(ntiles, NROWS * pb))
    return out


def device_run(design, wtile, NROWS, ncores, tiles_per_core, tiles, act, iters):
    """Bind once, dispatch `iters` times, HOLD the run object.

    Rebinding buffers per call costs 1363 us on an lm_head-sized dispatch --
    larger than everything this design is trying to save. Timing through the
    iron.jit callable measures that overhead, not the dispatch.
    """
    from pyxrt_design import PyxrtDesign

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
    from pyxrt_design import PyxrtDesign

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


def bench_oq4(NROWS, ncores, tiles_per_core, act, iters, tier="oq4"):
    """The oq4 tier through the SAME driver as the coarse one.

    Returns what `device_run` returns, so the caller gets timing and the device
    logits from one path -- a bandwidth number from a kernel whose output was
    never checked is worth nothing, and this tree has shipped exactly that.
    """
    nib, gscale = lc.oq4_tier()
    design, wtile = build(K_DIM, NROWS, ncores, tiles_per_core, tier=tier)
    tiles = pack_tiles_oq4(nib, gscale, NROWS, wtile)
    return device_run(design, wtile, NROWS, ncores, tiles_per_core,
                      tiles, act, iters) + (wtile, nib, gscale)


def bench_oq3(NROWS, ncores, tiles_per_core, act, iters, tier="oq3"):
    """The oq3 tier through the same driver, same as bench_oq4."""
    planes, gscale = lc.oq3_tier()
    design, wtile = build(K_DIM, NROWS, ncores, tiles_per_core, tier=tier)
    tiles = pack_tiles_oq3(planes, gscale, NROWS, wtile)
    return device_run(design, wtile, NROWS, ncores, tiles_per_core,
                      tiles, act, iters) + (wtile, planes, gscale)


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
    p.add_argument("--oq3-tier", default="oq3",
                   help="oq3 | oq3_acc (accumulator scaling instead of weight fold)")
    p.add_argument("--oq3-dump", help="save device oq3 logits to .npy and exit")
    p.add_argument("--oq3-sums", action="store_true",
                   help="isolate the scale plane read")
    p.add_argument("--oq3-dotq", action="store_true",
                   help="isolate to_float+MAC: unscaled q.act vs host")
    p.add_argument("--oq3-debug", action="store_true",
                   help="isolate the oq3 spread: device code sums vs host")
    p.add_argument("--vs-oq3", action="store_true",
                   help="build the oq3 tier and A/B it against coarse, INTERLEAVED")
    p.add_argument("--vs-oq4", action="store_true",
                   help="build the oq4 tier and A/B it against coarse, INTERLEAVED")
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

    if o.oq3_dump:
        d_logits, _, _, _, _, _, d_pl, d_gs = bench_oq3(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier=o.oq3_tier)
        np.save(o.oq3_dump, np.asarray(d_logits, np.float64))
        np.save(o.oq3_dump + ".act", np.asarray(xn, np.float32))
        print(f"  device oq3 logits ({o.oq3_tier}) -> {o.oq3_dump}")
        return 0

    if o.oq3_sums:
        d_logits, _, _, _, _, _, _, d_gs = bench_oq3(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier="oq3_sums")
        w = (np.arange(d_gs.shape[1], dtype=np.float64) + 1)
        want = (d_gs.astype(np.float64) * w).sum(1)
        got = np.asarray(d_logits, np.float64)
        e = float(np.abs(np.subtract(got, want)).max())
        print(f"\n  SCALE READ: max abs {e:.4e}  rel {e/np.abs(want).max():.3e}")
        bad = np.flatnonzero(np.abs(np.subtract(got, want)) > 1e-4 * np.abs(want).max())
        print(f"    {len(want)-len(bad)}/{len(want)} rows match")
        for r in bad[:4]:
            print(f"    row {r:6d}: device {got[r]:12.6f}  host {want[r]:12.6f}")
        return 0

    if o.oq3_dotq:
        d_logits, _, _, _, _, _, d_pl, d_gs = bench_oq3(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier="oq3_dotq")
        _q = lc.oq3_unpack(d_pl, d_gs.shape[1]).astype(np.float64)
        _a = np.asarray(xn, np.float32).astype(np.float64)
        want = _q @ _a
        got = np.asarray(d_logits, np.float64)
        e = float(np.abs(np.subtract(got, want)).max())
        print(f"\n  UNSCALED DOT: max abs {e:.4e}  rel {e/np.abs(want).max():.3e}")
        print(f"    device argmax {int(np.argmax(got))}  host {int(np.argmax(want))}")
        return 0

    if o.oq3_debug:
        d_logits, _, _, _, _, _, d_pl, d_gs = bench_oq3(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier="oq3_sumq")
        # The device weights each code by its LANE INDEX (i+1 within each
        # 32-block), so the host must too -- a plain sum here compared against a
        # weighted sum there is not a check, it is a mismatch that looks like one.
        _q = lc.oq3_unpack(d_pl, d_gs.shape[1]).astype(np.int64)
        _w = (np.arange(32, dtype=np.int64) + 1)
        want = (_q.reshape(_q.shape[0], -1, 32) * _w).sum((1, 2))
        got = np.asarray(d_logits, np.int64)
        n_bad = int((got != want).sum())
        print(f"\n  SPREAD ISOLATION: {len(want)-n_bad}/{len(want)} rows match "
              f"the host code sum")
        for r in np.flatnonzero(got != want)[:5]:
            print(f"    row {r:6d}: device {got[r]:8d}  host {want[r]:8d}  "
                  f"diff {got[r]-want[r]:+d}")
        return 0 if n_bad == 0 else 1

    if o.vs_oq3:
        q_logits, _, q_bytes, _, q_redo, q_wtile, q_pl, q_gs = \
            bench_oq3(NROWS, ncores, tiles_per_core, xn, o.iters, tier=o.oq3_tier)
        ref = lc.oq3_logits(q_pl, q_gs, xn)
        dev_am, ref_am = int(np.argmax(q_logits)), int(np.argmax(ref))
        err = float(np.abs(np.subtract(q_logits, ref)).max())
        print(f"\n  oq3 device vs host: max abs {err:.4e}, "
              f"rel {err/np.abs(ref).max():.3e}")
        print(f"  oq3 argmax {dev_am}, host oq3 argmax {ref_am}")
        # VALUES, not just the argmax. An argmax-only gate passed a kernel with
        # 28% relative error at NROWS=16 and only failed at 24 -- the dominant
        # logit survived a systematically wrong spread. 1e-2 is loose enough for
        # bf16 scale folding (oq4 measures 3.7e-3) and tight enough that a wrong
        # bit order cannot hide behind it.
        rel = err / np.abs(ref).max()
        assert dev_am == ref_am, "device and host oq3 disagree on the argmax"
        assert rel < 1e-2, f"oq3 device vs host rel {rel:.3e} -- kernel is wrong"

        _, _, _, _, d_redo, _, _, _ = bench_oq3(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier="oq3_1s")
        ca, qa, da = [], [], []
        for _ in range(o.rounds):
            ca.append(redo(o.iters)[1:])
            qa.append(q_redo(o.iters)[1:])
            da.append(d_redo(o.iters)[1:])
        ct, qt, dt = np.concatenate(ca), np.concatenate(qa), np.concatenate(da)
        cg = total / (ct.min() * 1e-6) / 1e9
        qg = q_bytes / (qt.min() * 1e-6) / 1e9
        dg = q_bytes / (dt.min() * 1e-6) / 1e9
        print(f"\n  interleaved A/B, {o.rounds} rounds x {o.iters-1} warm dispatches:")
        print(f"    coarse  min {ct.min():7.1f}  median {np.median(ct):7.1f}  "
              f"{total/1e6:6.1f} MB  {cg:.1f} GB/s")
        print(f"    oq3     min {qt.min():7.1f}  median {np.median(qt):7.1f}  "
              f"{q_bytes/1e6:6.1f} MB  {qg:.1f} GB/s   {q_wtile} B/tile")
        print(f"    oq3-1s  min {dt.min():7.1f}  median {np.median(dt):7.1f}  "
              f"{q_bytes/1e6:6.1f} MB  {dg:.1f} GB/s   <- SAME TILE, trivial arithmetic")
        print(f"\n  VERDICT: oq3 reaches {qg:.1f} GB/s against coarse's {cg:.1f} "
              f"({100*qg/cg-100:+.1f}%); control {dg:.1f}.")
        print("  -> " + ("the tile SHAPE streams fine; the gap is the unpack"
                         if dg > 0.9 * cg else
                         "the tile SHAPE itself does not stream at the coarse rate"))
        # TIME is the thing, not GB/s: fewer bytes at the same GB/s IS the win.
        print(f"  TIME vs coarse: {qt.min()/ct.min():.4f}x for "
              f"{q_bytes/total:.4f}x the bytes")

    if o.vs_oq4:
        # THE BANDWIDTH PROBE. Same activation, same driver, same row order,
        # same 4-bits-a-weight codes; the tiers differ only in the scale plane
        # (NROWS f32 per tile against NROWS*NG f16). 131.9 MB against 133.4 --
        # 1.2% apart -- so a GB/s difference is the tile SHAPE and nothing else.
        oq_logits, _, oq_bytes, _, oq_redo, oq_wtile, oq_nib, oq_gs = \
            bench_oq4(NROWS, ncores, tiles_per_core, xn, o.iters)
        # CORRECTNESS FIRST. A bandwidth number from an unchecked kernel is
        # worth nothing. Take the argmaxes BEFORE any subtraction: numpy 2.1.3
        # on 3.14 elides `a - b` into `a` when the refcount says throwaway, and
        # this file has already printed a bogus argmax from exactly that.
        ref = lc.oq4_logits(oq_nib, oq_gs, xn)
        dev_am, ref_am = int(np.argmax(oq_logits)), int(np.argmax(ref))
        err = float(np.abs(np.subtract(oq_logits, ref)).max())
        print(f"\n  oq4 device vs host: max abs {err:.4e}, "
              f"rel {err/np.abs(ref).max():.3e}")
        print(f"  oq4 argmax {dev_am}, host oq4 argmax {ref_am}")
        assert dev_am == ref_am, "device and host oq4 disagree on the argmax"

        # THE DMA CONTROL: identical tile, identical loads, coarse arithmetic.
        # Three-way in ONE process so all of it sees the same contention.
        _, _, _, _, dma_redo, _, _, _ = bench_oq4(
            NROWS, ncores, tiles_per_core, xn, o.iters, tier="oq4_1s")
        ca, qa, da = [], [], []
        for _ in range(o.rounds):
            ca.append(redo(o.iters)[1:])
            qa.append(oq_redo(o.iters)[1:])
            da.append(dma_redo(o.iters)[1:])
        ct, qt, dt = np.concatenate(ca), np.concatenate(qa), np.concatenate(da)
        dg = oq_bytes / (dt.min() * 1e-6) / 1e9
        cg = total / (ct.min() * 1e-6) / 1e9
        qg = oq_bytes / (qt.min() * 1e-6) / 1e9
        print(f"\n  interleaved A/B, {o.rounds} rounds x {o.iters-1} warm dispatches:")
        print(f"    coarse  min {ct.min():7.1f}  median {np.median(ct):7.1f}  "
              f"{total/1e6:6.1f} MB  {cg:.1f} GB/s   {oq_wtile and total//(ncores*tiles_per_core)} B/tile")
        print(f"    oq4     min {qt.min():7.1f}  median {np.median(qt):7.1f}  "
              f"{oq_bytes/1e6:6.1f} MB  {qg:.1f} GB/s   {oq_wtile} B/tile")
        print(f"    bytes oq4/coarse {oq_bytes/total:.4f}, time {qt.min()/ct.min():.4f}, "
              f"GB/s {qg/cg:.4f}")
        print(f"    oq4-1s  min {dt.min():7.1f}  median {np.median(dt):7.1f}  "
              f"{oq_bytes/1e6:6.1f} MB  {dg:.1f} GB/s   <- SAME TILE, coarse arithmetic")
        print(f"\n  VERDICT: oq4 reaches {qg:.1f} GB/s against coarse's {cg:.1f} "
              f"({100*qg/cg-100:+.1f}%).")
        print(f"  The same tile with COARSE arithmetic reaches {dg:.1f} GB/s "
              f"({100*dg/cg-100:+.1f}% vs coarse).")
        print("  -> " + ("the tile SHAPE streams fine; the gap is arithmetic"
                         if dg > 0.9 * cg else
                         "the tile SHAPE itself does not stream at the coarse rate"))

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
        # The default chunk, deliberately. chunk=16 looked 40% faster in a tight
        # loop over pre-gathered arrays (174.8 vs 293.1) and 120 sequential
        # repeats agreed (217.7 vs 338.0) — but INTERLEAVED, alternating the two
        # configurations pair by pair, it reverses: 304.3 against 249.9. The
        # sequential result was an order artifact, whichever ran first keeping
        # the cache. The in-situ gate run had said the same thing (202.5 against
        # 181.6) and was dismissed as noise.
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
