#!/usr/bin/env python3
"""The FFN half of a decoder layer as TWO chained phases in ONE dispatch.

**STATUS: the plumbing works, the numerics do NOT. This does not pass yet.**
It builds, places and runs end to end — two phases, one dispatch, one operand
fifo and one result fifo shared between them, P4's results drained into the
buffer P5's fills read. What is wrong is the values: relative error is p50 10.3%
and max 51% on the largest SwiGLU outputs, against an exp2 NLF floor of 5.86%,
so it is a bug and not the hardware transcendental. Two candidates not yet
separated: the in-place RMSNorm prologue (this is the first harness to use
`flm_norm_prepare` with a *shared* broadcast fifo across phases), and the
gate/up stash pairing under the shared operand fifo. Run with
`FFN_CHAIN_DIAG=1` for the relative-error breakdown. Do not build on this until
it passes.

Phases P4 and P5 of `docs/npu/flm-fused-layer-plan.md` §1.4, and the first time
real kernels are chained through the mechanism `chain_probe.py` validated with a
doubling stub:

    P4  norm2 + gate + up + SwiGLU   16 cores, 32 gate/up pairs each -> 8192
    P5  down, 4 chunks of K=2048     16 cores, 32 tiles each   -> 2048 + residual

P5's activation *is* P4's output. There is no host between them: P4's `drain`
lands in a DDR buffer and P5's `fill` reads it back, as buffer descriptors in
one command stream with `drain(wait=True)` as the barrier. That is 2 of the
layer's 5 phases and **29.2 of its 38.0 MB**, so it is most of the layer's
traffic and all of its inter-phase plumbing except the KV transform.

Everything is real: layer-0 weights from `model.q4nx`, the norm weight in the
broadcast's aux half, the residual arriving in P5's aux half and added by
`flm_gemv_flush` from the tile's `row_base`.

    python3 ffn_chain.py                 # verify, 16 cores
    python3 ffn_chain.py --bench

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
                      TaskGroup, Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
GATE_SRC = str(KDIR / "flm_gemv_gate.cc")
UPS_SRC = str(KDIR / "flm_gemv_up_swiglu.cc")
ACC_SRC = str(KDIR / "flm_gemv_acc.cc")
FLUSH_SRC = str(KDIR / "flm_gemv_flush.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
ASUM_SRC = str(KDIR / "flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
K_DIM, NROWS, BLK = 2048, 16, 32
D_MODEL, D_FF, NCHUNK = 2048, 8192, 4
EPS = 1e-5
FIXED_US = 92.9

rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


def build(ncores, host_norm=False, nrep=1):
    wt = q4nx.tile_bytes(K_DIM, NROWS)
    npairs = ncores // 2
    p4_pairs = D_FF // (ncores * NROWS)        # gate/up pairs per core  (32)
    p5_tiles = D_MODEL // (ncores * NROWS)     # tiles per core per chunk (8)
    accn = 2 * p5_tiles * NROWS
    # Where pair p's SwiGLU rows land in the broadcast BO. Each P5 chunk object
    # is [act K_DIM][aux K_DIM], so consecutive chunks' activation halves are
    # K_DIM apart with 2*K_DIM of stride. A pair owning more than one chunk's
    # worth of rows therefore needs a 2-D drain; at 16 cores it owns 1024 and
    # the pattern degenerates to a plain offset.
    rows_per_pair = D_FF // npairs
    nblk = max(1, rows_per_pair // K_DIM)
    blk = min(rows_per_pair, K_DIM)

    bc_ty = np.ndarray[(2 * K_DIM,), np.dtype[bfloat16]]     # [act][aux]
    wt_ty = np.ndarray[(wt,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[bfloat16]]
    wpair_ty = np.ndarray[(2 * wt,), np.dtype[np.uint8]]
    opair_ty = np.ndarray[(2 * NROWS,), np.dtype[bfloat16]]
    # P4 streams 2 tiles per pair-step, P5 streams NCHUNK x p5_tiles
    nw4 = 2 * p4_pairs
    nw5 = NCHUNK * p5_tiles
    w4_ty = np.ndarray[(2 * nw4 * wt,), np.dtype[np.uint8]]
    w5_ty = np.ndarray[(2 * nw5 * wt,), np.dtype[np.uint8]]
    o5_ty = np.ndarray[(2 * p5_tiles * NROWS,), np.dtype[bfloat16]]
    bc_all_ty = np.ndarray[((1 + NCHUNK) * 2 * K_DIM,), np.dtype[bfloat16]]

    flags = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_ACCN={accn}"]
    # A/B: with --host-norm the activation arrives already normalised and P4's
    # prologue is the plain block-sum one, which ffn_alt.py exercises at 16
    # cores. That isolates the in-place RMSNorm prologue from the rest.

    params = ", ".join(f"w4_{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"w5_{i}: In" for i in range(npairs))
    params += ", " + ", ".join(f"o5_{i}: Out" for i in range(npairs))
    src = f'''
def _design(bc: In, {params}):
    kg = ExternalFunction("flm_gemv_gate", source_file=GATE_SRC,
                          arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    ku = ExternalFunction("flm_gemv_up_swiglu", source_file=UPS_SRC,
                          arg_types=[bc_ty, wt_ty, o_ty], compile_flags=FLAGS)
    ka = ExternalFunction("flm_gemv_acc", source_file=ACC_SRC,
                          arg_types=[bc_ty, wt_ty], compile_flags=FLAGS)
    kf = ExternalFunction("flm_gemv_flush", source_file=FLUSH_SRC,
                          arg_types=[bc_ty, wt_ty, o_ty], compile_flags=FLAGS)
    kasum = ExternalFunction("flm_asum_prepare", source_file=ASUM_SRC,
                             arg_types=[bc_ty], compile_flags=FLAGS)
    # With --host-norm both prologues ARE flm_asum_prepare, and declaring it
    # twice is `redefinition of symbol named 'flm_asum_prepare'` — one
    # ExternalFunction per entry point, reused, not one per call site.
    knorm = kasum if {host_norm} else ExternalFunction(
        "flm_norm_prepare", source_file=NORM_SRC,
        arg_types=[bc_ty], compile_flags=FLAGS)

    f_bc = ObjectFifo(bc_ty, depth=1, name="bc")
    bc_cons = [f_bc.cons() for _ in range({ncores})]
    # ONE operand fifo and ONE result fifo per pair, reused by both phases.
    # A core tile has 2 input DMA channels and the broadcast takes one, so a
    # second weight fifo is a compile error, not a tight fit:
    #   "tile (0,3) requires 3 input/2 output DMA channels, but only 2 input/2
    #    output available". This is what the plan means by the topology being
    # identical in every phase.
    f_w = [ObjectFifo(wpair_ty, name=f"wp{{i}}") for i in range({npairs})]
    w_sub = [f.cons().split([0, {wt}], obj_types=[wt_ty, wt_ty]) for f in f_w]
    f_o = [ObjectFifo(opair_ty, name=f"op{{i}}") for i in range({npairs})]
    o_sub = [f.prod().join([0, {NROWS}], obj_types=[o_ty, o_ty]) for f in f_o]

    def core(bcc, wc, op, kgate, kups, kacc, kflush, kn, kas):
        for _ in range_({nrep}):
          # ---- P4: norm2 + gate/up + SwiGLU ----
          eb = bcc.acquire(1)
          kn(eb)                                  # RMSNorm in place + block sums
          for _ in range_({p4_pairs}):
              eg = wc.acquire(1)
              kgate(eb, eg)                       # gate -> 64 B in-core stash
              wc.release(1)
              eu = wc.acquire(1)
              eo = op.acquire(1)
              kups(eb, eu, eo)                    # up, then SwiGLU against g_gate
              op.release(1)
              wc.release(1)
          bcc.release(1)
          # ---- P5: down, 4 K-chunks; the last flushes with the residual ----
          for _ in range_({NCHUNK - 1}):
              eb = bcc.acquire(1)
              kas(eb)
              for _ in range_({p5_tiles}):
                  ew = wc.acquire(1)
                  kacc(eb, ew)
                  wc.release(1)
              bcc.release(1)
          eb = bcc.acquire(1)
          kas(eb)
          for _ in range_({p5_tiles}):
              ew = wc.acquire(1)
              eo = op.acquire(1)
              kflush(eb, ew, eo)
              op.release(1)
              wc.release(1)
          bcc.release(1)

    workers = []
    for p in range({npairs}):
        for j in range(2):
            workers.append(Worker(core,
                fn_args=[bc_cons[2 * p + j], w_sub[p][j].cons(),
                         o_sub[p][j].prod(), kg, ku, ka, kf, knorm, kasum],
                stack_size=4096))

    def sequence(*args):
        n = {npairs}
        bcb = args[0]
        w4b = [args[1 + i] for i in range(n)]
        w5b = [args[1 + n + i] for i in range(n)]
        o5b = [args[1 + 2 * n + i] for i in range(n)]
        bch = args[1 + 3 * n]
        wh = [args[2 + 3 * n + i] for i in range(n)]
        oh = [args[2 + 4 * n + i] for i in range(n)]

        for _rep in range({nrep}):
          # ---- P4 ------------------------------------------------------------
          # The results drain straight into the ACTIVATION halves of the P5
          # broadcast objects, in the same BO. `drain(offset=)` is what makes the
          # chain work without a host round trip: pair p owns a contiguous run of
          # SwiGLU rows, and the run lands where P5 chunk c will read it. The aux
          # halves of those same objects were written by the host with the
          # residual, and the drains do not touch them.
          tg = TaskGroup()
          bch.fill(bcb, group=tg, offset=0,
                   sizes=[1, 1, 1, {2 * K_DIM}], strides=[0, 0, 0, 1])
          for i in range(n):
              wh[i].fill(w4b[i], group=tg)
          for i in range(n):
              row = i * {rows_per_pair}
              oh[i].drain(bcb, wait=True, group=tg,
                          offset=(1 + row // {K_DIM}) * {2 * K_DIM} + row % {K_DIM},
                          sizes=[1, 1, {nblk}, {blk}],
                          strides=[0, 0, {2 * K_DIM}, 1])
          tg.finish()

          # ---- P5 ------------------------------------------------------------
          # One fill per chunk (the core acquires the broadcast once per chunk),
          # one weight fill for all four chunks (they share the operand fifo).
          tg = TaskGroup()
          for ch in range({NCHUNK}):
              bch.fill(bcb, group=tg, offset=(1 + ch) * {2 * K_DIM},
                       sizes=[1, 1, 1, {2 * K_DIM}], strides=[0, 0, 0, 1])
          for i in range(n):
              wh[i].fill(w5b[i], group=tg)
          for i in range(n):
              oh[i].drain(o5b[i], wait=True, group=tg)
          tg.finish()

    arg_types = [bc_all_ty] + [w4_ty] * {npairs} + [w5_ty] * {npairs}
    arg_types += [o5_ty] * {npairs}
    arg_types += [f_bc.prod(tile=AnyShimTile)]
    arg_types += [f.prod(tile=AnyShimTile) for f in f_w]
    arg_types += [f.cons(tile=AnyShimTile) for f in f_o]
    rt = Runtime(sequence, arg_types)
    return Program(iron.get_current_device(), rt, workers).resolve_program()
'''
    ns = dict(np=np, iron=iron, CompileTime=CompileTime, In=In, Out=Out,
              ObjectFifo=ObjectFifo, Program=Program, Runtime=Runtime,
              TaskGroup=TaskGroup, Worker=Worker, AnyShimTile=AnyShimTile,
              range_=range_, ExternalFunction=ExternalFunction,
              GATE_SRC=GATE_SRC, UPS_SRC=UPS_SRC, ACC_SRC=ACC_SRC,
              FLUSH_SRC=FLUSH_SRC, NORM_SRC=NORM_SRC, ASUM_SRC=ASUM_SRC,
              FLAGS=flags, bc_ty=bc_ty, wt_ty=wt_ty, o_ty=o_ty,
              wpair_ty=wpair_ty, opair_ty=opair_ty, w4_ty=w4_ty, w5_ty=w5_ty,
              o5_ty=o5_ty, bc_all_ty=bc_all_ty, nrep=nrep,
              __name__="flm_ffn_chain")
    exec(src, ns)
    return iron.jit(ns["_design"],
                    source_files=[GATE_SRC, UPS_SRC, ACC_SRC, FLUSH_SRC,
                                  NORM_SRC, ASUM_SRC], full_elf=True), wt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--repeat", type=int, default=1,
                   help="run the P4+P5 pair N times in ONE dispatch — the Task 8 "
                        "unroll shape. Throughput only; correctness is checked "
                        "at N=1 (later repeats overwrite the same buffers).")
    p.add_argument("--host-norm", action="store_true",
                   help="A/B: normalise on the host and use flm_asum_prepare, "
                        "isolating the in-place RMSNorm prologue")
    o = p.parse_args()
    ncores = o.cores
    npairs = ncores // 2
    p4_pairs = D_FF // (ncores * NROWS)
    p5_tiles = D_MODEL // (ncores * NROWS)

    c = q4nx.Q4nx(str(Q4NX))
    pre = f"model.layers.{o.layer}."
    nw2 = c.bf16(pre + "post_attention_layernorm.weight").astype(np.float32)[:D_MODEL]
    gd, gm, gc_ = load_linear(c, pre + "mlp.gate_proj.weight", D_FF, K_DIM)
    ud, um, uc_ = load_linear(c, pre + "mlp.up_proj.weight", D_FF, K_DIM)
    dd_, dm_, dc_ = c.blocks(pre + "mlp.down_proj.weight")
    nb8 = D_FF // BLK
    dd = dd_[:D_MODEL, :nb8].astype(np.float32)
    dm = dm_[:D_MODEL, :nb8].astype(np.float32)
    dc = dc_[:D_MODEL, :nb8]

    rng = np.random.default_rng(0)
    h = rnd(rng.standard_normal(D_MODEL) * 0.05)

    design, wt = build(ncores, o.host_norm, o.repeat)

    # ---- weight streams -------------------------------------------------
    # P4: core i owns rows [i*p4_pairs*NROWS, ...), gate/up alternating
    w4, w5 = [], []
    for pr in range(npairs):
        rpp4 = D_FF // npairs
        per = []
        for j in range(2):
            rows = [pr * rpp4 + t * 2 * NROWS + j * NROWS
                    for t in range(p4_pairs)]
            per.append(np.concatenate([
                np.concatenate([
                    q4nx.pack_tile(gd[r:r+NROWS], gm[r:r+NROWS], gc_[r:r+NROWS],
                                   row_base=r),
                    q4nx.pack_tile(ud[r:r+NROWS], um[r:r+NROWS], uc_[r:r+NROWS],
                                   row_base=r)])
                for r in rows]))
        buf = np.empty((2 * p4_pairs, 2, wt), np.uint8)
        buf[:, 0, :] = per[0].reshape(-1, wt)
        buf[:, 1, :] = per[1].reshape(-1, wt)
        w4.append(buf.reshape(-1))

        rpp5 = D_MODEL // npairs
        nbc = K_DIM // BLK
        per = []
        for j in range(2):
            rows = [pr * rpp5 + t * 2 * NROWS + j * NROWS
                    for t in range(p5_tiles)]
            per.append(np.concatenate([
                q4nx.pack_tile(dd[r:r+NROWS, ch*nbc:(ch+1)*nbc],
                               dm[r:r+NROWS, ch*nbc:(ch+1)*nbc],
                               dc[r:r+NROWS, ch*nbc:(ch+1)*nbc], row_base=r)
                for ch in range(NCHUNK) for r in rows]))
        buf = np.empty((NCHUNK * p5_tiles, 2, wt), np.uint8)
        buf[:, 0, :] = per[0].reshape(-1, wt)
        buf[:, 1, :] = per[1].reshape(-1, wt)
        w5.append(buf.reshape(-1))

    # ---- broadcast BO: P4's object, then P5's four chunk objects ---------
    # P4's activation half is h; its aux half is the norm weight. P5 chunk c
    # reads swiglu[2048c:...] as its activation and h as its aux (the residual).
    # The activation halves of the P5 objects are written BY THE DEVICE when
    # P4's results drain into this same BO — see the offsets below.
    hd0 = h.astype(np.float64)
    inv0 = np.float32(1.0 / np.sqrt((hd0 * hd0).mean() + EPS))
    hn0 = rnd(rnd(h * rnd(inv0)) * nw2)
    bc = np.zeros((1 + NCHUNK, 2 * K_DIM), np.float32)
    bc[0, :D_MODEL] = hn0 if o.host_norm else h
    bc[0, K_DIM:K_DIM + D_MODEL] = nw2
    for ch in range(NCHUNK):
        bc[1 + ch, K_DIM:K_DIM + D_MODEL] = h        # residual in aux
    bc_t = iron.tensor(bc.reshape(-1).astype(bfloat16), dtype=bfloat16,
                       device="npu")
    w4_ts = [iron.tensor(x, dtype=np.uint8, device="npu") for x in w4]
    w5_ts = [iron.tensor(x, dtype=np.uint8, device="npu") for x in w5]
    o5_ts = [iron.zeros(2 * p5_tiles * NROWS, dtype=bfloat16, device="npu")
             for _ in range(npairs)]

    args = (bc_t, *w4_ts, *w5_ts, *o5_ts)
    if o.bench:
        bench = run_iters(design, *args, warmup=2, iters=10)
        us = bench.npu.min_us if bench.npu else bench.e2e.min_us
    else:
        design(*args)
        us = None

    # ---- reference, float64 with the kernels' bf16 roundings -------------
    hd = h.astype(np.float64)
    inv = np.float32(1.0 / np.sqrt((hd * hd).mean() + EPS))
    hn = rnd(rnd(h * rnd(inv)) * nw2)
    g = np.concatenate([q4nx.gemv_reference_bf16(hn, gd[r:r+NROWS], gm[r:r+NROWS],
                                                 gc_[r:r+NROWS])
                        for r in range(0, D_FF, NROWS)])
    u = np.concatenate([q4nx.gemv_reference_bf16(hn, ud[r:r+NROWS], um[r:r+NROWS],
                                                 uc_[r:r+NROWS])
                        for r in range(0, D_FF, NROWS)])
    sw = rnd((g / (1.0 + np.exp(-g))) * u)
    x_out = np.concatenate([
        q4nx.gemv_reference_bf16(sw, dd[r:r+NROWS], dm[r:r+NROWS], dc[r:r+NROWS])
        for r in range(0, D_MODEL, NROWS)]) + hd

    # P4's output was drained INTO the broadcast BO, so read it back from
    # there — the activation halves of the four P5 chunk objects.
    bc_back = bc_t.numpy().astype(np.float64).reshape(1 + NCHUNK, 2 * K_DIM)
    got4 = bc_back[1:, :K_DIM].reshape(-1)
    e4 = np.abs(got4 - sw)
    _DIAG = __import__("os").environ.get("FFN_CHAIN_DIAG")
    if False:
        import sys as _s
        print(f"  DIAG sorted-multiset match: "
              f"{np.abs(np.sort(got4)-np.sort(sw)).max():.4e}", file=_s.stderr)
        big = np.abs(sw) > 0.05 * np.abs(sw).max()
        rel = np.abs(got4 - sw)[big] / np.abs(sw)[big]
        print(f"  DIAG |sw| max {np.abs(sw).max():.4f}, on the {big.sum()} "
              f"largest: rel err p50 {np.median(rel):.4f} p99 "
              f"{np.quantile(rel,0.99):.4f} max {rel.max():.4f}", file=_s.stderr)
        print(f"  DIAG AIE2P exp2 floor is 3.54% mean / 5.86% max",
              file=_s.stderr)
        # Is the error tracking the exp2 ARGUMENT? The kernel computes
        # silu(g) = g / (1 + exp2(-g*log2e)); ffn_alt.py fed unnormalised
        # activations so -g*log2e stayed near 0, where the NLF was calibrated.
        silu = lambda z: z / (1.0 + np.exp(-z))
        print("  DIAG  idx        g          u   silu(g)*u     device     ratio",
              file=_s.stderr)
        for i in list(range(6)) + [16, 17, 32, 1024, 1025]:
            r = got4[i] / (silu(g[i]) * u[i]) if silu(g[i]) * u[i] else float("nan")
            print(f"  DIAG {i:5d} {g[i]:10.5f} {u[i]:10.5f} "
                  f"{silu(g[i])*u[i]:11.6f} {got4[i]:10.6f} {r:9.4f}",
                  file=_s.stderr)
        cands = {
            "silu(g)*u  (current ref)": rnd(silu(g) * u),
            "silu(u)*g  (tiles swapped)": rnd(silu(u) * g),
            "g*u        (no silu)": rnd(g * u),
            "silu(g)    (u ignored)": rnd(silu(g)),
            "u          (gate ignored)": rnd(u),
            "silu(g)*u, g from PREV tile": rnd(silu(np.roll(g, 16)) * u),
        }
        for lab, v in cands.items():
            e = np.abs(got4 - v)
            print(f"  DIAG cand {lab:30s} max {e.max():.3e} "
                  f"exact {np.mean(e < 1e-6):.1%}", file=_s.stderr)
        arg = -g * np.log2(np.e)
        rel_all = np.abs(got4 - sw) / np.maximum(np.abs(sw), 1e-12)
        print(f"  DIAG exp2 arg range [{arg.min():.1f}, {arg.max():.1f}]  "
              f"|g| p50 {np.median(np.abs(g)):.3f} max {np.abs(g).max():.3f}",
              file=_s.stderr)
        for lo, hi in ((-40, -8), (-8, -2), (-2, 2), (2, 8), (8, 40)):
            m = (arg >= lo) & (arg < hi)
            if m.sum():
                print(f"  DIAG arg in [{lo:4d},{hi:4d}): {m.sum():5d} rows, "
                      f"rel err p50 {np.median(rel_all[m]):.4f} "
                      f"p95 {np.quantile(rel_all[m],0.95):.4f}", file=_s.stderr)
        rpp = D_FF // npairs
        idx = np.arange(D_FF)
        pr_i, within = idx // rpp, idx % rpp
        t_i, j_i = within // (2 * NROWS), (within % (2 * NROWS)) // NROWS
        bad = np.abs(got4 - sw) > 1e-3
        print("  DIAG err by pair : " + " ".join(
            f"{pr}:{bad[pr_i==pr].mean():.0%}" for pr in range(npairs)),
            file=_s.stderr)
        print("  DIAG err by core-in-pair : " + " ".join(
            f"{j}:{bad[j_i==j].mean():.0%}" for j in range(2)), file=_s.stderr)
        print("  DIAG err by tile (first 8): " + " ".join(
            f"{t}:{bad[t_i==t].mean():.0%}" for t in range(8)), file=_s.stderr)
        rel5 = np.abs(got5 - x_out) / np.maximum(np.abs(x_out), 1e-9)
        print(f"  DIAG x_out rel err p50 {np.median(rel5):.4f} "
              f"max {rel5.max():.4f}", file=_s.stderr)

    rpp5 = D_MODEL // npairs
    got5 = np.concatenate([o5_ts[pr].numpy().astype(np.float64)
                           for pr in range(npairs)])
    assert got5.size == D_MODEL, (got5.size, D_MODEL)
    e5 = np.abs(got5 - x_out)
    if _DIAG:
        import sys as _s
        big = np.abs(sw) > 0.05 * np.abs(sw).max()
        rel = np.abs(got4 - sw)[big] / np.abs(sw)[big]
        print(f"  DIAG |sw| max {np.abs(sw).max():.4f}; on the {big.sum()} "
              f"largest: rel p50 {np.median(rel):.4f} p99 "
              f"{np.quantile(rel,0.99):.4f} max {rel.max():.4f}", file=_s.stderr)
        # Is the error tracking the exp2 ARGUMENT? The kernel computes
        # silu(g) = g / (1 + exp2(-g*log2e)); ffn_alt.py fed unnormalised
        # activations so -g*log2e stayed near 0, where the NLF was calibrated.
        silu = lambda z: z / (1.0 + np.exp(-z))
        print("  DIAG  idx        g          u   silu(g)*u     device     ratio",
              file=_s.stderr)
        for i in list(range(6)) + [16, 17, 32, 1024, 1025]:
            r = got4[i] / (silu(g[i]) * u[i]) if silu(g[i]) * u[i] else float("nan")
            print(f"  DIAG {i:5d} {g[i]:10.5f} {u[i]:10.5f} "
                  f"{silu(g[i])*u[i]:11.6f} {got4[i]:10.6f} {r:9.4f}",
                  file=_s.stderr)
        cands = {
            "silu(g)*u  (current ref)": rnd(silu(g) * u),
            "silu(u)*g  (tiles swapped)": rnd(silu(u) * g),
            "g*u        (no silu)": rnd(g * u),
            "silu(g)    (u ignored)": rnd(silu(g)),
            "u          (gate ignored)": rnd(u),
            "silu(g)*u, g from PREV tile": rnd(silu(np.roll(g, 16)) * u),
        }
        for lab, v in cands.items():
            e = np.abs(got4 - v)
            print(f"  DIAG cand {lab:30s} max {e.max():.3e} "
                  f"exact {np.mean(e < 1e-6):.1%}", file=_s.stderr)
        arg = -g * np.log2(np.e)
        rel_all = np.abs(got4 - sw) / np.maximum(np.abs(sw), 1e-12)
        print(f"  DIAG exp2 arg range [{arg.min():.1f}, {arg.max():.1f}]  "
              f"|g| p50 {np.median(np.abs(g)):.3f} max {np.abs(g).max():.3f}",
              file=_s.stderr)
        for lo, hi in ((-40, -8), (-8, -2), (-2, 2), (2, 8), (8, 40)):
            m = (arg >= lo) & (arg < hi)
            if m.sum():
                print(f"  DIAG arg in [{lo:4d},{hi:4d}): {m.sum():5d} rows, "
                      f"rel err p50 {np.median(rel_all[m]):.4f} "
                      f"p95 {np.quantile(rel_all[m],0.95):.4f}", file=_s.stderr)
        rpp = D_FF // npairs
        idx = np.arange(D_FF)
        pr_i, within = idx // rpp, idx % rpp
        t_i, j_i = within // (2 * NROWS), (within % (2 * NROWS)) // NROWS
        bad = np.abs(got4 - sw) > 1e-3
        print("  DIAG err by pair : " + " ".join(
            f"{pr}:{bad[pr_i==pr].mean():.0%}" for pr in range(npairs)),
            file=_s.stderr)
        print("  DIAG err by core-in-pair : " + " ".join(
            f"{j}:{bad[j_i==j].mean():.0%}" for j in range(2)), file=_s.stderr)
        print("  DIAG err by tile (first 8): " + " ".join(
            f"{t}:{bad[t_i==t].mean():.0%}" for t in range(8)), file=_s.stderr)
        rel5 = np.abs(got5 - x_out) / np.maximum(np.abs(x_out), 1e-9)
        print(f"  DIAG x_out rel p50 {np.median(rel5):.4f} "
              f"p99 {np.quantile(rel5,0.99):.4f}", file=_s.stderr)
        print("  DIAG AIE2P exp2 floor: 3.54% mean / 5.86% max", file=_s.stderr)

    total = o.repeat * ncores * (2 * p4_pairs + NCHUNK * p5_tiles) * wt
    print(f"FFN half as 2 chained phases, 1 dispatch: {ncores} cores, "
          f"layer {o.layer}")
    print(f"  P4 {p4_pairs} gate/up pairs/core -> {D_FF};  "
          f"P5 {NCHUNK}x{p5_tiles} tiles/core -> {D_MODEL}")
    print(f"  P4 SwiGLU out : max err {e4.max():.4e}  mean|ref| "
          f"{np.abs(sw).mean():.5f}")
    print(f"  P5 x_out      : max err {e5.max():.4e}  mean|ref| "
          f"{np.abs(x_out).mean():.5f}")
    if us:
        print(f"  {total/1e6:.2f} MB  {total/(us*1e-6)/1e9:.1f} GB/s  {us:.1f} us "
              f"(marginal {us-FIXED_US:.1f}, 16-core ideal "
              f"{total/1e6*17.85:.1f})")
    # The gate is a POINTWISE RELATIVE error on the values that carry the
    # signal, not an absolute error scaled by the mean. SwiGLU's output has a
    # max/mean ratio of ~14, so "8% of the mean" is a far tighter bound on the
    # largest values than the exp2 NLF can meet, and an earlier version of this
    # file failed on that alone after the arithmetic was already right.
    def relgate(got, ref, name):
        big = np.abs(ref) > 0.05 * np.abs(ref).max()
        rel = (np.abs(got - ref)[big] / np.abs(ref)[big]).max()
        print(f"  {name}: max rel err on the {big.sum()} largest {rel:.4f}")
        return rel
    r4 = relgate(got4, sw, "P4 SwiGLU")
    # x_out = W_down.sw + h, and h is added EXACTLY, so all the error lives in
    # the projection term. Where the two nearly cancel, a pointwise relative
    # error on x_out diverges while the absolute error is unchanged — it reads
    # 15.8% on rows whose x_out is near zero. Scale by the term that carries
    # the error instead.
    d_ref = x_out - hd
    r5 = np.abs(got5 - x_out).max() / np.abs(d_ref).max()
    print(f"  P5 x_out : max err {np.abs(got5-x_out).max():.4e} vs "
          f"max|W_down.sw| {np.abs(d_ref).max():.4f} -> {r5:.4f}")
    ok = r4 <= 6e-2 and r5 <= 6e-2
    print(f"  gate: <= 6% (AIE2P exp2 NLF: 3.54% mean / 5.86% max) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
