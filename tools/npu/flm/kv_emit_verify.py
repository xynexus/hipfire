#!/usr/bin/env python3
"""k′ appended one token per dispatch, carrying across the dispatch boundary.

**Verified.** One design, N dispatches, every step exact — including the two
carries. Two faults had to be fixed to get here and both are worth knowing:

1. **Hoisting the acquires above the kernel calls.** Interleaved
   `acquire -> call -> acquire` made the first kernel read zeros from its
   acquired input, with no error; hoisting fixed it, and reverting broke it
   again. **The trigger is not understood** — `ffn_chain.py` interleaves the
   same way and is exact — so this is a known hazard with an unknown boundary,
   not a rule. If a kernel reads zeros from an acquired input, try hoisting.
2. **Build ONE design and reuse it.** A new design is a new program load, which
   clears core `.bss` — so a harness that rebuilds per token destroys the very
   carry it is trying to test. The drain offset is therefore fixed at column
   pair (0,1) here and every token writes there; an odd step's column 0 can only
   come from `g_kprev`, so the carry is still what is under test. Real decode
   reuses one design too, and will vary the offset with `offset_parameter=`
   (a `ScratchpadParameter`) rather than by rebuilding.

    python3 kv_emit_verify.py
    python3 kv_emit_verify.py --tokens 8

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
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
EMIT_SRC = str(KDIR / "flm_kv_emit.cc")
SEED_SRC = str(Path(__file__).parent / "_kv_seed.cc")
HEAD, TSEQ = 64, 32
# flm_kv_emit reads the position from the tile's trailer via tile_flags(),
# which offsets by TILE_BYTES — so it needs a real tile-sized buffer, not a
# bare trailer. Passing 64 bytes reads 20 KB out of bounds and the even/odd
# branch goes random, which looks like a broken carry.
TILE = q4nx.tile_bytes(2048, 16)
TRAILER_OFF = TILE - 64

Path(SEED_SRC).write_text("""// SPDX-License-Identifier: Apache-2.0
// Writes a head into g_stage, standing in for flm_gemv_qkv's rotate-and-stage
// epilogue. The append scheme is what is under test, not the projection.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
// DEFINES g_stage here. In the layer it is flm_gemv_qkv.cc's staging buffer and
// this kernel does not exist; isolating the append means something else has to
// own it.
alignas(64) bfloat16 g_stage[DIM_HEAD];
extern "C" __attribute__((noinline)) void
kv_seed(const bfloat16 *restrict in) {
#ifdef SEED_CONST
  // split test: ignore the input and write a constant. If the cache then shows
  // 5.0, the g_stage handoff works and the input fifo is at fault; if it stays
  // zero, the cross-TU g_stage link is.
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::broadcast<bfloat16, 32>(bfloat16(5.0f)));
#else
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::load_v<32>(in + i));
#endif
}
""")


def build(t):
    """One design per token: the drain offset is baked into the sequence."""
    in_ty = np.ndarray[(HEAD,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(TILE,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(2 * HEAD,), np.dtype[bfloat16]]
    import os as _os0
    k_ty = (np.ndarray[(2 * HEAD,), np.dtype[bfloat16]]
            if _os0.environ.get('KV_MATCHED')
            else np.ndarray[(HEAD * TSEQ,), np.dtype[bfloat16]])
    off = t - (t & 1)                    # even column of this token's pair
    import os as _osc
    SEEDC = ["-DSEED_CONST=1"] if _osc.environ.get("KV_SEEDCONST") else []
    SEEDTAG = 1 if SEEDC else 0    # local: the f-string interpolates it,
                                   # which is what gives the two variants
                                   # distinct jit cache keys
    import os as _os
    if _os.environ.get("KV_CONTIG"):     # control: is the drain running at all?
        SIZES, STRIDES = [1, 1, 1, 2 * HEAD], [0, 0, 0, 1]
    elif _os.environ.get("KV_MATCHED"):  # control: BO exactly one object
        SIZES, STRIDES = [1, 1, 1, 2 * HEAD], [0, 0, 0, 1]
    else:
        SIZES, STRIDES = [1, 1, HEAD, 2], [0, 0, TSEQ, 1]

    src = f'''
def _design(hin: In, wt: In, kc: Out):
    kseed = ExternalFunction("kv_seed", source_file=SEED_SRC,
                             arg_types=[in_ty], compile_flags=FLAGS)
    kemit = ExternalFunction("flm_kv_emit", source_file=EMIT_SRC,
                             arg_types=[wt_ty, out_ty], compile_flags=FLAGS)
    f_in = ObjectFifo(in_ty, depth=1, name="hin{t}_{SEEDTAG}")
    f_wt = ObjectFifo(wt_ty, depth=1, name="wt{t}")
    f_o = ObjectFifo(out_ty, depth=1, name="ko{t}")

    def core(ic, wc, op, ks, ke):
        # All three acquires precede both kernel calls. Interleaving them —
        # acquire, call, acquire — makes `ks` read ZEROS from `ei`, verified by
        # A/B in both directions. The exact trigger is NOT established:
        # ffn_chain.py interleaves in the same way and is exact, so "never
        # interleave" is not the rule. Treat it as a known hazard with an
        # unknown boundary; if a kernel reads zeros from an acquired input,
        # hoist the acquires first.
        ei = ic.acquire(1)
        ew = wc.acquire(1)
        eo = op.acquire(1)
        ks(ei)
        ke(ew, eo)
        op.release(1)
        wc.release(1)
        ic.release(1)

    w = Worker(core, fn_args=[f_in.cons(), f_wt.cons(), f_o.prod(),
                              kseed, kemit], stack_size=16384)

    def seq(hb, wb, kb, hh, wh, oh):
        tg = TaskGroup()
        hh.fill(hb, group=tg)
        wh.fill(wb, group=tg)
        # the pair write: 2 elements per destination, HEAD destinations TSEQ
        # apart, starting at the pair's even column
        oh.drain(kb, wait=True, group=tg, offset={off},
                 sizes={SIZES}, strides={STRIDES})
        tg.finish()

    rt = Runtime(seq, [in_ty, wt_ty, k_ty, f_in.prod(tile=AnyShimTile),
                       f_wt.prod(tile=AnyShimTile), f_o.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile, range_=range_,
              ExternalFunction=ExternalFunction, EMIT_SRC=EMIT_SRC,
              SEED_SRC=SEED_SRC,
              FLAGS=[f"-DDIM_K=2048", f"-DDIM_NROWS=16", f"-DDIM_HEAD={HEAD}",
                     f"-DDIM_ACT=2048"] + SEEDC,
              in_ty=in_ty, wt_ty=wt_ty, out_ty=out_ty, k_ty=k_ty,
              SIZES=SIZES, STRIDES=STRIDES, SEEDC=SEEDC,
              SEEDTAG=SEEDTAG,
              __name__=f"flm_kv_emit_t{t}")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[EMIT_SRC, SEED_SRC],
                    full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=5)
    o = p.parse_args()
    N = o.tokens
    if N > TSEQ - 1:
        raise SystemExit(f"--tokens must be < {TSEQ} (one KV tile)")

    rng = np.random.default_rng(0)
    ks = q4nx.bf16_to_f32(q4nx.f32_to_bf16(
        (rng.standard_normal((N, HEAD)) * 0.3).astype(np.float32)))
    if __import__("os").environ.get("KV_CONSTHEAD"):
        ks = np.full((N, HEAD), 5.0, np.float32)   # split: known input data

    # the KV tile persists across dispatches on the host side, like a real cache
    import os as _os1
    ksz = 2 * HEAD if _os1.environ.get("KV_MATCHED") else HEAD * TSEQ
    k_t = iron.zeros(ksz, dtype=bfloat16, device="npu")

    # ONE design for every token — the whole point. Building per token would
    # reload the program and clear .bss, which is what defeated the previous
    # version of this harness. The drain offset is therefore FIXED at column
    # pair (0,1) and every token writes there; that still exercises the carry,
    # because an odd step's column 0 can only come from g_kprev.
    design = build(0)
    snaps = []
    for t in range(N):
        trailer = np.zeros(TILE, np.uint8)
        trailer[TRAILER_OFF:TRAILER_OFF + 8].view(np.float32)[:] = [0.0, float(t)]
        design(iron.tensor(ks[t].astype(bfloat16), dtype=bfloat16, device="npu"),
               iron.tensor(trailer, dtype=np.uint8, device="npu"), k_t)
        snaps.append(k_t.numpy().astype(np.float64).reshape(HEAD, TSEQ)[:, :2].copy())

    print(f"k' over {N} dispatches of ONE design, all writing column pair (0,1)")
    ok = True
    for t in range(N):
        c0, c1 = snaps[t][:, 0], snaps[t][:, 1]
        if t % 2 == 0:
            e0 = np.abs(c0 - ks[t].astype(np.float64)).max()
            e1 = np.abs(c1).max()
            ok &= e0 == 0 and e1 == 0
            print(f"  t={t} even: col0=k{t} err {e0:.3e}   col1=0 err {e1:.3e}")
        else:
            e0 = np.abs(c0 - ks[t - 1].astype(np.float64)).max()
            e1 = np.abs(c1 - ks[t].astype(np.float64)).max()
            ok &= e0 == 0 and e1 == 0
            print(f"  t={t} odd : col0=k{t-1} err {e0:.3e}  <- THE CARRY   "
                  f"col1=k{t} err {e1:.3e}")
    print(f"  -> {'PASS: g_kprev survives the dispatch boundary; K stays bf16'
                 if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
