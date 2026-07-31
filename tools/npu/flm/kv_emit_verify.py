#!/usr/bin/env python3
"""k′ appended one token per dispatch, carrying across the dispatch boundary.

**STATUS: this harness does not work yet — it produces an all-zero cache.** The
kernel it exercises (`kernels/npu/flm_kv_emit.cc`) compiles clean at a 64 B
frame, and the two mechanisms it combines are each already proven elsewhere:
the paired strided append by `kv_append_probe.py`, and cross-dispatch `.bss`
persistence by `static_persist_probe.py`. What fails is this design: nothing is
written at all, with no error reported. Ruled out — the drain's stride pattern
(a plain contiguous drain is equally empty), the destination BO size (a BO sized
to exactly one object is equally empty), and the tile trailer offset (fixed, no
change). So the core or its fifos are not producing, upstream of the append.

The next attempt should **extend `qkv_route_probe.py`**, which produces and
routes correctly with one input-less core, rather than build a third design from
scratch; the difference here is two input fifos feeding the core.

`flm_kv_emit.cc` closes the P1→P2 seam. Appending a token to the channel-major K
cache is a stride-TSEQ scatter, and a DMA cannot do that one element at a time —
sizes must be multiples of 4 bytes and offsets 4-byte aligned, so a lone bf16 per
destination is an illegal size and an odd column an unreachable offset. The
narrowest legal write covers two columns from an even one, so:

    even t:  (k′_t, 0)          -> column pair (t, t+1)
    odd  t:  (k′_{t-1}, k′_t)   -> column pair (t-1, t)

Every column is written twice, once opening the pair and once closing it, and
the zero at even t is what attention's `npad` correction needs from a padded
position.

**Closing a pair needs the previous token's k′, from the previous dispatch.**
That is the one thing this scheme rests on and the one thing a single dispatch
cannot show, so this harness runs **N separate dispatches** — one per token,
exactly as decode does — and then checks the whole cache at once:

    GATE: K[:, t] == k′_t for every t < N, and every column >= N still zero

A carry that failed would leave the odd columns holding the wrong token, which
this catches per column rather than in aggregate.

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
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::load_v<32>(in + i));
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
    f_in = ObjectFifo(in_ty, depth=1, name="hin{t}")
    f_wt = ObjectFifo(wt_ty, depth=1, name="wt{t}")
    f_o = ObjectFifo(out_ty, depth=1, name="ko{t}")

    def core(ic, wc, op, ks, ke):
        ei = ic.acquire(1)
        ks(ei)
        ew = wc.acquire(1)
        eo = op.acquire(1)
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
                     f"-DDIM_ACT=2048"],
              in_ty=in_ty, wt_ty=wt_ty, out_ty=out_ty, k_ty=k_ty,
              SIZES=SIZES, STRIDES=STRIDES,
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

    # the KV tile persists across dispatches on the host side, like a real cache
    import os as _os1
    ksz = 2 * HEAD if _os1.environ.get("KV_MATCHED") else HEAD * TSEQ
    k_t = iron.zeros(ksz, dtype=bfloat16, device="npu")
    for t in range(N):
        trailer = np.zeros(TILE, np.uint8)
        trailer[TRAILER_OFF:TRAILER_OFF + 8].view(np.float32)[:] = [0.0, float(t)]
        design = build(t)
        design(iron.tensor(ks[t].astype(bfloat16), dtype=bfloat16, device="npu"),
               iron.tensor(trailer, dtype=np.uint8, device="npu"), k_t)

    if _os1.environ.get("KV_MATCHED"):
        v = k_t.numpy().astype(np.float64)
        print(f"  MATCHED-BO control: got[0:6] {v[:6].round(4)}  "
              f"want interleaved {ks[0][:3].round(4)} with zeros")
        return 0
    K = k_t.numpy().astype(np.float64).reshape(HEAD, TSEQ)
    if __import__("os").environ.get("KV_DIAG"):
        import sys as _s
        w0 = ks[0].astype(np.float64)
        print(f"  DIAG want k0[0:6] {w0[:6].round(4)}", file=_s.stderr)
        print(f"  DIAG K[0:6, 0]    {K[:6,0].round(4)}", file=_s.stderr)
        print(f"  DIAG K[0:6, 1]    {K[:6,1].round(4)}", file=_s.stderr)
        print(f"  DIAG K[0, 0:6]    {K[0,:6].round(4)}", file=_s.stderr)
        flat = k_t.numpy().astype(np.float64)
        hits = [int(i) for i in np.where(np.abs(flat - w0[0]) < 1e-6)[0][:6]]
        print(f"  DIAG k0[0]={w0[0]:.4f} appears at flat indices {hits} "
              f"(want {0*TSEQ+0}); TSEQ={TSEQ}", file=_s.stderr)
        hits1 = [int(i) for i in np.where(np.abs(flat - w0[1]) < 1e-6)[0][:6]]
        print(f"  DIAG k0[1]={w0[1]:.4f} appears at {hits1} (want {1*TSEQ})",
              file=_s.stderr)
    print(f"k' appended over {N} SEPARATE dispatches, one per token")
    ok = True
    for t in range(N):
        e = np.abs(K[:, t] - ks[t].astype(np.float64)).max()
        ok &= e == 0
        kind = "opens pair" if t % 2 == 0 else "closes pair (needs the carry)"
        print(f"  t={t}: column {t} max err {e:.3e}   [{kind}]")
    tail = np.abs(K[:, N:]).max()
    ok &= tail == 0
    print(f"  columns {N}..{TSEQ-1} still zero: {tail:.3e}  "
          f"(attention needs K=0 padding)")
    print(f"  -> {'PASS: the cross-dispatch carry works; K stays bf16'
                 if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
