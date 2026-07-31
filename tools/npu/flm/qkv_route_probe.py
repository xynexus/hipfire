#!/usr/bin/env python3
"""Can several drains split one result fifo three ways? (last P1→P2 primitive)

Phase P1 emits 48 heads down **one** result fifo — 32 q, 8 k, 8 v — and they go
to three unrelated places:

    q′  contiguous, into the buffer P2 reads as its query block
    k′  a stride-TSEQ scatter into the channel-major K cache
    v′  contiguous, into the position-major V cache

A drain consumes what it takes, so this needs *successive partial drains* from a
single fifo, each with its own destination buffer and access pattern. Nothing so
far has done that: every harness to date drains a fifo exactly once.

If it does not work, P1 needs a result fifo per destination — and a core tile
has 2 output DMA channels, so three of them do not fit and §1.4's phase schedule
changes.

The probe emits a distinguishable ramp per head, splits the stream 3 ways with
the real patterns, and checks each destination separately. A drain that took the
wrong count would shift the later ones, which the per-head ramp makes visible.

    GATE: all three destinations hold exactly their own heads

    python3 qkv_route_probe.py
    python3 qkv_route_probe.py --heads 6,2,2

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

HEAD, TSEQ = 64, 32
SRC = Path(__file__).parent / "_route_emit.cc"
SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
// Emits one head, tagged so the host can tell which one it was. The probe is
// about the routing, not the arithmetic.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
int g_head_ix;

extern "C" __attribute__((noinline)) void
route_emit(bfloat16 *restrict out) {
  // g_head_ix persists across calls (and across dispatches — see
  // static_persist_probe.py), so each head carries a distinct tag and a drain
  // that took the wrong count shifts the tags visibly.
  // Every element of head h is h+1 — a small integer, exact in bf16 (which is
  // only exact on integers to 256, so a 100*h+i ramp would add rounding noise
  // on top of any routing error and confuse the two).
  const bfloat16 tag = bfloat16(float(g_head_ix) + 1.0f);
  for (int i = 0; i < DIM_HEAD; i += 16)
    aie::store_v(out + i, aie::broadcast<bfloat16, 16>(tag));
  ++g_head_ix;
}
""")


def build(nq, nk, nv):
    h_ty = np.ndarray[(HEAD,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(nq * HEAD,), np.dtype[bfloat16]]
    k_ty = np.ndarray[(nk * HEAD * TSEQ,), np.dtype[bfloat16]]
    v_ty = np.ndarray[(nv * TSEQ * HEAD,), np.dtype[bfloat16]]
    total = nq + 2 * nk + nv   # a K tile takes 2 objects

    def _design(qb: Out, kb: Out, vb: Out):
        kern = ExternalFunction("route_emit", source_file=str(SRC),
                                arg_types=[h_ty],
                                compile_flags=[f"-DDIM_HEAD={HEAD}"])
        f_o = ObjectFifo(h_ty, depth=2, name="heads")

        def core(op, k):
            for _ in range_(total):
                eo = op.acquire(1)
                k(eo)
                op.release(1)

        # 16384: a scalar float->bf16 loop can spill the accumulator file, and
        # overflow here is silent. See stack_audit.py.
        w = Worker(core, fn_args=[f_o.prod(), kern], stack_size=16384)

        def seq(qbuf, kbuf, vbuf, oh):
            # THREE successive partial drains of ONE fifo, in emit order.
            tg = TaskGroup()
            # q': nq heads, contiguous
            oh.drain(qbuf, wait=True, group=tg,
                     sizes=[1, 1, 1, nq * HEAD], strides=[0, 0, 0, 1])
            # k': each head scattered down a column of its own [HEAD][TSEQ]
            # tile. Two columns per write at an even offset is the narrowest
            # legal bf16 form (kv_append_probe.py), so head h writes columns
            # 0 and 1 of tile h.
            oh.drain(kbuf, wait=True, group=tg,
                     sizes=[1, nk, HEAD, 2], strides=[0, HEAD * TSEQ, TSEQ, 1])
            # v': each head contiguous at the top of its own [TSEQ][HEAD] tile
            oh.drain(vbuf, wait=True, group=tg,
                     sizes=[1, nv, 1, HEAD], strides=[0, TSEQ * HEAD, 0, 1])
            tg.finish()

        # ONE shim consumer handle, three drains on it. A fifo may have only
        # one shim endpoint — asking for three is `redefinition of symbol named
        # 'heads_shim_alloc'` — but a handle can be drained repeatedly, which is
        # what splitting the stream needs.
        rt = Runtime(seq, [q_ty, k_ty, v_ty, f_o.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt,
                       workers=[w]).resolve_program()

    return iron.jit(_design, source_files=[str(SRC)], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heads", default="4,2,2", help="nq,nk,nv")
    o = p.parse_args()
    nq, nk, nv = (int(x) for x in o.heads.split(","))

    design = build(nq, nk, nv)
    q_t = iron.zeros(nq * HEAD, dtype=bfloat16, device="npu")
    k_t = iron.zeros(nk * HEAD * TSEQ, dtype=bfloat16, device="npu")
    v_t = iron.zeros(nv * TSEQ * HEAD, dtype=bfloat16, device="npu")
    design(q_t, k_t, v_t)

    head = lambda h: np.full(HEAD, float(h + 1))
    print(f"one result fifo split 3 ways: {nq} q + {nk} k (2 objects each) "
          f"+ {nv} v")
    ok = True

    got = q_t.numpy().astype(np.float64)
    want = np.concatenate([head(h) for h in range(nq)])
    e = np.abs(got - want).max()
    ok &= e == 0
    print(f"  q' contiguous          : max err {e:.3e}   "
          f"per-head tags got {[got[h*HEAD] for h in range(nq)]} "
          f"want {[h+1.0 for h in range(nq)]}")

    # A K tile consumes TWO objects, not one: the write is 2 elements per
    # destination (the narrowest legal channel-major bf16 form), so filling
    # HEAD destinations needs 2*HEAD source values. Channels 0..31 come from the
    # first object and 32..63 from the second, because the drain walks the
    # source linearly while striding the destination.
    K = k_t.numpy().astype(np.float64).reshape(nk, HEAD, TSEQ)
    for i in range(nk):
        t0, t1 = nq + 2 * i + 1.0, nq + 2 * i + 2.0
        e = max(np.abs(K[i, :HEAD // 2, :2] - t0).max(),
                np.abs(K[i, HEAD // 2:, :2] - t1).max())
        rest = np.abs(K[i, :, 2:]).max()
        ok &= e == 0 and rest == 0
        print(f"  k' tile {i}: cols 0-1 = {K[i,0,0]:.0f}/{K[i,HEAD//2,0]:.0f} "
              f"(want {t0:.0f}/{t1:.0f}, 2 objects per tile)   "
              f"cols 2.. zero {rest:.3e}")

    V = v_t.numpy().astype(np.float64).reshape(nv, TSEQ, HEAD)
    for i in range(nv):
        h = head(nq + 2 * nk + i)     # k consumed 2 objects per tile
        e = np.abs(V[i, 0] - h).max()
        ok &= e == 0
        print(f"  v' tile {i} row 0        : got {V[i,0,0]:.1f} want {h[0]:.1f}")

    print(f"  -> {'PASS: one fifo can feed three destinations'
                 if ok else 'FAIL: P1 needs a fifo per destination, and a core '
                           'tile has only 2 output DMA channels'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
