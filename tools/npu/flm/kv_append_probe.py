#!/usr/bin/env python3
"""Appending k′ to the channel-major KV cache: what the DMA will and will not do.

The last seam in the fused layer is P1 → P2 — k′ and v′ must land in the layout
attention reads, and those layouts are deliberately opposite
(`flm_attn_decode.cc`):

    K  channel-major  [HEAD][TSEQ]   scores accumulate across d, no reduce
    V  position-major [TSEQ][HEAD]   output accumulates across t

so appending one token is a contiguous 64-element write for v′ and a **stride-32
scatter** for k′. Whether a `drain` can express that scatter decides whether
§1.4's phase schedule survives.

**Measured: with a bf16 cache, it cannot.** Two independent DMA rules bite:

    sizes:   'aie.dma_bd' op Transfer sizes must be multiples of 4 bytes.
             1 elements at 2 bytes each equal 2 bytes, which is not divisible by 4
    offsets: 'aie.dma_bd' op Offset must be aligned to 4 byte boundary

A bf16 is 2 bytes, so one value per destination is an illegal *size*, and an odd
column is an illegal *offset*. The narrowest legal channel-major bf16 write
therefore covers **two columns starting at an even one** — verified working —
which a one-token-at-a-time decode cannot produce on its own.

**With an f32 K cache it works directly**, at every position including odd ones,
because one element is already 4 bytes. That is what this probe now checks. The
cost is not free — see `docs/npu/flm-refe-log.md` for the traffic comparison and
the cheaper alternative that needs a core-static value to survive between
dispatches.

    GATE: K[d][t] == ramp[d] for every d and t, and the untouched columns stay
          exactly zero (attention requires K=0 padding for its npad correction)

    python3 kv_append_probe.py --positions 0,1,2,3

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,  # noqa: E402
                      TaskGroup, Worker)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

HEAD, TSEQ = 64, 32
SRC = Path(__file__).parent / "_kv_emit.cc"
SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
// Emits one head. The probe is about the DMA pattern, not the arithmetic, so
// the body just forwards a head-sized object.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
extern "C" __attribute__((noinline)) void
kv_emit(const float *restrict in, float *restrict out) {
  for (int i = 0; i < DIM_HEAD; i += 16)
    aie::store_v(out + i, aie::load_v<16>(in + i));
}
""")


def build(npos):
    h_ty = np.ndarray[(HEAD,), np.dtype[np.float32]]      # k' as f32
    kv_ty = np.ndarray[(TSEQ * HEAD,), np.dtype[np.float32]]

    def _design(src: In, kv: Out):
        kern = ExternalFunction("kv_emit", source_file=str(SRC),
                                arg_types=[h_ty, h_ty],
                                compile_flags=[f"-DDIM_HEAD={HEAD}"])
        f_in = ObjectFifo(h_ty, name="hin")
        f_out = ObjectFifo(h_ty, name="hout")

        def core(ic, op, k):
            for _ in range_(npos):
                ei = ic.acquire(1)
                eo = op.acquire(1)
                k(ei, eo)
                op.release(1)
                ic.release(1)

        w = Worker(core, fn_args=[f_in.cons(), f_out.prod(), kern],
                   stack_size=4096)

        def seq(sb, kvb, sh, kvh):
            for i in range(npos):
                tg = TaskGroup()
                # k': HEAD values, destination stride TSEQ — the scatter under
                # test. Element d of the head belongs at K[d][t], which is
                # offset d*TSEQ + t.
                sh.fill(sb, group=tg)
                # ONE token per append, using a 2-element innermost run whose
                # second element is a deliberate ZERO.
                #
                # A single bf16 per destination is illegal — "Transfer sizes
                # must be multiples of 4 bytes" — so the narrowest legal
                # channel-major scatter touches two columns. Decode produces one
                # token at a time, so the second column would be clobbered; but
                # it is column t+1, which is *beyond the current sequence
                # length*, and the next token overwrites it. Writing a zero
                # there is not merely harmless, it is what attention already
                # requires: padded positions must hold K=0 so their softmax
                # contribution is exactly the exp2(-m) that flm_attn_finish
                # subtracts.
                kvh.drain(kvb, wait=True, group=tg, offset=i,
                          sizes=[1, 1, HEAD, 1], strides=[0, 0, TSEQ, 1])
                tg.finish()

        rt = Runtime(seq, [h_ty, kv_ty, f_in.prod(tile=AnyShimTile),
                           f_out.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt,
                       workers=[w]).resolve_program()

    return iron.jit(_design, source_files=[str(SRC)], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", default="0,1,2,3",
                   help="cache positions to append at (each gets one k'/v')")
    o = p.parse_args()
    pos = [int(x) for x in o.positions.split(",")]
    npos = len(pos)
    if pos != list(range(npos)):
        print("note: the probe appends at consecutive positions 0..n-1; "
              "--positions only sets how many")
    design = build(npos)

    # a ramp so a wrong stride is visible as a permutation, not as noise.
    # The k' source is (value, 0) per channel; the v' source is the plain ramp,
    # and it reads the first HEAD elements of the same buffer.
    r = np.arange(1, HEAD + 1, dtype=np.float32) / HEAD
    s_t = iron.tensor(r, dtype=np.float32, device="npu")
    kv_t = iron.zeros(TSEQ * HEAD, dtype=np.float32, device="npu")
    design(s_t, kv_t)
    got = kv_t.numpy().astype(np.float64)
    K = got.reshape(HEAD, TSEQ)                   # channel-major
    rd = r.astype(np.float64)

    print(f"KV append via strided drain: HEAD={HEAD} TSEQ={TSEQ}, "
          f"{npos} positions")
    ok = True
    for i in range(npos):
        ek = np.abs(K[:, i] - rd).max()
        ok &= ek == 0
        print(f"  t={i}: K[:,{i}] max err {ek:.3e}")
    # the trailing zero of the last append must leave column npos exactly zero
    untouched = np.abs(K[:, npos:]).max() if npos < TSEQ else 0.0
    print(f"  columns {npos}..{TSEQ-1} still zero (pad must be K=0): max |K| {untouched:.3e}")
    ok &= untouched == 0
    if not ok:
        print(f"  K[:,0] first 8 got {K[:8,0].round(4)}  want {rd[:8].round(4)}")
    print(f"  -> {'PASS: a strided drain can append k-prime channel-major'
                 if ok else 'FAIL: k-prime needs a core-side transpose'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
