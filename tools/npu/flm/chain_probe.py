#!/usr/bin/env python3
"""Can phase N+1 read what phase N wrote, inside ONE dispatch? (Task 7 falsifier)

The fused layer is five phases in one dispatch, and every phase but the first
consumes the previous phase's output: P1's q′ feeds P2, P2's attention output
feeds P3, P3's `h` feeds P4, P4's SwiGLU output feeds P5. There is no host
between them, so the only way that can work is for a phase's `drain` to land in
a DDR buffer that a later phase's `fill` reads back — all as buffer descriptors
in a single command stream, ordered by the in-dispatch barrier.

**That has never been tested, and the whole of Task 7 rests on it.** If a
`fill` issued after a `drain(wait=True)` on the same BO sees the drained data,
the plan is buildable as written. If it sees stale data, the fused layer needs
a different carrier for inter-phase values (memtile residency, or splitting the
layer back into more dispatches) and §1.4's phase schedule has to change.

The probe strips it to the bone: one pair of cores, a kernel that doubles its
input, and three phases chained A → B → C through the same fifos. If the chain
holds, C = 8·A. Any other answer localises the break — 2·A means only the first
phase ran, 4·A means the third fill read stale data.

    GATE: C == 8*A  -> Task 7 is buildable as designed
          otherwise -> the phase schedule needs a different inter-phase carrier

    python3 chain_probe.py
    python3 chain_probe.py --phases 5 --n 2048

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

SRC = Path(__file__).parent / "_chain_double.cc"
SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
// Doubles its input. The probe needs an operation whose repeated application
// is visible in the result, so that a broken chain is distinguishable from a
// chain that ran the wrong number of times.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_N
#define DIM_N 256
#endif
extern "C" __attribute__((noinline)) void
chain_double(const bfloat16 *restrict in, bfloat16 *restrict out) {
  for (int i = 0; i < DIM_N; i += 32)
    aie::store_v(out + i, aie::mul(aie::load_v<32>(in + i),
                                   bfloat16(2.0f)).to_vector<bfloat16>());
}
""")


def build(n, nphases):
    ty = np.ndarray[(n,), np.dtype[bfloat16]]
    flags = [f"-DDIM_N={n}"]

    def _design(a: In, b: Out):
        kern = ExternalFunction("chain_double", source_file=str(SRC),
                                arg_types=[ty, ty], compile_flags=flags)
        f_in = ObjectFifo(ty, name="cin")
        f_out = ObjectFifo(ty, name="cout")

        def core(ic, op, k):
            for _ in range_(nphases):
                ei = ic.acquire(1)
                eo = op.acquire(1)
                k(ei, eo)
                op.release(1)
                ic.release(1)

        w = Worker(core, fn_args=[f_in.cons(), f_out.prod(), kern],
                   stack_size=4096)

        def seq(ab, bb, ah, bh):
            # Phase 0 reads the host's A. Every later phase reads B — the buffer
            # the previous phase's drain just wrote. One TaskGroup per phase so
            # its BDs are freed before the next opens, and so the drain's
            # wait=True is a real barrier between them.
            for ph in range(nphases):
                tg = TaskGroup()
                ah.fill(ab if ph == 0 else bb, group=tg)
                bh.drain(bb, wait=True, group=tg)
                tg.finish()

        rt = Runtime(seq, [ty, ty, f_in.prod(tile=AnyShimTile),
                           f_out.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt,
                       workers=[w]).resolve_program()

    return iron.jit(_design, source_files=[str(SRC)], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256, help="bf16 elements per phase")
    p.add_argument("--phases", type=int, default=3)
    o = p.parse_args()

    design = build(o.n, o.phases)
    a = np.arange(1, o.n + 1, dtype=np.float32) / o.n
    a_t = iron.tensor(a.astype(bfloat16), dtype=bfloat16, device="npu")
    b_t = iron.zeros(o.n, dtype=bfloat16, device="npu")
    design(a_t, b_t)
    got = b_t.numpy().astype(np.float64)

    want = a.astype(np.float64) * (2.0 ** o.phases)
    print(f"chained phases in one dispatch: {o.phases} phases, {o.n} bf16 each")
    print(f"  a[0]={a[0]:.6f}  ->  got[0]={got[0]:.6f}   "
          f"expected {want[0]:.6f} (a x 2^{o.phases})")
    ratio = got[0] / a[0] if a[0] else float("nan")
    print(f"  realised gain {ratio:.3f}x, expected {2.0**o.phases:.0f}x")
    err = np.abs(got - want).max()
    ok = err <= 1e-2 * np.abs(want).mean()
    print(f"  max err {err:.3e}")
    if ok:
        print("  -> PASS: a later phase DOES read what an earlier phase drained.")
        print("     Task 7's inter-phase carrier works; build the 5-phase layer.")
    else:
        n_ran = int(round(np.log2(max(ratio, 1e-9))))
        print(f"  -> FAIL: the chain broke. The gain implies {n_ran} of "
              f"{o.phases} phases took effect.")
        print("     Task 7 needs a different inter-phase carrier (memtile")
        print("     residency, or more dispatches) and §1.4 must change.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
