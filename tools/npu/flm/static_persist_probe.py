#!/usr/bin/env python3
"""Does core-static data survive between dispatches? (decides the k′ append)

`kv_append_probe.py` showed that appending one token to the channel-major K
cache is not expressible as a bf16 DMA: transfer sizes must be multiples of 4
bytes and offsets must be 4-byte aligned, so a single 2-byte value per
destination is illegal and odd columns are unreachable. Two options remain:

  * **f32 K** — proven, and doubles KV traffic (5.26 → 10.52 MB/layer at
    S=2048, roughly 54 → 48 tok/s on the measured projection).
  * **paired append** — keep bf16 and always write two columns at an even
    offset: `(k′_t, 0)` at even `t`, `(k′_{t-1}, k′_t)` at offset `t−1` at odd
    `t`. Costs nothing.

The second is free but assumes the emitting core still holds `k′_{t-1}` **from
the previous dispatch**. Decode is one dispatch per token with the same xclbin
loaded throughout, so the question is whether a core's `.bss` survives from one
dispatch to the next or is re-initialised.

The probe is a kernel that increments a file-scope counter and emits it, invoked
as N separate dispatches on one loaded design:

    persists  -> 1, 2, 3, ...   the paired append is available, keep bf16
    reset     -> 1, 1, 1, ...   core state cannot carry a token; use f32 K

It also reports whether a **second design** sees the first's counter, which
would mean state leaks between programs rather than persisting usefully.

    python3 static_persist_probe.py
    python3 static_persist_probe.py --dispatches 8

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

N = 32
SRC = Path(__file__).parent / "_persist_bump.cc"
SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
// Increments a file-scope counter and reports it. The counter is ordinary .bss
// with external linkage, which is exactly what the k' carry would be.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_N
#define DIM_N 32
#endif
#ifndef PERSIST_TAG
#define PERSIST_TAG 0
#endif

alignas(64) float g_persist_count[DIM_N];

extern "C" __attribute__((noinline)) void
persist_bump(bfloat16 *restrict out) {
#if PERSIST_TAG == 1
  // CONTROL: no state at all. If repeated dispatches do not all read 7, the
  // probe's own output path is broken and it cannot answer anything about .bss.
  for (int i = 0; i < DIM_N; ++i)
    out[i] = bfloat16(7.0f);
#else
  for (int i = 0; i < DIM_N; ++i) {
    g_persist_count[i] += 1.0f;
    out[i] = bfloat16(g_persist_count[i]);
  }
#endif
}
""")


def build(tag):
    o_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    def _design(out: Out):
        # tag is interpolated into the design source so the two variants get
        # different cache keys — a -D value passed only through a runtime list
        # is invisible to iron.jit's key (see the tools README).
        kern = ExternalFunction("persist_bump", source_file=str(SRC),
                                arg_types=[o_ty],
                                compile_flags=[f"-DDIM_N={N}",
                                               f"-DPERSIST_TAG={tag}"])
        # depth=1: at the default depth of 2 the drain alternates between
        # buffers and every other dispatch reads the one the core did not
        # just write, which shows up as 0, 2, 0, 4 and looks like a
        # persistence failure rather than a fifo artifact.
        f_o = ObjectFifo(o_ty, depth=1, name=f"po{tag}")

        def core(op, k):
            eo = op.acquire(1)
            k(eo)
            op.release(1)

        w = Worker(core, fn_args=[f_o.prod(), kern], stack_size=16384)

        def seq(ob, oh):
            oh.drain(ob, wait=True)

        rt = Runtime(seq, [o_ty, f_o.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt,
                       workers=[w]).resolve_program()

    _design.__name__ = f"_design_tag{tag}"
    return iron.jit(_design, source_files=[str(SRC)], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dispatches", type=int, default=5)
    o = p.parse_args()

    def run(tag):
        # iron.jit keys on the design's code object, which is identical for both
        # tags (they differ only via the closure), so the cache must be cleared
        # between them — the trap recorded in the tools README.
        shutil.rmtree(Path.home() / ".npu" / "cache", ignore_errors=True)
        d = build(tag)
        out = []
        for _ in range(o.dispatches):
            t = iron.zeros(N, dtype=bfloat16, device="npu")
            d(t)
            out.append(float(t.numpy().astype(np.float64)[0]))
        return out

    ctrl = run(1)
    print(f"CONTROL, stateless kernel writing 7.0, {o.dispatches} dispatches:")
    print(f"  values: {ctrl}")
    if not all(v == 7.0 for v in ctrl):
        print("  -> the probe's OWN read path is unreliable; it cannot answer")
        print("     the persistence question. Do not read anything into the")
        print("     counter values below.")
        return 1
    print("  -> read path is sound\n")
    seen = run(0)

    print(f"core-static counter over {o.dispatches} separate dispatches, "
          f"one loaded design")
    print(f"  values: {seen}")
    persists = seen == [float(i + 1) for i in range(o.dispatches)]
    reset = all(v == 1.0 for v in seen)
    if persists:
        verdict = "PERSISTS — .bss survives between dispatches"
    elif reset:
        verdict = "RESET — .bss is re-initialised every dispatch"
    else:
        verdict = "NEITHER — see the values; do not rely on this either way"
    print(f"  -> {verdict}")

    print("\n  consequence for the k' append:")
    if persists:
        print("    the paired bf16 append is available: a core can carry")
        print("    k'_{t-1} across dispatches, so K stays bf16 and the KV")
        print("    traffic stays at 5.26 MB/layer (S=2048).")
    else:
        print("    a core cannot carry k'_{t-1}, so the paired append needs the")
        print("    value re-supplied by the host each step, or K becomes f32 at")
        print("    10.52 MB/layer (S=2048).")
    return 0 if persists or reset else 1


if __name__ == "__main__":
    raise SystemExit(main())
