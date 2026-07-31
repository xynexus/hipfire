#!/usr/bin/env python3
"""An acquire between two kernels sharing a global loses the handoff, silently.

Minimal reproduction of a bug that cost two ticks in `kv_emit_verify.py`. Two
kernels communicate through a core global — kernel A reads an acquired fifo
object and writes the global, kernel B reads the global and writes the output.
That is the pattern the fused layer uses three times over:

    flm_gemv_gate     -> g_gate   -> flm_gemv_up_swiglu
    flm_gemv_residual -> g_resid  -> flm_gemv_flush
    flm_gemv_qkv      -> g_stage  -> flm_qkv_emit / flm_kv_emit

**If a fifo `acquire` sits between the two calls, kernel B reads zeros.** No
error, no warning, no diagnostic — the output is simply the global's initial
value. Hoisting the acquires above both calls fixes it.

    ks(es); ew = wc.acquire(1); ke(...)        ->  0.0   WRONG
    ew = wc.acquire(1); ks(es); ke(...)        ->  42.0  right

An intervening `release` does **not** help, which rules out the obvious
"the object is still locked" explanation:

    ks(es); sc.release(1); ew = wc.acquire(1); ke(...)  ->  0.0   WRONG

**The boundary is not fully characterised, and one known case contradicts the
simple rule.** `ffn_chain.py` puts two acquires between `flm_gemv_gate` and
`flm_gemv_up_swiglu` and verifies exact. The difference from this probe is that
there both calls sit in the same `range_` loop iteration, whereas here the write
is outside the loop and the read inside it. That is a hypothesis, not a result.

    python3 global_handoff_probe.py            # all three variants
    python3 global_handoff_probe.py --variant hoist

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

N = 64
HERE = Path(__file__).parent
A_SRC, B_SRC = HERE / "_gh_a.cc", HERE / "_gh_b.cc"
# One entry point per translation unit — two in one file is `duplicate symbol`.
A_SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
alignas(64) bfloat16 g_hand[64];
extern "C" __attribute__((noinline)) void
gh_write(const bfloat16 *restrict in) {
  for (int i = 0; i < 64; i += 32)
    aie::store_v(g_hand + i, aie::load_v<32>(in + i));
}
""")
B_SRC.write_text("""// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
extern bfloat16 g_hand[];
extern "C" __attribute__((noinline)) void
gh_read(const bfloat16 *restrict unused, bfloat16 *restrict out) {
  (void)unused;
  for (int i = 0; i < 64; i += 32)
    aie::store_v(out + i, aie::load_v<32>(g_hand + i));
}
""")


def build(variant):
    ty = np.ndarray[(N,), np.dtype[bfloat16]]
    src = f'''
def _design(a: In, b: In, o: Out):
    kw = ExternalFunction("gh_write", source_file=str(A_SRC), arg_types=[ty])
    kr = ExternalFunction("gh_read", source_file=str(B_SRC), arg_types=[ty, ty])
    f_a = ObjectFifo(ty, depth=1, name="a_{variant}")
    f_b = ObjectFifo(ty, depth=1, name="b_{variant}")
    f_o = ObjectFifo(ty, depth=1, name="o_{variant}")

    def core(ac, bc, op, kwr, krd):
        ea = ac.acquire(1)
        if "{variant}" == "hoist":
            eb = bc.acquire(1); eo = op.acquire(1); kwr(ea)
        elif "{variant}" == "release":
            kwr(ea); ac.release(1); eb = bc.acquire(1); eo = op.acquire(1)
        else:                                   # "interleave" — the bug
            kwr(ea); eb = bc.acquire(1); eo = op.acquire(1)
        krd(eb, eo)
        op.release(1); bc.release(1)
        if "{variant}" != "release":
            ac.release(1)

    w = Worker(core, fn_args=[f_a.cons(), f_b.cons(), f_o.prod(), kw, kr],
               stack_size=16384)

    def seq(ab, bb, ob, ah, bh, oh):
        tg = TaskGroup()
        ah.fill(ab, group=tg); bh.fill(bb, group=tg)
        oh.drain(ob, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty, ty, ty, f_a.prod(tile=AnyShimTile),
                       f_b.prod(tile=AnyShimTile), f_o.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile,
              ExternalFunction=ExternalFunction, A_SRC=A_SRC, B_SRC=B_SRC,
              ty=ty, __name__=f"gh_{variant}")
    exec(src, ns)
    return iron.jit(ns["_design"], source_files=[str(A_SRC), str(B_SRC)],
                    full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["interleave", "release", "hoist"])
    o = p.parse_args()
    variants = [o.variant] if o.variant else ["interleave", "release", "hoist"]

    print("kernel A reads an acquired object -> global -> kernel B reads it")
    print(f"{'variant':<12s} {'acquire between the calls?':<28s} {'out[0]':>7s}  ")
    print("-" * 56)
    ok = True
    for v in variants:
        design = build(v)
        a = iron.tensor(np.full(N, 42.0, np.float32).astype(bfloat16),
                        dtype=bfloat16, device="npu")
        b = iron.zeros(N, dtype=bfloat16, device="npu")
        out = iron.zeros(N, dtype=bfloat16, device="npu")
        design(a, b, out)
        got = float(out.numpy().astype(np.float64)[0])
        desc = {"interleave": "yes", "release": "yes, after a release",
                "hoist": "no — hoisted above both"}[v]
        good = got == 42.0
        ok &= good if v == "hoist" else True
        print(f"{v:<12s} {desc:<28s} {got:7.1f}  {'ok' if good else 'LOST'}")
    print("\n42.0 is the value fed in; 0.0 means the handoff was lost.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
