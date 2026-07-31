#!/usr/bin/env python3
"""How accurate is `aie::exp2<bfloat16>`? The attention floor needs this number.

The attention error has resisted four normalizers (mean|ref|, max|V|, max|ref|,
and a fitted product). The mechanism was eventually read out of
`flm_attn_decode.cc` rather than fitted: the softmax weights `p` and the
online-rescale factor `corr` are both `aie::exp2<bfloat16>` results, and `corr`
multiplies the whole running accumulator on every max update. That explains the
SHAPE of the error — it tracks rescale count, not position count — but the
coefficient is still 3-4x off on some configurations.

The untested term is `aie::exp2` itself, which is a hardware approximation. This
measures it, separating the two errors folded into that one call:

    device vs float64 2**x        total error
    device vs bfloat16(2**x)      NLF error alone, with bf16 rounding removed

If the second column is zero, the NLF is correctly rounded and the attention
floor is pure bf16 arithmetic. If it is not, that residual is the missing term.

    python3 exp2_probe.py                # the range attention actually uses
    python3 exp2_probe.py --lo -30 --hi 4

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

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
SRC = str(KDIR / "flm_exp2_probe.cc")
N = 1024


def build():
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    flags = [f"-DDIM_NPROBE={N}"]

    def _design(a: In, o: Out):
        k = ExternalFunction("flm_exp2_probe", source_file=SRC,
                             arg_types=[in_ty, out_ty], compile_flags=flags)
        f_a = ObjectFifo(in_ty, depth=1, name="e2_in")
        f_o = ObjectFifo(out_ty, depth=1, name="e2_out")

        def core(ac, op, k):
            e = ac.acquire(1)
            eo = op.acquire(1)
            k(e, eo)
            op.release(1)
            ac.release(1)

        w = Worker(core, fn_args=[f_a.cons(), f_o.prod(), k], stack_size=4096)

        def seq(ab, ob, ah, oh):
            tg = TaskGroup()
            ah.fill(ab, group=tg)
            oh.drain(ob, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [in_ty, out_ty, f_a.prod(tile=AnyShimTile),
                           f_o.cons(tile=AnyShimTile)])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return iron.jit(_design, source_files=[SRC], full_elf=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lo", type=float, default=-20.0,
                   help="attention feeds exp2 (score - running_max), so <= 0")
    p.add_argument("--hi", type=float, default=0.0)
    p.add_argument("--dump", type=int, default=0,
                   help="print N sample (x, got, exact) triples — sanity-check\n"
                        "the probe itself before believing its headline")
    o = p.parse_args()

    x = np.linspace(o.lo, o.hi, N).astype(np.float32)
    a = iron.tensor(x, dtype=np.float32, device="npu")
    b = iron.zeros(N, dtype=bfloat16, device="npu")
    build()(a, b)

    got = b.numpy().astype(np.float64)
    exact = np.exp2(x.astype(np.float64))
    rounded = np.asarray(exact, dtype=bfloat16).astype(np.float64)

    # Relative, since exp2 spans decades over this range and an absolute error
    # would just report the largest x.
    nz = exact > 0
    rel_total = np.abs(got[nz] - exact[nz]) / exact[nz]
    rel_nlf = np.abs(got[nz] - rounded[nz]) / exact[nz]
    ulp = 2.0**-9                      # bf16 half-ulp, the correctly-rounded bound

    if o.dump:
        step = max(1, N // o.dump)
        for i in range(0, N, step):
            print(f"    x {x[i]:+9.5f}  got {got[i]:.8e}  "
                  f"exact {exact[i]:.8e}  rel {abs(got[i]-exact[i])/exact[i]:.3e}")
    print(f"aie::exp2<bfloat16> over x in [{o.lo}, {o.hi}], {N} points")
    print(f"  vs float64 2**x     : max rel {rel_total.max():.3e}"
          f"  ({rel_total.max() / ulp:.2f} half-ulp)   mean {rel_total.mean():.3e}")
    print(f"  vs bfloat16(2**x)   : max rel {rel_nlf.max():.3e}"
          f"  ({rel_nlf.max() / ulp:.2f} half-ulp)   mean {rel_nlf.mean():.3e}")
    exact_bits = int((got == rounded).sum())
    print(f"  bit-identical to correctly-rounded bf16: {exact_bits}/{N}"
          f"  ({100 * exact_bits / N:.1f}%)")
    if rel_nlf.max() <= 1e-12:
        print("  -> the NLF is correctly rounded; the attention floor is pure "
              "bf16 rounding")
    else:
        worst = int(np.argmax(rel_nlf))
        print(f"  -> the NLF carries its OWN error, worst at x = "
              f"{x[nz][worst]:.4f}: got {got[nz][worst]:.6e} vs "
              f"{rounded[nz][worst]:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
