#!/usr/bin/env python3
"""RMSNorm on the NPU against a float64 reference, on real llama weights.

Needed twice per decoder layer (input_layernorm, post_attention_layernorm) and
therefore by every candidate fused-layer design. Runs
`kernels/npu/flm_rmsnorm.cc`, a compile-time-shaped wrapper over the existing
`tools/npu/rms_norm_weighted_bf16.cc`, whose math already matches llama:

    out = x * rsqrt(mean(x^2) + 1e-5) * weight

and 1e-5 is llama-3.2-1B's `rms_norm_eps` exactly.

The weight is the real `model.layers.N.input_layernorm.weight` (BF16, stored
unquantized in FLM's container).

**What this is really measuring.** The underlying kernel broadcasts `inv_rms` in
bf16, so the normalisation scale carries ~0.4% error applied *uniformly* to the
whole vector — a systematic scale error on every activation entering the
projections, not a random one. This harness reports the error both ways: raw,
and after removing the best-fit scale. If almost all of the error is the scale
factor, that says the kernel is fine elementwise and the question is whether a
uniform 0.4% gain error matters downstream.

    python3 rmsnorm_verify.py
    python3 rmsnorm_verify.py --layer 5 --post

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_rmsnorm.cc")
AIE_INC = Path.home() / ".venv/lib/python3.14/site-packages/mlir_aie/include"
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
EPS = 1e-5


@iron.jit(source_files=[SRC])
def rmsnorm(x: In, w: In, out: Out, *, COLS: CompileTime[int] = 2048):
    ty = np.ndarray[(COLS,), np.dtype[bfloat16]]
    kern = ExternalFunction("flm_rmsnorm", source_file=SRC,
                            arg_types=[ty, ty, ty],
                            compile_flags=[f"-DDIM_COLS={COLS}"],
                            include_dirs=[str(AIE_INC)])
    f_x, f_w, f_o = (ObjectFifo(ty, name="x"), ObjectFifo(ty, name="w"),
                     ObjectFifo(ty, name="o"))

    def core(xc, wc, op, k):
        ex, ew, eo = xc.acquire(1), wc.acquire(1), op.acquire(1)
        k(ex, ew, eo)
        op.release(1); wc.release(1); xc.release(1)

    worker = Worker(core, fn_args=[f_x.cons(), f_w.cons(), f_o.prod(), kern],
                    stack_size=4096)

    def seq(a, b, c, xh, wh, oh):
        xh.fill(a); wh.fill(b); oh.drain(c, wait=True)

    rt = Runtime(seq, [ty, ty, ty, f_x.prod(), f_w.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--post", action="store_true",
                   help="post_attention_layernorm instead of input_layernorm")
    p.add_argument("--cols", type=int, default=2048)
    o = p.parse_args()

    which = "post_attention_layernorm" if o.post else "input_layernorm"
    name = f"model.layers.{o.layer}.{which}.weight"
    c = q4nx.Q4nx(str(Q4NX))
    w = c.bf16(name).astype(np.float32)[: o.cols]
    print(f"{name}: {w.shape[0]} weights, "
          f"range [{w.min():.4f}, {w.max():.4f}], mean {w.mean():.4f}")

    rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))
    rng = np.random.default_rng(0)
    x = rnd(rng.standard_normal(o.cols) * 0.05)

    x_t = iron.tensor(x.astype(bfloat16), dtype=bfloat16, device="npu")
    w_t = iron.tensor(w.astype(bfloat16), dtype=bfloat16, device="npu")
    o_t = iron.zeros(o.cols, dtype=bfloat16, device="npu")
    rmsnorm(x_t, w_t, o_t, COLS=o.cols)
    got = o_t.numpy().astype(np.float64)

    xd = x.astype(np.float64)
    ref = xd * (1.0 / np.sqrt((xd * xd).mean() + EPS)) * w.astype(np.float64)

    err = np.abs(got - ref)
    scale = np.abs(ref).mean()
    # Separate the systematic gain error from the elementwise error: fit the
    # single best scale, then re-measure. If nearly all the error is the gain,
    # the kernel is elementwise-correct and only the bf16 inv_rms is at fault.
    gain = float((got @ ref) / (ref @ ref))
    resid = np.abs(got - gain * ref)

    # RELATIVE error per element is the meaningful measure. Comparing an
    # absolute error against mean|ref| inflates it wherever the reference has
    # wide dynamic range, which it does here (the norm weight spans
    # -0.64..1.06). That framing produced a misleading "7% error" reading on
    # this kernel and on the SwiGLU before it.
    big = np.abs(ref) > 0.05 * np.abs(ref).max()
    rel = np.abs(got[big] - ref[big]) / np.abs(ref[big])
    rel_g = np.abs(got[big] - gain * ref[big]) / np.abs(ref[big])

    print(f"\nabsolute     : max {err.max():.4e}  mean {err.mean():.4e}")
    print(f"relative     : max {rel.max():.3%}  mean {rel.mean():.3%}  "
          f"(over {big.sum()} elements above 5% of peak)")
    print(f"best-fit gain: {gain:.6f}  ({abs(gain-1):.3%} off unity) "
          f"-- the bf16 inv_rms broadcast")
    print(f"after gain   : max {rel_g.max():.3%}  mean {rel_g.mean():.3%}")

    # Emulate the kernel's own bf16 chain: inv_rms rounded to bf16, then
    # x*inv_rms rounded, then *weight rounded. Three roundings, ~0.4% each.
    inv = np.float32(1.0 / np.sqrt((x.astype(np.float64) ** 2).mean() + EPS))
    emu = rnd(rnd(x * rnd(inv)) * w)
    erel = np.abs(emu.astype(np.float64)[big] - ref[big]) / np.abs(ref[big])
    print(f"bf16-chain emulation vs float64: max {erel.max():.3%}  "
          f"mean {erel.mean():.3%}")
    print(f"device vs that emulation       : max "
          f"{np.abs(got - emu.astype(np.float64)).max():.4e}")

    # The gate is the device matching its own bf16 chain, not float64.
    ok = np.abs(got - emu.astype(np.float64)).max() <= 2e-3 * np.abs(ref).max()
    print(f"\ndevice-vs-bf16-chain -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
