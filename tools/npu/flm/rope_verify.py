#!/usr/bin/env python3
"""llama-3.2-1B RoPE on the NPU against a float64 reference.

Runs `kernels/npu/flm_rope.cc` over all 32 query heads and 8 KV heads, at a
given token position, and checks against HF's rotate_half formulation in
float64.

**The llama3 frequency scaling comes out of the container, not a
reimplementation.** `rope_freqs.weight` [32] is the per-frequency llama3
DIVISOR — [1]*15, then 1.648, 3.297, 9.688, then [32]*14 — so

    inv_freq = 1 / theta**(2i/head_dim) / rope_freqs[i]

This harness re-derives the llama3 wavelength interpolation from config.json
independently and asserts the two agree, so the identification is checked rather
than assumed.

    python3 rope_verify.py --pos 1000
    python3 rope_verify.py --pos 100000     # past the 8192 original context

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
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_rope.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
HEAD, THETA = 64, 500000.0
FACTOR, LO_F, HI_F, OLD_CTX = 32.0, 1.0, 4.0, 8192


@iron.jit(source_files=[SRC])
def rope(x: In, cs: In, out: Out, *, NHEADS: CompileTime[int] = 32):
    head_ty = np.ndarray[(HEAD,), np.dtype[bfloat16]]
    all_ty = np.ndarray[(NHEADS * HEAD,), np.dtype[bfloat16]]
    kern = ExternalFunction("flm_rope", source_file=SRC,
                            arg_types=[head_ty, head_ty, head_ty],
                            compile_flags=[f"-DDIM_HEAD={HEAD}"])
    f_x, f_cs, f_o = (ObjectFifo(head_ty, name="x"),
                      ObjectFifo(head_ty, depth=1, name="cs"),
                      ObjectFifo(head_ty, name="o"))

    def core(xc, cc, op, k):
        # cs is the same for every head at this position: acquired once.
        ec = cc.acquire(1)
        for _ in range_(NHEADS):
            ex, eo = xc.acquire(1), op.acquire(1)
            k(ex, ec, eo)
            op.release(1); xc.release(1)
        cc.release(1)

    w = Worker(core, fn_args=[f_x.cons(), f_cs.cons(), f_o.prod(), kern],
               stack_size=4096)

    def seq(a, b, c, xh, ch, oh):
        xh.fill(a); ch.fill(b); oh.drain(c, wait=True)

    rt = Runtime(seq, [all_ty, head_ty, all_ty,
                       f_x.prod(), f_cs.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def llama3_inv_freq():
    """Independent re-derivation of the llama3-scaled frequencies."""
    base = 1.0 / (THETA ** (np.arange(0, HEAD, 2) / HEAD))
    wl = 2 * np.pi / base
    lo_wl, hi_wl = OLD_CTX / LO_F, OLD_CTX / HI_F
    out = np.where(wl > lo_wl, base / FACTOR, base)
    mid = (wl >= hi_wl) & (wl <= lo_wl)
    smooth = (OLD_CTX / wl - LO_F) / (HI_F - LO_F)
    return np.where(mid, (1 - smooth) * base / FACTOR + smooth * base, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pos", type=int, default=1000, help="token position")
    p.add_argument("--nheads", type=int, default=32, help="32 query, 8 kv")
    o = p.parse_args()

    c = q4nx.Q4nx(str(Q4NX))
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)
    base = 1.0 / (THETA ** (np.arange(0, HEAD, 2) / HEAD))
    inv_freq = base / divisor

    ref_freq = llama3_inv_freq()
    rel = np.abs(inv_freq - ref_freq) / np.abs(ref_freq)
    print(f"rope_freqs.weight as the llama3 divisor: max rel diff vs an "
          f"independent derivation {rel.max():.4%}")
    assert rel.max() < 0.01, "rope_freqs is not the llama3 divisor"
    print(f"  divisor: {divisor[:3]} ... {divisor[14:18]} ... {divisor[-2:]}")

    ang = o.pos * inv_freq
    cs = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32)
    cs_b = q4nx.bf16_to_f32(q4nx.f32_to_bf16(cs))

    rng = np.random.default_rng(0)
    x = q4nx.bf16_to_f32(q4nx.f32_to_bf16(
        rng.standard_normal(o.nheads * HEAD).astype(np.float32) * 0.5))

    x_t = iron.tensor(x.astype(bfloat16), dtype=bfloat16, device="npu")
    c_t = iron.tensor(cs_b.astype(bfloat16), dtype=bfloat16, device="npu")
    o_t = iron.zeros(o.nheads * HEAD, dtype=bfloat16, device="npu")
    rope(x_t, c_t, o_t, NHEADS=o.nheads)
    got = o_t.numpy().astype(np.float64).reshape(o.nheads, HEAD)

    xr = x.astype(np.float64).reshape(o.nheads, HEAD)
    cosv, sinv = np.cos(ang), np.sin(ang)
    lo = xr[:, :HEAD // 2] * cosv - xr[:, HEAD // 2:] * sinv
    hi = xr[:, HEAD // 2:] * cosv + xr[:, :HEAD // 2] * sinv
    ref = np.concatenate([lo, hi], axis=1)

    err = np.abs(got - ref)
    big = np.abs(ref) > 0.05 * np.abs(ref).max()
    relerr = err[big] / np.abs(ref[big])
    print(f"\npos={o.pos}  heads={o.nheads}")
    print(f"  absolute: max {err.max():.4e}  mean {err.mean():.4e}")
    print(f"  relative: max {relerr.max():.3%}  mean {relerr.mean():.3%} "
          f"(over {big.sum()} of {ref.size} elements above 5% of peak)")

    # The float64 reference is not the gate. `x*cos - y*sin` is a difference of
    # two similar-magnitude terms, so relative error blows up on whichever
    # element cancels hardest regardless of how exact the kernel is. The gate is
    # the device matching ITS OWN bf16 chain: bf16 operands, float accumulation
    # (mul then msc), one rounding on the store.
    cb = q4nx.bf16_to_f32(q4nx.f32_to_bf16(cosv.astype(np.float32))).astype(np.float64)
    sb = q4nx.bf16_to_f32(q4nx.f32_to_bf16(sinv.astype(np.float32))).astype(np.float64)
    rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))
    emu = np.concatenate([
        rnd(xr[:, :HEAD // 2] * cb - xr[:, HEAD // 2:] * sb),
        rnd(xr[:, HEAD // 2:] * cb + xr[:, :HEAD // 2] * sb)], axis=1).astype(np.float64)
    dev = np.abs(got - emu).max()
    print(f"  vs its own bf16 chain: max {dev:.4e}")

    ok = dev <= 1e-6 * max(np.abs(ref).max(), 1e-30)
    print(f"  device-vs-bf16-chain -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
