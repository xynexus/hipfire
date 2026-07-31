#!/usr/bin/env python3
"""Verify the q4_1 decode GEMV core body on silicon against a numpy reference.

Phase 2 milestone 4: `kernels/npu/flm_gemv_q4_1.cc` is the compute body of the
`layer.xclbin` reproduction. This runs it on real q4_1 data taken from FLM's own
`model.q4nx` — real scales, real mins, real codes, real distributions — and
compares against `q4nx.gemv_reference` accumulated in float64.

**What this does and does not establish.** It checks the arithmetic exactly: the
nibble unpack, the mask-only high-half with 1/16 folded into the scale, the
block-sum identity that removes the zero-point from the inner loop, and the
float accumulation. It does **not** check which output row / k-block each of
FLM's stored slots belongs to, because that mapping is unresolved (FLM's weights
are not a quantization of the published checkpoint — see `docs/npu/flm-refe-log.md`).
The bytes are real; their addressing is ours.

    python3 gemv_verify.py                    # K=2048, one tile of 8 rows
    python3 gemv_verify.py --n 512            # 64 tiles, one core

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

KERNEL_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_gemv_q4_1.cc")
PREP_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_asum_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"


BLK_C = 32


def tile_bytes(K, NROWS):
    return 2 * NROWS * (K // BLK_C) * 2 + NROWS * (K // 2)


import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402


# The shapes are CompileTime parameters, not closure captures. They have to be:
# the jit cache key is built from the generator plus its compile kwargs, so a
# shape closed over from an enclosing scope does NOT invalidate the cache, and
# the second shape run silently reuses the first shape's binary. `source_files`
# is here for the same reason -- ExternalFunction's own `source_file=` is not
# part of the key, so editing the kernel .cc alone reuses a stale xclbin.
@iron.jit(source_files=[KERNEL_SRC, PREP_SRC])
def flm_gemv(act: In, w: In, out: Out, *, K: CompileTime[int] = 2048,
             N: CompileTime[int] = 8, NROWS: CompileTime[int] = 8):
    NB = K // BLK_C
    wtile_bytes = tile_bytes(K, NROWS)
    ntiles = N // NROWS

    act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wtile_bytes,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[np.float32]]
    w_all_ty = np.ndarray[(ntiles * wtile_bytes,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(N,), np.dtype[np.float32]]

    kern = ExternalFunction(
        "flm_gemv_q4_1",
        source_file=KERNEL_SRC,
        arg_types=[act_ty, wt_ty, o_ty],
        compile_flags=[f"-DDIM_K={K}", f"-DDIM_NROWS={NROWS}"],
    )
    # Activation block-sums, computed once per acquire rather than per tile.
    prep = ExternalFunction(
        "flm_asum_prepare",
        source_file=PREP_SRC,
        arg_types=[act_ty],
        compile_flags=[f"-DDIM_K={K}", f"-DDIM_NROWS={NROWS}"],
    )

    # depth=1: the activation is acquired ONCE and held for the whole tile
    # loop, so there is nothing for a second buffer to overlap with — it is
    # dead weight in L1. At K=8192 that is 16384 B, which is the difference
    # between 2 and 4 rows per weight tile for down_proj.
    f_act = ObjectFifo(act_ty, depth=1, name="act")
    f_w = ObjectFifo(wt_ty, name="wt")
    f_o = ObjectFifo(o_ty, name="out")

    def core(a_cons, w_cons, o_prod, k, kprep):
        # The activation is acquired once and held across every weight tile: it
        # is the same vector for all of them, and re-acquiring it per tile would
        # put the broadcast on the critical path.
        ea = a_cons.acquire(1)
        kprep(ea)
        for _ in range_(ntiles):
            ew = w_cons.acquire(1)
            eo = o_prod.acquire(1)
            k(ea, ew, eo)
            o_prod.release(1)
            w_cons.release(1)
        a_cons.release(1)

    # IRON's default worker stack is 1024 B and overflowing it fails silently
    # (NaN in the first output rows, plausible garbage in the rest). 4096 is
    # ample now that the activation block-sums are a file-scope static rather
    # than a stack array -- and the stack counts against the same 64 KB as the
    # buffers, which is what pushed K=8192 at 4 rows/tile over the limit.
    worker = Worker(core, fn_args=[f_act.cons(), f_w.cons(), f_o.prod(), kern, prep],
                    stack_size=4096)

    def sequence(a, wbuf, obuf, ah, wh, oh):
        ah.fill(a)
        wh.fill(wbuf)
        oh.drain(obuf, wait=True)

    rt = Runtime(sequence, [act_ty, w_all_ty, o_all_ty,
                            f_act.prod(), f_w.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=8, help="output rows total")
    p.add_argument("--nrows", type=int, default=8, help="output rows per tile")
    p.add_argument("--tensor", default="model.layers.0.mlp.down_proj.weight")
    o = p.parse_args()

    K, N, NROWS = o.k, o.n, o.nrows
    NB = K // BLK_C
    if N % NROWS:
        raise SystemExit(f"--n {N} must divide by --nrows {NROWS}")

    # Real q4_1 data from FLM's container. A 5120-byte row carries 256 blocks;
    # take as many as this shape needs and regroup them into (rows, NB).
    c = q4nx.Q4nx(str(Q4NX))
    d_all, m_all, codes_all = c.blocks(o.tensor)
    need = N * NB
    flat_d = d_all.ravel()[:need].reshape(N, NB).astype(np.float32)
    flat_m = m_all.ravel()[:need].reshape(N, NB).astype(np.float32)
    flat_c = codes_all.reshape(-1, BLK_C)[:need].reshape(N, NB, BLK_C)
    print(f"{o.tensor}: using {need} real q4_1 blocks -> {N} rows x {NB} blocks, K={K}")
    print(f"  d in [{flat_d.min():.5g}, {flat_d.max():.5g}]  "
          f"m in [{flat_m.min():.5g}, {flat_m.max():.5g}]  "
          f"codes in [{flat_c.min()}, {flat_c.max()}]")

    rng = np.random.default_rng(0)
    act = rng.standard_normal(K).astype(np.float32)
    # Round the activation to bf16 first so the reference sees exactly the
    # values the device will see -- otherwise the comparison measures the host's
    # rounding, not the kernel.
    act = q4nx.bf16_to_f32(q4nx.f32_to_bf16(act))

    def by_tile(fn):
        return np.concatenate([
            fn(act, flat_d[i:i + NROWS], flat_m[i:i + NROWS], flat_c[i:i + NROWS])
            for i in range(0, N, NROWS)
        ])

    expected = by_tile(q4nx.gemv_reference_bf16)   # the correctness gate
    exact = by_tile(q4nx.gemv_reference)           # float64, for context only

    wtile_bytes = tile_bytes(K, NROWS)
    ntiles = N // NROWS
    wbuf = np.concatenate([
        q4nx.pack_tile(flat_d[i:i + NROWS], flat_m[i:i + NROWS], flat_c[i:i + NROWS])
        for i in range(0, N, NROWS)
    ])
    assert wbuf.size == ntiles * wtile_bytes, (wbuf.size, ntiles * wtile_bytes)

    a_t = iron.tensor(act.astype(bfloat16), dtype=bfloat16, device="npu")
    w_t = iron.tensor(wbuf, dtype=np.uint8, device="npu")
    o_t = iron.zeros(N, dtype=np.float32, device="npu")

    flm_gemv(a_t, w_t, o_t, K=K, N=N, NROWS=NROWS)
    got = o_t.numpy().astype(np.float64)

    err = np.abs(got - expected)
    scale = np.abs(expected).mean()
    fmt_err = np.abs(expected - exact)
    print(f"\n{'row':>5s} {'device':>14s} {'bf16 ref':>14s} {'exact f64':>14s} "
          f"{'abs err':>11s}")
    for i in range(min(N, 8)):
        print(f"{i:5d} {got[i]:14.6f} {expected[i]:14.6f} {exact[i]:14.6f} "
              f"{err[i]:11.3e}")
    print(f"\nvs bf16 reference : max {err.max():.4e}  mean {err.mean():.4e}")
    print(f"vs exact float64  : max {np.abs(got - exact).max():.4e}   "
          f"(the format's own cost is {fmt_err.max():.4e}, "
          f"{fmt_err.max()/scale:.2%} of |out|)")

    # Against the bf16-faithful reference the only remaining difference is the
    # order of the float accumulation, so the tolerance is tight. A body with a
    # real defect misses this by orders of magnitude -- the truncating rounding
    # mode and the dropped zero-point terms both showed up as ~100% errors.
    tol = 1e-2 * scale
    ok = err.max() <= tol
    print(f"tolerance {tol:.4e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
