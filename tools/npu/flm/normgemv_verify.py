#!/usr/bin/env python3
"""RMSNorm fused into the GEMV prologue — verified against numpy.

At the measured **92.9 us per dispatch**, a layer's RMSNorm moves 4 KB and is
~99.9% fixed cost. Two per layer over 16 layers is **2.97 ms/token**, a fifth of
the 13.55 ms streaming floor, for 128 KB of data. So it must ride along with a
large operator rather than take a dispatch.

`kernels/npu/flm_norm_prepare.cc` replaces `flm_asum_prepare`: it normalises the
activation **in place** in the ObjectFifo buffer and computes the q4_1 block sums
of the normalised activation in the same sweep. The GEMV that follows is
unchanged — it reads the same pointer.

Passes over the activation: standalone RMSNorm (2) + asum_prepare (1) = 3;
fused = 2. So it is one pass cheaper than the two operators were separately, on
top of saving the dispatch.

    python3 normgemv_verify.py
    python3 normgemv_verify.py --n 512 --layer 5

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from gemv_verify import BLK_C, tile_bytes  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
GEMV_SRC = str(KDIR / "flm_gemv_q4_1.cc")
NORM_SRC = str(KDIR / "flm_norm_prepare.cc")
Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
EPS = 1e-5


@iron.jit(source_files=[GEMV_SRC, NORM_SRC])
def norm_gemv(actnw: In, w: In, out: Out, *, K: CompileTime[int] = 2048,
              N: CompileTime[int] = 512, NROWS: CompileTime[int] = 16):
    wtile = tile_bytes(K, NROWS)
    ntiles = N // NROWS
    # One buffer carries [activation K][norm weight K]: a separate fifo for the
    # norm weight would be a third DMA input and a core tile has only two.
    act_ty = np.ndarray[(2 * K,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(wtile,), np.dtype[np.uint8]]
    o_ty = np.ndarray[(NROWS,), np.dtype[np.float32]]
    w_all_ty = np.ndarray[(ntiles * wtile,), np.dtype[np.uint8]]
    o_all_ty = np.ndarray[(N,), np.dtype[np.float32]]

    flags = [f"-DDIM_K={K}", f"-DDIM_NROWS={NROWS}"]
    kern = ExternalFunction("flm_gemv_q4_1", source_file=GEMV_SRC,
                            arg_types=[act_ty, wt_ty, o_ty], compile_flags=flags)
    # Replaces flm_asum_prepare: normalises in place AND fills g_asum.
    prep = ExternalFunction("flm_norm_prepare", source_file=NORM_SRC,
                            arg_types=[act_ty], compile_flags=flags)

    f_act = ObjectFifo(act_ty, depth=1, name="act")
    f_w = ObjectFifo(wt_ty, name="wt")
    f_o = ObjectFifo(o_ty, name="out")

    def core(ac, wc, op, k, kp):
        ea = ac.acquire(1)
        kp(ea)                          # normalise in place + block sums
        for _ in range_(ntiles):
            ew, eo = wc.acquire(1), op.acquire(1)
            k(ea, ew, eo)
            op.release(1); wc.release(1)
        ac.release(1)

    worker = Worker(core, fn_args=[f_act.cons(), f_w.cons(),
                                   f_o.prod(), kern, prep], stack_size=4096)

    def seq(a, wb, ob, ah, wh, oh):
        ah.fill(a); wh.fill(wb); oh.drain(ob, wait=True)

    rt = Runtime(seq, [act_ty, w_all_ty, o_all_ty,
                       f_act.prod(), f_w.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--nrows", type=int, default=16)
    p.add_argument("--layer", type=int, default=0)
    o = p.parse_args()
    K, N, NROWS = o.k, o.n, o.nrows
    nb = K // BLK_C

    c = q4nx.Q4nx(str(Q4NX))
    nw = c.bf16(f"model.layers.{o.layer}.input_layernorm.weight").astype(np.float32)[:K]
    d_all, m_all, codes_all = c.blocks(f"model.layers.{o.layer}.self_attn.q_proj.weight")
    need = N * nb
    d = d_all.ravel()[:need].reshape(N, nb).astype(np.float32)
    m = m_all.ravel()[:need].reshape(N, nb).astype(np.float32)
    codes = codes_all.reshape(-1, BLK_C)[:need].reshape(N, nb, BLK_C)

    rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))
    rng = np.random.default_rng(0)
    x = rnd(rng.standard_normal(K) * 0.05)

    wbuf = np.concatenate([q4nx.pack_tile(d[i:i + NROWS], m[i:i + NROWS],
                                          codes[i:i + NROWS])
                           for i in range(0, N, NROWS)])
    a_t = iron.tensor(np.concatenate([x, nw]).astype(bfloat16),
                      dtype=bfloat16, device="npu")
    w_t = iron.tensor(wbuf, dtype=np.uint8, device="npu")
    o_t = iron.zeros(N, dtype=np.float32, device="npu")
    norm_gemv(a_t, w_t, o_t, K=K, N=N, NROWS=NROWS)
    got = o_t.numpy().astype(np.float64)

    # Reference: the same two steps the kernel now does in one pass, with the
    # kernel's own bf16 roundings, so this is the correctness gate.
    xd = x.astype(np.float64)
    inv = np.float32(1.0 / np.sqrt((xd * xd).mean() + EPS))
    xn = rnd(rnd(x * rnd(inv)) * nw)
    ref = np.concatenate([
        q4nx.gemv_reference_bf16(xn, d[i:i + NROWS], m[i:i + NROWS],
                                 codes[i:i + NROWS])
        for i in range(0, N, NROWS)])

    err = np.abs(got - ref)
    scale = np.abs(ref).mean()
    print(f"norm+GEMV fused: K={K} N={N} rows/tile={NROWS}, layer {o.layer}")
    print(f"  norm weight range [{nw.min():.4f}, {nw.max():.4f}]")
    print(f"  vs norm-then-GEMV reference: max {err.max():.4e}  mean {err.mean():.4e}")
    print(f"  mean|ref| {scale:.5f}")
    ok = err.max() <= 1e-2 * scale
    print(f"  tolerance {1e-2*scale:.4e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
