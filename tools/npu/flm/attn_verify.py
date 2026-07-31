#!/usr/bin/env python3
"""Decode attention over a KV cache on the NPU, checked against numpy.

`attn.xclbin` was never rebuilt — this is the missing operator. It runs
`kernels/npu/flm_attn_decode.cc`: one GQA group per core (llama-3.2-1B has 32
query heads over 8 KV heads, ratio 4, head_dim 64), online softmax so the cache
streams once, and the two KV operands in the orientations the reverse
engineering found:

    K  channel-major  [HEAD][TSEQ]  -> scores accumulate across d, no reduce
    V  position-major [TSEQ][HEAD]  -> output accumulates across t

The 1/sqrt(head_dim) and log2(e) factors are folded into Q on the host so the
softmax exponential is the hardware `exp2` with no pre-multiply.

    python3 attn_verify.py                  # seq=512
    python3 attn_verify.py --seq 2048

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import math
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

K = Path(__file__).resolve().parents[3] / "kernels/npu"
SRC = str(K / "flm_attn_decode.cc")
# One ExternalFunction per source file: IRON compiles each separately, so
# several entry points in one file link several times (duplicate symbol).
BEGIN_SRC = str(K / "flm_attn_begin.cc")
FIN_SRC = str(K / "flm_attn_finish.cc")
GQA, HEAD, TSEQ = 4, 64, 32


@iron.jit(source_files=[SRC, BEGIN_SRC, FIN_SRC])
def attn(q: In, kv: In, out: Out, *, SEQ: CompileTime[int] = 512):
    """One GQA group. K and V tiles are interleaved in one stream so the cache
    arrives as a single sequential read, which is what it is in memory."""
    ntiles = SEQ // TSEQ
    q_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]
    # one tile of K (channel-major) immediately followed by one tile of V
    kv_tile_ty = np.ndarray[(2 * TSEQ * HEAD,), np.dtype[bfloat16]]
    kv_all_ty = np.ndarray[(ntiles * 2 * TSEQ * HEAD,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(GQA * HEAD,), np.dtype[bfloat16]]

    flags = [f"-DDIM_GQA={GQA}", f"-DDIM_HEAD={HEAD}", f"-DDIM_TSEQ={TSEQ}"]
    k_begin = ExternalFunction("flm_attn_begin", source_file=BEGIN_SRC,
                               arg_types=[], compile_flags=flags)
    k_tile = ExternalFunction("flm_attn_tile", source_file=SRC,
                              arg_types=[q_ty, kv_tile_ty],
                              compile_flags=flags)
    k_fin = ExternalFunction("flm_attn_finish", source_file=FIN_SRC,
                             arg_types=[o_ty], compile_flags=flags)

    f_q = ObjectFifo(q_ty, name="q")
    f_kv = ObjectFifo(kv_tile_ty, name="kv")
    f_o = ObjectFifo(o_ty, name="o")

    def core(qc, kvc, op, kb, kt, kf):
        eq = qc.acquire(1)
        kb()
        for _ in range_(ntiles):
            ekv = kvc.acquire(1)
            # one object = [K tile channel-major][V tile position-major];
            # the kernel indexes both halves from the single base
            kt(eq, ekv)
            kvc.release(1)
        eo = op.acquire(1)
        kf(eo)
        op.release(1)
        qc.release(1)

    w = Worker(core, fn_args=[f_q.cons(), f_kv.cons(), f_o.prod(),
                              k_begin, k_tile, k_fin], stack_size=4096)

    def seq_fn(qb, kvb, ob, qh, kvh, oh):
        qh.fill(qb)
        kvh.fill(kvb)
        oh.drain(ob, wait=True)

    rt = Runtime(seq_fn, [q_ty, kv_all_ty, o_ty,
                          f_q.prod(), f_kv.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=int, default=512, help="KV cache length")
    o = p.parse_args()
    SEQ = o.seq
    if SEQ % TSEQ:
        raise SystemExit(f"--seq must divide by {TSEQ}")

    rnd = lambda x: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(x, np.float32)))
    rng = np.random.default_rng(0)
    q = rnd(rng.standard_normal((GQA, HEAD)) * 0.3)
    K = rnd(rng.standard_normal((SEQ, HEAD)) * 0.3)
    V = rnd(rng.standard_normal((SEQ, HEAD)) * 0.3)

    print(f"decode attention: GQA={GQA} head_dim={HEAD} seq={SEQ} "
          f"(KV streamed {2*SEQ*HEAD*2/1e6:.2f} MB)")

    # Fold 1/sqrt(head_dim) and log2(e) into Q so the kernel's exp2 is exact.
    qs = rnd(q * (1.0 / math.sqrt(HEAD)) * math.log2(math.e))

    # Interleave [K tile channel-major][V tile position-major] per tile.
    buf = np.empty((SEQ // TSEQ, 2, TSEQ * HEAD), np.float32)
    for t in range(SEQ // TSEQ):
        kt = K[t * TSEQ:(t + 1) * TSEQ]           # [TSEQ][HEAD]
        buf[t, 0] = kt.T.reshape(-1)              # -> [HEAD][TSEQ]
        buf[t, 1] = V[t * TSEQ:(t + 1) * TSEQ].reshape(-1)
    kv_flat = buf.reshape(-1)

    q_t = iron.tensor(qs.reshape(-1).astype(bfloat16), dtype=bfloat16, device="npu")
    kv_t = iron.tensor(kv_flat.astype(bfloat16), dtype=bfloat16, device="npu")
    o_t = iron.zeros(GQA * HEAD, dtype=bfloat16, device="npu")
    attn(q_t, kv_t, o_t, SEQ=SEQ)
    got = o_t.numpy().astype(np.float64).reshape(GQA, HEAD)

    # reference: plain softmax attention in float64
    scores = (q.astype(np.float64) @ K.astype(np.float64).T) / math.sqrt(HEAD)
    pmax = scores.max(1, keepdims=True)
    e = np.exp(scores - pmax)
    ref = (e / e.sum(1, keepdims=True)) @ V.astype(np.float64)

    err = np.abs(got - ref)
    scale = np.abs(ref).mean()
    print(f"\n{'head':>5s} {'device[0]':>13s} {'ref[0]':>13s} {'max err':>11s}")
    for h in range(GQA):
        print(f"{h:5d} {got[h,0]:13.6f} {ref[h,0]:13.6f} {np.abs(got[h]-ref[h]).max():11.3e}")
    print(f"\nmax abs err {err.max():.4e}   mean {err.mean():.4e}   "
          f"mean|ref| {scale:.5f}")

    # The floor is NOT bf16 rounding — it is AIE2P's hardware exp2, measured at
    # **5.86% max / 3.54% mean** relative error over x in [-8, 0] against
    # numpy, where bf16 rounding alone would be 0.20%. The NLF unit is a coarse
    # piecewise approximation, ~18x worse than the format it returns. Softmax
    # probabilities inherit that directly, and the attention output is a
    # probability-weighted average, so it shows up undiminished. A more accurate
    # softmax needs range reduction plus a polynomial rather than the NLF —
    # note the existing silu_mul_bf16 already avoids the vector NLF on AIE2P.
    tol = 8e-2 * scale
    ok = err.max() <= tol
    print(f"tolerance {tol:.4e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
