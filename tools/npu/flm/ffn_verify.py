#!/usr/bin/env python3
"""The llama-3.2-1B FFN block on the NPU, end to end, on real q4_1 weights.

Phase 2 task 2: the decode GEMM is reproduced and verified, but nothing has run
a *block* yet. The FFN is the right first one — it is three of the layer's seven
linears and **31.5 MB of its 38.0 MB of weights**, and it needs no attention, no
KV cache and no RoPE:

    h -> gate_proj(h) , up_proj(h)          K=2048, N=8192   (GEMV)
      -> silu(gate) * up                    8192             (SwiGLU)
      -> down_proj(.)                       K=8192, N=2048   (GEMV)

Every linear runs `kernels/npu/flm_gemv_q4_1.cc`; the SwiGLU runs the existing
`tools/npu/silu_mul_bf16.cc`, which is already AIE2P-aware. Weights are real
q4_1 blocks from FLM's `model.q4nx`, repacked into our tile layout (FLM's own
block->(row, k) mapping is unresolved — see docs/npu/flm-reproduction-results.md
§2 — so the bytes are real and the addressing is ours).

**Scope.** This runs the block as separate dispatches with host glue between
stages, and checks the arithmetic end to end. Fusing the stages into one
dispatch — which is what FLM does, and what a tok/s number needs — is the next
step and needs the GEMV to emit bf16 and the intermediate to stay on device.

    python3 ffn_verify.py                 # full FFN block
    python3 ffn_verify.py --layer 5

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from gemv_verify import BLK_C, flm_gemv, tile_bytes  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
SILU_SRC = str(Path(__file__).resolve().parents[3] / "kernels/npu/flm_swiglu.cc")
AIE_INC = Path.home() / ".venv/lib/python3.14/site-packages/mlir_aie/include"


@iron.jit(source_files=[SILU_SRC])
def swiglu(gate: In, up: In, out: Out, *, N: CompileTime[int] = 8192,
           TILE: CompileTime[int] = 1024):
    """out[i] = silu(gate[i]) * up[i], streamed in TILE-sized chunks."""
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    kern = ExternalFunction(
        "flm_swiglu", source_file=SILU_SRC,
        arg_types=[tile_ty, tile_ty, tile_ty],
        compile_flags=[f"-DDIM_TILE={TILE}"],
        include_dirs=[str(AIE_INC)],
    )
    f_g, f_u, f_o = (ObjectFifo(tile_ty, name="g"), ObjectFifo(tile_ty, name="u"),
                     ObjectFifo(tile_ty, name="o"))

    def core(gc, uc, op, k):
        for _ in range_(N // TILE):
            eg, eu, eo = gc.acquire(1), uc.acquire(1), op.acquire(1)
            k(eg, eu, eo)
            op.release(1); uc.release(1); gc.release(1)

    w = Worker(core, fn_args=[f_g.cons(), f_u.cons(), f_o.prod(), kern],
               stack_size=4096)

    def seq(g, u, o, gh, uh, oh):
        gh.fill(g); uh.fill(u); oh.drain(o, wait=True)

    rt = Runtime(seq, [full_ty, full_ty, full_ty,
                       f_g.prod(), f_u.prod(), f_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def load_linear(c, name, N, K):
    """Real q4_1 blocks from the container, regrouped to (N, K/32)."""
    d_all, m_all, codes_all = c.blocks(name)
    nb = K // BLK_C
    need = N * nb
    avail = d_all.size
    reps = -(-need // avail)
    d = np.tile(d_all.ravel(), reps)[:need].reshape(N, nb).astype(np.float32)
    m = np.tile(m_all.ravel(), reps)[:need].reshape(N, nb).astype(np.float32)
    codes = np.tile(codes_all.reshape(-1, BLK_C), (reps, 1))[:need].reshape(N, nb, BLK_C)
    return d, m, codes


def run_gemv(act, d, m, codes, K, N, nrows):
    """One projection on the NPU; returns float32[N]."""
    wbuf = np.concatenate([
        q4nx.pack_tile(d[i:i + nrows], m[i:i + nrows], codes[i:i + nrows])
        for i in range(0, N, nrows)
    ])
    assert wbuf.size == (N // nrows) * tile_bytes(K, nrows)
    a_t = iron.tensor(act.astype(bfloat16), dtype=bfloat16, device="npu")
    w_t = iron.tensor(wbuf, dtype=np.uint8, device="npu")
    o_t = iron.zeros(N, dtype=np.float32, device="npu")
    flm_gemv(a_t, w_t, o_t, K=K, N=N, NROWS=nrows)
    return o_t.numpy().astype(np.float64)


def ref_gemv(act, d, m, codes, N, nrows):
    return np.concatenate([
        q4nx.gemv_reference_bf16(act, d[i:i + nrows], m[i:i + nrows],
                                 codes[i:i + nrows])
        for i in range(0, N, nrows)
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=8192)
    o = p.parse_args()

    H, I = o.hidden, o.inter
    c = q4nx.Q4nx(str(Q4NX))
    pre = f"model.layers.{o.layer}."
    print(f"llama-3.2-1B layer {o.layer} FFN: hidden={H} intermediate={I}")
    print(f"  weights streamed: gate {I*H*5/8/1e6:.1f} MB + up {I*H*5/8/1e6:.1f} MB"
          f" + down {H*I*5/8/1e6:.1f} MB = {3*I*H*5/8/1e6:.1f} MB")

    rng = np.random.default_rng(0)
    h = q4nx.bf16_to_f32(q4nx.f32_to_bf16(
        (rng.standard_normal(H) * 0.05).astype(np.float32)))

    # --- gate / up : K=hidden, N=intermediate -------------------------------
    NR1 = 16
    gd, gm, gc_ = load_linear(c, pre + "mlp.gate_proj.weight", I, H)
    ud, um, uc_ = load_linear(c, pre + "mlp.up_proj.weight", I, H)
    gate = run_gemv(h, gd, gm, gc_, H, I, NR1)
    up = run_gemv(h, ud, um, uc_, H, I, NR1)
    gate_ref = ref_gemv(h, gd, gm, gc_, I, NR1)
    up_ref = ref_gemv(h, ud, um, uc_, I, NR1)
    print(f"\n  gate_proj  max|err| {np.abs(gate - gate_ref).max():.3e}")
    print(f"  up_proj    max|err| {np.abs(up - up_ref).max():.3e}")

    # --- SwiGLU on device ---------------------------------------------------
    g_bf = q4nx.f32_to_bf16(gate.astype(np.float32))
    u_bf = q4nx.f32_to_bf16(up.astype(np.float32))
    g_t = iron.tensor(q4nx.bf16_to_f32(g_bf).astype(bfloat16), dtype=bfloat16, device="npu")
    u_t = iron.tensor(q4nx.bf16_to_f32(u_bf).astype(bfloat16), dtype=bfloat16, device="npu")
    s_t = iron.zeros(I, dtype=bfloat16, device="npu")
    swiglu(g_t, u_t, s_t, N=I, TILE=1024)
    act2 = s_t.numpy().astype(np.float32)

    gf, uf = q4nx.bf16_to_f32(g_bf), q4nx.bf16_to_f32(u_bf)
    sw_ref = (gf / (1.0 + np.exp(-gf.astype(np.float64)))) * uf
    sw_err = np.abs(act2.astype(np.float64) - sw_ref).max()
    print(f"  swiglu     max|err| {sw_err:.3e}  "
          f"(rel {sw_err/np.abs(sw_ref).mean():.2%} of mean |out|)")

    # --- down : K=intermediate, N=hidden ------------------------------------
    # NROWS=2. At K=8192 the activation alone is 16384 B (32768 double-buffered),
    # so the weight tile budget is much tighter than at K=2048: 4 rows is 20480 B
    # (40960 double-buffered) and overflows the 64 KB tile memory at 73728 B.
    # 2 rows is 10240 B -> 20480 + 32768 = 53248 B, which fits.
    NR2 = 2
    dd, dm, dc_ = load_linear(c, pre + "mlp.down_proj.weight", H, I)
    act2 = q4nx.bf16_to_f32(q4nx.f32_to_bf16(act2))
    out = run_gemv(act2, dd, dm, dc_, I, H, NR2)
    out_ref = ref_gemv(act2, dd, dm, dc_, H, NR2)
    err = np.abs(out - out_ref)
    print(f"  down_proj  max|err| {err.max():.3e}")

    scale = np.abs(out_ref).mean()
    ok = err.max() <= 1e-2 * scale
    print(f"\nFFN block output: mean|out| {scale:.5f}  max|err| {err.max():.3e}  "
          f"tol {1e-2*scale:.3e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
