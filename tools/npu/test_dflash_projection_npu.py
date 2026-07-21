#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase B: DFlash projection GEMM parity on the NPU (aie2p / Strix Halo).

Validates the TRUE int8 W8A8 projection at the DRAFTER's shapes — M = block_size
(16), K = hidden (4096) / intermediate (12288), N = the projection output dim —
against a numpy golden, on device. Built on the PROVEN `oq_gemm_design` int8 GEMM
(`aie::mmul<4,8,8,int8,int8,acc32>`, per-group G256 host rescale) — the same path
`test_oq_gemm_npu.py` validates bit-exact. (The mlir-aie `single_core` @iron.jit
example WEDGES this firmware — do not use it; oq_gemm's design is the good one.)

Per-group G256 is run as `ng` int8 contractions (one per 256-K group), each
rescaled by `sw[out,g]·sx[block,g]` and summed — exactly `opus_lowbit::dot_offset_fold`
/ the GPU OQ8 path. This is the per-group quality tier (best; ng launches). A
single full-K launch (per-row scale) is the efficient production tier — a follow-up
once the per-group numerics are confirmed on device.

Checks per projection shape:
  1. int32 contraction bit-exact vs numpy int64 (per group) — kernel correctness.
     This integer check (np.array_equal) is the PASS criterion and is robust.
  2. f32 rescale vs f32 reference `X @ W^T` — W8A8 SNR (int8 grid error, ~40 dB
     per projection). CAVEAT: halo's numpy 2.x mis-computes large float ops
     (`Xf@Wf.T`, big reductions) at N>=4096, so the reported SNR/cos is only
     reliable at N=1024 (k/v_proj: +40 dB on device). The rescale math is
     confirmed +40 dB on nix1 numpy 1.26 (tools/npu/dflash_int8_sim.py); the
     int32 check above is what proves the on-device kernel is correct.

Env (halo): source ~/.venv/bin/activate; export
  PYTHONPATH=~/build/mlir-aie/install/python:$PYTHONPATH; PATH=/opt/xilinx/xrt/bin:$PATH

Usage:
  python3 test_dflash_projection_npu.py                 # all drafter projections
  python3 test_dflash_projection_npu.py --proj q_proj   # one
  python3 test_dflash_projection_npu.py --block 16 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oq_gemm_design as design  # noqa: E402  (sets up pyxrt / LD_LIBRARY_PATH)

GROUP = 256

# Drafter (z-lab Qwen3.5-9B-DFlash) projection shapes: (K_in, N_out).
HIDDEN, INTER, QDIM, KVDIM, NE = 4096, 12288, 4096, 1024, 5
PROJECTIONS = {
    "q_proj": (HIDDEN, QDIM),
    "k_proj": (HIDDEN, KVDIM),
    "v_proj": (HIDDEN, KVDIM),
    "o_proj": (QDIM, HIDDEN),
    "gate_proj": (HIDDEN, INTER),
    "up_proj": (HIDDEN, INTER),
    "down_proj": (INTER, HIDDEN),
    "fc": (NE * HIDDEN, HIDDEN),
}


def quantize_group_symmetric(x_f32, bits=8):
    """Per-256-group symmetric int8. x:[rows,K] (K%256==0) -> (q int8, scale [rows,ng])."""
    qmax = (1 << (bits - 1)) - 1
    rows, K = x_f32.shape
    ng = K // GROUP
    xg = x_f32.reshape(rows, ng, GROUP)
    absmax = np.abs(xg).max(axis=2)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    q = np.round(xg / scale[:, :, None]).clip(-qmax, qmax).astype(np.int8)
    return q.reshape(rows, K), scale


def run_projection(name, K, N, block, seed):
    rng = np.random.default_rng(seed)
    # Realistic magnitudes: normed activations O(1), trained weights ~0.02 std.
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)   # [N_out, K]
    Xf = (rng.standard_normal((block, K)) * 1.0).astype(np.float32)  # [block, K]

    qw, sw = quantize_group_symmetric(Wf, 8)   # [N,K], [N,ng]
    qx, sx = quantize_group_symmetric(Xf, 8)   # [block,K], [block,ng]
    ng = K // GROUP

    # NPU: per-group int8 contraction. matmul_npu(A[M,K'],B[Nb,K']) -> C[M,Nb].
    # Use A=qw (rows=N_out), B=qx (rows=block) so C[N_out, block] per group.
    parts = np.empty((ng, N, block), dtype=np.int32)
    tile = None
    for g in range(ng):
        Wg = qw[:, g * GROUP:(g + 1) * GROUP]   # [N,256]
        Xg = qx[:, g * GROUP:(g + 1) * GROUP]   # [block,256]
        C, tile = design.matmul_npu(Wg, Xg)     # [N, block] int32
        parts[g] = C

    # 1. int32 bit-exactness vs numpy int64 (per group).
    qwg = qw.astype(np.int64).reshape(N, ng, GROUP)
    qxg = qx.astype(np.int64).reshape(block, ng, GROUP)
    int_ok = True
    for g in range(ng):
        ref = (qwg[:, g, :] @ qxg[:, g, :].T).astype(np.int64)  # [N, block]
        if not np.array_equal(parts[g].astype(np.int64), ref):
            int_ok = False
            d = np.abs(parts[g].astype(np.int64) - ref)
            print(f"    group {g}: int32 MISMATCH max|Δ|={d.max()} ({np.count_nonzero(d)} elems)")

    # 2. f32 rescale (per-group) -> Y[block, N], vs f32 reference X@W^T.
    Y = np.zeros((block, N), dtype=np.float32)
    for g in range(ng):
        P = parts[g].T.astype(np.float32)                     # [block, N]
        Y += (sx[:, g][:, None] * sw[:, g][None, :]) * P
    ref_f32 = Xf @ Wf.T                                        # [block, N]
    # Metrics via explicit np.sum on contiguous float64 — halo's numpy 2.x gives
    # wrong results from `@`/`np.linalg.norm` on these (non-contiguous, from .T)
    # arrays; nix1 numpy 1.26 does not. The kernel is proven by the int32
    # bit-exact check above; these are quality-report numbers only.
    Yd = np.ascontiguousarray(Y, dtype=np.float64)
    Rd = np.ascontiguousarray(ref_f32, dtype=np.float64)
    err = np.abs(Yd - Rd)
    num = float(np.sum((Yd - Rd) ** 2))
    den = float(np.sum(Rd ** 2))
    dot = float(np.sum(Yd * Rd))
    yy = float(np.sum(Yd * Yd))
    snr = 10 * np.log10(den / num) if num > 0 else float("inf")
    cos = dot / ((yy * den) ** 0.5) if yy > 0 and den > 0 else 0.0
    status = "PASS" if int_ok else "FAIL(int32)"
    print(f"  {name:10s} K={K:5d} N={N:5d} ng={ng:2d} tile={tile}  "
          f"int32={'exact' if int_ok else 'MISMATCH'}  "
          f"W8A8 SNR={snr:5.2f}dB cos={cos:.5f} max_abs={err.max():.2e}  [{status}]")
    return int_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", choices=list(PROJECTIONS) + ["all"], default="all")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[test_dflash_projection] block={args.block} (drafter M), per-group G256 W8A8 on NPU")
    names = list(PROJECTIONS) if args.proj == "all" else [args.proj]
    all_ok = True
    for name in names:
        K, N = PROJECTIONS[name]
        all_ok &= run_projection(name, K, N, args.block, args.seed)
    print(f"[test_dflash_projection] {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
