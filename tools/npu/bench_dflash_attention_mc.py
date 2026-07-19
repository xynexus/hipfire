#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Bench + numerics for single-core vs multi-core DFlash whole-layer attention.

Runs the real in-body shapes (q_len=groups*block, kv_len=ctx+block, n_iters=n_kv)
through `dflash_attn_all` (1 core, 8 kv-heads serial) and `dflash_attn_mc`
(n_cores workers, kv-heads split across columns). Reports per-dispatch ms over
>=3 reps with spread, and gates cosine against BOTH an f32 numpy reference and a
bf16-input precision-matched reference (per the Phase 0 numerics gate).

  source tools/npu/npuenv.sh
  npupy tools/npu/bench_dflash_attention_mc.py --reps 5 --cores 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

HEAD_DIM = 128


def ref_attention(q, k, v, groups, dtype):
    """Non-causal cross-attention reference. q [B,NH,HD], k/v [tot,NKV,HD]."""
    q = q.astype(dtype).astype(np.float32)
    k = k.astype(dtype).astype(np.float32)
    v = v.astype(dtype).astype(np.float32)
    B, NH, HD = q.shape
    scale = 1.0 / np.sqrt(HD)
    out = np.empty((B, NH * HD), np.float32)
    for h in range(NH):
        kvh = h // groups
        s = (q[:, h, :] @ k[:, kvh, :].T) * scale        # [B, tot]
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=1, keepdims=True)
        out[:, h * HD:(h + 1) * HD] = p @ v[:, kvh, :]
    return out


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=32)
    ap.add_argument("--n-q", type=int, default=32)
    ap.add_argument("--n-kv", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--variants", default="all,mc")
    args = ap.parse_args()

    # Import order matters: build_dflash_attention_sc -> oq_gemm_design runs the
    # pyxrt/libxrt_coreutil bootstrap, which must happen BEFORE aie.utils is
    # imported or DEFAULT_TENSOR_CLASS silently falls back to CPU and
    # device="npu" is rejected.
    import build_dflash_attention_sc as B
    import aie.iron as iron
    from ml_dtypes import bfloat16

    block, tot = args.block, args.ctx + args.block
    NH, NKV = args.n_q, args.n_kv
    groups = NH // NKV
    q_len = groups * block

    rng = np.random.default_rng(0)
    q = rng.standard_normal((block, NH, HEAD_DIM), np.float32) * 0.5
    k = rng.standard_normal((tot, NKV, HEAD_DIM), np.float32) * 0.5
    v = rng.standard_normal((tot, NKV, HEAD_DIM), np.float32) * 0.5

    ref_f32 = ref_attention(q, k, v, groups, np.float32)
    ref_bf16 = ref_attention(q, k, v, groups, bfloat16)

    print(f"[bench] block={block} ctx={args.ctx} tot={tot} n_q={NH} n_kv={NKV} "
          f"groups={groups} q_len={q_len} kv_len={tot} reps={args.reps}")
    print(f"[bench] f32-ref vs bf16-ref cos = {cos(ref_f32, ref_bf16):.6f} "
          "(precision floor of the bf16 kernel)")

    # Pack once — identical host layout for both variants (same driver ABI).
    qbuf, kvbuf = [], []
    for kvh in range(NKV):
        heads = range(kvh * groups, (kvh + 1) * groups)
        qbuf.append(np.concatenate([q[:, h, :] for h in heads], axis=0).reshape(-1))
        kvbuf.append(np.concatenate([k[:, kvh, :].reshape(-1), v[:, kvh, :].reshape(-1)]))
    Qt = iron.tensor(np.concatenate(qbuf).astype(bfloat16), dtype=bfloat16, device="npu")
    KVt = iron.tensor(np.concatenate(kvbuf).astype(bfloat16), dtype=bfloat16, device="npu")

    def unpack(Ot):
        o = np.array(Ot.numpy(), dtype=np.float32).reshape(NKV, q_len, HEAD_DIM)
        ctx = np.empty((block, NH * HEAD_DIM), np.float32)
        for kvh in range(NKV):
            for i, h in enumerate(range(kvh * groups, (kvh + 1) * groups)):
                ctx[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] = o[kvh, i * block:(i + 1) * block, :]
        return ctx

    for name in args.variants.split(","):
        name = name.strip()
        if name == "all":
            fn, kw, label = B.dflash_attn_all, dict(q_len=q_len, kv_len=tot, n_iters=NKV), \
                "dflash_attn_all (1 core)"
        elif name == "mc":
            fn, kw, label = B.dflash_attn_mc, dict(q_len=q_len, kv_len=tot, n_iters=NKV,
                                                   n_cores=args.cores), \
                f"dflash_attn_mc ({args.cores} cores)"
        else:
            raise SystemExit(f"unknown variant {name!r}")

        Ot = iron.zeros(NKV * q_len * HEAD_DIM, dtype=bfloat16, device="npu")
        try:
            fn(Qt, KVt, Ot, **kw)  # warm: JIT compile + first load
        except Exception as e:  # noqa: BLE001 — report the emitting stage verbatim
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")
            continue
        out = unpack(Ot)
        c_f32, c_bf16 = cos(out, ref_f32), cos(out, ref_bf16)

        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(Qt, KVt, Ot, **kw)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        print(f"[{label}] {ts.mean():7.3f} ms  min {ts.min():7.3f}  max {ts.max():7.3f}  "
              f"spread {ts.max() - ts.min():6.3f}  n={args.reps}")
        print(f"[{label}] cos vs f32-golden = {c_f32:.6f}   cos vs bf16-ref = {c_bf16:.6f}")


if __name__ == "__main__":
    main()
