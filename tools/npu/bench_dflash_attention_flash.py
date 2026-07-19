#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Bench + numerics for the flash-style (streaming-KV, mmul) DFlash attention.

Same gates and reference as `bench_dflash_attention_mc.py`: cosine against both
an f32 numpy golden and a bf16-input precision-matched reference. Reports
ms/dispatch and GFLOP/s so the result is directly comparable to the sc kernel's
0.97 GFLOP/s baseline.

Unlike the sc kernel this one has no `tot <= 55` cap, so `--tot-sweep` can walk
past it.

  source tools/npu/npuenv.sh
  npupy tools/npu/bench_dflash_attention_flash.py --tot-sweep 48,64,128,528
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
    q = q.astype(dtype).astype(np.float32)
    k = k.astype(dtype).astype(np.float32)
    v = v.astype(dtype).astype(np.float32)
    b, nh, hd = q.shape
    scale = 1.0 / np.sqrt(hd)
    out = np.empty((b, nh * hd), np.float32)
    for h in range(nh):
        kvh = h // groups
        s = (q[:, h, :] @ k[:, kvh, :].T) * scale
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=1, keepdims=True)
        out[:, h * hd:(h + 1) * hd] = p @ v[:, kvh, :]
    return out


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--n-q", type=int, default=32)
    ap.add_argument("--n-kv", type=int, default=8)
    ap.add_argument("--q-len", type=int, default=16)
    ap.add_argument("--kv-tile", type=int, default=16)
    ap.add_argument("--kv-depth", type=int, default=2)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tot-sweep", default="48",
                    help="comma-separated tot (= ctx + block) values")
    args = ap.parse_args()

    # Import order matters: build_* -> oq_gemm_design runs the pyxrt bootstrap,
    # which must happen BEFORE aie.utils is imported.
    import build_dflash_attention_flash as F
    import aie.iron as iron
    from ml_dtypes import bfloat16

    block, nh, nkv = args.block, args.n_q, args.n_kv
    groups = nh // nkv
    print(f"[flash] block={block} n_q={nh} n_kv={nkv} groups={groups} "
          f"q_len={args.q_len} kv_tile={args.kv_tile} kv_depth={args.kv_depth} "
          f"cores={args.cores}")

    for tot_s in args.tot_sweep.split(","):
        tot = int(tot_s)
        if tot % args.kv_tile:
            print(f"[tot={tot}] SKIP: kv_tile={args.kv_tile} does not divide tot")
            continue
        rng = np.random.default_rng(0)
        q = rng.standard_normal((block, nh, HEAD_DIM), np.float32) * 0.5
        k = rng.standard_normal((tot, nkv, HEAD_DIM), np.float32) * 0.5
        v = rng.standard_normal((tot, nkv, HEAD_DIM), np.float32) * 0.5
        ref_f32 = ref_attention(q, k, v, groups, np.float32)
        ref_bf16 = ref_attention(q, k, v, groups, bfloat16)

        qpk, kvpk, n_iters, n_tiles, hpi = F.pack_inputs(
            q, k, v, groups, args.q_len, args.kv_tile)
        Qt = iron.tensor(qpk.astype(bfloat16), dtype=bfloat16, device="npu")
        KVt = iron.tensor(kvpk.astype(bfloat16), dtype=bfloat16, device="npu")
        Ot = iron.zeros(n_iters * args.q_len * HEAD_DIM, dtype=bfloat16, device="npu")
        kw = dict(q_len=args.q_len, kv_tile=args.kv_tile, n_tiles=n_tiles,
                  n_iters=n_iters, n_cores=args.cores, kv_depth=args.kv_depth)
        try:
            F.dflash_attn_flash_mc(Qt, KVt, Ot, **kw)
        except Exception as e:  # noqa: BLE001 — report the emitting stage verbatim
            print(f"[tot={tot}] FAILED: {type(e).__name__}: {e}")
            continue
        out = F.unpack_output(Ot.numpy(), block, nh, args.q_len, hpi)
        c32, cbf = cos(out, ref_f32), cos(out, ref_bf16)

        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            F.dflash_attn_flash_mc(Qt, KVt, Ot, **kw)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        # 2 GEMMs (QKᵀ and PV), 2 FLOP per MAC, over all q-heads.
        flop = 2 * 2 * block * tot * HEAD_DIM * nh
        gflops = flop / (ts.mean() * 1e-3) / 1e9
        print(f"[tot={tot:4d}] n_tiles={n_tiles:3d} {ts.mean():8.3f} ms  "
              f"min {ts.min():8.3f}  max {ts.max():8.3f}  "
              f"{ts.mean() / tot:.4f} ms/row  {gflops:6.2f} GFLOP/s")
        print(f"[tot={tot:4d}] cos vs f32-golden = {c32:.6f}   "
              f"cos vs bf16-ref = {cbf:.6f}")


if __name__ == "__main__":
    main()
