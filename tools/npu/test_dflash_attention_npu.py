#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase C: DFlash non-causal cross-attention parity on the NPU (single-core).

Validates `dflash_attention_sc_bf16.cc` (built by build_dflash_attention_sc.py)
against the Phase-A golden l0 attention tensors: for each q-head, runs the
non-causal cross-attention (GQA: kv-head = q_head // (n_q/n_kv)) on the NPU and
compares to `rust_l0_attn_out`. The attention MATH already matches the golden at
cos=1.0 (numpy, --algo-only); this checks the on-device bf16 kernel.

Golden tensors come from `dflash_ref_dump` with HIPFIRE_DFLASH_GOLDEN_DIR set:
  rust_l0_q_roped [block, n_q*128], rust_l0_k_roped/v [ctx+block, n_kv*128],
  rust_l0_attn_out [block, n_q*128].

Env (nix1): PEANO_INSTALL_DIR + PYTHONPATH=~/mlir-aie-312/install/python,
  run with ~/mlir-aie-312/venv312/bin/python.

Usage:
  python tools/npu/test_dflash_attention_npu.py --golden-dir <ref_dir>/rust [--heads 1]
  python tools/npu/test_dflash_attention_npu.py --golden-dir <dir> --algo-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

HEAD_DIM = 128


def load_golden(gdir):
    q = np.load(gdir / "rust_l0_q_roped.npy")
    k = np.load(gdir / "rust_l0_k_roped.npy")
    v = np.load(gdir / "rust_l0_v.npy")
    ref = np.load(gdir / "rust_l0_attn_out.npy")
    block = q.shape[0]
    tot = k.shape[0]
    n_q = q.shape[1] // HEAD_DIM
    n_kv = k.shape[1] // HEAD_DIM
    q = q.reshape(block, n_q, HEAD_DIM)
    k = k.reshape(tot, n_kv, HEAD_DIM)
    v = v.reshape(tot, n_kv, HEAD_DIM)
    ref = ref.reshape(block, n_q, HEAD_DIM)
    return q, k, v, ref, block, tot, n_q, n_kv


def attn_ref(qh, kh, vh):
    scale = HEAD_DIM ** -0.5
    scores = (qh @ kh.T) * scale
    w = np.exp(scores - scores.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return w @ vh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", type=Path, required=True)
    ap.add_argument("--heads", type=int, default=0, help="0 = all q-heads")
    ap.add_argument("--algo-only", action="store_true", help="numpy algorithm check, no NPU")
    args = ap.parse_args()

    q, k, v, ref, block, tot, n_q, n_kv = load_golden(args.golden_dir)
    groups = n_q // n_kv
    n_heads = n_q if args.heads == 0 else min(args.heads, n_q)
    print(f"[test_dflash_attention] block={block} ctx+block={tot} n_q={n_q} n_kv={n_kv} "
          f"heads_tested={n_heads} {'(algo-only)' if args.algo_only else '(NPU)'}")

    if args.algo_only:
        out = np.stack([attn_ref(q[:, h, :], k[:, h // groups, :], v[:, h // groups, :])
                        for h in range(n_q)], axis=1)
        d = np.abs(out - ref)
        cos = float(out.reshape(-1) @ ref.reshape(-1) / (np.linalg.norm(out) * np.linalg.norm(ref)))
        print(f"  numpy algorithm vs golden: max_abs={d.max():.3e} cos={cos:.6f}")
        print("PASS" if cos > 0.999 else "FAIL")
        return 0 if cos > 0.999 else 1

    from build_dflash_attention_sc import run_attn_head
    from ml_dtypes import bfloat16

    def bf16(x):
        return x.astype(bfloat16).astype(np.float32)

    # The kernel computes bf16 attention. Gate it against a bf16-INPUT reference
    # (inputs cast to bf16, fp32 math, output cast to bf16) — this isolates kernel
    # correctness from the bf16-vs-f16 precision gap of the golden. A fixed
    # absolute max_abs vs the f16 golden sits BELOW the bf16 arithmetic floor for
    # large-magnitude heads (|attn_out| reaches ~39, and bf16's ~0.4% relative
    # precision alone forces ~0.2 abs there — an ideal bf16 kernel scores the same).
    # cos vs golden is reported too (informational).
    all_ok = True
    worst_g = 0.0
    worst_bf16_cos = 1.0
    for h in range(n_heads):
        kvh = h // groups
        npu = run_attn_head(q[:, h, :], k[:, kvh, :], v[:, kvh, :], block, tot)
        g = ref[:, h, :]
        # bf16-input reference for this head
        bref = attn_ref(bf16(q[:, h, :]), bf16(k[:, kvh, :]), bf16(v[:, kvh, :]))
        bref = bf16(bref)
        cb = float(npu.reshape(-1) @ bref.reshape(-1) /
                   (np.linalg.norm(npu) * np.linalg.norm(bref) + 1e-30))
        dg = np.abs(npu - g)
        cg = float(npu.reshape(-1) @ g.reshape(-1) /
                   (np.linalg.norm(npu) * np.linalg.norm(g) + 1e-30))
        ok = bool(cb > 0.999)  # kernel correctness gate (bf16-aware)
        all_ok &= ok
        worst_g = max(worst_g, dg.max())
        worst_bf16_cos = min(worst_bf16_cos, cb)
        if h < 4 or not ok:
            print(f"  head {h:2d} (kv {kvh}): cos_bf16ref={cb:.5f}  "
                  f"[golden cos={cg:.5f} max_abs={dg.max():.3e}]  [{'PASS' if ok else 'FAIL'}]")
    print(f"[test_dflash_attention] worst cos_bf16ref={worst_bf16_cos:.5f} "
          f"(golden worst max_abs={worst_g:.3e})  {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
