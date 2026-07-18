#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase D — GQA head-batched attention: 4 q-heads per dispatch (32 -> 8 /layer).

Attention is per-query independent given K/V, and GQA means `groups = n_q/n_kv`
q-heads SHARE one kv-head's K/V. So the queries of those `groups` heads can be
STACKED into a single `q_len = groups*block` call against that kv-head's KV —
mathematically identical, and it needs NO kernel change (the kernel already takes
q_len as a compile-time constant). That collapses attention from 32 dispatches per
layer (one per q-head) to 8 (one per kv-head).

Tile budget at groups=4, block=16, kv_len=48, head_dim=128 (bf16):
  Q 16 KB + KV[K|V] 24 KB + O 16 KB = 56 KB of the 64 KB tile. Fits.

Validates the stacked result against the Phase-A golden l0 attention, per q-head.

Usage:
  python tools/npu/test_dflash_attention_batched_npu.py --golden-dir <ref>/rust
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

HEAD_DIM = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", type=Path, required=True)
    ap.add_argument("--kv-heads-tested", type=int, default=0, help="0 = all kv-heads")
    args = ap.parse_args()

    g = args.golden_dir
    q = np.load(g / "rust_l0_q_roped.npy")
    k = np.load(g / "rust_l0_k_roped.npy")
    v = np.load(g / "rust_l0_v.npy")
    ref = np.load(g / "rust_l0_attn_out.npy")
    block, tot = q.shape[0], k.shape[0]
    n_q = q.shape[1] // HEAD_DIM
    n_kv = k.shape[1] // HEAD_DIM
    q = q.reshape(block, n_q, HEAD_DIM)
    k = k.reshape(tot, n_kv, HEAD_DIM)
    v = v.reshape(tot, n_kv, HEAD_DIM)
    ref = ref.reshape(block, n_q, HEAD_DIM)
    groups = n_q // n_kv
    q_len = groups * block

    n_kvh = n_kv if args.kv_heads_tested == 0 else min(args.kv_heads_tested, n_kv)
    print(f"[attn_batched] block={block} tot={tot} n_q={n_q} n_kv={n_kv} groups={groups} "
          f"-> q_len={q_len} per dispatch; kv_heads_tested={n_kvh} "
          f"(dispatches/layer: {n_q} -> {n_kv})")

    from build_dflash_attention_sc import run_attn_head
    from ml_dtypes import bfloat16

    def bf16(x):
        return x.astype(bfloat16).astype(np.float32)

    def attn_ref(qh, kh, vh):
        s = (qh @ kh.T) * (HEAD_DIM ** -0.5)
        w = np.exp(s - s.max(1, keepdims=True))
        return (w / w.sum(1, keepdims=True)) @ vh

    all_ok = True
    worst = 1.0
    for kvh in range(n_kvh):
        heads = list(range(kvh * groups, (kvh + 1) * groups))
        # stack the group's q-heads: [groups*block, 128]
        Qs = np.concatenate([q[:, h, :] for h in heads], axis=0)
        out = run_attn_head(Qs, k[:, kvh, :], v[:, kvh, :], q_len, tot)  # [q_len,128]
        for i, h in enumerate(heads):
            npu_h = out[i * block:(i + 1) * block, :]
            gh = ref[:, h, :]
            bref = bf16(attn_ref(bf16(q[:, h, :]), bf16(k[:, kvh, :]), bf16(v[:, kvh, :])))
            cb = float(npu_h.reshape(-1) @ bref.reshape(-1) /
                       (np.linalg.norm(npu_h) * np.linalg.norm(bref) + 1e-30))
            cg = float(npu_h.reshape(-1) @ gh.reshape(-1) /
                       (np.linalg.norm(npu_h) * np.linalg.norm(gh) + 1e-30))
            ok = cb > 0.999
            all_ok &= ok
            worst = min(worst, cb)
            if kvh == 0 or not ok:
                print(f"  kv{kvh} q-head {h:2d}: cos_bf16ref={cb:.5f} [golden cos={cg:.5f}] "
                      f"[{'PASS' if ok else 'FAIL'}]")
    print(f"[attn_batched] worst cos_bf16ref={worst:.5f}  {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
