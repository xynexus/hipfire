#!/usr/bin/env python3
# Lever 5 gate (shared by the low-rank track, Levers 3-4-5): does static rank-r
# low-rank hold at head_dim=256 on qwen3.5? Per kv-head, stack all captured cold
# tokens into an [N x 256] matrix, SVD, and report rank-r reconstruction relative
# Frobenius error + energy fraction for r in {32,64,128,192}.
#
# Expectation from the literature: post-RoPE K is NOT low-rank (RoPE spreads energy —
# why ReCalKV needs head-grouping/full-K reconstruction); V (no RoPE) is more low-rank.
#
# Capture format (HIPFIRE_KV_CAPTURE_{K,V}), repeated: [u32 base][u32 mb][u32 nkv][u32 HD][f32 mb*nkv*HD]
# Usage: lowrank_feasibility.py <kcap.bin> <vcap.bin>
import sys, struct
import numpy as np

HD = 256

def load_per_head(path):
    b = open(path, "rb").read()
    off = 0
    per_head = None
    while off < len(b):
        base, mb, nkv, hd = struct.unpack_from("<IIII", b, off); off += 16
        n = mb * nkv * hd
        arr = np.frombuffer(b, dtype="<f4", count=n, offset=off).reshape(mb, nkv, hd)
        off += n * 4
        if per_head is None:
            per_head = [[] for _ in range(nkv)]
        for kv in range(nkv):
            per_head[kv].append(arr[:, kv, :].astype(np.float64))
    return [np.concatenate(rows, axis=0) for rows in per_head]  # list of [N x HD]

def rank_r_report(name, mats, ranks=(32, 64, 128, 192)):
    # Precompute SVD once per head.
    svds = [np.linalg.svd(M, full_matrices=False) for M in mats]
    print(f"\n=== {name}: rank-r reconstruction (avg over {len(mats)} kv-heads, HD={HD}) ===")
    print("  rank   rel-Frob-err   energy-kept")
    err_at = {}
    for r in ranks:
        errs, engs = [], []
        for M, (U, S, Vt) in zip(mats, svds):
            Sr = S.copy(); Sr[r:] = 0.0
            Mr = (U * Sr) @ Vt
            errs.append(np.linalg.norm(M - Mr) / max(np.linalg.norm(M), 1e-12))
            engs.append((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-12))
        err_at[r] = float(np.mean(errs))
        print(f"  {r:4d}   {err_at[r]:9.4f}      {np.mean(engs):8.4f}")
    v = err_at.get(64, 1.0)
    print(f"  --> rank-64 rel-err = {v:.4f}  ({'LOW-RANK HOLDS (<0.10)' if v < 0.10 else 'NOT low-rank at r=64'})")

if __name__ == "__main__":
    kmats = load_per_head(sys.argv[1])
    vmats = load_per_head(sys.argv[2])
    print(f"K: {len(kmats)} heads, {kmats[0].shape[0]} tokens each; V likewise.")
    rank_r_report("K (post-RoPE)", kmats)
    rank_r_report("V (no RoPE)", vmats)
