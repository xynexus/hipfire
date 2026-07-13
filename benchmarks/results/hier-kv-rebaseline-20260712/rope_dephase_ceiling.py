#!/usr/bin/env python3
# RoPE-dephased-merge CEILING analysis (Phase 2, decision 1=A).
#
# Question: of the intra-merge-group variance that flat-averaging loses, how much
# is REMOVABLE RoPE phase (de-rotate to a common reference before averaging) vs
# irreducible CONTENT? Dephasing helps only the phase part, and qwen3.5 rotates
# only n_rot=64 of 256 dims (partial_rotary_factor=0.25), so the pass-through 192
# dims are pure content dephasing cannot touch.
#
# qwen3.5 text RoPE (kernels/src/rope_partial_interleaved_batched.hip):
#   interleaved: pair i rotates dims (2i, 2i+1); freq_i = 1/theta^(2i/n_rot);
#   angle = pos*freq_i;  v0'=v0*cos-v1*sin, v1'=v0*sin+v1*cos.  theta=1e7, n_rot=64.
#
# Capture format (HIPFIRE_KV_CAPTURE_K, kv_hier.rs migrate_n), repeated records:
#   [u32 base_pos][u32 mb][u32 nkv][u32 hd][f32 ck ...]  ck = [mb x nkv*hd] token-major
#
# Usage: rope_dephase_ceiling.py [capture.bin]   (no arg → run synthetic self-test only)
import sys, struct, math
import numpy as np

THETA = 1e7
N_ROT = 64
HD = 256
FOLD_M = 4  # merge group size (default HIPFIRE_KV_FOLD_M)

def rope_angles(pos):
    """Per-pair angle at absolute position `pos`. Returns [n_rot/2] angles."""
    i = np.arange(N_ROT // 2)
    freq = 1.0 / (THETA ** (2.0 * i / N_ROT))
    return pos * freq

def apply_rope(k, pos, sign=+1.0):
    """Rotate the first N_ROT dims of k by (sign * angle(pos)). sign=-1 de-rotates.
    k: [..., HD]. Interleaved pairs (2i,2i+1). Dims [N_ROT:] pass through."""
    out = k.copy()
    ang = sign * rope_angles(pos)
    c, s = np.cos(ang), np.sin(ang)
    v0 = k[..., 0:N_ROT:2]
    v1 = k[..., 1:N_ROT:2]
    out[..., 0:N_ROT:2] = v0 * c - v1 * s
    out[..., 1:N_ROT:2] = v0 * s + v1 * c
    return out

def derotate_to(k, pos, ref):
    """Move a post-RoPE k from its position `pos` to reference position `ref`:
    de-rotate by pos then re-rotate by ref = net rotate by (ref-pos)."""
    return apply_rope(k, ref - pos, sign=+1.0)

def group_variances(ks, positions):
    """ks: [m, HD] post-RoPE K at consecutive `positions`. Return
    (flat_rot_var, deph_rot_var, rot_var, total_var) summed over the group,
    measured on the ROTATED dims [0:N_ROT) for flat/deph, and rot-vs-total shares."""
    ref = positions[0]
    # Flat: variance of raw post-RoPE K around its mean.
    mean_flat = ks.mean(axis=0)
    var_total = ((ks - mean_flat) ** 2).sum()
    var_rot = ((ks[:, :N_ROT] - mean_flat[:N_ROT]) ** 2).sum()
    # Dephased: de-rotate each member to the group reference, then variance.
    deph = np.stack([derotate_to(ks[j], positions[j], ref) for j in range(len(ks))])
    mean_deph = deph.mean(axis=0)
    var_rot_deph = ((deph[:, :N_ROT] - mean_deph[:N_ROT]) ** 2).sum()
    var_rot_flat = var_rot  # rotated-dim variance before dephasing
    return var_rot_flat, var_rot_deph, var_rot, var_total

def analyze(records):
    """records: list of (base_pos, [mb x nkv x HD] arrays). Aggregate the ceiling."""
    tot_rot_flat = tot_rot_deph = tot_rot = tot_all = 0.0
    ngroups = 0
    for base_pos, ck in records:
        mb, nkv, _ = ck.shape
        for kv in range(nkv):
            # consecutive positions [base_pos, base_pos+mb); group in FOLD_M chunks
            for g0 in range(0, mb - FOLD_M + 1, FOLD_M):
                idx = list(range(g0, g0 + FOLD_M))
                ks = ck[idx, kv, :]
                positions = [base_pos + j for j in idx]
                rf, rd, rv, tv = group_variances(ks, positions)
                tot_rot_flat += rf; tot_rot_deph += rd; tot_rot += rv; tot_all += tv
                ngroups += 1
    phase_frac = 1.0 - tot_rot_deph / tot_rot_flat if tot_rot_flat > 0 else 0.0
    rot_share = tot_rot / tot_all if tot_all > 0 else 0.0
    # Net headroom: fraction of TOTAL merge variance removable by dephasing =
    # (rotated-dim share of variance) * (phase fraction within rotated dims).
    net = rot_share * phase_frac
    print(f"\n=== RoPE-dephase ceiling ({ngroups} groups, fold_m={FOLD_M}) ===")
    print(f"  rotated-dim share of intra-group variance : {rot_share:6.3f}  (cap from partial_rotary; pass-through dims are pure content)")
    print(f"  phase fraction WITHIN rotated dims         : {phase_frac:6.3f}  (1.0 = all phase, 0.0 = all content)")
    print(f"  --> NET removable by dephasing (of total)  : {net:6.3f}")
    verdict = ("DEPHASING WORTH IT" if net > 0.15 else
               "MARGINAL" if net > 0.05 else "DEPHASING DEAD (blur is content, not phase)")
    print(f"  verdict: {verdict}")
    return net

def self_test():
    """Round-trip + synthetic separability check. Build K = shared_content +
    per-position rope-phase applied to a fixed base; dephasing must recover the
    shared content (phase_frac ~ 1 on rotated dims, given identical base)."""
    rng = np.random.default_rng(0)
    # Round-trip: rotate then de-rotate = identity.
    k = rng.standard_normal(HD).astype(np.float64)
    k_rot = apply_rope(k, 1234.0, +1)
    k_back = apply_rope(k_rot, 1234.0, -1)
    assert np.allclose(k, k_back, atol=1e-9), "RoPE round-trip failed"
    # Pure-phase group: SAME underlying content, only position differs → after
    # de-rotation to a common ref, rotated dims should collapse (phase_frac≈1).
    base = rng.standard_normal(HD)
    positions = [1000, 1001, 1002, 1003]
    ks = np.stack([apply_rope(base, p, +1) for p in positions])
    rf, rd, rv, tv = group_variances(ks, positions)
    pf = 1.0 - rd / rf
    assert pf > 0.999, f"pure-phase group should give phase_frac~1, got {pf:.4f}"
    # Pure-content group: different content, same position → dephasing does nothing.
    ks2 = np.stack([apply_rope(rng.standard_normal(HD), 1000, +1) for _ in range(4)])
    rf2, rd2, _, _ = group_variances(ks2, [1000, 1000, 1000, 1000])
    pf2 = 1.0 - rd2 / rf2
    assert abs(pf2) < 1e-9, f"same-position group should give phase_frac~0, got {pf2:.6f}"
    print("self-test OK: RoPE round-trip exact; pure-phase→1.000; pure-content→0.000")

def read_captures(path):
    b = open(path, "rb").read()
    off = 0; recs = []
    while off < len(b):
        base, mb, nkv, hd = struct.unpack_from("<IIII", b, off); off += 16
        n = mb * nkv * hd
        ck = np.frombuffer(b, dtype="<f4", count=n, offset=off).reshape(mb, nkv, hd).astype(np.float64)
        off += n * 4
        recs.append((base, ck))
    return recs

if __name__ == "__main__":
    self_test()
    if len(sys.argv) > 1:
        recs = read_captures(sys.argv[1])
        print(f"loaded {len(recs)} capture records from {sys.argv[1]}")
        analyze(recs)
    else:
        print("(no capture file given — self-test only; pass a HIPFIRE_KV_CAPTURE_K dump to run the ceiling)")
