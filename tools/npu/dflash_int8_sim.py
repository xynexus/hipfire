#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""dflash_int8_sim.py — W8A16 / W8A8 quality simulation for the DFlash drafter.

Answers the design question BEFORE building the NPU int8 kernel or the OQ8++
Hessian pipeline: how much quality does a true int8 compute path cost on the
real z-lab drafter, and does the A8 activation quant (W8A8) meaningfully lose
vs near-lossless activations (W8A16)?

It reuses the validated numpy reference (`dflash_ref.py`) but replaces each
projection's f32 matmul with:

  W8A16 : per-group symmetric int8 WEIGHTS (clip-search scale), activations
          left in f32 (near-lossless). Isolates weight-quant error.
  W8A8  : same int8 weights AND per-group symmetric int8 ACTIVATIONS, matmul
          as an int32 fold (`Σ q_w·q_x · s_w·s_x`) — the exact math the AIE2
          `aie::mmul<...,int8,int8>` kernel + `opus_lowbit::dot_offset_fold`
          will run. This is the real W8A8 path.

Both are NON-ROTATED (no FWHT): at 8 bits, incoherence rotation buys ~nothing
and would force a per-block on-NPU activation FWHT. Weight quant is fully
offline; the runtime activation path is just per-group int8 quant.

Reports per-op and full-body error vs the F16/numpy golden so the ++ (Hessian
/LDLQ) and int8-kernel investment can be decided on evidence.

Usage:
  dflash_int8_sim.py --ref-dir <dir> --weights <safetensors> \
      [--group 256] [--mode w8a8|w8a16|both] [--clip-steps 24]
"""
import argparse
import os
import numpy as np

import dflash_ref as R  # reuse loader + math primitives


def quant_group_symmetric_int8(x, group, clip_steps=1, min_clip=0.6):
    """Per-group symmetric int8 (signed [-127,127]). Returns (q_int8, scale_per_group).

    Quantizes along the LAST axis in contiguous groups of `group`. clip_steps>1
    runs an unweighted clip-search (the '+'), minimizing per-group MSE.
    """
    orig = x.shape
    flat = x.reshape(-1, group).astype(np.float32)   # [n_groups, group]
    amax = np.abs(flat).max(axis=1, keepdims=True)    # [n_groups,1]
    if clip_steps <= 1:
        scale = np.where(amax > 0, amax / 127.0, 1.0)
    else:
        alphas = min_clip + (1.0 - min_clip) * (np.arange(clip_steps) / (clip_steps - 1))
        best_err = np.full((flat.shape[0], 1), np.inf)
        best_scale = np.where(amax > 0, amax / 127.0, 1.0)
        for a in alphas:
            clip = a * amax
            sc = np.where(clip > 0, clip / 127.0, 1.0)
            q = np.clip(np.round(flat / sc), -127, 127)
            err = ((flat - q * sc) ** 2).sum(axis=1, keepdims=True)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_scale = np.where(better, sc, best_scale)
        scale = best_scale
    q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
    return q.reshape(orig), scale.reshape(orig[:-1] + (orig[-1] // group,))


def quant_group_mixed4(x, group, n_out=3):
    """Per-group MIXED int4 bulk + sparse int8 overlay — the oq4.25 codec.

    Mirrors `quantize_oq4_mixed_plain` in crates/hipfire-quantize/src/bin/
    dflash_convert.rs (non-rotated). Returns (codes_int8, scale_per_group) in the
    SAME representation as quant_group_symmetric_int8: every value is
    `code * scale`, with codes clamped to +/-7 for the bulk and +/-127 for the
    n_out overlay slots. That means QLinear consumes it unchanged.

    n_out=3 over a 256-group is 130+2*3 = 136 B/group = 4.25 bits/weight.
    """
    orig = x.shape
    flat = x.reshape(-1, group).astype(np.float32)
    amax = np.abs(flat).max(axis=1, keepdims=True)

    def sse(scale, limits):
        q = np.clip(np.round(flat / scale), -limits, limits)
        return ((flat - q * scale) ** 2).sum(axis=1, keepdims=True)

    def clipsearch(grid, limits, seed):
        best_s = seed.copy()
        best_e = sse(best_s, limits)
        for c in grid:
            s = np.maximum(c * amax / 7.0, 1e-12)
            e = sse(s, limits)
            better = e < best_e
            best_e = np.where(better, e, best_e)
            best_s = np.where(better, s, best_s)
        return best_s

    def overlay_limits(scale):
        """Pick the n_out highest int8-upgrade-gain slots; return a per-element
        clamp limit array (127 at overlay slots, 7 elsewhere)."""
        q4 = np.clip(np.round(flat / scale), -7, 7)
        q8 = np.clip(np.round(flat / scale), -127, 127)
        e4 = (flat - q4 * scale) ** 2
        e8 = (flat - q8 * scale) ** 2
        gain = e4 - e8
        idx = np.argpartition(-gain, n_out - 1, axis=1)[:, :n_out]
        limits = np.full(flat.shape, 7.0, dtype=np.float32)
        np.put_along_axis(limits, idx, 127.0, axis=1)
        return limits

    # int4-only clip-search, then two rounds of (pick overlays -> refit scale),
    # matching the Rust mixed_clipsearch_plain exactly.
    grid4 = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    grid_mixed = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                  0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
    seed = np.where(amax > 0, amax / 7.0, 1.0)
    s = clipsearch(grid4, 7.0, seed)
    for _ in range(2):
        lim = overlay_limits(s)
        s = clipsearch(grid_mixed, lim, s)
    limits = overlay_limits(s)

    q = np.clip(np.round(flat / s), -limits, limits).astype(np.int8)
    return q.reshape(orig), s.reshape(orig[:-1] + (orig[-1] // group,))


def fwht_signs(seed, n=256):
    """Port of hipfire_primitives::fwht::gen_fwht_signs (LCG, bit 16 -> +/-1)."""
    state = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


# The engine's fixed sign table (gen_fwht_signs(42|1042, 256)); see
# crates/hipfire-quantize/src/rotate.rs.
SIGNS1 = fwht_signs(42)
SIGNS2 = fwht_signs(1042)


def fwht_groups(x, group=256, signs1=SIGNS1, signs2=SIGNS2):
    """Signed orthonormal FWHT over the LAST axis in contiguous `group` blocks.

    Port of cpu_fwht_256: signs1 pre-multiply, Hadamard butterfly, then
    1/sqrt(n)*signs2 post. Orthonormal, so for weights w and activations x
    rotated by the same H: (Hx).(Hw) == x.w exactly. Rotation is therefore
    loss-free on the product and only changes what the QUANTIZER sees --
    spreading outliers so a low-bit grid fits the distribution better.
    """
    orig = x.shape
    a = x.reshape(-1, group).astype(np.float32) * signs1
    m, n = a.shape
    s = 1
    while s < n:
        a = a.reshape(m, n // (2 * s), 2, s)
        u = a[:, :, 0, :]
        v = a[:, :, 1, :]
        a = np.concatenate([(u + v)[:, :, None, :], (u - v)[:, :, None, :]],
                           axis=2).reshape(m, n)
        s *= 2
    a = a * (signs2 / np.sqrt(n, dtype=np.float32))
    return a.reshape(orig)


def dequant_group(q, scale, group):
    orig = q.shape
    flat = q.reshape(-1, group).astype(np.float32)
    s = scale.reshape(-1, 1)
    return (flat * s).reshape(orig)


class QLinear:
    """Quantized-weight linear. Activation path: 'w8a16' (f32) or 'w8a8' (int8).

    `wq` selects the WEIGHT codec: None => symmetric int8 (8 b/w), or an int
    n_out => mixed int4 bulk + n_out int8 overlays (4.0625 + n_out/16 b/w).
    Both yield (int8 codes, per-group scale), so the activation paths below are
    identical either way.
    """

    def __init__(self, W, group, clip_steps, wq=None, rotate=False):
        # W: [out, in]; quantize along `in` (the contraction dim, grouped).
        self.group = group
        self.rotate = rotate
        # Rotation is applied to the weights OFFLINE here and to activations at
        # forward time. Orthonormal per group, so the product is unchanged; only
        # the quantization grid sees a better-conditioned distribution.
        Wq = fwht_groups(W, group) if rotate else W
        if wq is None:
            self.qW, self.sW = quant_group_symmetric_int8(Wq, group, clip_steps)
        else:
            self.qW, self.sW = quant_group_mixed4(Wq, group, n_out=wq)
        self.W_deq = dequant_group(self.qW, self.sW, group)  # [out,in]
        self.n_groups = W.shape[1] // group

    def w8a16(self, x):
        # x: [batch, in] f32; weights dequantized, exact f32 matmul.
        if self.rotate:
            x = fwht_groups(x, self.group)
        return x @ self.W_deq.T

    def w8a8(self, x, clip_steps=1):
        # x: [batch, in] f32 -> per-group int8; int32 fold with int8 weights.
        # Rotate BEFORE quantizing: rotation also conditions the activation
        # distribution, which is the other half of what incoherence buys.
        if self.rotate:
            x = fwht_groups(x, self.group)
        qx, sx = quant_group_symmetric_int8(x, self.group, clip_steps)  # [B,in],[B,ng]
        B = x.shape[0]
        out = self.W_deq.shape[0]
        qxf = qx.reshape(B, self.n_groups, self.group).astype(np.int32)
        qwf = self.qW.reshape(out, self.n_groups, self.group).astype(np.int32)
        # per (b,out,group) int dot, then * sW[out,g]*sx[b,g], sum over groups
        # einsum: [B,ng,g],[out,ng,g] -> [B,out,ng]
        iacc = np.einsum("bng,ong->bon", qxf, qwf).astype(np.float32)
        scal = sx[:, None, :] * self.sW[None, :, :]           # [B,out,ng]
        return (iacc * scal).sum(axis=2)                       # [B,out]


def run_forward(meta, inp, W, group, mode, clip_steps):
    H, NH, NKV = meta["hidden"], meta["n_heads"], meta["n_kv_heads"]
    HD, INTER, NL = meta["head_dim"], meta["intermediate"], meta["n_layers"]
    EPS, THETA = meta["norm_eps"], meta["rope_theta"]
    B, L, NE = meta["block_size"], meta["ctx_len"], meta["num_extract"]
    scaling = HD ** -0.5
    noise, th, pos_q, pos_k = inp
    # int8 activations whenever the mode ends in a8 (w8a8 / mixed4a8).
    act8 = mode.endswith("a8")
    act_steps = clip_steps if act8 else 1
    # mixed4* modes use the 4.25 b/w weight codec (3 overlays per 256-group).
    # "r" prefix => FWHT-rotated (incoherence) variant.
    rot = mode.startswith("r")
    base = mode[1:] if rot else mode
    wq = 3 if base.startswith("mixed4") else None

    def proj(name):
        return QLinear(W[name], group, clip_steps, wq=wq, rotate=rot)

    def apply(ql, x):
        return ql.w8a8(x, act_steps) if act8 else ql.w8a16(x)

    fc = proj("fc.weight")
    thp = apply(fc, th.reshape(L, NE * H))
    thp = R.rmsnorm(thp, W["hidden_norm.weight"], EPS)

    cos_k, sin_k = R.rope_cos_sin(pos_k, HD, THETA)
    cos_q, sin_q = R.rope_cos_sin(pos_q, HD, THETA)
    hidden = noise.astype(np.float32).copy()
    per_op = {}
    for li in range(NL):
        p = f"layers.{li}."
        residual = hidden
        x = R.rmsnorm(hidden, W[p + "input_layernorm.weight"], EPS)
        qp = proj(p + "self_attn.q_proj.weight")
        kp = proj(p + "self_attn.k_proj.weight")
        vp = proj(p + "self_attn.v_proj.weight")
        q = apply(qp, x).reshape(B, NH, HD)
        k_noise = apply(kp, x).reshape(B, NKV, HD)
        v_noise = apply(vp, x).reshape(B, NKV, HD)
        k_ctx = apply(kp, thp).reshape(L, NKV, HD)
        v_ctx = apply(vp, thp).reshape(L, NKV, HD)
        q = R.rmsnorm(q, W[p + "self_attn.q_norm.weight"], EPS)
        k = np.concatenate([k_ctx, k_noise], 0)
        v = np.concatenate([v_ctx, v_noise], 0)
        k = R.rmsnorm(k, W[p + "self_attn.k_norm.weight"], EPS)
        q = np.transpose(q, (1, 0, 2)); k = np.transpose(k, (1, 0, 2)); v = np.transpose(v, (1, 0, 2))
        q = R.apply_rope(q, cos_q, sin_q); k = R.apply_rope(k, cos_k, sin_k)
        groups = NH // NKV
        k_e = np.repeat(k, groups, 0); v_e = np.repeat(v, groups, 0)
        scores = np.einsum("hqd,hkd->hqk", q, k_e) * scaling
        attn = R.softmax(scores, -1)
        ctx = np.einsum("hqk,hkd->hqd", attn, v_e)
        ctx = np.transpose(ctx, (1, 0, 2)).reshape(B, NH * HD)
        op = proj(p + "self_attn.o_proj.weight")
        hidden = residual + apply(op, ctx)
        residual = hidden
        x2 = R.rmsnorm(hidden, W[p + "post_attention_layernorm.weight"], EPS)
        gp = proj(p + "mlp.gate_proj.weight"); up = proj(p + "mlp.up_proj.weight")
        g = apply(gp, x2); u = apply(up, x2)
        silu = g * (1.0 / (1.0 + np.exp(-g)))
        dp = proj(p + "mlp.down_proj.weight")
        hidden = residual + apply(dp, silu * u)
        if li == 0:
            per_op["l0_out"] = hidden.copy()
    final = R.rmsnorm(hidden, W["norm.weight"], EPS)
    return final, per_op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--group", type=int, default=256)
    ap.add_argument("--mode",
                    choices=["w8a16", "w8a8", "mixed4a16", "mixed4a8",
                             "rw8a16", "rw8a8", "rmixed4a16", "rmixed4a8",
                             "both", "all", "rot"],
                    default="both")
    ap.add_argument("--clip-steps", type=int, default=24)
    args = ap.parse_args()

    import json
    rd = args.ref_dir
    meta = json.load(open(os.path.join(rd, "ref_meta.json")))
    inp = (np.load(os.path.join(rd, "noise_embedding.npy")),
           np.load(os.path.join(rd, "target_hidden.npy")),
           np.load(os.path.join(rd, "positions_q.npy")),
           np.load(os.path.join(rd, "positions_k.npy")))
    golden = np.load(os.path.join(rd, "golden", "final_block_hidden.npy"))
    W = R.load_safetensors_f32(args.weights)

    if args.mode == "both":
        modes = ["w8a16", "w8a8"]
    elif args.mode == "all":
        modes = ["w8a16", "w8a8", "mixed4a16", "mixed4a8"]
    elif args.mode == "rot":
        modes = ["mixed4a16", "rmixed4a16", "mixed4a8", "rmixed4a8"]
    else:
        modes = [args.mode]
    print(f"[int8-sim] group={args.group} clip_steps={args.clip_steps} "
          f"vs golden final rms={np.sqrt((golden**2).mean()):.4f}")
    for mode in modes:
        final, _ = run_forward(meta, inp, W, args.group, mode, args.clip_steps)
        d = np.abs(final - golden)
        denom = np.maximum(np.abs(golden), 1e-3)
        cos = float(final.reshape(-1) @ golden.reshape(-1) /
                    (np.linalg.norm(final) * np.linalg.norm(golden)))
        # signal-to-noise in dB relative to golden energy
        snr = 10 * np.log10((golden ** 2).sum() / ((final - golden) ** 2).sum())
        print(f"  {mode:10s} max_abs={d.max():.4e} mean_abs={d.mean():.4e} "
              f"max_rel={(d/denom).max():.3e} cos={cos:.6f} SNR={snr:.2f}dB")


if __name__ == "__main__":
    main()
