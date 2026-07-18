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


def dequant_group(q, scale, group):
    orig = q.shape
    flat = q.reshape(-1, group).astype(np.float32)
    s = scale.reshape(-1, 1)
    return (flat * s).reshape(orig)


class QLinear:
    """int8-weight linear. mode 'w8a16' (f32 act) or 'w8a8' (int8 act fold)."""

    def __init__(self, W, group, clip_steps):
        # W: [out, in]; quantize along `in` (the contraction dim, grouped).
        self.group = group
        self.qW, self.sW = quant_group_symmetric_int8(W, group, clip_steps)
        self.W_deq = dequant_group(self.qW, self.sW, group)  # [out,in]
        self.n_groups = W.shape[1] // group

    def w8a16(self, x):
        # x: [batch, in] f32; weights dequantized, exact f32 matmul.
        return x @ self.W_deq.T

    def w8a8(self, x, clip_steps=1):
        # x: [batch, in] f32 -> per-group int8; int32 fold with int8 weights.
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
    act_steps = clip_steps if mode == "w8a8" else 1

    def proj(name):
        return QLinear(W[name], group, clip_steps)

    def apply(ql, x):
        return ql.w8a16(x) if mode == "w8a16" else ql.w8a8(x, act_steps)

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
    ap.add_argument("--mode", choices=["w8a16", "w8a8", "both"], default="both")
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

    modes = ["w8a16", "w8a8"] if args.mode == "both" else [args.mode]
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
        print(f"  {mode:6s} max_abs={d.max():.4e} mean_abs={d.mean():.4e} "
              f"max_rel={(d/denom).max():.3e} cos={cos:.6f} SNR={snr:.2f}dB")


if __name__ == "__main__":
    main()
