#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""dflash_ref.py — numpy golden reference for the DFlash NPU bring-up.

Reads the deterministic inputs dumped by `dflash_ref_dump.rs` (Phase A) plus
the ORIGINAL z-lab safetensors weights (bf16 -> f32, bit-exact to the F32 HFQ
the runtime loads), reproduces the DFlash block forward op-by-op in float32
numpy (mirroring dflash/model.py), and:

  1. validates its final block hidden against `block_hidden.npy` (the F32 GPU
     reference), proving numpy == runtime within tolerance, and
  2. writes every per-op intermediate to `<ref_dir>/golden/` so each NPU
     primitive (Phase B) can be checked against a real-weights golden slice.

Usage:
  dflash_ref.py --ref-dir <dir_from_dflash_ref_dump> \
                --weights <safetensors_or_dir> [--atol 2e-2]
"""
import argparse
import json
import os
import struct
import sys
import numpy as np


# ── safetensors reader (bf16/f16/f32 -> f32) ────────────────────────────────
def _resolve_st(path):
    if os.path.isdir(path):
        # HF snapshot dir or cache root
        cand = os.path.join(path, "model.safetensors")
        if os.path.exists(cand):
            return cand
        snaps = os.path.join(path, "snapshots")
        if os.path.isdir(snaps):
            for d in sorted(os.listdir(snaps)):
                c = os.path.join(snaps, d, "model.safetensors")
                if os.path.exists(c):
                    return c
        raise FileNotFoundError(f"no model.safetensors under {path}")
    return path


def load_safetensors_f32(path):
    path = _resolve_st(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dt in ("F16", "FP16"):
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt in ("F32", "FP32"):
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                raise ValueError(f"unsupported dtype {dt} for {name}")
            out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


# ── math primitives (fp32, mirroring Qwen3) ─────────────────────────────────
def rmsnorm(x, w, eps):
    # x: [..., d], normalize over last dim.
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(v + eps))) * w


def linear(x, w):
    # y = x @ w^T ; w: [out, in]
    return x @ w.T


def rope_cos_sin(positions, head_dim, theta):
    # HF "neox" layout: inv_freq over half dim, duplicated (cat), non-interleaved.
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    ang = np.outer(positions.astype(np.float64), inv_freq)  # [T, half]
    emb = np.concatenate([ang, ang], axis=-1)  # [T, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    # x: [heads, T, hd]; cos/sin: [T, hd]
    return x * cos[None, :, :] + rotate_half(x) * sin[None, :, :]


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--weights", required=True,
                    help="safetensors file or HF dir/snapshot")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    args = ap.parse_args()

    rd = args.ref_dir
    meta = json.load(open(os.path.join(rd, "ref_meta.json")))
    H = meta["hidden"]; NH = meta["n_heads"]; NKV = meta["n_kv_heads"]
    HD = meta["head_dim"]; INTER = meta["intermediate"]; NL = meta["n_layers"]
    EPS = meta["norm_eps"]; THETA = meta["rope_theta"]
    B = meta["block_size"]; L = meta["ctx_len"]; NE = meta["num_extract"]
    scaling = HD ** -0.5

    noise = np.load(os.path.join(rd, "noise_embedding.npy"))       # [B, H]
    th = np.load(os.path.join(rd, "target_hidden.npy"))            # [L, NE, H]
    pos_q = np.load(os.path.join(rd, "positions_q.npy"))           # [B]
    pos_k = np.load(os.path.join(rd, "positions_k.npy"))           # [L+B]
    ref_final = np.load(os.path.join(rd, "block_hidden.npy"))      # [B, H]

    W = load_safetensors_f32(args.weights)
    gd = os.path.join(rd, "golden")
    os.makedirs(gd, exist_ok=True)

    def g(name, arr):
        np.save(os.path.join(gd, name + ".npy"), arr.astype(np.float32))

    # ── fc + hidden_norm on context ─────────────────────────────────────────
    th_flat = th.reshape(L, NE * H)                                # [L, NE*H]
    thp = linear(th_flat, W["fc.weight"])                         # [L, H]
    thp = rmsnorm(thp, W["hidden_norm.weight"], EPS)              # [L, H]
    g("target_hidden_proj", thp)

    # rope tables over the full k-position span; q uses the last-B slice.
    cos_k, sin_k = rope_cos_sin(pos_k, HD, THETA)                 # [L+B, HD]
    cos_q, sin_q = rope_cos_sin(pos_q, HD, THETA)                 # [B, HD]

    hidden = noise.astype(np.float32).copy()                     # [B, H]
    tot = L + B
    for li in range(NL):
        p = f"layers.{li}."
        residual = hidden
        x = rmsnorm(hidden, W[p + "input_layernorm.weight"], EPS)
        if li == 0:
            g("l0_input_norm", x)

        # projections
        q = linear(x, W[p + "self_attn.q_proj.weight"]).reshape(B, NH, HD)
        k_noise = linear(x, W[p + "self_attn.k_proj.weight"]).reshape(B, NKV, HD)
        v_noise = linear(x, W[p + "self_attn.v_proj.weight"]).reshape(B, NKV, HD)
        k_ctx = linear(thp, W[p + "self_attn.k_proj.weight"]).reshape(L, NKV, HD)
        v_ctx = linear(thp, W[p + "self_attn.v_proj.weight"]).reshape(L, NKV, HD)

        # per-head norms
        q = rmsnorm(q, W[p + "self_attn.q_norm.weight"], EPS)     # [B, NH, HD]
        k = np.concatenate([k_ctx, k_noise], axis=0)             # [tot, NKV, HD]
        v = np.concatenate([v_ctx, v_noise], axis=0)             # [tot, NKV, HD]
        k = rmsnorm(k, W[p + "self_attn.k_norm.weight"], EPS)

        # to [heads, T, hd]
        q = np.transpose(q, (1, 0, 2))                           # [NH, B, HD]
        k = np.transpose(k, (1, 0, 2))                           # [NKV, tot, HD]
        v = np.transpose(v, (1, 0, 2))                           # [NKV, tot, HD]

        # rope
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        if li == 0:
            g("l0_q_roped", np.transpose(q, (1, 0, 2)))          # [B, NH, HD]
            g("l0_k_roped", np.transpose(k, (1, 0, 2)))          # [tot, NKV, HD]
            g("l0_v", np.transpose(v, (1, 0, 2)))

        # GQA expand kv
        groups = NH // NKV
        k_e = np.repeat(k, groups, axis=0)                       # [NH, tot, HD]
        v_e = np.repeat(v, groups, axis=0)

        # non-causal attention
        scores = np.einsum("hqd,hkd->hqk", q, k_e) * scaling     # [NH, B, tot]
        attn = softmax(scores, axis=-1)
        ctx = np.einsum("hqk,hkd->hqd", attn, v_e)               # [NH, B, HD]
        ctx = np.transpose(ctx, (1, 0, 2)).reshape(B, NH * HD)   # [B, NH*HD]
        if li == 0:
            g("l0_attn_out", ctx)

        attn_proj = linear(ctx, W[p + "self_attn.o_proj.weight"])  # [B, H]
        hidden = residual + attn_proj
        if li == 0:
            g("l0_post_attn_residual", hidden)

        # MLP (SwiGLU)
        residual = hidden
        x2 = rmsnorm(hidden, W[p + "post_attention_layernorm.weight"], EPS)
        gate = linear(x2, W[p + "mlp.gate_proj.weight"])          # [B, INTER]
        up = linear(x2, W[p + "mlp.up_proj.weight"])
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))
        ff = linear(silu * up, W[p + "mlp.down_proj.weight"])     # [B, H]
        if li == 0:
            g("l0_swiglu_down", ff)
        hidden = residual + ff
        g(f"l{li}_out", hidden)

    final = rmsnorm(hidden, W["norm.weight"], EPS)
    g("final_block_hidden", final)

    # ── validate numpy final vs Rust GPU reference ──────────────────────────
    diff = np.abs(final - ref_final)
    denom = np.maximum(np.abs(ref_final), 1e-3)
    rel = diff / denom
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    mean_abs = float(diff.mean())
    print(f"[dflash_ref] shapes: B={B} L={L} H={H} NL={NL}")
    print(f"[dflash_ref] numpy-vs-Rust final: max_abs={max_abs:.4e} "
          f"mean_abs={mean_abs:.4e} max_rel={max_rel:.4e}")
    print(f"[dflash_ref] wrote {len(os.listdir(gd))} golden tensors -> {gd}")

    ok = max_abs <= args.atol or max_rel <= args.rtol
    if not ok:
        # deeper: report where
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"[dflash_ref] FAIL at {idx}: numpy={final[idx]:.5f} "
              f"rust={ref_final[idx]:.5f}", file=sys.stderr)
        sys.exit(1)
    print("[dflash_ref] OK (numpy reference matches GPU reference)")


if __name__ == "__main__":
    main()
