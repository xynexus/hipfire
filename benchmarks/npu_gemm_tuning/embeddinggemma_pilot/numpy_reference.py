#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numpy reference for one EmbeddingGemma (Gemma3) encoder layer.

Executable spec / oracle for the IRON graph. Reproduces layer 0 from the HF
captured input + weights and verifies each stage against HF's captured
intermediates (reference_layer0.npz). Layer 0 is a LOCAL (sliding) layer;
at M256 < window the attention is full bidirectional. RMSNorm uses HF's
(1 + weight) convention.
"""
import os, numpy as np
from safetensors import safe_open

HERE = os.path.dirname(__file__)
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))

H, NH, NKV, HD, FF = 768, 3, 1, 256, 1152
EPS = 1e-6
ROPE_BASE = 10000.0   # layer 0 is local/sliding
SCALE = 1.0 / np.sqrt(HD)  # 0.0625

W = {}
with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
    for k in st.keys():
        if "layers.0." in k:
            W[k.split("layers.0.")[1]] = st.get_tensor(k).astype(np.float32)

def rmsnorm(x, w):  # HF Gemma: x * rsqrt(mean(x^2)+eps) * (1 + w)
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return (x * r) * (1.0 + w)

def gelu_tanh(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v**3)))

def lin(x, w):  # torch nn.Linear weight [out,in], y = x @ w.T
    return x @ w.T

def rope(t, base):  # t: [m, nheads, hd], rotate-half (NeoX split-half)
    m, nh, hd = t.shape
    half = hd // 2
    i = np.arange(half)
    inv = 1.0 / (base ** (2 * i / hd))
    pos = np.arange(m)[:, None]
    ang = pos * inv[None, :]            # [m, half]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x1, x2 = t[..., :half], t[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def chk(name, got, key):
    exp = ref[key][0] if ref[key].ndim == 3 else ref[key]
    err = np.abs(got - exp).max()
    rel = err / (np.abs(exp).max() + 1e-9)
    print(f"  {name:28s} max_abs={err:.3e} rel={rel:.2e} {'OK' if rel < 2e-3 else 'MISMATCH'}")
    return rel < 2e-3

x = ref["layer0__in"][0]  # [256, 768]
print("=== stage-by-stage vs HF ===")

# --- attention block ---
xn = rmsnorm(x, W["input_layernorm.weight"]); chk("input_layernorm", xn, "layer0__input_layernorm")
q = lin(xn, W["self_attn.q_proj.weight"]); chk("q_proj", q, "layer0__self_attn__q_proj")
k = lin(xn, W["self_attn.k_proj.weight"]); chk("k_proj", k, "layer0__self_attn__k_proj")
v = lin(xn, W["self_attn.v_proj.weight"]); chk("v_proj", v, "layer0__self_attn__v_proj")

m = x.shape[0]
qh = q.reshape(m, NH, HD); kh = k.reshape(m, NKV, HD); vh = v.reshape(m, NKV, HD)
qh = rmsnorm(qh, W["self_attn.q_norm.weight"])   # per-head over HD
kh = rmsnorm(kh, W["self_attn.k_norm.weight"])
qh = rope(qh, ROPE_BASE); kh = rope(kh, ROPE_BASE)
# GQA rep=3: broadcast single kv head to 3 q heads
kb = np.repeat(kh, NH // NKV, axis=1); vb = np.repeat(vh, NH // NKV, axis=1)
# scores [nh, m, m]
scores = np.einsum("qhd,khd->hqk", qh, kb) * SCALE
scores -= scores.max(-1, keepdims=True)
p = np.exp(scores); p /= p.sum(-1, keepdims=True)
ctx = np.einsum("hqk,khd->qhd", p, vb).reshape(m, NH * HD)
o = lin(ctx, W["self_attn.o_proj.weight"]); chk("self_attn(o_proj)", o, "layer0__self_attn")
xa = x + rmsnorm(o, W["post_attention_layernorm.weight"])

# --- FFN block ---
fn = rmsnorm(xa, W["pre_feedforward_layernorm.weight"])
g = lin(fn, W["mlp.gate_proj.weight"]); u = lin(fn, W["mlp.up_proj.weight"])
d = lin(gelu_tanh(g) * u, W["mlp.down_proj.weight"]); chk("mlp", d, "layer0__mlp")
out = xa + rmsnorm(d, W["post_feedforward_layernorm.weight"])

print("=== final layer output vs HF ===")
ok = chk("layer0 output", out, "layer0")
print("\nREFERENCE", "VERIFIED ✅" if ok else "MISMATCH ❌")
