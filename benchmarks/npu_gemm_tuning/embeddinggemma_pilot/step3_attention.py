#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rung A step 3: the GQA attention core (MHA) on the NPU, verified vs the oracle.

Prepares qk-norm + RoPE (base 1e4) in numpy from the oracle q/k/v_proj outputs
(these ops are already HF-verified in numpy_reference), feeds the IRON MHA operator
(num_heads=3, seq=256, d=256, num_KV_heads=1; internal scale 1/sqrt(256)=0.0625,
bidirectional), and compares its attention context to the numpy softmax attention.
MHA layout: Q[heads,seq,d], K/V[kv_heads, seq*d], O=Q. o_proj is step 4 (GEMM).
"""
import os, numpy as np, ml_dtypes
from safetensors import safe_open
from iron.common.context import AIEContext          # aie/iron before torch
from iron.operators.mha.op import MHA
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
BF16 = ml_dtypes.bfloat16
M, NH, NKV, HD = 256, 3, 1, 256
EPS, ROPE_BASE = 1e-6, 10000.0
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))

def Wt(name):
    with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
        return st.get_tensor(next(k for k in st.keys() if k.endswith("layers.0." + name))).astype(np.float32)

def rmsnorm(x, w):
    return (x * (1.0 / np.sqrt(np.mean(x*x, -1, keepdims=True) + EPS))) * (1.0 + w)

def rope(t, base):
    m, nh, hd = t.shape; half = hd // 2
    inv = 1.0 / (base ** (2*np.arange(half)/hd)); ang = np.arange(m)[:, None]*inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = t[..., :half], t[..., half:]
    return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], -1)

def fill(t, a):
    t.torch_view().copy_(torch.from_numpy(np.ascontiguousarray(a).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))

# --- prepare inputs in numpy (qk-norm + rope), from the oracle q/k/v proj ---
q = ref["layer0__self_attn__q_proj"][0].reshape(M, NH, HD)
k = ref["layer0__self_attn__k_proj"][0].reshape(M, NKV, HD)
v = ref["layer0__self_attn__v_proj"][0].reshape(M, NKV, HD)
qh = rope(rmsnorm(q, Wt("self_attn.q_norm.weight")), ROPE_BASE)   # [256,3,256]
kh = rope(rmsnorm(k, Wt("self_attn.k_norm.weight")), ROPE_BASE)   # [256,1,256]

# numpy reference attention context (bidirectional GQA)
kb, vb = np.repeat(kh, NH//NKV, 1), np.repeat(v, NH//NKV, 1)
sc = np.einsum("qhd,khd->hqk", qh, kb) / np.sqrt(HD); sc -= sc.max(-1, keepdims=True)
p = np.exp(sc); p /= p.sum(-1, keepdims=True)
ctx_ref = np.einsum("hqk,khd->qhd", p, vb)                        # [256,3,256]

# --- MHA on the NPU: Q[heads,seq,d], K/V[kv, seq*d], O=Q ---
print("compiling+dispatching MHA (h=3,kv=1,seq=256,d=256) ...")
op = MHA(num_heads=NH, seq_len=M, d=HD, num_KV_heads=NKV, context=AIEContext()).compile()
call = op.get_callable()
Q = XRTTensor((NH*M*HD,), dtype=BF16); fill(Q, np.transpose(qh, (1, 0, 2)))       # [h,s,d]
K = XRTTensor((NKV*M*HD,), dtype=BF16); fill(K, np.transpose(kh, (1, 0, 2)))      # [kv,s,d]->flat
V = XRTTensor((NKV*M*HD,), dtype=BF16); fill(V, np.transpose(v, (1, 0, 2)))
O = XRTTensor((NH*M*HD,), dtype=BF16); fill(O, np.zeros(NH*M*HD, np.float32))
call(Q, K, V, O)
out = O.to_torch().float().cpu().numpy().reshape(NH, M, HD).transpose(1, 0, 2)    # [s,h,d]

rel = np.abs(out - ctx_ref).max() / (np.abs(ctx_ref).max() + 1e-9)
print(f"NPU MHA context vs numpy attention: max_abs={np.abs(out-ctx_ref).max():.3e} rel={rel:.3e}")
print("STEP 3", "PASS ✅ (bf16 tol)" if rel < 5e-2 else f"CHECK rel={rel:.3e}")
