#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rung A step 3 (composed): GQA attention from verified primitives on the NPU.

Since IRON MHA is head_dim=64-only, attention is composed as (per q-head):
  scores = GEMM(Q_scaled, K, b_col_maj=True)  # Q @ K^T, scale 1/sqrt(256) folded into Q
  P      = softmax(scores)                     # over the 256 keys
  ctx    = GEMM(P, V, b_col_maj=False)         # P @ V
GQA rep 3 (3 q-heads share the 1 kv-head), bidirectional. qk-norm + RoPE prepared in
numpy (HF-verified). Verified vs the numpy attention context. o_proj is step 4.
"""
import os, numpy as np, ml_dtypes
from safetensors import safe_open
from iron.common.context import AIEContext          # aie/iron before torch
from iron.operators.gemm.op import GEMM
from iron.operators.softmax.op import Softmax
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
BF16 = ml_dtypes.bfloat16
M, NH, NKV, HD = 256, 3, 1, 256
EPS, ROPE_BASE, SCALE = 1e-6, 10000.0, 1.0/np.sqrt(256)
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))

def Wt(n):
    with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
        return st.get_tensor(next(k for k in st.keys() if k.endswith("layers.0."+n))).astype(np.float32)
def rmsnorm(x, w): return (x*(1.0/np.sqrt(np.mean(x*x,-1,keepdims=True)+EPS)))*(1.0+w)
def rope(t, base):
    m,nh,hd=t.shape; half=hd//2; inv=1.0/(base**(2*np.arange(half)/hd)); ang=np.arange(m)[:,None]*inv[None,:]
    c,s=np.cos(ang)[:,None,:],np.sin(ang)[:,None,:]; x1,x2=t[...,:half],t[...,half:]
    return np.concatenate([x1*c-x2*s, x1*s+x2*c],-1)
def fill(t,a): t.torch_view().copy_(torch.from_numpy(np.ascontiguousarray(a).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))
def npx(t): return t.to_torch().float().cpu().numpy()

# prepare q/k/v (qk-norm + rope) in numpy from oracle projections
q = rope(rmsnorm(ref["layer0__self_attn__q_proj"][0].reshape(M,NH,HD), Wt("self_attn.q_norm.weight")), ROPE_BASE)
k = rope(rmsnorm(ref["layer0__self_attn__k_proj"][0].reshape(M,NKV,HD), Wt("self_attn.k_norm.weight")), ROPE_BASE)[:,0,:]  # [256,256]
v = ref["layer0__self_attn__v_proj"][0].reshape(M,NKV,HD)[:,0,:]  # [256,256]

# numpy reference context (bidirectional GQA)
sc = np.einsum("qhd,kd->hqk", q, k)*SCALE; sc -= sc.max(-1,keepdims=True)
p = np.exp(sc); p /= p.sum(-1,keepdims=True); ctx_ref = np.einsum("hqk,kd->qhd", p, v)  # [256,3,256]

print("compiling scores-GEMM / softmax / context-GEMM (reused across 3 heads) ...")
g_ty = dict(M=M, K=HD, N=HD, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=4)
scores_gemm = GEMM(**g_ty, b_col_maj=True, context=AIEContext()).compile().get_callable()   # Q@K^T
sm = Softmax(rows=M, cols=HD, num_aie_columns=8, context=AIEContext()).compile().get_callable()
ctx_gemm = GEMM(**g_ty, b_col_maj=False, context=AIEContext()).compile().get_callable()      # P@V

Kb = XRTTensor((M*HD,), dtype=BF16); fill(Kb, k)
Vb = XRTTensor((M*HD,), dtype=BF16); fill(Vb, v)
out = np.zeros((M, NH, HD), np.float32)
for h in range(NH):
    Qh = XRTTensor((M*HD,), dtype=BF16); fill(Qh, q[:,h,:]*SCALE)   # scale folded into Q
    S  = XRTTensor((M*HD,), dtype=BF16); fill(S, np.zeros(M*HD,np.float32))
    scores_gemm(Qh, Kb, S)                                          # [256,256] scores
    P  = XRTTensor((M*HD,), dtype=BF16); fill(P, np.zeros(M*HD,np.float32))
    sm(S, P)                                                        # softmax rows
    C  = XRTTensor((M*HD,), dtype=BF16); fill(C, np.zeros(M*HD,np.float32))
    ctx_gemm(P, Vb, C)                                             # [256,256] context
    out[:, h, :] = npx(C).reshape(M, HD)

rel = np.abs(out - ctx_ref).max() / (np.abs(ctx_ref).max() + 1e-9)
print(f"NPU composed attention vs numpy: max_abs={np.abs(out-ctx_ref).max():.3e} rel={rel:.3e}")
print("STEP 3 (composed)", "PASS ✅ (bf16 tol)" if rel < 6e-2 else f"CHECK rel={rel:.3e}")
