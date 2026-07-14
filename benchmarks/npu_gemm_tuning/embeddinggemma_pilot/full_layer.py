#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rung A culmination: one full EmbeddingGemma layer, heavy ops on the NPU.

Assembles the whole Gemma3 encoder layer using verified IRON operators (RMSNorm,
GEMM, softmax, gelu[tanh], elementwise_mul) wired in Python, with the trivial glue
(residual adds, qk-norm, RoPE, reshapes) in numpy. Verified vs the HF oracle's final
layer-0 output. Proves the operators compose to the correct layer before fusion
(step 6) collapses them into one ELF. Everything bf16 (NPU-native).
"""
import os, numpy as np, ml_dtypes
from safetensors import safe_open
from iron.common.context import AIEContext          # aie/iron before torch
from iron.operators.gemm.op import GEMM
from iron.operators.rms_norm.op import RMSNorm
from iron.operators.softmax.op import Softmax
from iron.operators.gelu.op import GELU
from iron.operators.elementwise_mul.op import ElementwiseMul
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
BF16 = ml_dtypes.bfloat16
M, H, NH, NKV, HD, FF = 256, 768, 3, 1, 256, 1152
EPS, RB, SC = 1e-6, 10000.0, 1.0/np.sqrt(256)
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))
_W = {}
with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
    for k in st.keys():
        if "layers.0." in k: _W[k.split("layers.0.")[1]] = st.get_tensor(k).astype(np.float32)
def rms_np(x, w): return (x*(1.0/np.sqrt(np.mean(x*x,-1,keepdims=True)+EPS)))*(1.0+w)
def rope(t, base):
    m,nh,hd=t.shape; half=hd//2; inv=1.0/(base**(2*np.arange(half)/hd)); a=np.arange(m)[:,None]*inv[None,:]
    c,s=np.cos(a)[:,None,:],np.sin(a)[:,None,:]; x1,x2=t[...,:half],t[...,half:]
    return np.concatenate([x1*c-x2*s, x1*s+x2*c],-1)
def fill(t,a): t.torch_view().copy_(torch.from_numpy(np.ascontiguousarray(a).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))
def npx(t): return t.to_torch().float().cpu().numpy()
def T(shape): x=XRTTensor((int(np.prod(shape)),),dtype=BF16); return x

_cache={}
def gemm(x, w_oi, Mm, Kk, Nn, bcm):  # y = x @ w_oi^T  (w_oi is HF [out,in])
    key=("g",Mm,Kk,Nn,bcm)
    if key not in _cache:
        cols=6 if Nn==1152 else 4
        _cache[key]=GEMM(M=Mm,K=Kk,N=Nn,tile_m=64,tile_k=64,tile_n=64,num_aie_columns=cols,
                         b_col_maj=bcm,context=AIEContext()).compile().get_callable()
    A=T((Mm*Kk,)); fill(A,x); B=T((Nn*Kk,)); fill(B,w_oi); C=T((Mm*Nn,)); fill(C,np.zeros(Mm*Nn))
    _cache[key](A,B,C); return npx(C).reshape(Mm,Nn)
def rmsn(x, w):  # weighted RMSNorm over H=768, weight already (1+w)
    if "r" not in _cache:
        _cache["r"]=RMSNorm(size=M*H,num_aie_columns=8,num_channels=1,tile_size=H,weighted=True,context=AIEContext()).compile().get_callable()
    A=T((M*H,)); fill(A,x); Bw=T((H,)); fill(Bw,w); C=T((M*H,)); fill(C,np.zeros(M*H))
    _cache["r"](A,Bw,C); return npx(C).reshape(M,H)
def softmax(s):
    if "s" not in _cache:
        _cache["s"]=Softmax(rows=M,cols=HD,num_aie_columns=8,context=AIEContext()).compile().get_callable()
    A=T((M*HD,)); fill(A,s); C=T((M*HD,)); fill(C,np.zeros(M*HD)); _cache["s"](A,C); return npx(C).reshape(M,HD)
def gelu(x):
    if "ge" not in _cache:
        _cache["ge"]=GELU(size=M*FF,num_aie_columns=8,num_channels=1,tile_size=FF,context=AIEContext()).compile().get_callable()
    A=T((M*FF,)); fill(A,x); C=T((M*FF,)); fill(C,np.zeros(M*FF)); _cache["ge"](A,C); return npx(C).reshape(M,FF)
def emul(a,b):
    if "m" not in _cache:
        _cache["m"]=ElementwiseMul(size=M*FF,num_aie_columns=8,tile_size=FF,context=AIEContext()).compile().get_callable()
    A=T((M*FF,)); fill(A,a); B=T((M*FF,)); fill(B,b); C=T((M*FF,)); fill(C,np.zeros(M*FF)); _cache["m"](A,B,C); return npx(C).reshape(M,FF)

def chk(name, got, key):
    e = ref[key][0].astype(np.float32) if ref[key].ndim==3 else ref[key].astype(np.float32)
    e = e.reshape(got.shape)
    r = np.abs(got-e).max()/(np.abs(e).max()+1e-9)
    c = float((got.reshape(-1)@e.reshape(-1))/(np.linalg.norm(got)*np.linalg.norm(e)+1e-9))
    print(f"  [chk] {name:14s} rel={r:.3e} cos={c:.5f}")

print("running full layer (heavy ops on NPU) ...")
x = ref["layer0__in"][0].astype(np.float32)
xn = rmsn(x, 1.0+_W["input_layernorm.weight"])
q = gemm(xn,_W["self_attn.q_proj.weight"],M,H,H,True)
k = gemm(xn,_W["self_attn.k_proj.weight"],M,H,HD,True)
v = gemm(xn,_W["self_attn.v_proj.weight"],M,H,HD,True)
qh = rope(rms_np(q.reshape(M,NH,HD),_W["self_attn.q_norm.weight"]),RB)
kh = rope(rms_np(k.reshape(M,NKV,HD),_W["self_attn.k_norm.weight"]),RB)[:,0,:]
vv = v.reshape(M,HD)
ctx = np.zeros((M,NH,HD),np.float32)
for h in range(NH):
    sc = gemm(qh[:,h,:]*SC, kh, M, HD, HD, True)       # Q@K^T
    p  = softmax(sc)
    ctx[:,h,:] = gemm(p, vv, M, HD, HD, False)          # P@V
chk("q_proj", q, "layer0__self_attn__q_proj"); chk("k_proj", k, "layer0__self_attn__k_proj")
o = gemm(ctx.reshape(M,H),_W["self_attn.o_proj.weight"],M,H,H,True)
chk("self_attn", o, "layer0__self_attn")
xa = x + rmsn(o, 1.0+_W["post_attention_layernorm.weight"])
fn = rmsn(xa, 1.0+_W["pre_feedforward_layernorm.weight"])
g = gemm(fn,_W["mlp.gate_proj.weight"],M,H,FF,True)
u = gemm(fn,_W["mlp.up_proj.weight"],M,H,FF,True)
chk("gate_proj", g, "layer0__mlp__gate_proj"); chk("up_proj", u, "layer0__mlp__up_proj")
hid = emul(gelu(g), u)
d = gemm(hid,_W["mlp.down_proj.weight"],M,FF,H,True)  # Wdown is [out,in] -> b_col_maj=True
chk("mlp(down)", d, "layer0__mlp")
out = xa + rmsn(d, 1.0+_W["post_feedforward_layernorm.weight"])

exp = ref["layer0"][0].astype(np.float32)
rel = np.abs(out-exp).max()/(np.abs(exp).max()+1e-9)
cos = float((out.reshape(-1)@exp.reshape(-1))/(np.linalg.norm(out)*np.linalg.norm(exp)))
print(f"FULL LAYER (NPU) vs HF: max_abs={np.abs(out-exp).max():.3e} rel={rel:.3e} cos={cos:.6f}")
print("RUNG A", "PASS ✅ full layer on NPU matches HF" if (rel<8e-2 and cos>0.999) else f"CHECK rel={rel:.3e} cos={cos:.5f}")
