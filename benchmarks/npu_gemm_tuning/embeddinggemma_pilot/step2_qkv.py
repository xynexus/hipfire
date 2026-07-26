#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rung A step 2: the QKV projections on the NPU, verified vs the oracle.

Three IRON GEMMs (q_proj 768->768, k_proj/v_proj 768->256) at M256, b_col_maj=True
(HF weight [out,in] fed directly for y = x @ W^T). Input is the oracle's
input_layernorm output; outputs verified vs oracle q/k/v_proj (bf16 accum tol).
tile_n=64 * num_aie_columns=4 -> min_N=256 divides both 768 and 256.
"""
import os, numpy as np, ml_dtypes
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
# aie/iron BEFORE torch — otherwise torch's ROCm libLLVM and mlir_aie's LLVM
# double-register llvm::cl options and segfault (same fix as the llama wrapper).
from iron.common.context import AIEContext
from iron.operators.gemm.op import GEMM
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

BF16 = ml_dtypes.bfloat16
M, K = 256, 768
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))
xn = ref["layer0__input_layernorm"][0].astype(np.float32)  # [256,768] (isolates the GEMM)

def W(name):
    with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
        key = next(k for k in st.keys() if k.endswith("layers.0." + name))
        return st.get_tensor(key).astype(np.float32)

def fill(t, arr):
    t.torch_view().copy_(torch.from_numpy(
        np.ascontiguousarray(arr).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))

def gemm_npu(x, w_out_in, N):
    ctx = AIEContext()
    op = GEMM(M=M, K=K, N=N, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=4,
              b_col_maj=True, context=ctx).compile()
    call = op.get_callable()
    A = XRTTensor((M * K,), dtype=BF16); fill(A, x)
    B = XRTTensor((N * K,), dtype=BF16); fill(B, w_out_in)   # HF [out=N, in=K]
    C = XRTTensor((M * N,), dtype=BF16); fill(C, np.zeros(M * N, np.float32))
    call(A, B, C)
    return C.to_torch().float().cpu().numpy().reshape(M, N)

allok = True
for name, N, okey in [("self_attn.q_proj.weight", 768, "layer0__self_attn__q_proj"),
                      ("self_attn.k_proj.weight", 256, "layer0__self_attn__k_proj"),
                      ("self_attn.v_proj.weight", 256, "layer0__self_attn__v_proj")]:
    print(f"compiling+dispatching GEMM M256 K768 N{N} ({name.split('.')[1]}) ...")
    out = gemm_npu(xn, W(name), N)
    exp = ref[okey][0].astype(np.float32)
    rel = np.abs(out - exp).max() / (np.abs(exp).max() + 1e-9)
    ok = rel < 4e-2
    allok &= ok
    print(f"  {name.split('.')[1]:8s} vs HF: max_abs={np.abs(out-exp).max():.3e} rel={rel:.3e} {'OK' if ok else 'CHECK'}")

print("\nSTEP 2", "PASS ✅ (bf16 accum tolerance)" if allok else "CHECK")
