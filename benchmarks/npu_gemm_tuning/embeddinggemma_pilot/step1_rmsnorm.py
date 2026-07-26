#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rung A step 1: RMSNorm on the NPU at EmbeddingGemma dims, verified vs the oracle.

Instantiates the IRON weighted rms_norm operator at [256, 768], dispatches it on
the aie2p NPU, and compares to numpy_reference's input_layernorm (which matches HF).
Proves the operator-reuse + NPU compile/dispatch/verify loop at EmbeddingGemma dims.

Run via run_step1.sh (sets the March mlir_aie overlay + pyxrt). Single-op xclbin —
no full-ELF, so the pinned aiecc is fine.
"""
import os, numpy as np, ml_dtypes

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"

from iron.common.context import AIEContext
from iron.operators.rms_norm.op import RMSNorm
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

M, H = 256, 768
BF16 = ml_dtypes.bfloat16

# oracle input + weight
ref = np.load(os.path.join(HERE, "reference_layer0.npz"))
x = ref["layer0__in"][0].astype(np.float32)                       # [256,768]
exp = ref["layer0__input_layernorm"][0].astype(np.float32)        # HF output
from safetensors import safe_open
with safe_open(os.path.join(SNAP, "model.safetensors"), "np") as st:
    w_raw = st.get_tensor("model.layers.0.input_layernorm.weight").astype(np.float32) \
        if "model.layers.0.input_layernorm.weight" in st.keys() \
        else st.get_tensor("layers.0.input_layernorm.weight").astype(np.float32)
w = (1.0 + w_raw)  # HF Gemma applies (1 + weight)

print("compiling IRON weighted RMSNorm [256,768] ...")
ctx = AIEContext()
op = RMSNorm(size=M * H, num_aie_columns=8, num_channels=1, tile_size=H,
             weighted=True, context=ctx).compile()
call = op.get_callable()

A = XRTTensor((M * H,), dtype=BF16)
B = XRTTensor((H,), dtype=BF16)
C = XRTTensor((M * H,), dtype=BF16)
A.torch_view().copy_(__import__("torch").from_numpy(x.reshape(-1).astype(BF16).view(np.uint16)).view(__import__("torch").bfloat16))
B.torch_view().copy_(__import__("torch").from_numpy(w.astype(BF16).view(np.uint16)).view(__import__("torch").bfloat16))

print("dispatching on NPU ...")
call(A, B, C)
out = C.to_torch().float().cpu().numpy().reshape(M, H)

err = np.abs(out - exp)
rel = err.max() / (np.abs(exp).max() + 1e-9)
print(f"NPU rms_norm vs HF oracle: max_abs={err.max():.4e} rel={rel:.3e}")
print("STEP 1", "PASS ✅ (bf16 tolerance)" if rel < 2e-2 else f"CHECK rel={rel:.3e}")
