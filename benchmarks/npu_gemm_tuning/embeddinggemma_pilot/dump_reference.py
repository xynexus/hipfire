#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth reference for the EmbeddingGemma fused-encoder pilot.

Loads google/embeddinggemma-300m via transformers (offline, from the local HF
snapshot), runs a fixed tokenized prompt, and captures — for layer 0 — the input
hidden state, the output hidden state, and per-submodule intermediates
(norm/q/k/v/attn/o/ffn). Also dumps the config scalars an IRON reimplementation
must match. Everything saved to reference_layer0.npz for numeric verification of
the IRON graph.
"""
import os, sys, json
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

SNAP = "/srv/huggingface/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
OUT = os.path.join(os.path.dirname(__file__), "reference_layer0.npz")

cfg = AutoConfig.from_pretrained(SNAP)
print("=== config scalars an IRON layer must match ===")
for k in ["hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "intermediate_size",
          "query_pre_attn_scalar", "sliding_window", "rope_theta",
          "rope_local_base_freq", "hidden_activation", "rms_norm_eps",
          "attention_bias", "layer_types", "sliding_window_pattern",
          "final_logit_softcapping", "attn_logit_softcapping"]:
    print(f"  {k}: {getattr(cfg, k, '<absent>')}")

torch.manual_seed(0)
model = AutoModel.from_pretrained(SNAP, torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(SNAP)

# Use a long passage so the sequence fills all 256 positions with no padding ->
# attention_mask is all ones -> pure full bidirectional attention (no mask), which
# keeps the IRON layer simple and faithful for the pilot.
para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "the sun sets slowly behind the distant purple mountains and birds sing. ")
text = para * 40
enc = tok(text, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
assert int(enc["attention_mask"].sum()) == 256, "want a full 256-token, no-pad input"
print("\ninput_ids shape:", tuple(enc["input_ids"].shape),
      "n_real_tokens:", int(enc["attention_mask"].sum()))

# Find the decoder layer list (model.layers on Gemma3TextModel).
base = model
for attr in ("model", "text_model"):
    if hasattr(base, attr):
        base = getattr(base, attr)
layers = base.layers
layer0 = layers[0]
print("layer0 type:", type(layer0).__name__)

caps = {}
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        caps[name] = (o.detach().float().cpu().numpy() if torch.is_tensor(o) else None)
        if isinstance(inp, tuple) and len(inp) and torch.is_tensor(inp[0]):
            caps[name + "__in"] = inp[0].detach().float().cpu().numpy()
    return f

handles = [layer0.register_forward_hook(hook("layer0"))]
for sub in ["input_layernorm", "self_attn", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj", "post_attention_layernorm",
            "pre_feedforward_layernorm", "mlp", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj", "post_feedforward_layernorm"]:
    m = layer0
    for p in sub.split("."):
        m = getattr(m, p)
    handles.append(m.register_forward_hook(hook("layer0." + sub)))

with torch.no_grad():
    model(**enc)
for h in handles:
    h.remove()

save = {"input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy()}
for k, v in caps.items():
    if v is not None:
        save[k.replace(".", "__")] = v
        print(f"  captured {k}: {v.shape}")
np.savez(OUT, **save)
print("\nsaved", OUT)
print("layer0 in ->", save["layer0__in"].shape, " out ->", save["layer0"].shape)
