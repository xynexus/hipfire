"""Compute Gemma 4 26B-A4B layer 0 forward in Python via safetensors.

Compares against Rust HIPFIRE_DUMP_LAYER0/HIPFIRE_DUMP_MOE_VALS dumps to
localize where the engine diverges.

Computes only layer 0 (sliding) for prompt token <bos>=2 at pos=0 (no KV
context). All ops in float32 for clean comparison.
"""
import json, os, math, sys, glob
import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAP = sorted(glob.glob('/mnt/nas/kaden/cache/huggingface/hub/models--google--gemma-4-26B-A4B/snapshots/*/'))[0]

def load(name):
    idx = json.load(open(os.path.join(SNAP,'model.safetensors.index.json')))
    fn = idx['weight_map'][name]
    with safe_open(os.path.join(SNAP, fn), framework='pt') as f:
        return f.get_tensor(name).to(torch.float32)

def stat(t, label):
    v = t.flatten().float()
    print(f'  {label:38} n={v.numel():6}  rms={float(v.pow(2).mean().sqrt()):.4f}  '
          f'min={float(v.min()):.4f}  max={float(v.max()):.4f}  '
          f'abs_mean={float(v.abs().mean()):.4f}')

def rmsnorm(x, weight=None, eps=1e-6):
    """Gemma4RMSNorm: x / sqrt(mean(x^2)+eps), then optionally * weight (no +1)."""
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).pow(-0.5)
    out = x * rms
    if weight is not None:
        out = out * weight
    return out

# ── Config (from config.json) ──
cfg = json.load(open(os.path.join(SNAP,'config.json')))['text_config']
hidden = cfg['hidden_size']  # 2816
n_heads = cfg['num_attention_heads']  # 16
head_dim = cfg['head_dim']  # 256 (sliding)
n_kv = cfg['num_key_value_heads']  # 8 (sliding)
intermediate = cfg['intermediate_size']  # 2112
mi = cfg['moe_intermediate_size']  # 704
n_experts = cfg['num_experts']  # 128
top_k = cfg['top_k_experts']  # 8
eps = cfg['rms_norm_eps']  # 1e-6
embed_scale = math.sqrt(hidden)  # 53.07

print(f'Config: hidden={hidden} n_heads={n_heads} head_dim={head_dim} n_kv={n_kv} intermediate={intermediate}')
print(f'        moe_intermediate={mi} n_experts={n_experts} top_k={top_k} embed_scale={embed_scale:.4f}')

# ── Load layer 0 weights ──
embed = load('model.language_model.embed_tokens.weight')              # [vocab, hidden]
input_norm_w = load('model.language_model.layers.0.input_layernorm.weight')
post_attn_norm_w = load('model.language_model.layers.0.post_attention_layernorm.weight')
pre_ffn_norm_w = load('model.language_model.layers.0.pre_feedforward_layernorm.weight')
post_ffn_norm_w = load('model.language_model.layers.0.post_feedforward_layernorm.weight')
post_norm_1_w = load('model.language_model.layers.0.post_feedforward_layernorm_1.weight')
post_norm_2_w = load('model.language_model.layers.0.post_feedforward_layernorm_2.weight')
pre_norm_2_w = load('model.language_model.layers.0.pre_feedforward_layernorm_2.weight')

q_proj = load('model.language_model.layers.0.self_attn.q_proj.weight')      # [4096, 2816]
k_proj = load('model.language_model.layers.0.self_attn.k_proj.weight')      # [2048, 2816]
v_proj = load('model.language_model.layers.0.self_attn.v_proj.weight')      # [2048, 2816]
o_proj = load('model.language_model.layers.0.self_attn.o_proj.weight')      # [2816, 4096]
q_norm_w = load('model.language_model.layers.0.self_attn.q_norm.weight')    # [256]
k_norm_w = load('model.language_model.layers.0.self_attn.k_norm.weight')    # [256]

gate_proj = load('model.language_model.layers.0.mlp.gate_proj.weight')      # [2112, 2816]
up_proj = load('model.language_model.layers.0.mlp.up_proj.weight')          # [2112, 2816]
down_proj = load('model.language_model.layers.0.mlp.down_proj.weight')      # [2816, 2112]

router_proj = load('model.language_model.layers.0.router.proj.weight')      # [128, 2816]
router_scale = load('model.language_model.layers.0.router.scale')           # [2816]
per_expert_scale = load('model.language_model.layers.0.router.per_expert_scale')  # [128]
experts_gate_up = load('model.language_model.layers.0.experts.gate_up_proj') # [128, 1408, 2816]
experts_down = load('model.language_model.layers.0.experts.down_proj')      # [128, 2816, 704]

layer_scalar = float(load('model.language_model.layers.0.layer_scalar'))
print(f'Layer 0: layer_scalar = {layer_scalar:.4f}')

# ── Embedding lookup ──
token = 2  # <bos>
x = embed[token].clone() * embed_scale
stat(x, 'embed[<bos>] * sqrt(hidden)')

# ── Layer 0 sliding ──
residual = x.clone()

# input_layernorm
h = rmsnorm(x, input_norm_w, eps)
stat(h, 'input_layernorm output')

# Q/K/V projections (single-token, batch=1)
q = F.linear(h, q_proj).view(n_heads, head_dim)            # [16, 256]
k = F.linear(h, k_proj).view(n_kv, head_dim)               # [8, 256]
v = F.linear(h, v_proj).view(n_kv, head_dim)               # [8, 256]
stat(q.flatten(), 'q (post q_proj)')
stat(k.flatten(), 'k (post k_proj)')
stat(v.flatten(), 'v (post v_proj)')

# q_norm, k_norm, v_norm (per-head RMSNorm, k_eq_v doesn't apply: sliding has v_proj separate)
q_n = rmsnorm(q, q_norm_w, eps)
k_n = rmsnorm(k, k_norm_w, eps)
v_n = rmsnorm(v, None, eps)  # v_norm = with_scale=False (no weight)
stat(q_n.flatten(), 'q (post q_norm)')
stat(k_n.flatten(), 'k (post k_norm)')
stat(v_n.flatten(), 'v (post v_norm)')

# Skip RoPE/attention for now — at pos=0 with no context, attention is just V_n.
# attn_output = V_n[group_for_each_head], reshape to [n_heads*head_dim]
group = n_heads // n_kv  # 16/8 = 2
attn_out = v_n.repeat_interleave(group, dim=0).reshape(n_heads * head_dim)  # [4096]
stat(attn_out, 'attn_output (= V_n at pos=0, GQA-expanded)')

# o_proj
o = F.linear(attn_out, o_proj)
stat(o, 'o_proj output')

# post_attention_layernorm (sandwich)
o_normed = rmsnorm(o, post_attn_norm_w, eps)
stat(o_normed, 'post_attention_layernorm output')

# residual + attn_block
x = residual + o_normed
stat(x, 'post-attn-residual (== `attn_out` for FFN)')

# Save residual for FFN block
residual2 = x.clone()

# pre_feedforward_layernorm
h = rmsnorm(x, pre_ffn_norm_w, eps)
stat(h, 'pre_feedforward_layernorm output')

# Dense MLP (SwiGLU with gelu_pytorch_tanh)
gate = F.linear(h, gate_proj)         # [2112]
up = F.linear(h, up_proj)             # [2112]
gate_act = F.gelu(gate, approximate='tanh')
ffn_hidden = gate_act * up            # [2112]
ffn_out = F.linear(ffn_hidden, down_proj)  # [2816]
stat(ffn_out, 'ffn_out (post-SwiGLU dense MLP)')

# === MoE branch ===
# cur_mlp = post_feedforward_layernorm_1(ffn_out)
cur_mlp = rmsnorm(ffn_out, post_norm_1_w, eps)
stat(cur_mlp, 'cur_mlp (post_norm_1)')

# Router operates on residual2 (post-attn pre-FFN-norm), with raw RMSNorm + scale + sqrt-shrink
router_in = rmsnorm(residual2, None, eps) * router_scale * (hidden ** -0.5)
stat(router_in, 'router_in (rmsnorm * scale / sqrt(d))')

router_logits = F.linear(router_in, router_proj)  # [128]
router_probs = F.softmax(router_logits, dim=-1)
top_w, top_idx = torch.topk(router_probs, k=top_k, dim=-1)
top_w = top_w / top_w.sum(dim=-1, keepdim=True)
top_w = top_w * per_expert_scale[top_idx]
print(f'  router top-{top_k} idx = {top_idx.tolist()}')
print(f'  router top-{top_k} w   = {[f"{w:.4f}" for w in top_w.tolist()]}')

# pre_feedforward_layernorm_2(residual2)
moe_pre2 = rmsnorm(residual2, pre_norm_2_w, eps)
stat(moe_pre2, 'moe_pre2 (pre_norm_2(residual))')

# Per-expert FFN
cur_moe = torch.zeros(hidden)
for k_pos, e in enumerate(top_idx.tolist()):
    e_gate_up = experts_gate_up[e]   # [1408, 2816]
    e_down = experts_down[e]         # [2816, 704]
    g_u = F.linear(moe_pre2, e_gate_up)
    g, u = g_u.chunk(2, dim=-1)
    e_hid = F.gelu(g, approximate='tanh') * u  # [704]
    e_out = F.linear(e_hid, e_down)            # [2816]
    cur_moe = cur_moe + e_out * top_w[k_pos]

stat(cur_moe, 'cur_moe (raw expert-summed)')

cur_moe_n = rmsnorm(cur_moe, post_norm_2_w, eps)
stat(cur_moe_n, 'cur_moe (post_norm_2)')

combined = cur_mlp + cur_moe_n
stat(combined, 'combined = cur_mlp + cur_moe')

post = rmsnorm(combined, post_ffn_norm_w, eps)
stat(post, 'post_ffn_norm(combined)')

# Residual + scale
x = residual2 + post
stat(x, 'post-FFN-residual (before layer_scalar)')

x_scaled = x * layer_scalar
stat(x_scaled, f'layer 0 OUTPUT (after layer_scalar={layer_scalar:.4f})')
