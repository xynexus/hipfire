"""Emit a qwen4_exp reference forward: weights + per-layer activations.

Runs the PINNED upstream implementation (transformers @5f8ab9bb, the revision
`third_party/transformers-qwen4_exp/PROVENANCE.md` names) on a tiny config that
keeps the real STRUCTURE — 3 GatedDeltaNet layers then 1 sparse-attention layer,
one PLE layer, 4-wide hyper-connections, a routed MoE — at toy dimensions.

Output: `oracle.bin` (flat f32, little-endian) + `oracle.json` (manifest).
The Rust side differences its CPU trunk against this.
"""
import json, os, sys, torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

torch.manual_seed(0)

cfg = Qwen4ExpTextConfig(
    vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    full_attention_interval=4,
    # QSA: budget 8 / ratio 4 => block_topk 2 of 4 blocks at 16 tokens, so the
    # selection actually EXCLUDES rather than degenerating to dense.
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
    indexer_budget=8, indexer_compress_ratio=4,
    hc_count=4, hc_lowrank=16,
    ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=2,
    ngram_vocab_size_base=2000, make_ngram_vocab_size_divisible_by=8,
    # Match the shipped checkpoint's shard count. 8064 % 128 == 0, so the
    # sharded and concatenated forms correspond exactly and the test can compare
    # the port's per-shard plan against the reference's single table.
    split_ngram_parts=128,
    # The shipped checkpoint sets this; leaving it unset makes the reference fall
    # back to `hidden_act` (silu) and the oracle would then test the wrong gate.
    output_gate_type="sigmoid",
    seed=1234,
    rms_norm_eps=1e-6, max_position_embeddings=256, eos_token_id=2,
    tie_word_embeddings=False,   # the real model has an untied head
)
print("layer_types:", cfg.layer_types, file=sys.stderr)

# ForCausalLM, not the bare text model: it carries `lm_head` (the real model's
# head is untied) and produces logits, which is the end-to-end comparison point.
model = Qwen4ExpForCausalLM(cfg).to(torch.float32).eval()
# Deterministic, well-scaled weights: default init leaves some buffers zero,
# which would hide sign/order bugs.
g = torch.Generator().manual_seed(7)
with torch.no_grad():
    for name, p in model.named_parameters():
        p.copy_(torch.empty(p.shape).uniform_(-0.5, 0.5, generator=g))

# Per-submodule capture. The recorded `hidden_states` give layer INPUTS and the
# final collapsed state, which is not enough to localise a mismatch: it cannot
# isolate the mixer, and inside a layer it cannot say whether PLE, the token
# mixer, or the MoE drifted. Hook every seam instead.
CAP = {}


def hook(tag):
    def fn(_m, args, out):
        if isinstance(out, tuple):
            # Keep EVERY element: the rotary module returns (cos, sin) and
            # dropping one would make the pair untestable.
            for n, t in enumerate(out):
                if torch.is_tensor(t):
                    CAP[f"{tag}.{n}" if n else tag] = t.detach().clone()
        else:
            CAP[tag] = out.detach().clone()
        if args and torch.is_tensor(args[0]):
            CAP[f"{tag}.in"] = args[0].detach().clone()
    return fn


trunk = model.model.language_model if hasattr(model.model, "language_model") else model.model
trunk.hyper_connection_mixer.register_forward_hook(hook("mixer"))
for i, lyr in enumerate(trunk.layers):
    lyr.register_forward_hook(hook(f"layer{i}"))
    lyr.mlp.register_forward_hook(hook(f"layer{i}.mlp"))
    if getattr(lyr, "ple", None) is not None:
        lyr.ple.register_forward_hook(hook(f"layer{i}.ple"))
        # The n-gram table lookup on its own, so the HASHING can be differenced
        # without the projections and the conv on top of it.
        lyr.ple.ple_embedding.register_forward_hook(hook(f"layer{i}.ple_embedding"))
        # `.in` of the inner nn.Embedding IS the hashed n-gram row ids.
        lyr.ple.ple_embedding.ngram_embedding.register_forward_hook(
            hook(f"layer{i}.ngram_rows"))
    if hasattr(lyr, "linear_attn"):
        lyr.linear_attn.register_forward_hook(hook(f"layer{i}.linear_attn"))
        # Stage taps inside the mixer, so a mismatch localises to the projection,
        # the conv, the delta rule, or the output norm.
        for sub in ("in_proj_qkv", "conv1d", "norm", "out_proj"):
            getattr(lyr.linear_attn, sub).register_forward_hook(
                hook(f"layer{i}.la.{sub}"))
    else:
        lyr.self_attn.register_forward_hook(hook(f"layer{i}.self_attn"))
        # Stage taps: the indexer's token mask, and the projections. Lets the
        # attention block be differenced without also reimplementing mrope.
        lyr.self_attn.indexer.register_forward_hook(hook(f"layer{i}.sa.indexer"))
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(lyr.self_attn, sub).register_forward_hook(hook(f"layer{i}.sa.{sub}"))
trunk.rotary_emb.register_forward_hook(hook("rotary"))

ids = torch.tensor([[3, 17, 42, 5, 99, 7, 61, 23,
                     11, 2, 88, 34, 6, 71, 19, 55]], dtype=torch.long)
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

blob, manifest = bytearray(), {}
def put(name, t):
    t = t.detach().contiguous().cpu()
    # Integer tensors go into the manifest as EXACT values. The n-gram hash
    # multipliers run to ~7e16 (max_int64 // vocab_size); rounding them through
    # f32 would silently change every hash, and the test would then be checking
    # the wrong arithmetic.
    if not t.is_floating_point():
        # `.long()` first: a bool tensor would otherwise serialise as JSON
        # true/false, which is not an integer on the reading side.
        manifest[name] = {"shape": list(t.shape), "ints": t.long().flatten().tolist()}
        return
    t = t.to(torch.float32)
    manifest[name] = {"shape": list(t.shape), "offset": len(blob), "numel": t.numel()}
    blob.extend(t.numpy().tobytes())

for name, p in model.named_parameters():
    put(f"w.{name}", p)
for name, b in model.named_buffers():
    put(f"b.{name}", b)
put("input_ids", ids)
for i, h in enumerate(out.hidden_states):
    put(f"h.{i}", h)
put("out.logits", out.logits)
for tag, t in CAP.items():
    put(f"a.{tag}", t)

# ── vision tower ────────────────────────────────────────────────────────────
# A separate subsystem with its own config; the reference DOES implement it
# (unlike the MTP head, which it drops on load), so it gets a real oracle too.
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

vcfg = Qwen4ExpVisionConfig(
    depth=2, hidden_size=64, num_heads=4, intermediate_size=128,
    in_channels=3, patch_size=4, temporal_patch_size=2,
    # out_hidden MUST equal the text hidden: merged vision tokens are scattered
    # directly into the text embedding stream.
    spatial_merge_size=2, out_hidden_size=64,
    num_position_embeddings=64, hidden_act="gelu_pytorch_tanh",
)
vmodel = Qwen4ExpVisionModel(vcfg).to(torch.float32).eval()
with torch.no_grad():
    for _, prm in vmodel.named_parameters():
        prm.copy_(torch.empty(prm.shape).uniform_(-0.5, 0.5, generator=g))

VCAP = {}


def vhook(tag):
    def fn(_m, args, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            VCAP[tag] = t.detach().clone()
        if args and torch.is_tensor(args[0]):
            VCAP[f"{tag}.in"] = args[0].detach().clone()
    return fn


vmodel.patch_embed.register_forward_hook(vhook("patch_embed"))
# The rotary tables and the interpolated position embedding, so the blocks can be
# differenced without also reimplementing the grid interpolation.
vmodel.rotary_pos_emb.register_forward_hook(vhook("rotary"))
vmodel.pos_embed.register_forward_hook(vhook("pos_embed"))
vmodel.merger.register_forward_hook(vhook("merger"))
for i, blk in enumerate(vmodel.blocks):
    blk.register_forward_hook(vhook(f"block{i}"))
    blk.attn.register_forward_hook(vhook(f"block{i}.attn"))
    blk.mlp.register_forward_hook(vhook(f"block{i}.mlp"))

# One 4x4-patch image: grid_thw = [t, h, w]; patches carry
# in_channels * temporal_patch_size * patch_size^2 values each.
grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
n_patch = int(grid[0].prod())
per = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
pixels = torch.empty(n_patch, per).uniform_(-1.0, 1.0, generator=g)
with torch.no_grad():
    vout = vmodel(pixels, grid_thw=grid)

for name, prm in vmodel.named_parameters():
    put(f"vw.{name}", prm)
for name, b in vmodel.named_buffers():
    put(f"vb.{name}", b)
put("v.pixels", pixels)
put("v.grid_thw", grid)
put("v.last_hidden_state", vout.last_hidden_state)
put("v.pooler_output", vout.pooler_output)
for tag, t in VCAP.items():
    put(f"va.{tag}", t)
print(f"vision: {n_patch} patches -> {tuple(vout.pooler_output.shape)} merged", file=sys.stderr)

# ── text/vision fusion ──────────────────────────────────────────────────────
# The seam between the two towers: merged vision tokens are scattered into the
# text embedding at the placeholder positions. Captured off the FULL model, since
# that is where the logic lives.
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

IMG_TOK = 60
full_cfg = Qwen4ExpConfig(
    text_config=cfg.to_dict(), vision_config=vcfg.to_dict(),
    image_token_id=IMG_TOK, video_token_id=IMG_TOK + 1,
    vision_start_token_id=IMG_TOK + 2, vision_end_token_id=IMG_TOK + 3,
)
fmodel = Qwen4ExpModel(full_cfg).to(torch.float32).eval()
with torch.no_grad():
    for _, prm in fmodel.named_parameters():
        prm.copy_(torch.empty(prm.shape).uniform_(-0.5, 0.5, generator=g))

FCAP = {}


def pre_hook(_m, args, kw):
    if "inputs_embeds" in kw and torch.is_tensor(kw["inputs_embeds"]):
        FCAP["fused_embeds"] = kw["inputs_embeds"].detach().clone()
    return None


fmodel.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
fmodel.visual.register_forward_hook(
    lambda _m, a, out: FCAP.__setitem__(
        "image_embeds", (out.pooler_output if hasattr(out, "pooler_output") else out).detach().clone()
    )
)

# 4 merged vision tokens (16 patches / merge^2), so the prompt needs 4 placeholders.
f_ids = torch.tensor([[5, 9, IMG_TOK, IMG_TOK, IMG_TOK, IMG_TOK, 11, 3]], dtype=torch.long)
# `mm_token_type_ids` marks each position's modality (0 = text, 1 = image); M-RoPE
# needs it to give image tokens their 2-D grid positions rather than a running
# text index. The processor normally emits it alongside `input_ids`.
mm_types = (f_ids == IMG_TOK).to(torch.int32)
with torch.no_grad():
    fmodel(
        input_ids=f_ids, pixel_values=pixels, image_grid_thw=grid,
        mm_token_type_ids=mm_types, use_cache=False,
    )
put("f.mm_token_type_ids", mm_types)

put("f.input_ids", f_ids)
put("f.image_token_id", torch.tensor([IMG_TOK], dtype=torch.long))
put("f.embed_tokens", fmodel.language_model.embed_tokens.weight)
for tag, t in FCAP.items():
    put(f"f.{tag}", t)
print(f"fusion: {int((f_ids == IMG_TOK).sum())} placeholders", file=sys.stderr)

open(f"{OUT}/oracle.bin", "wb").write(bytes(blob))
# `text_config`-wrapped so `Qwen4ExpConfig::from_json` reads it unmodified —
# the same parser the shipped checkpoint goes through.
json.dump({"config": {"text_config": cfg.to_dict(), "vision_config": vcfg.to_dict()},
           "tensors": manifest},
          open(f"{OUT}/oracle.json", "w"), indent=1, default=str)
print(f"wrote {len(blob)} B, {len(manifest)} tensors, {len(out.hidden_states)} hidden states",
      file=sys.stderr)
print("logits[0,-1,:6]:", out.logits[0, -1, :6].tolist(), file=sys.stderr)

# ── Reproducing this ────────────────────────────────────────────────────────
# `qwen4_exp` is not in the transformers release installed on this box (5.2.0),
# and the vendored copy under third_party/ targets a NEWER PreTrainedConfig, so
# it cannot simply be dropped in. Install the pinned revision beside the main
# venv (which stays untouched) and run against that:
#
#   pip install --no-deps --target ./tfpin \
#     "git+https://github.com/huggingface/transformers@5f8ab9bb53ec9e0c9329153d18bd825ff1db80f9"
#   pip install --target ./tfpin --upgrade "tokenizers>=0.23.1,<0.24.0" "safetensors>=0.8.0" regex
#   PYTHONPATH=./tfpin python scripts/qwen4exp_oracle.py \
#     crates/hipfire-arch-qwen4exp/tests/oracle
#
# Do NOT shim the missing names into the installed transformers instead: the
# skew reaches `create_recurrent_attention_mask`, which is compute-semantic, and
# a stubbed oracle silently stops being the reference.
