#!/usr/bin/env python3
"""dump_reference_activations.py — capture per-layer activations from a HF
reference model for layer-by-layer correctness validation of a hipfire
arch port.

Loads model via transformers (bf16 reference), registers forward_pre and
forward hooks on every transformer block sub-module, runs a fixed prompt
through forward, and saves each captured tensor to disk as safetensors.

Output layout:
  <out-dir>/
    manifest.json                        # what was captured + dtype/shape
    input_ids.safetensors                # the prompt's tokens
    layer_<NN>/
      pre_attn_norm.input.safetensors
      pre_attn_norm.output.safetensors
      q_proj.output.safetensors          # after Q projection (and any q_norm)
      k_proj.output.safetensors
      v_proj.output.safetensors
      attention.output.safetensors        # after attention before o_proj
      o_proj.output.safetensors
      post_attn_norm.input.safetensors
      post_attn_norm.output.safetensors
      mlp.gate.output.safetensors
      mlp.up.output.safetensors
      mlp.down.output.safetensors
      residual_out.safetensors
    final_logits.safetensors

The hipfire-side verifier loads these tensors, runs the modular forward
to the same step, and computes NRMSE = sqrt(MSE) / sqrt(var(ref)) per
(layer, step). First divergence above threshold = the broken kernel.

Usage:
    python3 scripts/arch-intake/dump_reference_activations.py \\
        --model google/gemma-4-31B \\
        --prompt "The capital of France is" \\
        --out-dir /tmp/gemma4-ref \\
        --max-seq 32

Designed to run on hiptrx (~/venv-torch with torch 2.11.0+rocm7.2,
transformers 5.8, 4× R9700 with device_map="auto"). For Gemma 4 31B at
bf16 (~58 GB) you need 2× R9700 visible.

The hook system is conservative: it captures EVERY known step name pattern
across architectures (gemma4 sliding/full, qwen35 hybrid+MoE, llama, etc.).
Steps that don't exist on a given arch produce no files. The manifest.json
records what was actually captured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


def _safe_save(tensor, path: Path) -> None:
    """Save a single torch tensor as safetensors (one entry "x")."""
    import torch
    from safetensors.torch import save_file
    if not isinstance(tensor, torch.Tensor):
        return
    # Move to CPU + cast to fp32 for portability across hipfire-side dequant.
    t = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": t}, str(path))


# Ordered list of (regex over module name, step-name we save it as).
# The order matters — first match wins. We aim to be conservative: capture
# every plausible step on common archs (gemma4 / qwen3.5 / llama / mistral).
STEP_PATTERNS = [
    # input/output norms — Gemma 4 has sandwich norms (pre + post)
    (r"\.input_layernorm$",                 "pre_attn_norm"),
    (r"\.post_attention_layernorm$",        "post_attn_norm"),
    (r"\.pre_feedforward_layernorm$",       "pre_mlp_norm"),       # Gemma 4 sandwich
    (r"\.post_feedforward_layernorm$",      "post_mlp_norm"),      # Gemma 4 sandwich
    # q/k/v + their norms (Gemma 4 has q_norm, k_norm)
    (r"\.self_attn\.q_proj$",               "q_proj"),
    (r"\.self_attn\.k_proj$",               "k_proj"),
    (r"\.self_attn\.v_proj$",               "v_proj"),
    (r"\.self_attn\.q_norm$",               "q_norm"),
    (r"\.self_attn\.k_norm$",               "k_norm"),
    (r"\.self_attn\.o_proj$",               "o_proj"),
    # whole self_attn block (captures attention output before o_proj)
    (r"\.self_attn$",                       "self_attn_block"),
    # MLP gate/up/down (Gemma 4 + Qwen 3.5 dense pattern)
    (r"\.mlp\.gate_proj$",                  "mlp_gate"),
    (r"\.mlp\.up_proj$",                    "mlp_up"),
    (r"\.mlp\.down_proj$",                  "mlp_down"),
    # MLP whole block
    (r"\.mlp$",                             "mlp_block"),
    # whole transformer block — captures residual_in (forward_pre_hook) and
    # residual_out (forward_hook)
    (r"^model\.(language_model\.)?layers\.(\d+)$", "block"),
    (r"^layers\.(\d+)$",                    "block"),
]


def classify_step(name: str) -> tuple[int, str] | None:
    """Map a transformers module name to (layer_idx, step_name) tuple, or
    None if the module is outside the transformer block layers we care
    about (vision tower, embed, output projection, etc.)."""
    # Capture layer index first.
    m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if not m:
        return None
    layer_idx = int(m.group(1))
    for pat, step in STEP_PATTERNS:
        if re.search(pat, name):
            return (layer_idx, step)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="HF model ID, e.g. google/gemma-4-31B")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="prompt to forward")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-seq", type=int, default=32,
                    help="truncate prompt to this many tokens")
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# loading {args.model} ({args.dtype}, device_map=auto)", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache,
                                               trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.hf_cache,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    embed_device = next(model.parameters()).device
    print(f"# loaded in {time.time()-t0:.1f}s; embed on {embed_device}", file=sys.stderr)

    # Walk modules, register hooks. For each module we care about:
    #   - forward_pre_hook → save the .input
    #   - forward_hook     → save the .output
    captures: list[dict[str, Any]] = []
    handles = []
    n_hooks = 0
    for name, module in model.named_modules():
        cls = classify_step(name)
        if cls is None:
            continue
        layer_idx, step = cls

        def make_pre_hook(layer_idx, step, name):
            def hook(mod, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                path = out_dir / f"layer_{layer_idx:02d}" / f"{step}.input.safetensors"
                _safe_save(x, path)
                captures.append({
                    "layer_idx": layer_idx,
                    "step": step,
                    "side": "input",
                    "module_name": name,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "path": str(path.relative_to(out_dir)),
                })
            return hook

        def make_post_hook(layer_idx, step, name):
            def hook(mod, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    return
                path = out_dir / f"layer_{layer_idx:02d}" / f"{step}.output.safetensors"
                _safe_save(output, path)
                captures.append({
                    "layer_idx": layer_idx,
                    "step": step,
                    "side": "output",
                    "module_name": name,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "path": str(path.relative_to(out_dir)),
                })
            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx, step, name)))
        handles.append(module.register_forward_hook(make_post_hook(layer_idx, step, name)))
        n_hooks += 2

    print(f"# registered {n_hooks} hooks across {n_hooks // 2} modules",
          file=sys.stderr)

    # Tokenize prompt + save input_ids
    tokens = tokenizer(args.prompt, return_tensors="pt", truncation=True,
                       max_length=args.max_seq)
    input_ids = tokens.input_ids.to(embed_device)
    _safe_save(input_ids, out_dir / "input_ids.safetensors")
    print(f"# prompt: {args.prompt!r}", file=sys.stderr)
    print(f"# tokens: {input_ids.tolist()}", file=sys.stderr)

    # Forward.
    t_fwd = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    elapsed = time.time() - t_fwd
    print(f"# forward in {elapsed:.1f}s; logits shape {out.logits.shape}",
          file=sys.stderr)

    # Save final logits.
    _safe_save(out.logits, out_dir / "final_logits.safetensors")

    # Detach hooks.
    for h in handles:
        h.remove()

    # Manifest.
    manifest = {
        "model": args.model,
        "prompt": args.prompt,
        "input_ids": input_ids.tolist(),
        "n_tokens": int(input_ids.shape[1]),
        "dtype": args.dtype,
        "embed_device": str(embed_device),
        "config_summary": {
            "architectures": getattr(model.config, "architectures", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "head_dim": getattr(model.config, "head_dim", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
        },
        "captures": captures,
        "n_captures": len(captures),
        "wall_time_seconds": round(elapsed, 1),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"# wrote {out_dir / 'manifest.json'} ({len(captures)} captures)",
          file=sys.stderr)

    # Summary by step.
    by_step: dict[str, int] = {}
    by_layer: dict[int, int] = {}
    for c in captures:
        key = f"{c['step']}.{c['side']}"
        by_step[key] = by_step.get(key, 0) + 1
        by_layer[c["layer_idx"]] = by_layer.get(c["layer_idx"], 0) + 1
    print("# by step:", file=sys.stderr)
    for k in sorted(by_step):
        print(f"  {k}: {by_step[k]}", file=sys.stderr)
    print(f"# layers covered: {sorted(by_layer)[:5]} ... {sorted(by_layer)[-5:]} "
          f"(total {len(by_layer)})",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
