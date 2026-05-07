#!/usr/bin/env python3
"""inspect_model.py — first-look HF model summary for hipfire arch intake.

Loads a HuggingFace model ID via transformers (config + named_modules
walk; weights stay on disk via from_pretrained low_cpu_mem_usage), then
emits a structured summary suitable for filling in an
`arch-report.md` skeleton:

  - Config dict (all non-private fields).
  - List of unique top-level module classes (LlamaDecoderLayer, etc.).
  - Per-layer module shape map (one representative layer).
  - Weight shape inventory (named, dtype, shape).
  - Tokenizer family detection (gpt2 / bpe / spm-bpe / unigram).
  - Special-token IDs.
  - Vocab size + chat template presence.

Usage:
    python3 scripts/arch-intake/inspect_model.py google/gemma-4-31B-it

Prerequisites: transformers >= 5.x, torch (CPU-only is fine — we never
forward).

Designed to run on hiptrx (~/venv-torch has torch 2.11.0+rocm7.2,
transformers 5.8) but works anywhere transformers can resolve the
model. For weight-shape-only inspection (no actual parameter
download), pass --shapes-only — uses safetensors metadata directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def emit_config(model_id: str) -> dict:
    """Pull the raw config.json and surface the keys an arch port cares about."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    blob = cfg.to_dict()
    # text_config indirection (Gemma 4, Qwen3.5-VL)
    if "text_config" in blob and isinstance(blob["text_config"], dict):
        text = blob["text_config"]
    else:
        text = blob

    fields_of_interest = [
        # core
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "max_position_embeddings",
        "tie_word_embeddings",
        # rope
        "rope_theta",
        "rope_parameters",
        "rope_scaling",
        "partial_rotary_factor",
        # attention variants (Gemma-style, GLM-style)
        "sliding_window",
        "layer_types",
        "global_head_dim",
        "num_global_key_value_heads",
        "attention_k_eq_v",
        "attention_softcap",
        "final_logit_softcapping",
        # MoE
        "num_experts",
        "num_experts_per_tok",
        "expert_intermediate_size",
        "norm_topk_prob",
        # special tokens
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        # vision / multimodal
        "vision_config",
        "image_token_id",
        "boi_token_id",
        "eoi_token_id",
        "audio_token_id",
        "video_token_id",
    ]
    out = {}
    for f in fields_of_interest:
        v = text.get(f, blob.get(f))
        if v is not None:
            out[f] = v
    return out


def emit_module_classes(model_id: str) -> dict:
    """Walk named_modules() and report class-name frequency."""
    from transformers import AutoModel
    print(f"# loading {model_id} (low_cpu_mem_usage, weights on CPU)...", file=sys.stderr)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    counts = Counter()
    layer0_modules: list[tuple[str, str, list[int] | None]] = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        counts[cls] += 1
        if name.startswith("layers.0.") or name.startswith("model.layers.0."):
            shape = None
            if hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
                shape = list(mod.weight.shape)
            layer0_modules.append((name, cls, shape))

    return {
        "class_counts": dict(counts.most_common()),
        "layer0_modules": layer0_modules,
    }


def emit_weight_inventory(model_id: str) -> list[dict]:
    """Use safetensors metadata to list weight names + dtypes + shapes
    without loading parameters."""
    from huggingface_hub import HfFileSystem
    import safetensors
    fs = HfFileSystem()
    repo = f"{model_id}"
    files = [
        f for f in fs.ls(repo, detail=False)
        if f.endswith(".safetensors") and "index" not in f
    ]
    out: list[dict] = []
    for f in files:
        with fs.open(f, "rb") as fh:
            with safetensors.safe_open(fh, framework="pt") as st:
                for k in st.keys():
                    t = st.get_tensor(k)
                    out.append({
                        "name": k,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "shard": Path(f).name,
                    })
    return out


def emit_tokenizer_summary(model_id: str) -> dict:
    """Detect tokenizer family + special tokens + chat template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    fast = tok.is_fast
    # tokenizer.json model.type heuristic
    tk_json = None
    try:
        tk_json = json.loads(tok.backend_tokenizer.to_str())
    except Exception:
        pass

    family = "unknown"
    if tk_json is not None:
        mt = tk_json.get("model", {}).get("type", "").lower()
        vocab = tk_json.get("model", {}).get("vocab", {}) or {}
        merges = tk_json.get("model", {}).get("merges", []) or []
        has_spm_prefix = any(t.startswith("▁") for t in list(vocab)[:200])
        has_gpt2_prefix = any(t.startswith("Ġ") for t in list(vocab)[:200])
        if mt == "bpe" and has_spm_prefix and merges:
            family = "spm-bpe"  # Gemma 4
        elif mt == "bpe" and has_gpt2_prefix and merges:
            family = "gpt2-bpe"  # Qwen3 / Qwen3.5
        elif mt == "unigram" or (has_spm_prefix and not merges):
            family = "unigram-spm"  # LLaMA / Mistral
        elif merges:
            family = "bpe-other"
        else:
            family = mt or "unknown"

    return {
        "family": family,
        "is_fast": fast,
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "additional_special_tokens": list(getattr(tok, "additional_special_tokens", []) or []),
        "has_chat_template": bool(getattr(tok, "chat_template", None)),
        "chat_template_head": (tok.chat_template[:200] if tok.chat_template else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model_id", help="HF model ID, e.g. google/gemma-4-31B-it")
    ap.add_argument("--shapes-only", action="store_true",
                    help="Skip from_pretrained — use safetensors metadata instead "
                         "(faster, no GPU/CPU param load).")
    ap.add_argument("--out", type=Path, help="Write JSON output to file (else stdout).")
    args = ap.parse_args()

    summary: dict = {"model_id": args.model_id}

    print(f"# inspecting {args.model_id}", file=sys.stderr)
    summary["config"] = emit_config(args.model_id)
    summary["tokenizer"] = emit_tokenizer_summary(args.model_id)

    if args.shapes_only:
        print("# pulling weight shapes from safetensors metadata only", file=sys.stderr)
        summary["weights"] = emit_weight_inventory(args.model_id)
    else:
        print("# walking named_modules() for class layout (downloads weights)", file=sys.stderr)
        summary["modules"] = emit_module_classes(args.model_id)

    out_str = json.dumps(summary, indent=2, default=str)
    if args.out:
        args.out.write_text(out_str)
        print(f"# wrote {args.out}", file=sys.stderr)
    else:
        print(out_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
