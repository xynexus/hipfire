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


def _load_config_dict(model_id: str) -> dict:
    """Load config.json as a plain dict. Tries transformers AutoConfig
    first (handles config_class quirks); falls back to reading config.json
    directly via HfFileSystem when AutoConfig doesn't recognize model_type
    yet (the "transformers too old for this arch" case that --shapes-only
    is specifically meant to handle).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        return cfg.to_dict()
    except Exception as e:
        print(f"# AutoConfig failed ({type(e).__name__}); reading config.json via HF filesystem",
              file=sys.stderr)
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with fs.open(f"{model_id}/config.json", "rb") as fh:
            return json.loads(fh.read())


def emit_config(model_id: str) -> dict:
    """Pull the raw config.json and surface the keys an arch port cares about."""
    blob = _load_config_dict(model_id)
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
    """List weight names + dtypes + shapes from safetensors HEADER only —
    NO tensor data downloaded.

    Uses `huggingface_hub.parse_safetensors_file_metadata`, which fetches
    only the first ~few KB of each shard (the JSON header) via a Range
    request. For a 35B model that's ~kB total instead of GB.

    Falls back to `HfApi().get_safetensors_metadata` (single roll-up call)
    on huggingface_hub versions that expose it.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    # Preferred: one-shot metadata (modern hf hub).
    if hasattr(api, "get_safetensors_metadata"):
        try:
            meta = api.get_safetensors_metadata(model_id)
            out: list[dict] = []
            # `meta.weight_map` maps tensor-name → shard filename.
            wmap = getattr(meta, "weight_map", {}) or {}
            # `meta.files_metadata` maps shard filename → SafetensorsFileMetadata
            files_meta = getattr(meta, "files_metadata", {}) or {}
            seen = set()
            for shard_name, file_meta in files_meta.items():
                tensors = getattr(file_meta, "tensors", {}) or {}
                for name, t in tensors.items():
                    if name in seen:
                        continue
                    seen.add(name)
                    out.append({
                        "name": name,
                        "dtype": getattr(t, "dtype", None),
                        "shape": list(getattr(t, "shape", []) or []),
                        "shard": shard_name,
                    })
            if out:
                return out
            # If files_metadata was empty for some reason, fall through.
        except Exception as e:
            print(f"# get_safetensors_metadata failed ({e}); falling back to per-file parse",
                  file=sys.stderr)

    # Fallback: per-file header parse via parse_safetensors_file_metadata.
    from huggingface_hub import HfFileSystem
    try:
        from huggingface_hub import parse_safetensors_file_metadata
    except ImportError:
        # Older hf hub (<0.20) doesn't have it; we have to read the
        # header ourselves via a Range request through HfFileSystem.
        parse_safetensors_file_metadata = None  # type: ignore

    fs = HfFileSystem()
    files = [
        f.removeprefix(f"{model_id}/")
        for f in fs.ls(model_id, detail=False)
        if f.endswith(".safetensors") and "index" not in f
    ]
    if not files:
        # Some repos list .safetensors only via the index — fall back to
        # the index file's weight_map.
        try:
            with fs.open(f"{model_id}/model.safetensors.index.json", "rb") as fh:
                idx = json.loads(fh.read())
            shard_names = sorted(set(idx.get("weight_map", {}).values()))
            files = list(shard_names)
        except Exception:
            pass

    out = []
    for shard in files:
        if parse_safetensors_file_metadata is not None:
            file_meta = parse_safetensors_file_metadata(model_id, shard)
            tensors = getattr(file_meta, "tensors", {}) or {}
            for name, t in tensors.items():
                out.append({
                    "name": name,
                    "dtype": getattr(t, "dtype", None),
                    "shape": list(getattr(t, "shape", []) or []),
                    "shard": shard,
                })
        else:
            # Manual header read: u64 LE size + JSON.
            import struct
            with fs.open(f"{model_id}/{shard}", "rb") as fh:
                hdr_size = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hdr_size))
            for name, t in hdr.items():
                if name == "__metadata__":
                    continue
                out.append({
                    "name": name,
                    "dtype": t.get("dtype"),
                    "shape": list(t.get("shape", [])),
                    "shard": shard,
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
