#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Capture a 31B Gemma 4 BF16 oracle one upstream decoder layer at a time.

This is the memory-bounded equivalent of `capture_transformers_reference.py`
for long boundary prompts. It uses the released Transformers Gemma 4 modules,
BF16 source tensors, SDPA attention, masks, RoPE, final norm, tied head, and
softcap unchanged; only weight residency is streamed. It intentionally supports
one greedy token because the Phase 5 SWA boundary cases need the first decision.
"""

from __future__ import annotations

import argparse
from collections import UserDict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F
import transformers
from transformers import Gemma4Config
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextDecoderLayer,
    Gemma4TextRotaryEmbedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-chunk-rows", type=int, default=4096)
    return parser.parse_args()


def exact_input(path: Path) -> tuple[list[int], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if isinstance(value, dict) and "input_ids_pattern" in value:
        pattern = value["input_ids_pattern"]
        length = value.get("length")
        if not isinstance(pattern, list) or not pattern or not isinstance(length, int) or length <= 0:
            raise ValueError("input_ids_pattern and length must both be nonzero")
        values = [pattern[index % len(pattern)] for index in range(length)]
    else:
        values = value.get("input_ids") if isinstance(value, dict) else value
    if not isinstance(values, list) or not values:
        raise ValueError("--input-ids must resolve to a nonempty token array")
    if any(not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF for value in values):
        raise ValueError("every exact input token must be a u32")
    return values, hashlib.sha256(raw).hexdigest()


def parse_layers(raw: str, count: int) -> list[int]:
    layers = sorted({int(value) for value in raw.split(",")})
    if any(not 0 <= layer < count for layer in layers):
        raise ValueError(f"capture layers {layers} exceed layer count {count}")
    return layers


class TensorSource:
    def __init__(self, model: Path):
        self.model = model
        index = json.loads(model.joinpath("model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        try:
            filename = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"source tensor {name} is missing") from error
        # Do not retain shard handles: an open mmap keeps every previously read
        # layer resident in host memory even after its module is evicted.
        with safe_open(self.model / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def rows(self, name: str, start: int, end: int) -> torch.Tensor:
        try:
            filename = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"source tensor {name} is missing") from error
        with safe_open(self.model / filename, framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[start:end]


def load_layer(source: TensorSource, config, layer_index: int, device: torch.device):
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, layer_index)
    prefix = f"model.language_model.layers.{layer_index}."
    state = {
        name: source.tensor(prefix + name)
        for name in layer.state_dict()
    }
    layer.load_state_dict(state, strict=True, assign=True)
    return layer.to(device=device, dtype=torch.bfloat16).eval()


def load_norm(source: TensorSource, config, device: torch.device):
    with torch.device("meta"):
        norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.load_state_dict(
        {"weight": source.tensor("model.language_model.norm.weight")},
        strict=True,
        assign=True,
    )
    return norm.to(device=device, dtype=torch.bfloat16).eval()


def embedding_rows(
    source: TensorSource, input_ids: list[int], config, device: torch.device
) -> torch.Tensor:
    name = "model.language_model.embed_tokens.weight"
    unique = sorted(set(input_ids))
    rows = {token: source.rows(name, token, token + 1) for token in unique}
    hidden = torch.cat([rows[token] for token in input_ids], dim=0)
    scale = torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16)
    return (hidden * scale).unsqueeze(0).to(device)


def tied_logits(
    source: TensorSource,
    hidden: torch.Tensor,
    config,
    device: torch.device,
    chunk_rows: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, config.vocab_size, chunk_rows):
        end = min(start + chunk_rows, config.vocab_size)
        weight = source.rows(
            "model.language_model.embed_tokens.weight", start, end
        ).to(device)
        logits = F.linear(hidden, weight)
        if config.final_logit_softcapping is not None:
            cap = config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        chunks.append(logits.float().cpu())
        del weight, logits
    return torch.cat(chunks, dim=-1)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 1:
        raise ValueError("the streaming boundary oracle supports exactly one greedy token")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA/ROCm device requested but unavailable")

    input_ids, input_sha256 = exact_input(args.input_ids)
    wrapper = Gemma4Config.from_pretrained(args.model, local_files_only=True)
    config = wrapper.text_config
    config._attn_implementation = "sdpa"
    layers = parse_layers(args.layers, config.num_hidden_layers)
    device = torch.device(args.device)

    with torch.no_grad():
        source = TensorSource(args.model)
        hidden_states = embedding_rows(source, input_ids, config, device)
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
        mask_kwargs = {
            "config": config,
            "inputs_embeds": hidden_states,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        rotary = Gemma4TextRotaryEmbedding(config, device=device).to(device)
        position_embeddings = {
            layer_type: rotary(hidden_states, position_ids, layer_type)
            for layer_type in set(config.layer_types)
        }
        shared_kv_states = UserDict()
        selected: dict[str, np.ndarray] = {}

        for layer_index in range(config.num_hidden_layers):
            layer = load_layer(source, config, layer_index, device)
            layer_type = config.layer_types[layer_index]
            hidden_states = layer(
                hidden_states,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=masks[layer_type],
                position_ids=position_ids,
                past_key_values=None,
            )
            if layer_index in layers:
                selected[f"hidden_layer_{layer_index}"] = (
                    hidden_states[:, -1:, :].float().cpu().numpy()
                )
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"streaming oracle layer {layer_index + 1}/{config.num_hidden_layers}",
                flush=True,
            )

        norm = load_norm(source, config, device)
        final_hidden = norm(hidden_states[:, -1:, :])
        final_logits = tied_logits(
            source,
            final_hidden,
            config,
            device,
            args.head_chunk_rows,
        )
        next_token = int(torch.argmax(final_logits[0, 0]).item())

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "capture.npz",
        input_ids=np.asarray(input_ids, dtype=np.uint32),
        final_hidden=final_hidden.float().cpu().numpy(),
        final_logits=final_logits.numpy(),
        generated_ids=np.asarray([input_ids + [next_token]], dtype=np.uint32),
        **selected,
    )
    revision = None
    cache_ref = args.model.parent.parent / "refs" / "main"
    if cache_ref.is_file():
        revision = cache_ref.read_text().strip()
    metadata = {
        "schema": "hipfire.gemma4.transformers-streaming-reference.v1",
        "model": str(args.model.resolve()),
        "revision": revision,
        "dtype": "bfloat16",
        "device": str(device),
        "execution": "layer_streamed_full_sequence",
        "attention_implementation": "sdpa",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "input_kind": "exact_token_ids",
        "input_path": str(args.input_ids.resolve()),
        "input_sha256": input_sha256,
        "input_ids": input_ids,
        "input_token_count": len(input_ids),
        "captured_layers": layers,
        "selected_decoder_layers": layers,
        "max_new_tokens": 1,
        "generated_ids": [next_token],
        "head_chunk_rows": args.head_chunk_rows,
    }
    args.output.joinpath("capture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"streaming Gemma 4 oracle: PASS ({args.output})")


if __name__ == "__main__":
    main()
