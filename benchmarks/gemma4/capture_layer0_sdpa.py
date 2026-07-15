#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Run only Gemma 4 decoder layer 0 on captured Hipfire embeddings.

This is a lightweight numerical probe, not a second admission oracle. It loads
only layer 0 from the pinned safetensors checkpoint, feeds the exact per-token
``pre_layer`` rows captured by Hipfire, and writes the Q/K/V tensors entering
ROCm SDPA plus SDPA's attention output. The raw tensors use Hipfire's
position-major F32 layout for direct HIP kernel parity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer,
    Gemma4TextRotaryEmbedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hipfire", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_pre_layer(capture: Path, hidden_size: int) -> np.ndarray:
    metadata = json.loads(capture.joinpath("capture.json").read_text())
    positions = int(metadata["operator_positions"])
    if metadata.get("operator_layer") != 0 or positions < 1:
        raise ValueError("Hipfire input must be a per-position layer-0 trace")
    rows = []
    for position in range(positions):
        path = capture / f"operator_position_{position}_pre_layer.f32"
        row = np.fromfile(path, dtype="<f4")
        if row.size != hidden_size:
            raise ValueError(
                f"{path} has {row.size} values; expected hidden_size={hidden_size}"
            )
        rows.append(row)
    return np.stack(rows)


def load_layer(model: Path, config, device: torch.device) -> Gemma4TextDecoderLayer:
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = "model.language_model.layers.0."
    names = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not names:
        raise ValueError(f"{index_path} has no {prefix} tensors")
    shards = {shard for shard in names.values()}
    state = {}
    for shard in shards:
        with safe_open(str(model / shard), framework="pt", device=str(device)) as tensors:
            for full_name, mapped_shard in names.items():
                if mapped_shard == shard:
                    state[full_name.removeprefix(prefix)] = tensors.get_tensor(full_name)

    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, 0)
    missing, unexpected = layer.load_state_dict(state, assign=True, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"layer-0 checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
    return layer.eval()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().numpy().astype("<f4", copy=False).tofile(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device requested but unavailable")

    outer_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = outer_config.text_config
    config._attn_implementation = "sdpa"
    hidden = torch.from_numpy(load_pre_layer(args.hipfire, config.hidden_size))
    hidden = hidden.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    layer = load_layer(args.model, config, device)
    rotary = Gemma4TextRotaryEmbedding(config, device=device)
    position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
    position_embeddings = rotary(
        hidden, position_ids, layer_type=config.layer_types[0]
    )

    captured: dict[str, torch.Tensor] = {}
    boundary_modules = {
        "input_norm": layer.input_layernorm,
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "q_norm": layer.self_attn.q_norm,
        "k_norm": layer.self_attn.k_norm,
        "v_norm": layer.self_attn.v_norm,
        "o_proj": layer.self_attn.o_proj,
        "post_attention_norm": layer.post_attention_layernorm,
        "pre_ffn_norm": layer.pre_feedforward_layernorm,
        "gate": layer.mlp.gate_proj,
        "up": layer.mlp.up_proj,
        "gelu": layer.mlp.act_fn,
        "down": layer.mlp.down_proj,
        "post_ffn_norm": layer.post_feedforward_layernorm,
    }

    def save_boundary(name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{name} hook produced {type(output).__name__}")
            captured[name] = output.detach()

        return hook

    handles = [
        module.register_forward_hook(save_boundary(name))
        for name, module in boundary_modules.items()
    ]
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture_sdpa(module, query, key, value, attention_mask, **kwargs):
        captured["q_rope"] = query.transpose(1, 2).contiguous()
        captured["k_rope"] = key.transpose(1, 2).contiguous()
        captured["v_attention"] = value.transpose(1, 2).contiguous()
        output, weights = original_sdpa(
            module, query, key, value, attention_mask, **kwargs
        )
        captured["attention_raw"] = output
        return output, weights

    ALL_ATTENTION_FUNCTIONS["sdpa"] = capture_sdpa
    try:
        with torch.no_grad():
            layer_output = layer(
                hidden,
                shared_kv_states={},
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
            )
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
        for handle in handles:
            handle.remove()

    required = {
        *boundary_modules,
        "q_rope",
        "k_rope",
        "v_attention",
        "attention_raw",
    }
    if captured.keys() != required:
        raise RuntimeError(f"incomplete SDPA capture: {sorted(captured)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        write_f32(args.output / f"operator_{name}.f32", tensor)
    write_f32(args.output / "hidden_layer_0.f32", layer_output)
    metadata = {
        "model": str(args.model.resolve()),
        "hipfire_input": str(args.hipfire.resolve()),
        "dtype": "bfloat16",
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "sequence_length": hidden.shape[1],
        "hidden_size": config.hidden_size,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "scale": 1.0,
    }
    args.output.joinpath("capture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote layer-0 SDPA parity tensors to {args.output}")


if __name__ == "__main__":
    main()
