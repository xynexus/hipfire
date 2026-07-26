#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Capture selected Transformers Gemma 4 layers from frozen transition inputs.

Only the requested decoder-layer weights are materialized. Each layer receives
the exact input rows prepared by ``prepare_layer_transition_inputs.py`` and
writes its major operator boundaries for direct comparison with Hipfire's
position-by-position transition trace. This is a diagnostic, not a replacement
admission oracle.
"""

from __future__ import annotations

import argparse
import hashlib
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


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(value) for value in raw.split(",")})
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layer(
    model: Path, config, device: torch.device, layer_idx: int
) -> Gemma4TextDecoderLayer:
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    names = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not names:
        raise ValueError(f"{index_path} has no {prefix} tensors")
    state = {}
    for shard in set(names.values()):
        with safe_open(str(model / shard), framework="pt", device=str(device)) as tensors:
            for full_name, mapped_shard in names.items():
                if mapped_shard == shard:
                    state[full_name.removeprefix(prefix)] = tensors.get_tensor(full_name)

    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, layer_idx)
    missing, unexpected = layer.load_state_dict(state, assign=True, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"layer-{layer_idx} checkpoint mismatch: missing={missing} "
            f"unexpected={unexpected}"
        )
    return layer.eval()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().numpy().astype("<f4", copy=False).tofile(path)


def capture_layer(
    model: Path,
    inputs: Path,
    output: Path,
    config,
    device: torch.device,
    layer_idx: int,
    positions: int,
) -> None:
    input_path = inputs / f"input_layer_{layer_idx}.f32"
    raw = np.fromfile(input_path, dtype="<f4")
    expected = positions * config.hidden_size
    if raw.size != expected:
        raise ValueError(f"{input_path} has {raw.size} values, expected {expected}")
    hidden = torch.from_numpy(raw.reshape(positions, config.hidden_size).copy())
    hidden = hidden.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    layer = load_layer(model, config, device, layer_idx)
    rotary = Gemma4TextRotaryEmbedding(config, device=device)
    position_ids = torch.arange(positions, device=device).unsqueeze(0)
    position_embeddings = rotary(
        hidden, position_ids, layer_type=config.layer_types[layer_idx]
    )

    captured: dict[str, torch.Tensor] = {"pre_layer": hidden.detach()}
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
        "post_ffn_norm": layer.post_feedforward_layernorm,
    }
    boundary_modules = {
        name: module for name, module in boundary_modules.items() if module is not None
    }

    def save_boundary(name: str):
        def hook(_module, _inputs, value):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} hook produced {type(value).__name__}")
            captured[name] = value.detach()

        return hook

    def save_input(name: str):
        def hook(_module, values):
            captured[name] = values[0].detach()

        return hook

    def save_layer_output(_module, _inputs, value):
        hidden_output = value[0] if isinstance(value, tuple) else value
        captured["layer_output"] = hidden_output.detach()

    handles = [
        module.register_forward_hook(save_boundary(name))
        for name, module in boundary_modules.items()
    ]
    handles.extend(
        [
            layer.pre_feedforward_layernorm.register_forward_pre_hook(
                save_input("post_attention_residual")
            ),
            layer.mlp.down_proj.register_forward_pre_hook(save_input("geglu")),
            layer.register_forward_hook(save_layer_output),
        ]
    )

    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture_sdpa(module, query, key, value, attention_mask, **kwargs):
        captured["q_rope"] = query.transpose(1, 2).contiguous().detach()
        captured["k_rope"] = key.transpose(1, 2).contiguous().detach()
        captured["v_attention"] = value.transpose(1, 2).contiguous().detach()
        attention, weights = original_sdpa(
            module, query, key, value, attention_mask, **kwargs
        )
        captured["attention_raw"] = attention.detach()
        return attention, weights

    ALL_ATTENTION_FUNCTIONS["sdpa"] = capture_sdpa
    try:
        with torch.no_grad():
            layer(
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
        "pre_layer",
        *boundary_modules,
        "post_attention_residual",
        "geglu",
        "layer_output",
        "q_rope",
        "k_rope",
        "v_attention",
        "attention_raw",
    }
    if captured.keys() != required:
        raise RuntimeError(
            f"layer {layer_idx} capture mismatch: got={sorted(captured)} "
            f"expected={sorted(required)}"
        )

    layer_output = output / f"layer_{layer_idx}"
    layer_output.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        write_f32(layer_output / f"operator_{name}.f32", tensor)
    rope_cos, rope_sin = position_embeddings
    write_f32(layer_output / "operator_rope_cos.f32", rope_cos)
    write_f32(layer_output / "operator_rope_sin.f32", rope_sin)
    metadata = {
        "schema": "hipfire.gemma4.transformers-layer-transition-operators.v1",
        "model": str(model.resolve()),
        "input_manifest": str(inputs.joinpath("manifest.json").resolve()),
        "input_sha256": sha256(input_path),
        "layer": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "dtype": "bfloat16",
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "positions": positions,
        "hidden_size": config.hidden_size,
        "head_dim": layer.self_attn.head_dim,
        "boundaries": sorted(captured),
    }
    layer_output.joinpath("capture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device requested but unavailable")
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hipfire.gemma4.layer-transition-inputs.v1":
        raise ValueError("transition input manifest has the wrong schema")
    positions = int(manifest["positions"])

    outer_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = outer_config.text_config
    config._attn_implementation = "sdpa"
    invalid = [layer for layer in args.layers if not 0 <= layer < config.num_hidden_layers]
    if invalid:
        raise ValueError(f"layers out of range: {invalid}")
    for layer_idx in args.layers:
        capture_layer(
            args.model,
            args.inputs,
            args.output,
            config,
            device,
            layer_idx,
            positions,
        )
        print(f"captured Transformers layer {layer_idx} operators in {args.output}")


if __name__ == "__main__":
    main()
