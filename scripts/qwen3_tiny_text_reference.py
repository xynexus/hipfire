#!/usr/bin/env python3
"""Emit a deterministic tiny Qwen3 text tower and Transformers parity trace."""

import argparse
import enum
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


def stub_broken_torchvision() -> None:
    """Keep text-only Transformers independent of this host's mismatched torchvision."""

    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    io = types.ModuleType("torchvision.io")
    io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 2
        BILINEAR = 3
        HAMMING = 4
        BICUBIC = 5
        LANCZOS = 6

    transforms.InterpolationMode = InterpolationMode
    torchvision.transforms = transforms
    torchvision.io = io
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.io"] = io


def flattened(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stub_broken_torchvision()
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = args.output.resolve()
    torch.manual_seed(0x3103)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        # Keep one layer after the last selected state so hidden_states[27] is a
        # post-layer state, not the model's final RMSNorm output.
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    token_ids = torch.tensor([[4, 7, 3, 9, 2, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        output = model.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    root.mkdir(parents=True, exist_ok=True)
    for component in ["text_encoder", "transformer", "vae", "scheduler", "tokenizer"]:
        (root / component).mkdir(exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    (root / "text_encoder" / "config.json").write_text(
        config.to_json_string(), encoding="utf-8"
    )
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 8}),
        encoding="utf-8",
    )
    (root / "vae" / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLFlux2", "latent_channels": 2}),
        encoding="utf-8",
    )
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}),
        encoding="utf-8",
    )
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        root / "text_encoder" / "model.safetensors",
    )
    selected_indices = (9, 18, 27)
    selected = [output.hidden_states[index] for index in selected_indices]
    concatenated = torch.cat(selected, dim=-1)
    (root / "reference.json").write_text(
        json.dumps(
            {
                "token_ids": token_ids.reshape(-1).tolist(),
                "attention_mask": attention_mask.reshape(-1).tolist(),
                "layer_9": flattened(selected[0]),
                "layer_18": flattened(selected[1]),
                "layer_27": flattened(selected[2]),
                "concatenated": flattened(concatenated),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
