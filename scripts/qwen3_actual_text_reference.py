#!/usr/bin/env python3
"""Capture selected hidden states from an actual local Qwen3 text tower."""

import argparse
import enum
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch


def stub_broken_torchvision() -> None:
    """Keep text-only Transformers independent of this host's torchvision."""

    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms", loader=None
    )
    io = types.ModuleType("torchvision.io")
    io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)
    functional = types.ModuleType("torchvision.transforms.functional")
    functional.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms.functional", loader=None
    )

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 2
        BILINEAR = 3
        HAMMING = 4
        BICUBIC = 5
        LANCZOS = 6

    class ImageReadMode(enum.Enum):
        UNCHANGED = 0
        GRAY = 1
        GRAY_ALPHA = 2
        RGB = 3
        RGB_ALPHA = 4

    def decode_image(*_args, **_kwargs):
        raise RuntimeError("torchvision image decoding is unavailable in text-only reference")

    def pil_to_tensor(*_args, **_kwargs):
        raise RuntimeError("torchvision conversion is unavailable in text-only reference")

    transforms.InterpolationMode = InterpolationMode
    torchvision.transforms = transforms
    torchvision.io = io
    io.ImageReadMode = ImageReadMode
    io.decode_image = decode_image
    functional.pil_to_tensor = pil_to_tensor
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.io"] = io
    sys.modules["torchvision.transforms.functional"] = functional


def flattened(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="local Qwen3 model directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--prompt")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()

    stub_broken_torchvision()
    model_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if "text_config" in config:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model.resolve(),
            dtype=model_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        if hasattr(model.model, "visual"):
            del model.model.visual
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model.resolve(),
            dtype=model_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
    if args.prompt is None:
        token_ids = torch.tensor([[args.token_id]], dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
    else:
        from transformers import AutoTokenizer

        tokenizer_path = args.model.parent / "tokenizer"
        if not tokenizer_path.is_dir():
            tokenizer_path = args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.resolve(), local_files_only=True)
        template_path = tokenizer_path / "chat_template.jinja"
        if tokenizer.chat_template is None and template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        token_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        output = model.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    selected_indices = (9, 18, 27)
    selected = [output.hidden_states[index] for index in selected_indices]
    args.output.write_text(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "dtype": args.dtype,
                "token_ids": token_ids.reshape(-1).tolist(),
                "attention_mask": attention_mask.reshape(-1).tolist(),
                **{
                    f"layer_{index}": flattened(hidden)
                    for index, hidden in zip(selected_indices, selected)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
