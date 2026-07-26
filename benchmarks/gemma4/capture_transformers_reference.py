#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt

"""Capture the pinned BF16 Transformers oracle for Gemma 4 text execution.

This offline evidence tool writes token IDs, selected decoder-layer boundary
states, the final-position logits, and greedy generated IDs. Runtime code never
imports it. Inputs can be rendered prompt bytes or a committed exact-token JSON
fixture, which lets SWA-1/SWA/SWA+1 cases share precisely the same IDs with the
Hipfire candidate capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--prompt-file", type=Path)
    inputs.add_argument("--input-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers", default="0,-1", help="comma-separated decoder layer indices"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--operator-layer", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="place the full model on --device or let Accelerate split it",
    )
    parser.add_argument("--gpu-max-memory")
    parser.add_argument("--cpu-max-memory")
    parser.add_argument("--offload-folder", type=Path)
    return parser.parse_args()


def tokenizer_for(model: Path) -> PreTrainedTokenizerFast:
    cfg = json.loads(model.joinpath("tokenizer_config.json").read_text())
    template_path = model / "chat_template.jinja"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model / "tokenizer.json"),
        bos_token=cfg["bos_token"],
        eos_token=cfg["eos_token"],
        pad_token=cfg["pad_token"],
        chat_template=template_path.read_text() if template_path.is_file() else None,
    )


def capture_input(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast
) -> tuple[list[int], str, str | None, str]:
    if args.input_ids is not None:
        raw = args.input_ids.read_bytes()
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
            raise ValueError("--input-ids must contain a nonempty JSON array")
        if any(not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF for value in values):
            raise ValueError("every exact input token must be a u32")
        return values, "exact_token_ids", str(args.input_ids.resolve()), hashlib.sha256(raw).hexdigest()

    if args.prompt_file is not None:
        raw = args.prompt_file.read_bytes()
        prompt = raw.decode("utf-8")
        source = str(args.prompt_file.resolve())
    else:
        prompt = args.prompt
        raw = prompt.encode()
        source = None
    return (
        tokenizer.encode(prompt, add_special_tokens=False),
        "rendered_prompt",
        source,
        hashlib.sha256(raw).hexdigest(),
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but PyTorch has no ROCm/CUDA device")

    tokenizer = tokenizer_for(args.model)
    input_ids, input_kind, input_path, input_sha256 = capture_input(args, tokenizer)
    if not input_ids:
        raise ValueError("capture input contains zero token IDs")
    device = torch.device(args.device)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    device_map: dict[str, torch.device] | str
    device_map = "auto" if args.device_map == "auto" else {"": device}
    max_memory = None
    if args.device_map == "auto" and (
        args.gpu_max_memory is not None or args.cpu_max_memory is not None
    ):
        max_memory = {}
        if args.gpu_max_memory is not None:
            max_memory[0] = args.gpu_max_memory
        if args.cpu_max_memory is not None:
            max_memory["cpu"] = args.cpu_max_memory
    if args.offload_folder is not None:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=args.offload_folder,
        offload_state_dict=args.offload_folder is not None,
    )
    model.eval()

    text_model = getattr(getattr(model, "model", None), "language_model", None)
    if text_model is None:
        text_model = getattr(model, "language_model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise RuntimeError("Transformers model did not expose decoder layers")

    selected: dict[str, np.ndarray] = {}
    attention_inputs: dict[str, np.ndarray] = {}
    hooks = []
    resolved_layers = []
    for raw_index in args.layers.split(","):
        index = int(raw_index)
        resolved = index if index >= 0 else len(layers) + index
        if not 0 <= resolved < len(layers):
            raise ValueError(f"decoder layer index {index} is out of range")
        resolved_layers.append(resolved)

        def capture_boundary(_module, _inputs, output, *, layer=resolved):
            hidden = output[0] if isinstance(output, tuple) else output
            selected[f"hidden_layer_{layer}"] = hidden.float().cpu().numpy()

        hooks.append(layers[resolved].register_forward_hook(capture_boundary))

    if args.operator_layer is not None:
        operator_layer = args.operator_layer
        if not 0 <= operator_layer < len(layers):
            raise ValueError(f"operator layer {operator_layer} is out of range")
        layer = layers[operator_layer]

        def capture_pre_layer(_module, inputs):
            selected["operator_pre_layer"] = inputs[0].float().cpu().numpy()

        def capture_post_attention_residual(_module, inputs):
            selected["operator_post_attention_residual"] = (
                inputs[0].float().cpu().numpy()
            )

        def capture_output(name):
            def hook(_module, _inputs, output):
                selected[f"operator_{name}"] = output.float().cpu().numpy()

            return hook

        def capture_geglu(_module, inputs):
            selected["operator_geglu"] = inputs[0].float().cpu().numpy()

        def capture_attention_raw(_module, inputs):
            selected["operator_attention_raw"] = inputs[0].float().cpu().numpy()

        def capture_layer_output(module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            selected["operator_layer_output"] = hidden.float().cpu().numpy()
            selected["operator_post_ffn_residual"] = (
                hidden.float() / module.layer_scalar.float()
            ).cpu().numpy()

        hooks.extend(
            [
                layer.register_forward_pre_hook(capture_pre_layer),
                layer.input_layernorm.register_forward_hook(
                    capture_output("input_norm")
                ),
                layer.self_attn.q_proj.register_forward_hook(capture_output("q_proj")),
                layer.self_attn.k_proj.register_forward_hook(capture_output("k_proj")),
                *(
                    [
                        layer.self_attn.v_proj.register_forward_hook(
                            capture_output("v_proj")
                        )
                    ]
                    if layer.self_attn.v_proj is not None
                    else []
                ),
                layer.self_attn.q_norm.register_forward_hook(capture_output("q_norm")),
                layer.self_attn.k_norm.register_forward_hook(capture_output("k_norm")),
                layer.self_attn.v_norm.register_forward_hook(capture_output("v_norm")),
                layer.self_attn.o_proj.register_forward_pre_hook(capture_attention_raw),
                layer.self_attn.o_proj.register_forward_hook(capture_output("o_proj")),
                layer.post_attention_layernorm.register_forward_hook(
                    capture_output("post_attention_norm")
                ),
                layer.pre_feedforward_layernorm.register_forward_pre_hook(
                    capture_post_attention_residual
                ),
                layer.pre_feedforward_layernorm.register_forward_hook(
                    capture_output("pre_ffn_norm")
                ),
                layer.mlp.gate_proj.register_forward_hook(capture_output("gate")),
                layer.mlp.up_proj.register_forward_hook(capture_output("up")),
                layer.mlp.down_proj.register_forward_pre_hook(capture_geglu),
                layer.post_feedforward_layernorm.register_forward_hook(
                    capture_output("post_ffn_norm")
                ),
                layer.register_forward_hook(capture_layer_output),
            ]
        )

    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    capture_attention_inputs = args.operator_layer is not None

    def capture_sdpa(module, query, key, value, attention_mask, **kwargs):
        if capture_attention_inputs and module.layer_idx == args.operator_layer:
            # Preserve the tensors exactly as they enter PyTorch SDPA. Transpose
            # to Hipfire's position-major layout so the raw files can feed a
            # HIP kernel parity probe without an additional conversion step.
            attention_inputs["q_rope"] = (
                query.transpose(1, 2).contiguous().float().cpu().numpy()
            )
            attention_inputs["k_rope"] = (
                key.transpose(1, 2).contiguous().float().cpu().numpy()
            )
            attention_inputs["v_attention"] = (
                value.transpose(1, 2).contiguous().float().cpu().numpy()
            )
        return original_sdpa(
            module, query, key, value, attention_mask, **kwargs
        )

    if capture_attention_inputs:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = capture_sdpa

    try:
        with torch.no_grad():
            # Gemma 4's 262K vocabulary makes all-position logits prohibitively
            # large for the SWA-boundary prompts. The admission contract compares
            # the committed final position, so ask upstream to project only it.
            forward = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        if capture_attention_inputs:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
        for hook in hooks:
            hook.remove()
    missing = [layer for layer in resolved_layers if f"hidden_layer_{layer}" not in selected]
    if missing:
        raise RuntimeError(f"decoder hooks did not capture layers {missing}")

    with torch.no_grad():
        generated = model.generate(
            input_ids=ids,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "capture.npz",
        input_ids=np.asarray(input_ids, dtype=np.uint32),
        final_logits=forward.logits[:, -1, :].float().cpu().numpy(),
        generated_ids=generated.cpu().numpy().astype(np.uint32),
        **selected,
    )
    for name, values in attention_inputs.items():
        values.astype("<f4", copy=False).tofile(
            args.output / f"operator_{name}.f32"
        )
    if "operator_attention_raw" in selected:
        selected["operator_attention_raw"].astype("<f4", copy=False).tofile(
            args.output / "operator_attention_raw.f32"
        )

    revision = None
    cache_ref = args.model.parent.parent / "refs" / "main"
    if cache_ref.is_file():
        revision = cache_ref.read_text().strip()
    metadata = {
        "model": str(args.model.resolve()),
        "revision": revision,
        "dtype": "bfloat16",
        "device": str(device),
        "device_map": args.device_map,
        "max_memory": max_memory,
        "offload_folder": (
            str(args.offload_folder.resolve())
            if args.offload_folder is not None
            else None
        ),
        "resolved_device_map": {
            key: str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "input_kind": input_kind,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "input_token_count": len(input_ids),
        "selected_decoder_layers": sorted(resolved_layers),
        "operator_layer": args.operator_layer,
        "operator_attention_inputs": sorted(attention_inputs),
        "max_new_tokens": args.max_new_tokens,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=False),
    }
    args.output.joinpath("capture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.output / 'capture.npz'} and {args.output / 'capture.json'}")


if __name__ == "__main__":
    main()
