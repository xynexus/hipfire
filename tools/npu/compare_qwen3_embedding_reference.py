#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Compare Qwen3 NPU embeddings with BF16 and dequantized-HFQ ROCm oracles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


GROUP = 256
BLOCK = 258


@dataclass(frozen=True)
class TensorInfo:
    quant_type: int
    shape: tuple[int, ...]
    group_size: int
    offset: int
    size: int


class Hfq:
    def __init__(self, path: Path) -> None:
        self._file = path.open("rb")
        self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:4] != b"HFQM":
            raise ValueError(f"{path} is not an HFQM container")
        metadata_offset, data_offset = struct.unpack_from("<QQ", self.data, 16)
        json_end = self._json_end(metadata_offset, data_offset)
        position = metadata_offset + json_end
        count = struct.unpack_from("<I", self.data, position)[0]
        position += 4
        self.tensors: dict[str, TensorInfo] = {}
        for _ in range(count):
            name_bytes = struct.unpack_from("<H", self.data, position)[0]
            position += 2
            name = self.data[position : position + name_bytes].decode()
            position += name_bytes
            quant_type, rank = struct.unpack_from("<BB", self.data, position)
            position += 2
            shape = struct.unpack_from(f"<{rank}I", self.data, position)
            position += rank * 4
            group_size = struct.unpack_from("<I", self.data, position)[0]
            size, offset_div32 = struct.unpack_from("<QQ", self.data, position + 4)
            position += 20
            self.tensors[name] = TensorInfo(
                quant_type=quant_type,
                shape=shape,
                group_size=group_size,
                offset=offset_div32 * 32,
                size=size,
            )

    def close(self) -> None:
        self.data.close()
        self._file.close()

    def _json_end(self, start: int, end: int) -> int:
        depth = 0
        in_string = False
        escaped = False
        for relative, byte in enumerate(self.data[start:end]):
            if escaped:
                escaped = False
            elif byte == ord("\\") and in_string:
                escaped = True
            elif byte == ord('"'):
                in_string = not in_string
            elif not in_string and byte == ord("{"):
                depth += 1
            elif not in_string and byte == ord("}"):
                depth -= 1
                if depth == 0:
                    return relative + 1
        raise ValueError("HFQM metadata JSON did not end")

    def tensor(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        payload = memoryview(self.data)[info.offset : info.offset + info.size]
        elements = int(np.prod(info.shape))
        if info.quant_type == 1:
            values = np.frombuffer(payload, dtype="<f2", count=elements).astype(np.float32)
        elif info.quant_type == 2:
            values = np.frombuffer(payload, dtype="<f4", count=elements).copy()
        elif info.quant_type == 3:
            values = dequant_q8f16(payload, elements)
        elif info.quant_type == 16:
            bits = np.frombuffer(payload, dtype="<u2", count=elements).astype(np.uint32)
            values = (bits << 16).view(np.float32)
        elif info.quant_type in (35, 43):
            values = dequant_oq8(payload, info)
        else:
            raise ValueError(f"{name}: unsupported quant_type={info.quant_type}")
        return values.reshape(info.shape)


def signs(seed: int) -> np.ndarray:
    result = np.empty(GROUP, dtype=np.float32)
    state = seed
    for index in range(GROUP):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        result[index] = 1.0 if (state >> 16) & 1 else -1.0
    return result


SIGNS_1 = signs(42)
SIGNS_2 = signs(1042)


def inverse_fwht(groups: np.ndarray) -> np.ndarray:
    output = np.empty_like(groups, dtype=np.float32)
    # Keep each vectorized slab modest. Some ROCm environment NumPy builds have
    # produced incorrect large strided slice assignments above 256 rows.
    for offset in range(0, len(groups), 256):
        values = groups[offset : offset + 256] * SIGNS_2
        stride = 1
        while stride < GROUP:
            for start in range(0, GROUP, 2 * stride):
                left = values[:, start : start + stride].copy()
                right = values[:, start + stride : start + 2 * stride].copy()
                values[:, start : start + stride] = left + right
                values[:, start + stride : start + 2 * stride] = left - right
            stride *= 2
        output[offset : offset + len(values)] = values * (SIGNS_1 / 16.0)
    return output


def dequant_oq8(payload: memoryview, info: TensorInfo) -> np.ndarray:
    if info.group_size != GROUP or len(info.shape) != 2:
        raise ValueError(f"invalid OQ8 geometry shape={info.shape} group={info.group_size}")
    rows, columns = info.shape
    groups_per_row = (columns + GROUP - 1) // GROUP
    expected = rows * groups_per_row * BLOCK
    if len(payload) != expected:
        raise ValueError(f"OQ8 payload has {len(payload)} bytes, expected {expected}")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, BLOCK)
    scales = blocks[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
    quantized = blocks[:, 2:].view(np.int8).astype(np.float32)
    plain = inverse_fwht(quantized * scales[:, None])
    return plain.reshape(rows, groups_per_row * GROUP)[:, :columns].reshape(-1)


def dequant_q8f16(payload: memoryview, elements: int) -> np.ndarray:
    groups = (elements + 31) // 32
    if len(payload) != groups * 34:
        raise ValueError(f"Q8F16 payload has {len(payload)} bytes, expected {groups * 34}")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(groups, 34)
    scales = blocks[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
    quantized = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (quantized * scales[:, None]).reshape(-1)[:elements]


def apply_hfq(model: torch.nn.Module, hfq: Hfq) -> None:
    state = model.state_dict()
    missing = [name for name in state if name not in hfq.tensors]
    if missing:
        raise ValueError(f"HFQ is missing model tensors: {missing[:8]}")
    with torch.no_grad():
        for index, (name, destination) in enumerate(state.items(), 1):
            values = hfq.tensor(name)
            sidecar = (
                f"{name[: -len('.weight')]}.awq_scale.weight"
                if name.endswith(".weight")
                else f"{name}.awq_scale.weight"
            )
            if sidecar in hfq.tensors:
                scales = hfq.tensor(sidecar)
                if values.ndim != 2 or scales.shape != (values.shape[1],):
                    raise ValueError(f"{name}: incompatible AWQ sidecar shape {scales.shape}")
                values = values / scales[None, :]
            destination.copy_(torch.from_numpy(values).to(destination.dtype))
            if index == len(state) or index % 25 == 0:
                print(f"loaded {index}/{len(state)} HFQ tensors", flush=True)


def encode(
    model: torch.nn.Module,
    token_ids: list[list[int]],
    device: str,
    capture_layers: bool = False,
    capture_stages: bool = False,
    initial_last_layer_hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    lengths = torch.tensor([len(tokens) for tokens in token_ids], device=device)
    width = int(lengths.max().item())
    input_ids = torch.zeros((len(token_ids), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(token_ids):
        input_ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
        attention_mask[row, : len(tokens)] = 1
    layer_rows: list[torch.Tensor] = []
    stage_rows: dict[str, torch.Tensor] = {}
    hooks = []
    if capture_layers:

        def capture(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"unexpected Qwen3 layer output {type(hidden)}")
            selected = hidden[torch.arange(len(token_ids), device=device), lengths - 1]
            layer_rows.append(selected.float().cpu())

        layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError(f"{type(model).__name__} has no decoder layer list")
        hooks = [layer.register_forward_hook(capture) for layer in layers]
        if capture_stages:
            last = layers[-1]

            def selected(hidden: torch.Tensor) -> torch.Tensor:
                rows = hidden[torch.arange(len(token_ids), device=device), lengths - 1]
                return rows.reshape(len(token_ids), -1).float().cpu()

            def capture_named(name: str):
                def hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError(f"unexpected {name} output {type(hidden)}")
                    stage_rows[name] = selected(hidden)

                return hook

            def capture_named_input(name: str):
                def hook(_module: torch.nn.Module, inputs: tuple) -> None:
                    hidden = inputs[0]
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError(f"unexpected {name} input {type(hidden)}")
                    stage_rows[name] = selected(hidden)

                return hook

            hooks.extend(
                [
                    last.input_layernorm.register_forward_hook(capture_named("normalized_input")),
                    last.self_attn.q_proj.register_forward_hook(capture_named("query_projection")),
                    last.self_attn.k_proj.register_forward_hook(capture_named("key_projection")),
                    last.self_attn.v_proj.register_forward_hook(capture_named("value_projection")),
                    last.self_attn.q_norm.register_forward_hook(capture_named("query_headnorm_rope")),
                    last.self_attn.k_norm.register_forward_hook(capture_named("key_headnorm_rope")),
                    last.self_attn.o_proj.register_forward_pre_hook(capture_named_input("attention")),
                    last.self_attn.o_proj.register_forward_hook(capture_named("attention_delta")),
                    last.post_attention_layernorm.register_forward_pre_hook(capture_named_input("post_attention")),
                    last.post_attention_layernorm.register_forward_hook(capture_named("ffn_input")),
                    last.mlp.gate_proj.register_forward_hook(capture_named("gate_projection")),
                    last.mlp.up_proj.register_forward_hook(capture_named("up_projection")),
                    last.mlp.down_proj.register_forward_pre_hook(capture_named_input("swiglu")),
                    last.mlp.down_proj.register_forward_hook(capture_named("down_projection")),
                    last.register_forward_hook(capture_named("completed")),
                ]
            )
    with torch.inference_mode():
        try:
            if initial_last_layer_hidden is None:
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                hidden_input = initial_last_layer_hidden.to(
                    device=device, dtype=next(model.parameters()).dtype
                ).reshape(len(token_ids), 1, -1)
                position_ids = torch.zeros((len(token_ids), 1), dtype=torch.long, device=device)
                position_embeddings = model.rotary_emb(hidden_input, position_ids)
                hidden = model.layers[-1](
                    hidden_input,
                    attention_mask=None,
                    position_ids=position_ids,
                    cache_position=torch.zeros(1, dtype=torch.long, device=device),
                    position_embeddings=position_embeddings,
                )
        finally:
            for hook in hooks:
                hook.remove()
        pooled = hidden[torch.arange(len(token_ids), device=device), lengths - 1].float()
        embeddings = torch.nn.functional.normalize(pooled, dim=-1).cpu()
        trace = torch.stack(layer_rows) if capture_layers else None
        return embeddings, trace, stage_rows


def encode_padding_invariant(
    model: torch.nn.Module,
    token_ids: list[list[int]],
    device: str,
    capture_layers: bool = False,
    capture_stages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Encode mixed lengths independently so reference padding cannot alter results."""
    if len({len(tokens) for tokens in token_ids}) <= 1:
        return encode(model, token_ids, device, capture_layers, capture_stages)

    embeddings: list[torch.Tensor] = []
    layer_traces: list[torch.Tensor] = []
    stage_traces: dict[str, list[torch.Tensor]] = {}
    for tokens in token_ids:
        embedding, layers, stages = encode(model, [tokens], device, capture_layers, capture_stages)
        embeddings.append(embedding)
        if layers is not None:
            layer_traces.append(layers)
        for name, values in stages.items():
            stage_traces.setdefault(name, []).append(values)
    combined_layers = torch.cat(layer_traces, dim=1) if layer_traces else None
    combined_stages = {name: torch.cat(values, dim=0) for name, values in stage_traces.items()}
    return torch.cat(embeddings, dim=0), combined_layers, combined_stages


def cosine(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    return torch.nn.functional.cosine_similarity(left.float(), right.float()).tolist()


def comparison_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, list[float]]:
    left = left.float()
    right = right.float()
    error = (left - right).abs()
    return {
        "cosine": cosine(left, right),
        "max_abs": error.amax(dim=-1).tolist(),
        "mean_abs": error.mean(dim=-1).tolist(),
        "left_norm": torch.linalg.vector_norm(left, dim=-1).tolist(),
        "right_norm": torch.linalg.vector_norm(right, dim=-1).tolist(),
    }


def replay_exact_suffix(
    model: torch.nn.Module,
    completed: torch.Tensor,
    next_layer: int,
    device: str,
) -> torch.Tensor:
    hidden = completed.to(device=device, dtype=next(model.parameters()).dtype).unsqueeze(1)
    position_ids = torch.zeros((len(completed), 1), dtype=torch.long, device=device)
    position_embeddings = model.rotary_emb(hidden, position_ids)
    with torch.inference_mode():
        for layer in model.layers[next_layer:]:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                cache_position=torch.zeros(1, dtype=torch.long, device=device),
                position_embeddings=position_embeddings,
            )
        final = model.norm(hidden[:, 0]).float()
        return torch.nn.functional.normalize(final, dim=-1).cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hfq", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dequant-only",
        action="store_true",
        help="instantiate from config and skip loading the source BF16 weights",
    )
    args = parser.parse_args()

    npu_record = json.loads(args.npu_output.read_text())
    token_ids = npu_record["token_ids"]
    npu = torch.tensor(npu_record["embeddings"], dtype=torch.float32)
    capture_layers = npu_record.get("layer_last_token_residuals") is not None
    capture_stages = npu_record.get("last_layer_stages") is not None
    if args.dequant_only:
        config = AutoConfig.from_pretrained(args.source)
        model = AutoModel.from_config(
            config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        bf16 = None
        bf16_layers = None
        bf16_stages = None
    else:
        model = AutoModel.from_pretrained(
            args.source,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(args.device)
        model.eval()
        bf16, bf16_layers, bf16_stages = encode_padding_invariant(
            model, token_ids, args.device, capture_layers, capture_stages
        )

    hfq = Hfq(args.hfq)
    try:
        model.to("cpu")
        torch.cuda.empty_cache()
        apply_hfq(model, hfq)
    finally:
        hfq.close()
    model.to(args.device)
    quantized, quantized_layers, quantized_stages = encode_padding_invariant(
        model, token_ids, args.device, capture_layers, capture_stages
    )
    result = {
        "documents": len(token_ids),
        "dimensions": npu.shape[1],
        "dequant_hfq_gpu_vs_npu_cosine": cosine(quantized, npu),
    }
    if bf16 is not None:
        result.update(
            {
                "dequant_hfq_gpu_vs_bf16_cosine": cosine(quantized, bf16),
                "npu_vs_bf16_cosine": cosine(npu, bf16),
            }
        )
    if capture_layers:
        npu_layers = torch.tensor(npu_record["layer_last_token_residuals"], dtype=torch.float32).reshape(
            len(quantized_layers), len(token_ids), -1
        )
        result.update(
            {
                "layer_dequant_hfq_gpu_vs_npu_cosine": [
                    cosine(gpu, npu_layer) for gpu, npu_layer in zip(quantized_layers, npu_layers, strict=True)
                ],
                "layer_dequant_hfq_gpu_vs_bf16_cosine": [
                    cosine(gpu, source) for gpu, source in zip(quantized_layers, bf16_layers, strict=True)
                ],
                "layer_npu_vs_bf16_cosine": [
                    cosine(npu_layer, source) for npu_layer, source in zip(npu_layers, bf16_layers, strict=True)
                ],
            }
        )
    if capture_stages:
        npu_stages = npu_record["last_layer_stages"]
        result["last_layer_stage_dequant_hfq_gpu_vs_npu_cosine"] = {
            name: cosine(
                quantized_stages[name],
                torch.tensor(values, dtype=torch.float32).reshape(len(token_ids), -1),
            )
            for name, values in npu_stages.items()
        }
        result["last_layer_stage_dequant_hfq_gpu_vs_bf16_cosine"] = {
            name: cosine(quantized_stages[name], bf16_stages[name]) for name in npu_stages
        }
        result["last_layer_stage_npu_vs_bf16_cosine"] = {
            name: cosine(
                torch.tensor(values, dtype=torch.float32).reshape(len(token_ids), -1),
                bf16_stages[name],
            )
            for name, values in npu_stages.items()
        }
    if capture_stages and all(len(tokens) == 1 for tokens in token_ids):
        npu_stages = npu_record["last_layer_stages"]
        npu_layers = torch.tensor(npu_record["layer_last_token_residuals"], dtype=torch.float32).reshape(
            len(quantized_layers), len(token_ids), -1
        )
        _, _, replayed_stages = encode(
            model,
            [[0] for _ in token_ids],
            args.device,
            capture_layers=True,
            capture_stages=True,
            initial_last_layer_hidden=npu_layers[-2],
        )
        with torch.inference_mode():
            replayed_completed = replayed_stages["completed"].to(
                device=args.device, dtype=next(model.parameters()).dtype
            )
            replayed_final = model.norm(replayed_completed).float()
            replayed_embedding = torch.nn.functional.normalize(replayed_final, dim=-1).cpu()
        result["dequant_hfq_gpu_from_npu_layer26_vs_dequant_hfq_gpu_cosine"] = cosine(replayed_embedding, quantized)
        suffix_starts = sorted({0, 1, 2, 4, 8, 12, 16, 20, 24, len(npu_layers) - 2})
        result["dequant_hfq_gpu_suffix_from_npu_layer_cosine"] = {
            str(start): cosine(
                replay_exact_suffix(model, npu_layers[start], start + 1, args.device),
                quantized,
            )
            for start in suffix_starts
        }
        result["last_layer_stage_dequant_hfq_gpu_from_npu_input_vs_npu_cosine"] = {
            name: cosine(
                replayed_stages[name],
                torch.tensor(values, dtype=torch.float32).reshape(len(token_ids), -1),
            )
            for name, values in npu_stages.items()
        }
        result["last_layer_stage_dequant_hfq_gpu_from_npu_input_vs_npu_metrics"] = {
            name: comparison_metrics(
                replayed_stages[name],
                torch.tensor(values, dtype=torch.float32).reshape(len(token_ids), -1),
            )
            for name, values in npu_stages.items()
        }
        last = model.layers[-1]

        def npu_stage(name: str) -> torch.Tensor:
            return (
                torch.tensor(npu_stages[name], device=args.device)
                .to(dtype=next(model.parameters()).dtype)
                .reshape(len(token_ids), 1, -1)
            )

        layer_input = (
            npu_layers[-2].to(device=args.device, dtype=next(model.parameters()).dtype).reshape(len(token_ids), 1, -1)
        )
        with torch.inference_mode():
            value = npu_stage("value_projection").reshape(len(token_ids), 1, model.config.num_key_value_heads, -1)
            attention = value.repeat_interleave(last.self_attn.num_key_value_groups, dim=2).reshape(
                len(token_ids), 1, -1
            )
            isolated = {
                "normalized_input": last.input_layernorm(layer_input),
                "query_projection": last.self_attn.q_proj(npu_stage("normalized_input")),
                "key_projection": last.self_attn.k_proj(npu_stage("normalized_input")),
                "value_projection": last.self_attn.v_proj(npu_stage("normalized_input")),
                # Position zero makes RoPE the identity, so these isolate
                # the Q/K RMSNorm arithmetic for the length-one probe.
                "query_headnorm_rope": last.self_attn.q_norm(
                    npu_stage("query_projection").reshape(len(token_ids), 1, model.config.num_attention_heads, -1)
                ).reshape(len(token_ids), 1, -1),
                "key_headnorm_rope": last.self_attn.k_norm(
                    npu_stage("key_projection").reshape(len(token_ids), 1, model.config.num_key_value_heads, -1)
                ).reshape(len(token_ids), 1, -1),
                "attention": attention,
                "attention_delta": last.self_attn.o_proj(npu_stage("attention")),
                "post_attention": layer_input + npu_stage("attention_delta"),
                "ffn_input": last.post_attention_layernorm(npu_stage("post_attention")),
                "gate_projection": last.mlp.gate_proj(npu_stage("ffn_input")),
                "up_projection": last.mlp.up_proj(npu_stage("ffn_input")),
                "swiglu": last.mlp.act_fn(npu_stage("gate_projection")) * npu_stage("up_projection"),
                "down_projection": last.mlp.down_proj(npu_stage("swiglu")),
                "completed": npu_stage("post_attention") + npu_stage("down_projection"),
            }
        result["last_layer_stage_isolated_dequant_hfq_gpu_vs_npu_metrics"] = {
            name: comparison_metrics(
                exact.reshape(len(token_ids), -1).float().cpu(),
                torch.tensor(npu_stages[name], dtype=torch.float32).reshape(len(token_ids), -1),
            )
            for name, exact in isolated.items()
        }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
