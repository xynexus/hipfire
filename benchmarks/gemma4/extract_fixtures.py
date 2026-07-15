#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt

"""Reproduce Phase-0 Gemma 4 truth fixtures without reading weight data.

Standard checkpoints resolve from a Hugging Face cache root and must match the
revisions frozen below. Safetensors manifests come from headers only. The
unified 12B text config is fetched from its pinned Hub revision when absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import PreTrainedTokenizerFast


SOURCES = {
    "gemma-4-E2B": ("google/gemma-4-E2B", "63db66a33dc06d58c02b1e887446e103c202602c"),
    "gemma-4-E2B-it": ("google/gemma-4-E2B-it", "70af34e20bd4b7a91f0de6b22675850c43922a03"),
    "gemma-4-E4B": ("google/gemma-4-E4B", "a24c9379fd3839ae84e97f0b6aa3152fce9bd033"),
    "gemma-4-E4B-it": ("google/gemma-4-E4B-it", "fee6332c1abaafb77f6f9624236c63aa2f1d0187"),
    "gemma-4-26B-A4B": ("google/gemma-4-26B-A4B", "f1102d7de421741c6eafcda46d1806a7a65b83a3"),
    "gemma-4-26B-A4B-it": ("google/gemma-4-26B-A4B-it", "20da991ab4afab98e8f910c4a2e8f4fbefc404ad"),
    "gemma-4-31B": ("google/gemma-4-31B", "02e15e4990e8c452f8543fb26beff15b1daf8f3d"),
    "gemma-4-31B-it": ("google/gemma-4-31B-it", "3548789868c5356dbf307c98e6f609007b82b3eb"),
}

UNIFIED_SOURCE = ("google/gemma-4-12B", "1dd69cd087619018c29fbfe2c30c3cd3530479fb")

# layers, hidden, context, SWA, shared tail, MoE, K=V, double-wide tail
EXPECTED_VARIANTS = {
    "gemma-4-E2B": (35, 1536, 131072, 512, 20, False, False, True),
    "gemma-4-E2B-it": (35, 1536, 131072, 512, 20, False, False, True),
    "gemma-4-E4B": (42, 2560, 131072, 512, 18, False, False, False),
    "gemma-4-E4B-it": (42, 2560, 131072, 512, 18, False, False, False),
    "gemma-4-12B": (48, 3840, 262144, 1024, 0, False, True, False),
    "gemma-4-26B-A4B": (30, 2816, 262144, 1024, 0, True, True, False),
    "gemma-4-26B-A4B-it": (30, 2816, 262144, 1024, 0, True, True, False),
    "gemma-4-31B": (60, 5376, 262144, 1024, 0, False, True, False),
    "gemma-4-31B-it": (60, 5376, 262144, 1024, 0, False, True, False),
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Read the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit.",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city"],
        },
    },
}

WEATHER_CALL = {
    "id": "call_weather_1",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": {"city": "Paris", "unit": "celsius"},
    },
}

PROMPT_CASES: dict[str, dict[str, Any]] = {
    "plain": {
        "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
        "add_generation_prompt": True,
    },
    "system": {
        "messages": [
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": "What is two plus two?"},
        ],
        "add_generation_prompt": True,
    },
    "thinking_on": {
        "messages": [{"role": "user", "content": "Count the letters in banana."}],
        "enable_thinking": True,
        "add_generation_prompt": True,
    },
    "thinking_off": {
        "messages": [{"role": "user", "content": "Count the letters in banana."}],
        "enable_thinking": False,
        "add_generation_prompt": True,
    },
    "multi_turn": {
        "messages": [
            {"role": "user", "content": "My favorite color is amber."},
            {"role": "assistant", "content": "I will remember that."},
            {"role": "user", "content": "What color did I name?"},
        ],
        "add_generation_prompt": True,
    },
    "tool_declaration": {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "add_generation_prompt": True,
    },
    "tool_call": {
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should query the weather tool.",
                "tool_calls": [WEATHER_CALL],
            },
        ],
        "add_generation_prompt": False,
    },
    "tool_response": {
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": "{\"condition\":\"clear\",\"temperature\":18}",
            },
        ],
        "add_generation_prompt": True,
    },
    "assistant_continuation": {
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": "{\"condition\":\"clear\",\"temperature\":18}",
            },
            {"role": "assistant", "content": "It is 18 C and clear in Paris."},
        ],
        "add_generation_prompt": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/srv/huggingface"))
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "fixtures"
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def local_snapshot(root: Path, repo: str, revision: str) -> Path:
    cache_name = "models--" + repo.replace("/", "--")
    snapshot = root / cache_name / "snapshots" / revision
    if not snapshot.joinpath("config.json").is_file():
        raise FileNotFoundError(f"missing pinned snapshot: {snapshot}")
    ref = root / cache_name / "refs" / "main"
    if ref.is_file() and ref.read_text().strip() != revision:
        raise RuntimeError(f"{repo} refs/main no longer matches {revision}")
    return snapshot


def fetch_json(repo: str, revision: str, filename: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"failed to fetch pinned source {url}: {error}") from error


def distill_config(
    name: str,
    repo: str,
    revision: str,
    config: dict[str, Any],
    generation: dict[str, Any] | None,
    local_available: bool,
) -> dict[str, Any]:
    text = config.get("text_config", config)
    expected = EXPECTED_VARIANTS[name]
    actual = (
        text["num_hidden_layers"],
        text["hidden_size"],
        text["max_position_embeddings"],
        text["sliding_window"],
        text["num_kv_shared_layers"],
        text["enable_moe_block"],
        text["attention_k_eq_v"],
        text["use_double_wide_mlp"],
    )
    if actual != expected:
        raise AssertionError(f"{name}: execution shape {actual!r} != {expected!r}")
    if len(text["layer_types"]) != text["num_hidden_layers"]:
        raise AssertionError(f"{name}: layer_types length mismatch")
    if text["hidden_size"] % text["num_attention_heads"]:
        raise AssertionError(f"{name}: hidden size is not divisible by Q heads")
    if "full_attention" in text["layer_types"]:
        if text.get("global_head_dim") is None:
            raise AssertionError(f"{name}: global_head_dim is required")
        full_rope = text.get("rope_parameters", {}).get("full_attention", {})
        if full_rope.get("rope_type") != "proportional":
            raise AssertionError(f"{name}: global layers require proportional RoPE")
    return {
        "source": {
            "repository": repo,
            "revision": revision,
            "local_available_at_freeze": local_available,
        },
        "wrapper": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
        },
        "text_config": text,
        "generation_config": generation,
    }


def tensor_manifest(repo: str, revision: str, snapshot: Path) -> dict[str, Any]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        files = sorted(snapshot.glob("*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError(f"{repo}: no unambiguous safetensors manifest")
        with safe_open(files[0], framework="np", device="cpu") as handle:
            weight_map = {key: files[0].name for key in handle.keys()}

    by_shard: dict[str, list[str]] = {}
    for tensor_name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(tensor_name)
    tensors = []
    for shard, names in sorted(by_shard.items()):
        with safe_open(snapshot / shard, framework="np", device="cpu") as handle:
            for tensor_name in sorted(names):
                tensor_slice = handle.get_slice(tensor_name)
                tensors.append(
                    {
                        "name": tensor_name,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": tensor_slice.get_dtype(),
                        "shard": shard,
                    }
                )
    return {
        "source": {"repository": repo, "revision": revision},
        "header_only": True,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def assert_manifest_shape(name: str, manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Prove the source tensor topology distinguishes each execution family."""
    tensors = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    prefix = "model.language_model"
    text = config.get("text_config", config)

    if name.startswith(("gemma-4-E2B", "gemma-4-E4B")):
        ple = tensors.get(f"{prefix}.embed_tokens_per_layer.weight")
        expected = [text["vocab_size"], text["num_hidden_layers"] * text["hidden_size_per_layer_input"]]
        if ple is None or ple["shape"] != expected:
            raise AssertionError(f"{name}: PLE table shape does not match {expected}")
        for layer in range(text["num_hidden_layers"]):
            gate = f"{prefix}.layers.{layer}.per_layer_input_gate.weight"
            projection = f"{prefix}.layers.{layer}.per_layer_projection.weight"
            if gate not in tensors or projection not in tensors:
                raise AssertionError(f"{name}: missing PLE weights for layer {layer}")
    elif any("per_layer_input" in tensor_name for tensor_name in tensors):
        raise AssertionError(f"{name}: dense/MoE variant unexpectedly contains PLE weights")

    expert_gate = f"{prefix}.layers.0.experts.gate_up_proj"
    expert_down = f"{prefix}.layers.0.experts.down_proj"
    if text["enable_moe_block"]:
        expected_gate = [text["num_experts"], 2 * text["moe_intermediate_size"], text["hidden_size"]]
        expected_down = [text["num_experts"], text["hidden_size"], text["moe_intermediate_size"]]
        if tensors.get(expert_gate, {}).get("shape") != expected_gate:
            raise AssertionError(f"{name}: stacked expert gate/up shape mismatch")
        if tensors.get(expert_down, {}).get("shape") != expected_down:
            raise AssertionError(f"{name}: stacked expert down shape mismatch")
    elif expert_gate in tensors or expert_down in tensors:
        raise AssertionError(f"{name}: dense variant unexpectedly contains routed experts")

    # K=V is visible in the source: global layers retain K but omit V. PLE
    # checkpoints are intentionally different; their source files still carry
    # tail projection tensors even where the lowered cache plan will share KV.
    for layer, layer_type in enumerate(text["layer_types"]):
        v_name = f"{prefix}.layers.{layer}.self_attn.v_proj.weight"
        if text["attention_k_eq_v"] and layer_type == "full_attention":
            if v_name in tensors:
                raise AssertionError(f"{name}: K=V global layer {layer} contains V projection")
        elif v_name not in tensors:
            raise AssertionError(f"{name}: separate-V layer {layer} lacks V projection")


def tokenizer_for(snapshot: Path) -> PreTrainedTokenizerFast:
    cfg = json.loads(snapshot.joinpath("tokenizer_config.json").read_text())
    return PreTrainedTokenizerFast(
        tokenizer_file=str(snapshot / "tokenizer.json"),
        bos_token=cfg["bos_token"],
        eos_token=cfg["eos_token"],
        pad_token=cfg["pad_token"],
        chat_template=snapshot.joinpath("chat_template.jinja").read_text(),
    )


def render_prompts(repo: str, revision: str, snapshot: Path) -> dict[str, Any]:
    template = snapshot.joinpath("chat_template.jinja").read_bytes()
    tokenizer = tokenizer_for(snapshot)
    rendered_cases = {}
    for case_name, case in PROMPT_CASES.items():
        kwargs = {
            key: value
            for key, value in case.items()
            if key not in {"messages", "add_generation_prompt"}
        }
        rendered = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=case["add_generation_prompt"],
            **kwargs,
        )
        rendered_cases[case_name] = {
            "rendered": rendered,
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "token_ids": tokenizer.encode(rendered, add_special_tokens=False),
        }
    return {
        "source": {"repository": repo, "revision": revision},
        "template_sha256": hashlib.sha256(template).hexdigest(),
        "cases": rendered_cases,
    }


def generate(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Path] = {}
    for name, (repo, revision) in SOURCES.items():
        snapshot = local_snapshot(root, repo, revision)
        snapshots[name] = snapshot
        config = json.loads(snapshot.joinpath("config.json").read_text())
        generation_path = snapshot / "generation_config.json"
        generation = json.loads(generation_path.read_text()) if generation_path.is_file() else None
        write_json(
            output / "configs" / f"{name}.json",
            distill_config(name, repo, revision, config, generation, True),
        )
        manifest = tensor_manifest(repo, revision, snapshot)
        assert_manifest_shape(name, manifest, config)
        write_json(output / "manifests" / f"{name}.json", manifest)
        if name.endswith("-it"):
            write_json(
                output / "prompts" / f"{name}.json",
                render_prompts(repo, revision, snapshot),
            )

    unified_repo, unified_revision = UNIFIED_SOURCE
    unified_name = "gemma-4-12B"
    cache = root / ("models--" + unified_repo.replace("/", "--")) / "snapshots" / unified_revision
    if cache.joinpath("config.json").is_file():
        unified_config = json.loads(cache.joinpath("config.json").read_text())
        local_available = True
    else:
        unified_config = fetch_json(unified_repo, unified_revision, "config.json")
        local_available = False
    write_json(
        output / "configs" / f"{unified_name}.json",
        distill_config(
            unified_name,
            unified_repo,
            unified_revision,
            unified_config,
            None,
            local_available,
        ),
    )

    template_sources = {
        "gemma-4-edge-it.jinja": (*SOURCES["gemma-4-E2B-it"], snapshots["gemma-4-E2B-it"]),
        "gemma-4-large-it.jinja": (*SOURCES["gemma-4-31B-it"], snapshots["gemma-4-31B-it"]),
    }
    records = {}
    for filename, (repo, revision, snapshot) in template_sources.items():
        data = snapshot.joinpath("chat_template.jinja").read_bytes()
        target = output / "templates" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records[filename] = {
            "repository": repo,
            "revision": revision,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    write_json(output / "templates" / "sources.json", records)
    write_json(output / "prompts" / "cases.json", PROMPT_CASES)


def compare_trees(expected: Path, actual: Path) -> None:
    expected_files = {p.relative_to(expected) for p in expected.rglob("*") if p.is_file()}
    actual_files = {p.relative_to(actual) for p in actual.rglob("*") if p.is_file()}
    if expected_files != actual_files:
        raise SystemExit(
            f"fixture set differs: missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    mismatches = [
        str(path)
        for path in sorted(expected_files)
        if expected.joinpath(path).read_bytes() != actual.joinpath(path).read_bytes()
    ]
    if mismatches:
        raise SystemExit("fixture bytes differ: " + ", ".join(mismatches))


def main() -> None:
    args = parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="gemma4-fixtures-") as temp:
            generated = Path(temp) / "fixtures"
            generate(args.source_root, generated)
            compare_trees(args.output, generated)
        print(f"Gemma 4 fixtures reproduce exactly from {args.source_root}")
        return
    if args.output.exists():
        shutil.rmtree(args.output)
    generate(args.source_root, args.output)
    print(f"wrote Gemma 4 fixtures to {args.output}")


if __name__ == "__main__":
    main()
