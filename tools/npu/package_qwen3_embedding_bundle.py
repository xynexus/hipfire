#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Package compiled Qwen3 component images under one exact embedding cache key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


COMPONENTS = (
    "residual-rmsnorm",
    "projection-q",
    "projection-kv",
    "projection-o",
    "projection-gate-up",
    "projection-down",
    "headnorm-rope",
    "query-pack",
    "kv-pack",
    "attention",
    "attention-unpack",
    "swiglu",
    "final-pool-l2",
)


def component(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("component must be NAME=PATH") from error
    if name not in COMPONENTS:
        raise argparse.ArgumentTypeError(f"unknown component {name!r}")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--quant-format", default="oq8+")
    parser.add_argument("--component", action="append", type=component, default=[])
    args = parser.parse_args()
    supplied = dict(args.component)
    missing = [name for name in COMPONENTS if name not in supplied]
    if missing:
        parser.error(f"missing components: {', '.join(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in COMPONENTS:
        source = supplied[name]
        xclbin = source / "final.xclbin"
        instructions = source / "insts.bin"
        if not xclbin.is_file():
            candidates = list(source.glob("*.xclbin"))
            if len(candidates) == 1:
                xclbin = candidates[0]
        if not instructions.is_file():
            candidates = list(source.glob("*instr*.bin"))
            if len(candidates) == 1:
                instructions = candidates[0]
        if not xclbin.is_file() or not instructions.is_file():
            parser.error(f"component {name} is missing final.xclbin or insts.bin in {source}")
        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xclbin, destination / "final.xclbin")
        shutil.copy2(instructions, destination / "insts.bin")
        if (source / "manifest.json").is_file():
            shutil.copy2(source / "manifest.json", destination / "component-manifest.json")
        if (source / ".hipfire-build.json").is_file():
            shutil.copy2(source / ".hipfire-build.json", destination / "component-build.json")
    shutil.copy2(args.output / "residual-rmsnorm/final.xclbin", args.output / "final.xclbin")
    shutil.copy2(args.output / "residual-rmsnorm/insts.bin", args.output / "insts.bin")
    manifest = {
        "schema": "hipfire.npu_embedding_image.v1",
        "runtime_abi": "hipfire.full_embedding_encoder.v1",
        "key": {
            "npu_architecture": "aie2p",
            "model_geometry": {
                "architecture": "qwen3",
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.layers,
                "num_attention_heads": args.query_heads,
                "num_key_value_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "intermediate_size": args.intermediate_size,
            },
            "quant_format": args.quant_format,
            "sequence_bucket": args.bucket,
            "dispatch_batch": args.batch,
        },
        "xclbin": "final.xclbin",
        "instructions": "insts.bin",
        "components": list(COMPONENTS),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
