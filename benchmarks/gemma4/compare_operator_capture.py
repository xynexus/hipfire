#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Compare one Gemma 4 decoder layer's major operator boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_bf16_captures import vector_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--hipfire", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.hipfire.joinpath("capture.json").read_text())
    layer = metadata.get("operator_layer")
    names = metadata.get("operator_boundaries", [])
    if layer is None or not names:
        raise SystemExit("Hipfire capture has no operator trace")

    oracle = np.load(args.oracle / "capture.npz")
    results: dict[str, object] = {"operator_layer": layer, "boundaries": {}}
    for name in names:
        key = f"operator_{name}"
        if key not in oracle:
            raise SystemExit(f"oracle capture is missing {key}")
        reference = oracle[key][0, -1]
        candidate = np.fromfile(args.hipfire / f"{key}.f32", dtype="<f4")
        if reference.size != candidate.size:
            raise SystemExit(
                f"{name} size differs: oracle={reference.size} hipfire={candidate.size}"
            )
        results["boundaries"][name] = vector_metrics(reference, candidate)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
