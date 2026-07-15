#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Compare final-row operator boundaries for isolated Gemma 4 transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


BOUNDARIES = [
    ("pre_layer", "pre_layer"),
    ("input_norm", "input_norm"),
    ("q_proj", "q_proj"),
    ("k_proj", "k_proj"),
    ("v_proj", "v_proj"),
    ("q_norm", "q_norm"),
    ("k_norm", "k_norm"),
    ("v_attention", "v_norm"),
    ("q_rope", "q_rope_scaled"),
    ("k_rope", "k_rope"),
    ("attention_raw", "attention_raw"),
    ("o_proj", "o_proj"),
    ("post_attention_norm", "post_attention_norm"),
    ("post_attention_residual", "post_attention_residual"),
    ("pre_ffn_norm", "pre_ffn_norm"),
    ("gate", "gate"),
    ("up", "up"),
    ("geglu", "geglu"),
    ("post_ffn_norm", "post_ffn_norm"),
    ("layer_output", "layer_output"),
]


def parse_layers(raw: str) -> list[int]:
    return sorted({int(value) for value in raw.split(",")})


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    non_finite = int(reference.size - np.count_nonzero(finite))
    if not np.any(finite):
        return {
            "maximum_absolute_error": None,
            "normalized_rmse": None,
            "cosine": None,
            "non_finite_values": non_finite,
            "values": int(reference.size),
        }
    reference = reference[finite]
    candidate = candidate[finite]
    difference = reference - candidate
    reference_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidate)
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "normalized_rmse": float(np.linalg.norm(difference) / reference_norm),
        "cosine": float(
            np.dot(reference, candidate) / (reference_norm * candidate_norm)
        ),
        "non_finite_values": non_finite,
        "values": int(reference.size),
    }


def final_oracle_row(path: Path, positions: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size % positions:
        raise ValueError(f"{path} has {values.size} values for {positions} positions")
    width = values.size // positions
    return values.reshape(positions, width)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--hipfire", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    layer_results = []
    for layer in args.layers:
        oracle_dir = args.oracle / f"layer_{layer}"
        hipfire_dir = args.hipfire / f"layer_{layer}"
        oracle_metadata = json.loads(oracle_dir.joinpath("capture.json").read_text())
        hipfire_metadata = json.loads(hipfire_dir.joinpath("capture.json").read_text())
        positions = int(oracle_metadata["positions"])
        if positions != int(hipfire_metadata["positions"]):
            raise ValueError(f"layer {layer} position counts differ")
        position = positions - 1
        head_scale = math.sqrt(float(oracle_metadata["head_dim"]))

        boundaries = []
        for oracle_name, hipfire_name in BOUNDARIES:
            oracle_path = oracle_dir / f"operator_{oracle_name}.f32"
            hipfire_path = hipfire_dir / f"position_{position}_{hipfire_name}.f32"
            if not oracle_path.exists() or not hipfire_path.exists():
                raise FileNotFoundError(
                    f"layer {layer} missing {oracle_path.name} or {hipfire_path.name}"
                )
            reference = final_oracle_row(oracle_path, positions)
            candidate = np.fromfile(hipfire_path, dtype="<f4")
            if oracle_name == "q_rope":
                candidate = candidate / head_scale
            if reference.shape != candidate.shape:
                raise ValueError(
                    f"layer {layer} {oracle_name}/{hipfire_name} shapes differ: "
                    f"{reference.shape} vs {candidate.shape}"
                )
            result = metrics(reference, candidate)
            boundaries.append(
                {
                    "boundary": oracle_name,
                    "hipfire_boundary": hipfire_name,
                    "metrics": result,
                }
            )
            print(
                f"layer={layer:02} boundary={oracle_name:<25} "
                f"max_abs={result['maximum_absolute_error']:.9f} "
                f"nrmse={result['normalized_rmse']:.9f} "
                f"cosine={result['cosine']:.9f}"
            )
        layer_results.append({"layer": layer, "boundaries": boundaries})

    report = {
        "schema": "hipfire.gemma4.layer-transition-operator-parity.v1",
        "oracle": str(args.oracle.resolve()),
        "hipfire": str(args.hipfire.resolve()),
        "layers": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
