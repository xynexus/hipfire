#!/usr/bin/env python3
"""Prepare exact per-layer Gemma 4 transition inputs from frozen captures.

For layer 0, the input rows come from Hipfire's captured post-embedding
``pre_layer`` values. For every later layer, the input is the preceding frozen
Transformers decoder boundary. Expected outputs are always the corresponding
Transformers decoder boundaries. The resulting raw F32 files feed the
benchmark-only Rust layer-transition runner without putting Python in inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layer_zero_inputs(capture: Path) -> tuple[np.ndarray, dict[str, object]]:
    metadata = json.loads(capture.joinpath("capture.json").read_text())
    positions = int(metadata["operator_positions"])
    hidden_size = int(metadata["hidden_size"])
    if metadata.get("operator_layer") != 0 or positions < 1:
        raise ValueError("layer-zero capture must contain operator history for layer 0")

    rows = []
    for position in range(positions):
        path = capture / f"operator_position_{position}_pre_layer.f32"
        row = np.fromfile(path, dtype="<f4")
        if row.size != hidden_size:
            raise ValueError(
                f"{path} has {row.size} values, expected hidden size {hidden_size}"
            )
        rows.append(row)
    return np.stack(rows), metadata


def normalized_boundary(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{name} has shape {values.shape}, expected [positions, hidden]")
    return np.ascontiguousarray(values, dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--layer-zero-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oracle_npz = args.oracle / "capture.npz"
    oracle_json = args.oracle / "capture.json"
    layer_zero_json = args.layer_zero_input / "capture.json"
    layer_zero, layer_zero_metadata = load_layer_zero_inputs(args.layer_zero_input)

    with np.load(oracle_npz, allow_pickle=False) as oracle:
        layer_indices = sorted(
            int(name.removeprefix("hidden_layer_"))
            for name in oracle.files
            if name.startswith("hidden_layer_")
        )
        if layer_indices != list(range(len(layer_indices))):
            raise ValueError(f"oracle decoder layers are not contiguous: {layer_indices}")
        input_ids = np.asarray(oracle["input_ids"], dtype=np.uint32).reshape(-1)
        capture_ids = np.asarray(layer_zero_metadata["input_ids"], dtype=np.uint32)
        if not np.array_equal(input_ids, capture_ids):
            raise ValueError("oracle and layer-zero captures have different input ids")

        boundaries = [
            normalized_boundary(oracle[f"hidden_layer_{layer}"], f"hidden_layer_{layer}")
            for layer in layer_indices
        ]

    if not boundaries:
        raise ValueError("oracle capture has no decoder boundaries")
    if layer_zero.shape != boundaries[0].shape:
        raise ValueError(
            f"layer-zero input shape {layer_zero.shape} does not match {boundaries[0].shape}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    for layer, expected in enumerate(boundaries):
        inputs = layer_zero if layer == 0 else boundaries[layer - 1]
        input_path = args.output / f"input_layer_{layer}.f32"
        expected_path = args.output / f"expected_layer_{layer}.f32"
        np.ascontiguousarray(inputs, dtype="<f4").tofile(input_path)
        expected.tofile(expected_path)
        artifacts[str(layer)] = {
            "input": input_path.name,
            "input_sha256": sha256(input_path),
            "expected": expected_path.name,
            "expected_sha256": sha256(expected_path),
        }

    metadata = {
        "schema": "hipfire.gemma4.layer-transition-inputs.v1",
        "oracle": str(args.oracle.resolve()),
        "oracle_capture_sha256": sha256(oracle_npz),
        "oracle_metadata_sha256": sha256(oracle_json),
        "layer_zero_input": str(args.layer_zero_input.resolve()),
        "layer_zero_metadata_sha256": sha256(layer_zero_json),
        "input_ids": input_ids.tolist(),
        "positions": int(layer_zero.shape[0]),
        "hidden_size": int(layer_zero.shape[1]),
        "layers": len(boundaries),
        "artifacts": artifacts,
    }
    args.output.joinpath("manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"prepared {len(boundaries)} layer transitions x {layer_zero.shape[0]} positions "
        f"in {args.output}"
    )


if __name__ == "__main__":
    main()
