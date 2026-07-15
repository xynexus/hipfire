#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Compare a Hipfire Gemma 4 candidate with the pinned BF16 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--hipfire", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).with_name("bf16-thresholds.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = reference.astype(np.float64, copy=False).reshape(-1)
    candidate = candidate.astype(np.float64, copy=False).reshape(-1)
    finite = int((~np.isfinite(candidate)).sum())
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    cosine = float(np.dot(reference, candidate) / denominator) if denominator else 1.0
    rmse = float(np.sqrt(np.mean(np.square(candidate - reference))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    return {
        "cosine": cosine,
        "normalized_rmse": rmse / reference_rms if reference_rms else rmse,
        "maximum_absolute_error": float(np.max(np.abs(candidate - reference))),
        "non_finite_values": finite,
    }


def compare_capture(
    oracle_dir: Path, hipfire_dir: Path, thresholds_path: Path
) -> dict[str, object]:
    thresholds = json.loads(thresholds_path.read_text())
    metadata = json.loads(hipfire_dir.joinpath("capture.json").read_text())
    oracle = np.load(oracle_dir / "capture.npz")
    failures: list[str] = []
    results: dict[str, object] = {}

    input_ids = np.asarray(metadata["input_ids"], dtype=np.uint32)
    if not np.array_equal(input_ids, oracle["input_ids"]):
        failures.append("input token IDs differ")

    hidden_limits = thresholds["hidden_states"]
    hidden_results = {}
    for layer in metadata["captured_layers"]:
        candidate = np.fromfile(
            hipfire_dir / f"hidden_layer_{layer}.f32", dtype="<f4"
        )
        reference = oracle[f"hidden_layer_{layer}"][0, -1]
        metrics = vector_metrics(reference, candidate)
        hidden_results[str(layer)] = metrics
        if metrics["cosine"] < hidden_limits["minimum_cosine"]:
            failures.append(f"layer {layer} cosine below threshold")
        if metrics["normalized_rmse"] > hidden_limits["maximum_normalized_rmse"]:
            failures.append(f"layer {layer} normalized RMSE above threshold")
        if metrics["non_finite_values"] > hidden_limits["non_finite_values_allowed"]:
            failures.append(f"layer {layer} contains non-finite values")
    results["hidden_states"] = hidden_results

    logits_reference = oracle["final_logits"].reshape(-1)
    logits_candidate = np.fromfile(
        hipfire_dir / "final_logits.f32", dtype="<f4"
    ).reshape(-1)
    logits = vector_metrics(logits_reference, logits_candidate)
    logits["reference_argmax"] = int(np.argmax(logits_reference))
    logits["candidate_argmax"] = int(np.argmax(logits_candidate))
    reference_top5 = set(np.argpartition(logits_reference, -5)[-5:].tolist())
    candidate_top5 = set(np.argpartition(logits_candidate, -5)[-5:].tolist())
    logits["top5_overlap"] = len(reference_top5 & candidate_top5)
    logit_limits = thresholds["final_logits"]
    if logits["cosine"] < logit_limits["minimum_cosine"]:
        failures.append("final-logit cosine below threshold")
    if logits["maximum_absolute_error"] > logit_limits["maximum_absolute_error"]:
        failures.append("final-logit maximum absolute error above threshold")
    if logits["non_finite_values"] > logit_limits["non_finite_values_allowed"]:
        failures.append("final logits contain non-finite values")
    if logit_limits["require_argmax_match"] and (
        logits["reference_argmax"] != logits["candidate_argmax"]
    ):
        failures.append("final-logit argmax differs")
    if logits["top5_overlap"] < logit_limits["minimum_top5_overlap"]:
        failures.append("final-logit top-5 overlap below threshold")
    results["final_logits"] = logits

    max_new = int(metadata["max_new_tokens"])
    oracle_generated = oracle["generated_ids"][0, len(input_ids) : len(input_ids) + max_new]
    candidate_generated = np.asarray(metadata["generated_ids"], dtype=np.uint32)
    generation_match = bool(np.array_equal(oracle_generated, candidate_generated))
    results["greedy_generation"] = {
        "match": generation_match,
        "oracle": oracle_generated.tolist(),
        "hipfire": candidate_generated.tolist(),
    }
    if thresholds["greedy_generation"]["require_exact_token_ids"] and not generation_match:
        failures.append("greedy generated token IDs differ")

    results["status"] = "pass" if not failures else "fail"
    results["failures"] = failures
    return results


def main() -> None:
    args = parse_args()
    results = compare_capture(args.oracle, args.hipfire, args.thresholds)
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if results["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
