#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Validate the layer-streamed Gemma 4 BF16 oracle against a resident capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_bf16_captures import vector_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--streaming", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.999999)
    parser.add_argument("--maximum-normalized-rmse", type=float, default=0.002)
    parser.add_argument("--maximum-logit-absolute-error", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resident = np.load(args.resident / "capture.npz")
    streaming = np.load(args.streaming / "capture.npz")
    metadata = json.loads(args.streaming.joinpath("capture.json").read_text())
    failures: list[str] = []

    if not np.array_equal(resident["input_ids"], streaming["input_ids"]):
        failures.append("input token IDs differ")

    hidden_results = {}
    for layer in metadata["captured_layers"]:
        key = f"hidden_layer_{layer}"
        metrics = vector_metrics(resident[key][0, -1], streaming[key][0, -1])
        hidden_results[str(layer)] = metrics
        if metrics["cosine"] < args.minimum_cosine:
            failures.append(f"layer {layer} cosine below validation threshold")
        if metrics["normalized_rmse"] > args.maximum_normalized_rmse:
            failures.append(f"layer {layer} normalized RMSE above validation threshold")
        if metrics["non_finite_values"]:
            failures.append(f"layer {layer} contains non-finite values")

    resident_logits = resident["final_logits"].reshape(-1)
    streaming_logits = streaming["final_logits"].reshape(-1)
    logits = vector_metrics(resident_logits, streaming_logits)
    logits["resident_argmax"] = int(np.argmax(resident_logits))
    logits["streaming_argmax"] = int(np.argmax(streaming_logits))
    resident_top5 = set(np.argpartition(resident_logits, -5)[-5:].tolist())
    streaming_top5 = set(np.argpartition(streaming_logits, -5)[-5:].tolist())
    logits["top5_overlap"] = len(resident_top5 & streaming_top5)
    if logits["cosine"] < args.minimum_cosine:
        failures.append("final-logit cosine below validation threshold")
    if logits["maximum_absolute_error"] > args.maximum_logit_absolute_error:
        failures.append("final-logit maximum absolute error above validation threshold")
    if logits["non_finite_values"]:
        failures.append("final logits contain non-finite values")
    if logits["resident_argmax"] != logits["streaming_argmax"]:
        failures.append("final-logit argmax differs")
    if logits["top5_overlap"] != 5:
        failures.append("final-logit top-5 set differs")

    prompt_length = len(resident["input_ids"])
    resident_token = int(resident["generated_ids"][0, prompt_length])
    streaming_token = int(streaming["generated_ids"][0, prompt_length])
    generation = {
        "match": resident_token == streaming_token,
        "resident": resident_token,
        "streaming": streaming_token,
    }
    if not generation["match"]:
        failures.append("first greedy generated token differs")

    result = {
        "schema": "hipfire.gemma4.transformers-oracle-validation.v1",
        "resident": str(args.resident.resolve()),
        "streaming": str(args.streaming.resolve()),
        "thresholds": {
            "minimum_cosine": args.minimum_cosine,
            "maximum_normalized_rmse": args.maximum_normalized_rmse,
            "maximum_logit_absolute_error": args.maximum_logit_absolute_error,
            "require_argmax_match": True,
            "require_top5_set_match": True,
            "require_first_greedy_token_match": True,
        },
        "hidden_states": hidden_results,
        "final_logits": logits,
        "greedy_generation": generation,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
