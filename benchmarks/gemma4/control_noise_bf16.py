#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Measure observed Gemma 4 BF16 reference-execution variability.

This is a *control* measurement, not an oracle. It runs the same BF16 checkpoint
under several mathematically-equivalent settings that differ only in reduction
order / internal accumulation:

  * attention implementation: sdpa vs eager;
  * GEMM/attention tiling: the prompt processed alone vs right-padded in a longer
    masked sequence (padding changes tile boundaries and reduction order but,
    with a correct attention mask, does not change the math for the real tokens).

These are reference-implementation controls, not an admission oracle. The tool
records every pairwise comparison and, separately, the envelope around the
canonical unpadded SDPA sample used by Hipfire's frozen comparator. It does not
rewrite or recommend admission thresholds: a finite set of implementations and
one prompt establishes an observed variability band, not an irreducible floor.

Offline evidence tool. Runtime code never imports it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--impls", default="sdpa,eager")
    parser.add_argument(
        "--canonical-sample",
        default="sdpa",
        help="sample whose envelope matches the pinned comparison oracle",
    )
    parser.add_argument(
        "--pad-to",
        type=int,
        default=16,
        help="right-pad the prompt to this length as a second, equally-correct tiling",
    )
    return parser.parse_args()


def tokenizer_for(model: Path) -> PreTrainedTokenizerFast:
    cfg = json.loads(model.joinpath("tokenizer_config.json").read_text())
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model / "tokenizer.json"),
        bos_token=cfg["bos_token"],
        eos_token=cfg["eos_token"],
        pad_token=cfg["pad_token"],
    )


def capture_run(model, ids, attention_mask, last_real_index):
    """Return (final_logits[f32], {layer: hidden[f32]}) at the real last token."""
    text_model = getattr(getattr(model, "model", None), "language_model", None)
    if text_model is None:
        text_model = getattr(model, "language_model", None)
    layers = text_model.layers

    captured: dict[int, np.ndarray] = {}
    hooks = []
    for index in range(len(layers)):

        def capture(_m, _i, output, *, layer=index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, last_real_index].float().cpu().numpy()

        hooks.append(layers[index].register_forward_hook(capture))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        final = out.logits[0, last_real_index, :].float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return final, captured


def load(model_path, device, impl):
    m = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map={"": device},
        attn_implementation=impl,
    )
    m.eval()
    return m


def floor_metrics(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    ref_rms = float(np.sqrt(np.mean(np.square(a))))
    rmse = float(np.sqrt(np.mean(np.square(a - b))))
    return {
        "maximum_absolute_error": float(np.max(np.abs(a - b))),
        "cosine": float(np.dot(a, b) / denom) if denom else 1.0,
        "normalized_rmse": rmse / ref_rms if ref_rms else rmse,
    }


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values.astype("<f4", copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def sample_summary(final: np.ndarray, hidden: dict[int, np.ndarray]) -> dict:
    top5 = np.argpartition(final, -5)[-5:]
    finite_hidden = sum(int((~np.isfinite(values)).sum()) for values in hidden.values())
    hidden_digest = hashlib.sha256()
    for layer in sorted(hidden):
        hidden_digest.update(layer.to_bytes(4, "little"))
        hidden_digest.update(
            np.ascontiguousarray(hidden[layer].astype("<f4", copy=False)).tobytes()
        )
    return {
        "final_logits_sha256": array_sha256(final),
        "hidden_states_sha256": hidden_digest.hexdigest(),
        "final_argmax": int(np.argmax(final)),
        "final_top5": sorted(int(value) for value in top5),
        "final_non_finite_values": int((~np.isfinite(final)).sum()),
        "hidden_non_finite_values": finite_hidden,
    }


def compare_samples(
    label_a: str,
    sample_a: tuple,
    label_b: str,
    sample_b: tuple,
) -> dict:
    final_a, hidden_a = sample_a
    final_b, hidden_b = sample_b
    final = floor_metrics(final_a, final_b)
    top5_a = set(np.argpartition(final_a, -5)[-5:].tolist())
    top5_b = set(np.argpartition(final_b, -5)[-5:].tolist())
    final.update(
        {
            "reference_argmax": int(np.argmax(final_a)),
            "candidate_argmax": int(np.argmax(final_b)),
            "top5_overlap": len(top5_a & top5_b),
        }
    )

    per_layer = {
        str(layer): floor_metrics(hidden_a[layer], hidden_b[layer])
        for layer in sorted(hidden_a)
    }
    worst_nrmse_layer, worst_nrmse = max(
        per_layer.items(), key=lambda item: item[1]["normalized_rmse"]
    )
    minimum_cosine_layer, minimum_cosine = min(
        per_layer.items(), key=lambda item: item[1]["cosine"]
    )
    return {
        "reference": label_a,
        "candidate": label_b,
        "final_logits": final,
        "hidden_states": {
            "worst_normalized_rmse_layer": int(worst_nrmse_layer),
            "worst_normalized_rmse": worst_nrmse["normalized_rmse"],
            "minimum_cosine_layer": int(minimum_cosine_layer),
            "minimum_cosine": minimum_cosine["cosine"],
            "per_layer": per_layer,
        },
    }


def envelope(comparisons: list[dict]) -> dict:
    if not comparisons:
        raise ValueError("at least one comparison is required")
    final_mae = max(
        comparison["final_logits"]["maximum_absolute_error"]
        for comparison in comparisons
    )
    final_cos = min(
        comparison["final_logits"]["cosine"] for comparison in comparisons
    )
    hidden_nrmse = max(
        comparison["hidden_states"]["worst_normalized_rmse"]
        for comparison in comparisons
    )
    hidden_cos = min(
        comparison["hidden_states"]["minimum_cosine"]
        for comparison in comparisons
    )
    return {
        "final_logit_maximum_absolute_error": final_mae,
        "final_logit_minimum_cosine": final_cos,
        "hidden_worst_normalized_rmse": hidden_nrmse,
        "hidden_minimum_cosine_any_layer": hidden_cos,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but PyTorch has no ROCm/CUDA device")

    impls = [x.strip() for x in args.impls.split(",") if x.strip()]
    if not impls:
        raise SystemExit("--impls must contain at least one implementation")
    tokenizer = tokenizer_for(args.model)
    real_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    n = len(real_ids)
    device = torch.device(args.device)
    pad_id = tokenizer.pad_token_id or 0

    unpadded_ids = torch.tensor([real_ids], dtype=torch.long, device=device)
    unpadded_mask = torch.ones((1, n), dtype=torch.long, device=device)

    padded = list(real_ids) + [pad_id] * max(0, args.pad_to - n)
    padded_ids = torch.tensor([padded], dtype=torch.long, device=device)
    padded_mask = torch.tensor(
        [[1] * n + [0] * (len(padded) - n)], dtype=torch.long, device=device
    )

    # sample label -> (final_logits, {layer: hidden}). Each is an independently
    # valid Transformers BF16 reference execution for this diagnostic.
    samples: dict[str, tuple] = {}
    for impl in impls:
        model = load(args.model, device, impl)
        samples[f"{impl}"] = capture_run(model, unpadded_ids, unpadded_mask, n - 1)
        if args.pad_to > n:
            samples[f"{impl}_pad{args.pad_to}"] = capture_run(
                model, padded_ids, padded_mask, n - 1
            )
        # Drop main's own reference before the next load, or the previous 58 GiB
        # model is still live when from_pretrained warms the allocator -> OOM.
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = list(samples)
    pairs = list(itertools.combinations(labels, 2))
    if not pairs:
        raise SystemExit("the selected implementations/padding produced fewer than 2 samples")
    if args.canonical_sample not in samples:
        raise SystemExit(
            f"--canonical-sample {args.canonical_sample!r} is not in samples {labels}"
        )

    pairwise = [compare_samples(a, samples[a], b, samples[b]) for a, b in pairs]
    canonical = [
        compare_samples(
            args.canonical_sample,
            samples[args.canonical_sample],
            label,
            samples[label],
        )
        for label in labels
        if label != args.canonical_sample
    ]

    # Retain compact per-layer extrema for convenient plotting. The complete
    # pairwise metrics above preserve which pair produced each observation.
    layer_ids = sorted(samples[labels[0]][1])
    hidden_floor = {}
    worst_layer = None
    worst_nrmse = -1.0
    min_cos = 2.0
    for layer in layer_ids:
        max_nrmse = 0.0
        max_mae = 0.0
        lo_cos = 2.0
        for a, b in pairs:
            m = floor_metrics(samples[a][1][layer], samples[b][1][layer])
            max_nrmse = max(max_nrmse, m["normalized_rmse"])
            max_mae = max(max_mae, m["maximum_absolute_error"])
            lo_cos = min(lo_cos, m["cosine"])
        hidden_floor[str(layer)] = {
            "worst_normalized_rmse": max_nrmse,
            "worst_maximum_absolute_error": max_mae,
            "minimum_cosine": lo_cos,
        }
        if max_nrmse > worst_nrmse:
            worst_nrmse, worst_layer = max_nrmse, layer
        min_cos = min(min_cos, lo_cos)

    logit_floor = {"maximum_absolute_error": 0.0, "minimum_cosine": 2.0}
    for a, b in pairs:
        m = floor_metrics(samples[a][0], samples[b][0])
        logit_floor["maximum_absolute_error"] = max(
            logit_floor["maximum_absolute_error"], m["maximum_absolute_error"]
        )
        logit_floor["minimum_cosine"] = min(logit_floor["minimum_cosine"], m["cosine"])

    summaries = {
        label: sample_summary(samples[label][0], samples[label][1]) for label in labels
    }
    report = {
        "scope": "Gemma 4 observed BF16 reference-execution variability",
        "admission_use": "diagnostic only; does not modify frozen thresholds",
        "model": str(args.model.resolve()),
        "prompt": args.prompt,
        "input_token_count": n,
        "samples": labels,
        "canonical_sample": args.canonical_sample,
        "pairs_compared": len(pairs),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "rocm_version": torch.version.hip,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "sample_summaries": summaries,
        "canonical_reference_envelope": envelope(canonical),
        "all_pairwise_envelope": envelope(pairwise),
        "pairwise_comparisons": pairwise,
        "final_logits_floor": logit_floor,
        "hidden_states_floor": {
            "worst_layer": worst_layer,
            "worst_normalized_rmse": worst_nrmse,
            "minimum_cosine_any_layer": min_cos,
            "per_layer": hidden_floor,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"samples: {labels}  ({len(pairs)} reference-vs-reference pairs)")
    canonical_band = report["canonical_reference_envelope"]
    all_band = report["all_pairwise_envelope"]
    print(f"canonical sample: {args.canonical_sample}")
    print(f"canonical reference envelope: {canonical_band}")
    print(f"all-pairwise observed envelope: {all_band}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
