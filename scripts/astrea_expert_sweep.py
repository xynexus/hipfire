#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Immutable Astrea contracts for routed-expert calibration sweeps.

This module only plans workflow commands. The calibration data plane and model
forward remain in ``hipfire-coexistence calibrate``; quantization remains in
``hipfire-quantize`` through the native two-pass wrapper.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from string import Formatter


EXPERT_SWEEP_PLAN_SCHEMA = "hipfire.astrea.expert_calibration_sweep_plan.v0"
DEFAULT_QUANT_FORMAT = "oq4.25++"
DEFAULT_MINIMUM_ROWS = (512, 1024, 2048, 4096)
DEFAULT_CAPTURE_TARGETS = (2048, 4096, 8192)
DEFAULT_FIXED_CAPTURE_TARGET = 4096
DEFAULT_QUANT_ARGS = ("--awq", "--ldlq")
DEFAULT_SEQUENCES = 128
DEFAULT_CONTEXT = 2048
DEFAULT_SEQUENCE_BATCH = 64
DEFAULT_TIME_TILE = 32
DEFAULT_MAX_ROWS = 2048
DEFAULT_LAYER_PREFETCH_BYTES = 16 * 1024**3
DEFAULT_KLDREF_TOPK = 64
DEFAULT_EXPERT_CAPTURE_TILE_ROWS = 256
DEFAULT_REQUIRED_EXPERT_FRACTION = 1.0
DEFAULT_SAMPLING_SEED = 1
DEFAULT_EXPERT_COVERAGE_POLICY = "preserve-undercovered"

_ARTIFACT_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_REQUIRED_EVAL_FIELDS = {
    "candidate",
    "reference_model",
    "evaluation_dataset",
    "evaluation_output",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _dataset(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"dataset does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _positive_unique(values, label: str) -> list[int]:
    result = sorted(set(int(value) for value in values))
    if not result or any(value < 1 for value in result):
        raise ValueError(f"{label} must contain positive rows")
    return result


def _template_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }


def _render_evaluation_command(template: str, values: dict[str, str]) -> list[str]:
    fields = _template_fields(template)
    missing = sorted(_REQUIRED_EVAL_FIELDS - fields)
    unknown = sorted(fields - values.keys())
    if missing:
        raise ValueError(
            "evaluation command template is missing required placeholders: " + ", ".join(missing)
        )
    if unknown:
        raise ValueError("evaluation command template has unknown placeholders: " + ", ".join(unknown))
    command = shlex.split(template.format(**values))
    if not command:
        raise ValueError("evaluation command template produced an empty command")
    return command


def _two_pass_command(
    *,
    model: Path,
    calibration_artifact: Path,
    quantized_artifact: Path,
    manifest: Path,
    quant_format: str,
    calibration_dataset: Path,
    sequences: int,
    context: int,
    sequence_batch: int,
    time_tile: int,
    max_rows: int,
    layer_prefetch_bytes: int,
    kldref_topk: int,
    minimum_rows: int,
    capture_target: int,
    capture_tile_rows: int,
    required_expert_fraction: float,
    sampling_seed: int,
    expert_coverage_policy: str,
    quant_args: list[str],
) -> list[str]:
    return [
        "python3",
        "scripts/two_pass_quantize.py",
        "--model",
        str(model.resolve()),
        "--calib",
        str(calibration_artifact.resolve()),
        "--output",
        str(quantized_artifact.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--format",
        quant_format,
        "--corpus",
        str(calibration_dataset.resolve()),
        "--n-sequences",
        str(sequences),
        "--ctx-len",
        str(context),
        "--batch-size",
        str(sequence_batch),
        "--time-tile",
        str(time_tile),
        "--max-rows",
        str(max_rows),
        "--layer-prefetch-bytes",
        str(layer_prefetch_bytes),
        "--kldref-topk",
        str(kldref_topk),
        "--min-expert-activations",
        str(minimum_rows),
        "--expert-capture-target",
        str(capture_target),
        "--expert-capture-tile-rows",
        str(capture_tile_rows),
        "--required-expert-fraction",
        str(required_expert_fraction),
        "--sampling-seed",
        str(sampling_seed),
        "--expert-coverage-policy",
        expert_coverage_policy,
        "--",
        *quant_args,
    ]


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _plan_fingerprint(payload: dict) -> str:
    stable = json.loads(json.dumps(payload))
    if isinstance(stable.get("engine"), dict):
        stable["engine"].pop("captured_at_utc", None)
    return _fingerprint(stable)


def build_plan(
    *,
    model,
    artifact_stem,
    calibration_dataset,
    evaluation_dataset,
    reference_model,
    output_dir,
    evaluation_command_template,
    axis,
    minimum_rows=None,
    capture_targets=None,
    selected_minimum=None,
    fixed_capture_target=None,
    quant_format=DEFAULT_QUANT_FORMAT,
    quant_args=None,
    sequences=DEFAULT_SEQUENCES,
    context=DEFAULT_CONTEXT,
    sequence_batch=DEFAULT_SEQUENCE_BATCH,
    time_tile=DEFAULT_TIME_TILE,
    max_rows=DEFAULT_MAX_ROWS,
    layer_prefetch_bytes=DEFAULT_LAYER_PREFETCH_BYTES,
    kldref_topk=DEFAULT_KLDREF_TOPK,
    capture_tile_rows=DEFAULT_EXPERT_CAPTURE_TILE_ROWS,
    required_expert_fraction=DEFAULT_REQUIRED_EXPERT_FRACTION,
    sampling_seed=DEFAULT_SAMPLING_SEED,
    expert_coverage_policy=DEFAULT_EXPERT_COVERAGE_POLICY,
    hipfire="target/release/hipfire",
    evaluation_owns_resource_lease=False,
    engine=None,
    command=None,
) -> dict:
    """Freeze a minimum-floor or capture-target expert calibration sweep."""

    model = Path(model)
    calibration_dataset = Path(calibration_dataset)
    evaluation_dataset = Path(evaluation_dataset)
    reference_model = Path(reference_model)
    output_dir = Path(output_dir)
    if not _ARTIFACT_STEM.fullmatch(str(artifact_stem)) or str(artifact_stem).endswith(".hfq"):
        raise ValueError("artifact stem must be a canonical filename stem without a path or .hfq suffix")
    if axis not in {"minimum", "capture"}:
        raise ValueError("axis must be 'minimum' or 'capture'")
    if expert_coverage_policy not in {"strict", "preserve-undercovered"}:
        raise ValueError("expert coverage policy must be strict or preserve-undercovered")
    geometry = [sequences, context, sequence_batch, time_tile, max_rows, kldref_topk, capture_tile_rows]
    if any(int(value) < 1 for value in geometry):
        raise ValueError("sequence, context, geometry, top-k, and capture tile values must be positive")
    if sequence_batch * time_tile > max_rows:
        raise ValueError("sequence_batch * time_tile must not exceed max_rows")
    if layer_prefetch_bytes < 0:
        raise ValueError("layer_prefetch_bytes must be nonnegative")
    if not 0.0 < required_expert_fraction <= 1.0:
        raise ValueError("required_expert_fraction must be in (0, 1]")
    if sampling_seed < 0:
        raise ValueError("sampling_seed must be nonnegative")

    datasets = {
        "calibration": _dataset(calibration_dataset),
        "evaluation": _dataset(evaluation_dataset),
    }
    if datasets["calibration"]["sha256"] == datasets["evaluation"]["sha256"]:
        raise ValueError("calibration and evaluation datasets must have distinct content")

    if axis == "minimum":
        floors = _positive_unique(minimum_rows or DEFAULT_MINIMUM_ROWS, "minimum rows")
        target = int(fixed_capture_target or DEFAULT_FIXED_CAPTURE_TARGET)
        if target < max(floors):
            raise ValueError("fixed capture target must be at least every swept minimum")
        points = [(floor, target) for floor in floors]
        canonical_axis = "minimum_expert_activations"
        selection_contract = {
            "capture_target_held_fixed": target,
            "selection_evidence_required": True,
            "selection_rule": "select the lowest sufficient floor from held-out KLD/PPL and low-traffic expert evidence",
        }
    else:
        if selected_minimum is None or int(selected_minimum) < 1:
            raise ValueError("capture sweep requires a positive selected minimum")
        floor = int(selected_minimum)
        targets = _positive_unique(capture_targets or DEFAULT_CAPTURE_TARGETS, "capture targets")
        if any(target < floor for target in targets):
            raise ValueError("capture target is below the selected minimum")
        points = [(floor, target) for target in targets]
        canonical_axis = "expert_capture_target"
        selection_contract = {
            "selected_minimum": floor,
            "selection_evidence_required": True,
            "selection_rule": "select the lowest sufficient capture target from held-out quality and capture-cost evidence",
        }

    quant_args = list(DEFAULT_QUANT_ARGS if quant_args is None else quant_args)
    recipe = {
        "quant_format": str(quant_format),
        "quant_args": quant_args,
        "sequences": int(sequences),
        "context": int(context),
        "sequence_batch": int(sequence_batch),
        "time_tile": int(time_tile),
        "max_rows": int(max_rows),
        "layer_prefetch_bytes": int(layer_prefetch_bytes),
        "kldref_topk": int(kldref_topk),
        "expert_capture_tile_rows": int(capture_tile_rows),
        "required_expert_fraction": float(required_expert_fraction),
        "sampling_seed": int(sampling_seed),
        "expert_coverage_policy": expert_coverage_policy.replace("-", "_"),
        "evaluation_locking": (
            "resource_lease_owned_by_command" if evaluation_owns_resource_lease else "hipfire_flock"
        ),
    }

    variants = []
    for floor, target in points:
        variant_id = f"min{floor}-cap{target}"
        calibration_artifact = output_dir / f"{artifact_stem}.{variant_id}.calib.hfq"
        quantized_artifact = output_dir / f"{artifact_stem}.{variant_id}.{quant_format}.hfq"
        two_pass_manifest = output_dir / "manifests" / f"{variant_id}.two-pass.json"
        evaluation_output = output_dir / "evaluation" / variant_id
        values = {
            "variant": variant_id,
            "candidate": str(quantized_artifact.resolve()),
            "calibration_artifact": str(calibration_artifact.resolve()),
            "reference_model": str(reference_model.resolve()),
            "calibration_dataset": datasets["calibration"]["path"],
            "evaluation_dataset": datasets["evaluation"]["path"],
            "evaluation_output": str(evaluation_output.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        evaluation_command = _render_evaluation_command(evaluation_command_template, values)
        if not evaluation_owns_resource_lease:
            evaluation_command = [
                str(hipfire),
                "lock",
                "run",
                f"expert-calibration-sweep-{variant_id}",
                "--",
                *evaluation_command,
            ]
        variants.append(
            {
                "id": variant_id,
                "minimum_expert_activations": floor,
                "expert_capture_target": target,
                "calibration_artifact": values["calibration_artifact"],
                "quantized_artifact": values["candidate"],
                "two_pass_manifest": str(two_pass_manifest.resolve()),
                "evaluation_output": values["evaluation_output"],
                "two_pass_command": _two_pass_command(
                    model=model,
                    calibration_artifact=calibration_artifact,
                    quantized_artifact=quantized_artifact,
                    manifest=two_pass_manifest,
                    quant_format=quant_format,
                    calibration_dataset=calibration_dataset,
                    sequences=sequences,
                    context=context,
                    sequence_batch=sequence_batch,
                    time_tile=time_tile,
                    max_rows=max_rows,
                    layer_prefetch_bytes=layer_prefetch_bytes,
                    kldref_topk=kldref_topk,
                    minimum_rows=floor,
                    capture_target=target,
                    capture_tile_rows=capture_tile_rows,
                    required_expert_fraction=required_expert_fraction,
                    sampling_seed=sampling_seed,
                    expert_coverage_policy=expert_coverage_policy,
                    quant_args=quant_args,
                ),
                "evaluation_command": evaluation_command,
            }
        )

    body = {
        "schema": EXPERT_SWEEP_PLAN_SCHEMA,
        "status": "planned_heldout_untouched",
        "axis": canonical_axis,
        "model": {"path": str(model.resolve())},
        "reference_model": {"path": str(reference_model.resolve())},
        "artifact_stem": str(artifact_stem),
        "output_dir": str(output_dir.resolve()),
        "datasets": datasets,
        "engine": engine or {},
        "recipe": recipe,
        "selection_contract": selection_contract,
        "comparison_contract": {
            "one_axis_at_a_time": True,
            "evaluation_dataset_frozen_before_execution": True,
            "evaluation_dataset_must_remain_untouched_by_calibration": True,
            "required_metrics": [
                "mean_kld",
                "ppl",
                "low_traffic_expert_sensitivity",
                "artifact_size_bytes",
                "capture_seconds",
                "reduction_launches",
            ],
            "promotion_eligible_without_complete_metrics": False,
        },
        "variants": variants,
        "command_argv": list(command or []),
    }
    return {**body, "plan_fingerprint": _plan_fingerprint(body)}
