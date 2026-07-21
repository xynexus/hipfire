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
import math
import re
import shlex
import struct
from pathlib import Path
from string import Formatter


EXPERT_SWEEP_PLAN_SCHEMA = "hipfire.astrea.expert_calibration_sweep_plan.v1"
EXPERT_SWEEP_VERIFY_SCHEMA = "hipfire.astrea.expert_calibration_sweep_verify.v1"
EXPERT_SWEEP_RESULTS_SCHEMA = "hipfire.astrea.expert_calibration_sweep_results.v1"
EXPERT_SWEEP_ANALYSIS_SCHEMA = "hipfire.astrea.expert_calibration_sweep_analysis.v1"
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
_MAX_CONTROL_REGION_BYTES = 512 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_region(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = int(length)
    with path.open("rb") as source:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"short read while fingerprinting {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve_safetensors_root(path: Path) -> Path:
    requested = path.expanduser().resolve()
    if not requested.is_dir():
        raise ValueError(f"source model directory does not exist: {requested}")
    if (requested / "model.safetensors.index.json").is_file() or any(requested.glob("*.safetensors")):
        return requested

    main_ref = requested / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = requested / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot.resolve()

    snapshots = requested / "snapshots"
    candidates = (
        sorted(child.resolve() for child in snapshots.iterdir() if child.is_dir()) if snapshots.is_dir() else []
    )
    candidates = [
        child
        for child in candidates
        if (child / "model.safetensors.index.json").is_file() or any(child.glob("*.safetensors"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"cannot resolve a unique safetensors snapshot under {requested}")


def _safetensors_header_identity(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors shard is smaller than its header prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length > size - 8 or header_length > _MAX_CONTROL_REGION_BYTES:
            raise ValueError(f"safetensors shard has an invalid header length: {path}")
    return _sha256_region(path, 8 + header_length)


def _source_shard_identity(path: Path, root: Path) -> dict:
    relative = str(path.relative_to(root))
    size = path.stat().st_size
    if path.is_symlink():
        blob = path.readlink().name
        is_digest = len(blob) in {40, 64} and all(character in "0123456789abcdefABCDEF" for character in blob)
        if is_digest:
            return {
                "file": relative,
                "bytes": size,
                "identity_kind": "huggingface_blob_digest",
                "identity": blob.lower(),
            }
    return {
        "file": relative,
        "bytes": size,
        "identity_kind": "safetensors_header_hash",
        "identity": _safetensors_header_identity(path),
    }


def _safetensors_manifest_identity(path: Path) -> dict:
    root = _resolve_safetensors_root(path)
    config = root / "config.json"
    if not config.is_file():
        raise ValueError(f"source model has no config.json: {root}")
    index = root / "model.safetensors.index.json"
    control_files = [{"file": "config.json", "sha256": _sha256_file(config)}]
    if index.is_file():
        parsed = json.loads(index.read_text(encoding="utf-8"))
        shard_names = sorted(set(parsed.get("weight_map", {}).values()))
        if not shard_names:
            raise ValueError(f"safetensors index has no weight_map entries: {index}")
        control_files.append({"file": index.name, "sha256": _sha256_file(index)})
        shards = [root / name for name in shard_names]
    else:
        shards = sorted(root.glob("*.safetensors"))
    if not shards or any(not shard.is_file() for shard in shards):
        raise ValueError(f"source model has missing safetensors shards: {root}")
    shard_records = [_source_shard_identity(shard, root) for shard in shards]
    stable = {"control_files": control_files, "shards": shard_records}
    return {
        "kind": "safetensors_manifest",
        "resolved_root": str(root),
        **stable,
        "fingerprint": _canonical_sha256(stable),
    }


def _reference_identity(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"reference model does not exist: {resolved}")
    size = resolved.stat().st_size
    with resolved.open("rb") as source:
        header = source.read(32)
    if len(header) == 32 and header[:4] == b"HFQM":
        _, version, arch_id, tensor_count, metadata_offset, data_offset = struct.unpack("<4sIIIQQ", header)
        if not 32 <= metadata_offset <= data_offset <= size:
            raise ValueError(f"reference HFQ has invalid metadata/data offsets: {resolved}")
        if data_offset > _MAX_CONTROL_REGION_BYTES:
            raise ValueError(f"reference HFQ control region is unreasonably large: {resolved}")
        return {
            "kind": "hfq_control_region",
            "bytes": size,
            "version": version,
            "arch_id": arch_id,
            "tensor_count": tensor_count,
            "data_offset": data_offset,
            "sha256": _sha256_region(resolved, data_offset),
        }
    return {
        "kind": "complete_file",
        "bytes": size,
        "sha256": _sha256_file(resolved),
    }


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
    return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None}


def _render_evaluation_command(template: str, values: dict[str, str]) -> list[str]:
    fields = _template_fields(template)
    missing = sorted(_REQUIRED_EVAL_FIELDS - fields)
    unknown = sorted(fields - values.keys())
    if missing:
        raise ValueError("evaluation command template is missing required placeholders: " + ", ".join(missing))
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


def _command_value(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
        return command[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"variant two-pass command is missing {flag}") from error


def verify_plan(plan, *, current_engine=None) -> dict:
    """Validate a frozen sweep contract immediately before model execution."""

    if not isinstance(plan, dict) or plan.get("schema") != EXPERT_SWEEP_PLAN_SCHEMA:
        raise ValueError("unsupported expert sweep plan schema")
    fingerprint = plan.get("plan_fingerprint")
    body = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    expected = _plan_fingerprint(body)
    if fingerprint != expected:
        raise ValueError(f"plan fingerprint mismatch: recorded {fingerprint}, computed {expected}")

    observed_datasets = {}
    for role in ("calibration", "evaluation"):
        record = plan.get("datasets", {}).get(role)
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"plan has no {role} dataset identity")
        path = Path(record["path"])
        if not path.is_file():
            raise ValueError(f"{role} dataset is missing: {path}")
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise ValueError(f"{role} dataset hash drift: recorded {record['sha256']}, observed {observed}")
        observed_datasets[role] = observed
    if observed_datasets["calibration"] == observed_datasets["evaluation"]:
        raise ValueError("calibration and evaluation datasets no longer have distinct content")

    model_record = plan.get("model", {})
    reference_record = plan.get("reference_model", {})
    model = Path(model_record.get("path", ""))
    reference = Path(reference_record.get("path", ""))
    if not model.is_dir():
        raise ValueError(f"source model directory is missing: {model}")
    if not reference.is_file():
        raise ValueError(f"reference model is missing: {reference}")
    observed_source_identity = _safetensors_manifest_identity(model)
    if model_record.get("identity") != observed_source_identity:
        raise ValueError("source model identity drift")
    observed_reference_identity = _reference_identity(reference)
    if reference_record.get("identity") != observed_reference_identity:
        raise ValueError("reference model identity drift")

    planned_engine = plan.get("engine", {}).get("fingerprint_id")
    observed_engine = (current_engine or {}).get("fingerprint_id")
    if current_engine is not None and planned_engine != observed_engine:
        raise ValueError(f"engine fingerprint drift: recorded {planned_engine}, observed {observed_engine}")

    variants = plan.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("expert sweep plan has no variants")
    ids = [variant.get("id") for variant in variants]
    if any(not variant_id for variant_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("expert sweep variant ids must be non-empty and unique")
    outputs = [variant.get("quantized_artifact") for variant in variants]
    if any(not output for output in outputs) or len(outputs) != len(set(outputs)):
        raise ValueError("expert sweep quantized outputs must be non-empty and unique")

    minimums = set()
    targets = set()
    source_model = str(model.resolve())
    reference_model = str(reference.resolve())
    calibration_dataset = plan["datasets"]["calibration"]["path"]
    evaluation_dataset = plan["datasets"]["evaluation"]["path"]
    recipe = plan.get("recipe", {})
    for variant in variants:
        minimum = int(variant.get("minimum_expert_activations", 0))
        target = int(variant.get("expert_capture_target", 0))
        if minimum < 1 or target < minimum:
            raise ValueError(f"variant {variant.get('id')} has an invalid minimum/capture target")
        minimums.add(minimum)
        targets.add(target)
        two_pass = variant.get("two_pass_command")
        evaluation = variant.get("evaluation_command")
        if not isinstance(two_pass, list) or not isinstance(evaluation, list):
            raise ValueError(f"variant {variant.get('id')} has malformed commands")
        if _command_value(two_pass, "--min-expert-activations") != str(minimum):
            raise ValueError(f"variant {variant.get('id')} minimum disagrees with its command")
        if _command_value(two_pass, "--expert-capture-target") != str(target):
            raise ValueError(f"variant {variant.get('id')} capture target disagrees with its command")
        command_bindings = (
            ("--model", source_model, "source model"),
            ("--calib", variant["calibration_artifact"], "calibration artifact"),
            ("--output", variant["quantized_artifact"], "quantized artifact"),
            ("--manifest", variant["two_pass_manifest"], "two-pass manifest"),
            ("--format", recipe.get("quant_format"), "quant format"),
            ("--corpus", calibration_dataset, "calibration dataset"),
            ("--n-sequences", str(recipe.get("sequences")), "sequence count"),
            ("--ctx-len", str(recipe.get("context")), "context"),
            ("--batch-size", str(recipe.get("sequence_batch")), "sequence batch"),
            ("--time-tile", str(recipe.get("time_tile")), "time tile"),
            ("--max-rows", str(recipe.get("max_rows")), "row budget"),
            (
                "--layer-prefetch-bytes",
                str(recipe.get("layer_prefetch_bytes")),
                "layer prefetch bytes",
            ),
            ("--kldref-topk", str(recipe.get("kldref_topk")), "KLDREF top-k"),
            (
                "--expert-capture-tile-rows",
                str(recipe.get("expert_capture_tile_rows")),
                "expert capture tile",
            ),
            (
                "--required-expert-fraction",
                str(recipe.get("required_expert_fraction")),
                "required expert fraction",
            ),
            ("--sampling-seed", str(recipe.get("sampling_seed")), "sampling seed"),
            (
                "--expert-coverage-policy",
                str(recipe.get("expert_coverage_policy", "")).replace("_", "-"),
                "expert coverage policy",
            ),
        )
        for flag, expected_value, label in command_bindings:
            if _command_value(two_pass, flag) != expected_value:
                raise ValueError(f"variant {variant.get('id')} {label} disagrees with its command")
        separator = two_pass.index("--") if "--" in two_pass else len(two_pass)
        if two_pass[separator + 1 :] != recipe.get("quant_args"):
            raise ValueError(f"variant {variant.get('id')} quant args disagree with its command")
        for expected_path in (
            variant["quantized_artifact"],
            variant["evaluation_output"],
            evaluation_dataset,
            reference_model,
        ):
            if expected_path not in evaluation:
                raise ValueError(f"variant {variant.get('id')} evaluator is not bound to {expected_path}")

    axis = plan.get("axis")
    if axis == "minimum_expert_activations":
        if len(targets) != 1:
            raise ValueError("minimum sweep must hold the capture target fixed")
        frozen = int(plan.get("selection_contract", {}).get("capture_target_held_fixed", 0))
        if targets != {frozen}:
            raise ValueError("minimum sweep capture target disagrees with its selection contract")
    elif axis == "expert_capture_target":
        if len(minimums) != 1:
            raise ValueError("capture sweep must hold the selected minimum fixed")
        frozen = int(plan.get("selection_contract", {}).get("selected_minimum", 0))
        if minimums != {frozen}:
            raise ValueError("capture sweep minimum disagrees with its selection contract")
    else:
        raise ValueError(f"unsupported expert sweep axis: {axis}")

    return {
        "schema": EXPERT_SWEEP_VERIFY_SCHEMA,
        "status": "verified_not_run",
        "plan_fingerprint": fingerprint,
        "engine_fingerprint": planned_engine,
        "dataset_sha256": observed_datasets,
        "variant_ids": ids,
    }


def _finite_metric(record: dict, field: str, *, positive: bool = False) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expert sweep result {record.get('id')!r} has invalid {field}")
    value = float(value)
    if (positive and value <= 0.0) or (not positive and value < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"expert sweep result {record.get('id')!r} {field} must be {qualifier}")
    return value


def _preserve_set(record: dict) -> tuple[tuple[int, int], ...]:
    values = record.get("preserve_high_precision")
    if not isinstance(values, list):
        raise ValueError(
            f"expert sweep result {record.get('id')!r} preserve_high_precision must be a list"
        )
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("preserve_high_precision entries must be objects")
        layer = value.get("layer")
        expert = value.get("expert")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
        ):
            raise ValueError(f"invalid preserve_high_precision entry: {value!r}")
        result.append((layer, expert))
    if len(set(result)) != len(result):
        raise ValueError(f"expert sweep result {record.get('id')!r} repeats a preserved expert")
    return tuple(sorted(result))


def _statistic_stability(
    record: dict,
    variant: dict,
    reference_variant: dict,
) -> tuple[float, str]:
    report = record.get("statistic_stability_report")
    if not isinstance(report, dict):
        raise ValueError(
            f"expert sweep result {record.get('id')!r} has no statistic_stability_report"
        )
    if report.get("schema") != "hipfire.calibration_expert_statistic_stability.v1":
        raise ValueError(
            f"expert sweep result {record.get('id')!r} has an unsupported stability report"
        )
    required_true = (
        "valid",
        "provenance_complete",
        "fallback_set_match",
        "capture_target_order_valid",
    )
    if any(report.get(field) is not True for field in required_true):
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability report is not valid"
        )
    if report.get("reference") != reference_variant["calibration_artifact"]:
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability reference does not match the plan"
        )
    if report.get("candidate") != variant["calibration_artifact"]:
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability candidate does not match the plan"
        )
    if report.get("reference_capture_target") != reference_variant["expert_capture_target"]:
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability reference target does not match the plan"
        )
    if report.get("candidate_capture_target") != variant["expert_capture_target"]:
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability candidate target does not match the plan"
        )
    tensor_count = report.get("compared_expert_tensors")
    value_count = report.get("compared_values")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count < 1
        or isinstance(value_count, bool)
        or not isinstance(value_count, int)
        or value_count < 1
        or report.get("non_finite_values") != 0
    ):
        raise ValueError(
            f"expert sweep result {record.get('id')!r} stability report has no finite expert statistics"
        )
    metric = _finite_metric(report, "relative_l2_error")
    return metric, _fingerprint(report)


def build_results(plan: dict, variant_records: list[dict]) -> dict:
    """Normalize complete measured rows into a fingerprinted sweep result set."""

    if not isinstance(plan, dict) or plan.get("schema") != EXPERT_SWEEP_PLAN_SCHEMA:
        raise ValueError("unsupported expert sweep plan schema")
    expected_variants = plan.get("variants")
    if not isinstance(expected_variants, list) or not expected_variants:
        raise ValueError("expert sweep plan has no variants")
    if not isinstance(variant_records, list):
        raise ValueError("expert sweep variant records must be a list")
    records_by_id = {}
    for record in variant_records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("expert sweep result has no variant id")
        if record["id"] in records_by_id:
            raise ValueError(f"duplicate expert sweep result {record['id']}")
        records_by_id[record["id"]] = record
    expected_ids = [variant["id"] for variant in expected_variants]
    if set(records_by_id) != set(expected_ids):
        raise ValueError(
            f"expert sweep results do not match plan variants: expected={expected_ids}, "
            f"observed={sorted(records_by_id)}"
        )

    normalized = []
    capture_axis = plan.get("axis") == "expert_capture_target"
    stability_reference = (
        max(expected_variants, key=lambda variant: variant["expert_capture_target"])
        if capture_axis
        else None
    )
    for variant in expected_variants:
        record = records_by_id[variant["id"]]
        artifact_size = record.get("artifact_size_bytes")
        launches = record.get("reduction_launches")
        if isinstance(artifact_size, bool) or not isinstance(artifact_size, int) or artifact_size < 1:
            raise ValueError(f"expert sweep result {record['id']!r} has invalid artifact_size_bytes")
        if isinstance(launches, bool) or not isinstance(launches, int) or launches < 0:
            raise ValueError(f"expert sweep result {record['id']!r} has invalid reduction_launches")
        preserved = _preserve_set(record)
        row = {
            "id": record["id"],
            "minimum_expert_activations": variant["minimum_expert_activations"],
            "expert_capture_target": variant["expert_capture_target"],
            "mean_kld": _finite_metric(record, "mean_kld"),
            "ppl": _finite_metric(record, "ppl", positive=True),
            "artifact_size_bytes": artifact_size,
            "calibration_seconds": _finite_metric(record, "calibration_seconds", positive=True),
            "reduction_launches": launches,
            "preserve_high_precision": [
                {"layer": layer, "expert": expert} for layer, expert in preserved
            ],
        }
        if capture_axis:
            stability, report_fingerprint = _statistic_stability(
                record,
                variant,
                stability_reference,
            )
            row["statistic_stability"] = stability
            row["statistic_stability_report_fingerprint"] = report_fingerprint
            row["statistic_stability_report"] = record["statistic_stability_report"]
        normalized.append(row)

    body = {
        "schema": EXPERT_SWEEP_RESULTS_SCHEMA,
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "axis": plan.get("axis"),
        "variants": normalized,
    }
    return {**body, "results_fingerprint": _fingerprint(body)}


def analyze_results(plan: dict, results: dict) -> dict:
    """Compare a complete one-axis sweep without inventing a promotion threshold."""

    verify_plan(plan)
    if not isinstance(results, dict) or results.get("schema") != EXPERT_SWEEP_RESULTS_SCHEMA:
        raise ValueError("unsupported expert sweep results schema")
    if results.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ValueError("expert sweep results plan fingerprint mismatch")
    body = {key: value for key, value in results.items() if key != "results_fingerprint"}
    if results.get("results_fingerprint") != _fingerprint(body):
        raise ValueError("expert sweep results fingerprint mismatch")
    rebuilt = build_results(plan, results.get("variants"))
    if rebuilt != results:
        raise ValueError("expert sweep results are not in canonical plan order")

    axis = plan["axis"]
    axis_field = (
        "minimum_expert_activations"
        if axis == "minimum_expert_activations"
        else "expert_capture_target"
    )
    rows = sorted(results["variants"], key=lambda row: row[axis_field])
    reference = rows[-1]
    reference_preserved = set(_preserve_set(reference))
    analyzed_rows = []
    for row in rows:
        analyzed = dict(row)
        analyzed["quality_vs_reference"] = {
            "mean_kld_delta": row["mean_kld"] - reference["mean_kld"],
            "ppl_ratio": row["ppl"] / reference["ppl"],
        }
        analyzed["fallback_expert_count"] = len(_preserve_set(row))
        analyzed_rows.append(analyzed)

    cohorts = []
    for lower, higher in zip(rows, rows[1:]):
        lower_preserved = set(_preserve_set(lower))
        higher_preserved = set(_preserve_set(higher))
        if axis == "minimum_expert_activations":
            if not lower_preserved.issubset(higher_preserved):
                raise ValueError(
                    f"preserve_high_precision is not monotonic from {lower['id']} to {higher['id']}"
                )
            cohort = sorted(higher_preserved - lower_preserved)
            kld_penalty = lower["mean_kld"] - higher["mean_kld"]
            cohorts.append(
                {
                    "lower_variant": lower["id"],
                    "higher_variant": higher["id"],
                    "newly_low_bit_experts": [
                        {"layer": layer, "expert": expert} for layer, expert in cohort
                    ],
                    "expert_count": len(cohort),
                    "mean_kld_penalty": kld_penalty,
                    "mean_kld_penalty_per_expert": (
                        kld_penalty / len(cohort) if cohort else None
                    ),
                    "ppl_ratio": lower["ppl"] / higher["ppl"],
                }
            )
        elif lower_preserved != higher_preserved:
            raise ValueError(
                f"capture sweep fallback set drifted from {lower['id']} to {higher['id']}"
            )

    return {
        "schema": EXPERT_SWEEP_ANALYSIS_SCHEMA,
        "status": "complete_selection_required",
        "plan_fingerprint": plan["plan_fingerprint"],
        "results_fingerprint": results["results_fingerprint"],
        "axis": axis,
        "reference_variant": reference["id"],
        "required_metrics_complete": True,
        "selection_contract": plan["selection_contract"],
        "variants": analyzed_rows,
        "low_traffic_cohorts": cohorts,
        "reference_preserved_expert_count": len(reference_preserved),
    }


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
        "model": {
            "path": str(model.resolve()),
            "identity": _safetensors_manifest_identity(model),
        },
        "reference_model": {
            "path": str(reference_model.resolve()),
            "identity": _reference_identity(reference_model),
        },
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
            "required_metrics": (
                [
                    "mean_kld",
                    "ppl",
                    "low_traffic_expert_sensitivity",
                    "artifact_size_bytes",
                    "calibration_seconds",
                    "reduction_launches",
                ]
                if canonical_axis == "minimum_expert_activations"
                else [
                    "mean_kld",
                    "ppl",
                    "statistic_stability",
                    "artifact_size_bytes",
                    "calibration_seconds",
                    "reduction_launches",
                ]
            ),
            "promotion_eligible_without_complete_metrics": False,
        },
        "variants": variants,
        "command_argv": list(command or []),
    }
    return {**body, "plan_fingerprint": _plan_fingerprint(body)}
