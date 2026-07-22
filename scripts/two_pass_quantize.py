#!/usr/bin/env python3
"""Two source-checkpoint passes: native streamed calibration, then quantize.

The first pass uses hipfire's family-neutral Rust engine to read each source
tensor once and emits a
unified `.calib.hfq` containing Hessian/imatrix/router/KLDREF data. The second
pass is the existing `hipfire-quantize` ingestion pass. Candidate evaluation
may load the resulting HFQ later, but does not reread the BF16 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUANT_FORMAT = "oq4.25++"
DEFAULT_LAYER_PREFETCH_BYTES = 16 * 1024**3
DEFAULT_MIN_EXPERT_ACTIVATIONS = 2048
DEFAULT_EXPERT_CAPTURE_TARGET = 4096
DEFAULT_EXPERT_CAPTURE_TILE_ROWS = 256
DEFAULT_REQUIRED_EXPERT_FRACTION = 1.0
DEFAULT_SAMPLING_SEED = 1
DEFAULT_EXPERT_COVERAGE_POLICY = "preserve-undercovered"
DEFAULT_CALIBRATION_SEGMENT_RELEASE_SECONDS = 5
PASS_TWO_FIXED_SAFETY_BYTES = 64 * 1024**3
PASS_TWO_RELATIVE_SAFETY = 0.10
PASS_TWO_CONTAINER_OVERHEAD_BYTES = 16 * 1024**2
PASS_TWO_TENSOR_ALIGNMENT_BYTES = 4096


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _resolve_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        candidate = path / "snapshots" / main_ref.read_text().strip()
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (candidate for candidate in snapshots.iterdir() if (candidate / "config.json").is_file()),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        if candidates:
            return candidates[-1].resolve()
    raise FileNotFoundError(f"no Hugging Face snapshot/config.json under {path}")


def _safetensors_index(model: Path) -> list[dict]:
    snapshot = _resolve_snapshot(model)
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors files under {snapshot}")
    tensors = []
    for shard in shards:
        with shard.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise RuntimeError(f"truncated safetensors header prefix: {shard}")
            header_len = struct.unpack("<Q", prefix)[0]
            if header_len > 1024**3:
                raise RuntimeError(f"unreasonable safetensors header size {header_len}: {shard}")
            encoded = source.read(header_len)
        if len(encoded) != header_len:
            raise RuntimeError(f"truncated safetensors header: {shard}")
        header = json.loads(encoded)
        for name, value in header.items():
            if name == "__metadata__":
                continue
            shape = value.get("shape")
            offsets = value.get("data_offsets")
            dtype = value.get("dtype")
            if (
                not isinstance(shape, list)
                or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
                or offsets[1] < offsets[0]
                or not isinstance(dtype, str)
            ):
                raise RuntimeError(f"invalid safetensors index entry {name!r} in {shard}")
            numel = math.prod(shape)
            byte_len = offsets[1] - offsets[0]
            dtype_bytes = _DTYPE_BYTES.get(dtype)
            if dtype_bytes is not None and byte_len != numel * dtype_bytes:
                raise RuntimeError(
                    f"safetensors byte length mismatch for {name}: {byte_len} != {numel}*{dtype_bytes}"
                )
            tensors.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "numel": numel,
                    "source_bytes": byte_len,
                }
            )
    return tensors


def _routed_expert_identity(name: str) -> tuple[int, int | None, str] | None:
    parts = name.split(".")
    try:
        layer_at = parts.index("layers")
        layer = int(parts[layer_at + 1])
        expert_at = parts.index("experts", layer_at + 2)
    except (ValueError, IndexError):
        return None
    suffix = parts[expert_at + 1 :]
    if not suffix:
        return None
    expert = None
    if suffix[0].isdigit():
        expert = int(suffix.pop(0))
    if not suffix:
        return None
    projection = suffix[0]
    if projection in {"gate_up_proj", "gate_proj", "up_proj", "w1", "w3"}:
        role = "gate_up"
    elif projection in {"down_proj", "w2"}:
        role = "down"
    else:
        return None
    return layer, expert, role


def _oq_block_bytes(quant_format: str) -> tuple[float, int]:
    base = quant_format.removesuffix("++").removesuffix("+")
    if base == "oq4":
        return 4.0625, 130
    if base == "oq8":
        return 8.0625, 258
    match = re.fullmatch(r"oq(\d+\.\d+)", base)
    if match is None:
        raise RuntimeError(
            f"pass-two storage admission does not know the on-disk block size for {quant_format!r}"
        )
    requested = float(match.group(1))
    overlays = round((requested - 4.0625) * 16)
    if not 1 <= overlays <= 62 or abs((4.0625 + overlays / 16) - requested) > 1e-6:
        raise RuntimeError(f"invalid mixed Opus storage width {quant_format!r}")
    return requested, 130 + 2 * overlays


def _source_precision_output_bytes(tensor: dict) -> int:
    if tensor["dtype"] in {"BF16", "F16", "F32"}:
        return tensor["numel"] * 2
    return tensor["source_bytes"]


def _quantized_tensor_bytes(numel: int, block_bytes: int, *, q8: bool = False) -> int:
    effective_block = 34 if q8 else block_bytes
    group = 32 if q8 else 256
    return math.ceil(numel / group) * effective_block


def _nearest_existing_path(path: Path) -> Path:
    candidate = path if path.is_dir() else path.parent
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing parent for output path {path}")
        candidate = parent
    return candidate.resolve()


def pass_two_storage_preflight(
    *,
    model: Path,
    output: Path,
    quant_format: str,
    calibration: dict,
    available_bytes: int | None = None,
) -> dict:
    """Estimate pass-two disk demand without reading tensor payloads.

    The source scan reads safetensors headers only. Routed-expert tensors are
    recognized structurally (layer/expert/projection components), including
    grouped `[experts,...]` and already-split layouts. Experts declared by the
    audited calibration artifact are costed at F16/BF16 rather than at the
    requested OQ width.
    """

    storage_bits, block_bytes = _oq_block_bytes(quant_format)
    tensors = _safetensors_index(model)
    preserved_values = calibration.get("metadata", {}).get("preserve_high_precision", [])
    if not isinstance(preserved_values, list):
        raise RuntimeError("calibration preserve_high_precision is not a list")
    preserved = set()
    for value in preserved_values:
        if not isinstance(value, dict) or not isinstance(value.get("layer"), int) or not isinstance(value.get("expert"), int):
            raise RuntimeError(f"invalid calibration preserve_high_precision entry: {value!r}")
        preserved.add((value["layer"], value["expert"]))

    payload_bytes = 0
    nominal_payload_bytes = 0
    preserve_output_bytes = 0
    preserve_nominal_bytes = 0
    matched_roles: dict[tuple[int, int], set[str]] = {key: set() for key in preserved}
    output_tensors = 0
    source_payload_bytes = 0
    source_parameters = 0

    for tensor in tensors:
        source_payload_bytes += tensor["source_bytes"]
        source_parameters += tensor["numel"]
        identity = _routed_expert_identity(tensor["name"])
        if identity is not None:
            layer, explicit_expert, role = identity
            if explicit_expert is None:
                if len(tensor["shape"]) < 2 or tensor["shape"][0] < 1:
                    raise RuntimeError(f"grouped routed-expert tensor has no expert dimension: {tensor['name']}")
                expert_count = tensor["shape"][0]
                expert_numel = math.prod(tensor["shape"][1:])
                experts = range(expert_count)
            else:
                expert_numel = tensor["numel"]
                experts = (explicit_expert,)
            for expert in experts:
                nominal = _quantized_tensor_bytes(expert_numel, block_bytes)
                nominal_payload_bytes += nominal
                key = (layer, expert)
                if key in preserved:
                    full = expert_numel * 2
                    payload_bytes += full
                    preserve_output_bytes += full
                    preserve_nominal_bytes += nominal
                    matched_roles[key].add(role)
                else:
                    payload_bytes += nominal
                output_tensors += 1
            continue

        is_weight = len(tensor["shape"]) >= 2 and tensor["name"].endswith(".weight")
        if is_weight:
            # K-map, embedding, router, architecture-critical and divisibility
            # fallbacks can widen a nominal OQ matrix. Q8F16 is the widest
            # compressed result those branches emit. Cost every non-expert
            # matrix at that ceiling rather than mirroring evolving K-map
            # policy in this orchestration wrapper.
            encoded = _quantized_tensor_bytes(tensor["numel"], block_bytes, q8=True)
        else:
            encoded = _source_precision_output_bytes(tensor)
        payload_bytes += encoded
        nominal_payload_bytes += encoded
        output_tensors += 1

    if preserved:
        missing = sorted(key for key, roles in matched_roles.items() if not roles)
        incomplete = sorted((key, sorted(roles)) for key, roles in matched_roles.items() if roles and roles != {"gate_up", "down"})
        if missing:
            raise RuntimeError(
                f"calibration preserves {len(missing)} experts with no routed-expert source tensors: {missing[:8]}"
            )
        if incomplete:
            raise RuntimeError(
                f"calibration preserved experts lack both routed roles in source index: {incomplete[:8]}"
            )

    alignment_bytes = output_tensors * PASS_TWO_TENSOR_ALIGNMENT_BYTES
    artifact_estimate = payload_bytes + alignment_bytes + PASS_TWO_CONTAINER_OVERHEAD_BYTES
    safety_margin = max(PASS_TWO_FIXED_SAFETY_BYTES, math.ceil(artifact_estimate * PASS_TWO_RELATIVE_SAFETY))
    required_free = artifact_estimate + safety_margin
    probe_path = _nearest_existing_path(output)
    if available_bytes is None:
        stats = os.statvfs(probe_path)
        available_bytes = stats.f_bavail * stats.f_frsize
    sufficient = available_bytes >= required_free
    return {
        "schema": "hipfire.pass_two_storage_preflight.v1",
        "index_only": True,
        "payload_values_read": False,
        "format": quant_format,
        "storage_bits_per_weight": storage_bits,
        "source": {
            "snapshot": str(_resolve_snapshot(model)),
            "tensors": len(tensors),
            "parameters": source_parameters,
            "payload_bytes": source_payload_bytes,
        },
        "preserve_high_precision": {
            "requested_experts": len(preserved),
            "matched_experts": sum(roles == {"gate_up", "down"} for roles in matched_roles.values()),
            "output_bytes": preserve_output_bytes,
            "nominal_quantized_bytes": preserve_nominal_bytes,
            "delta_bytes": preserve_output_bytes - preserve_nominal_bytes,
        },
        "estimate": {
            "nominal_payload_bytes": nominal_payload_bytes,
            "mixed_payload_bytes": payload_bytes,
            "nonexpert_weight_ceiling": "q8f16",
            "tensor_alignment_bytes": alignment_bytes,
            "fixed_container_overhead_bytes": PASS_TWO_CONTAINER_OVERHEAD_BYTES,
            "completed_artifact_estimate_bytes": artifact_estimate,
            "safety_margin_bytes": safety_margin,
            "required_free_bytes": required_free,
        },
        "filesystem": {
            "probe_path": str(probe_path),
            "available_bytes": available_bytes,
            "required_free_bytes": required_free,
            "sufficient": sufficient,
        },
    }


def require_pass_two_storage(preflight: dict) -> None:
    filesystem = preflight["filesystem"]
    if filesystem["sufficient"] is not True:
        raise RuntimeError(
            "insufficient output storage for pass two: "
            f"{filesystem['available_bytes']} bytes available at {filesystem['probe_path']}, "
            f"{filesystem['required_free_bytes']} bytes required by the preserved-expert-aware estimate"
        )


def build_commands(
    *,
    coexistence: str,
    quantizer: str,
    model: Path,
    calib: Path,
    output: Path,
    quant_format: str,
    corpus: str,
    n_sequences: int,
    ctx_len: int,
    batch_size: int,
    time_tile: int,
    max_rows: int,
    layer_prefetch_bytes: int,
    kldref_topk: int,
    min_expert_activations: int,
    expert_capture_target: int,
    expert_capture_tile_rows: int,
    required_expert_fraction: float,
    sampling_seed: int,
    expert_coverage_policy: str,
    quant_args: list[str],
) -> tuple[list[str], list[str]]:
    collect_cmd = [
        coexistence,
        "calibrate",
        "--model",
        str(model),
        "--output",
        str(calib),
        "--corpus",
        corpus,
        "--sequences",
        str(n_sequences),
        "--context",
        str(ctx_len),
        "--sequence-batch",
        str(batch_size),
        "--time-tile",
        str(time_tile),
        "--max-rows",
        str(max_rows),
        "--layer-prefetch-bytes",
        str(layer_prefetch_bytes),
        "--kldref",
        "--kldref-topk",
        str(kldref_topk),
        "--min-expert-activations",
        str(min_expert_activations),
        "--expert-capture-target",
        str(expert_capture_target),
        "--expert-capture-tile-rows",
        str(expert_capture_tile_rows),
        "--required-expert-fraction",
        str(required_expert_fraction),
        "--sampling-seed",
        str(sampling_seed),
        "--expert-coverage-policy",
        expert_coverage_policy,
        "--resume",
    ]
    quant_cmd = [
        quantizer,
        "--input",
        str(model),
        "--output",
        str(output),
        "--format",
        quant_format,
        "--hessian",
        str(calib),
        *quant_args,
    ]
    return collect_cmd, quant_cmd


def _print_command(label: str, command: list[str]) -> None:
    print(f"{label}: {shlex.join(command)}", flush=True)


def scope_gpu_commands(hipfire: str, collect_cmd: list[str], quant_cmd: list[str]) -> tuple[list[str], list[str]]:
    # `hipfire-coexistence calibrate` owns a native FlockGuard. Wrapping it in
    # another process-level lock would deadlock against its own child. The
    # quantizer has no internal guard, so the workflow owns that lock exactly once.
    return collect_cmd, [hipfire, "lock", "run", "two-pass-quantization", "--", *quant_cmd]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def recipe_manifest(
    *,
    model: Path,
    calib: Path,
    output: Path,
    quant_format: str,
    corpus: Path,
    n_sequences: int,
    ctx_len: int,
    batch_size: int,
    time_tile: int,
    max_rows: int,
    layer_prefetch_bytes: int,
    kldref_topk: int,
    min_expert_activations: int,
    expert_capture_target: int,
    expert_capture_tile_rows: int,
    required_expert_fraction: float,
    sampling_seed: int,
    expert_coverage_policy: str,
    quant_args: list[str],
) -> dict:
    recipe = {
        "model": str(model.resolve()),
        "calibration_artifact": str(calib.resolve()),
        "quantized_artifact": str(output.resolve()),
        "quant_format": quant_format,
        "corpus": str(corpus.resolve()),
        "corpus_sha256": _sha256_file(corpus),
        "sequences": n_sequences,
        "context": ctx_len,
        "sequence_batch": batch_size,
        "time_tile": time_tile,
        "max_rows": max_rows,
        "layer_prefetch_bytes": layer_prefetch_bytes,
        "kldref_topk": kldref_topk,
        "min_expert_activations": min_expert_activations,
        "expert_capture_target": expert_capture_target,
        "expert_capture_tile_rows": expert_capture_tile_rows,
        "required_expert_fraction": required_expert_fraction,
        "sampling_seed": sampling_seed,
        "expert_coverage_policy": expert_coverage_policy.replace("-", "_"),
        "quant_args": quant_args,
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return {**recipe, "recipe_fingerprint": f"sha256:{hashlib.sha256(encoded).hexdigest()}"}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _get(value: dict | None, *keys: str):
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _merge_calibration_execution(previous: dict | None, current: dict) -> dict:
    if not isinstance(previous, dict):
        return current
    identity = ("mode", "process_segment_layers", "release_seconds")
    if any(previous.get(key) != current.get(key) for key in identity):
        return current
    merged = {**previous, **current}
    segments = []
    seen = set()
    for segment in [*previous.get("segments", []), *current.get("segments", [])]:
        if not isinstance(segment, dict):
            continue
        key = (
            segment.get("started_after_layer"),
            segment.get("pause_after_layer"),
            segment.get("completed_layers"),
            segment.get("artifact_complete"),
        )
        if key not in seen:
            segments.append(segment)
            seen.add(key)
    if segments:
        merged["segments"] = segments
    return merged


def update_manifest(
    path: Path,
    *,
    recipe: dict,
    phase: str,
    calibration: dict | None = None,
    calibration_audit: dict | None = None,
    storage_preflight: dict | None = None,
    quantized: dict | None = None,
    calibration_execution: dict | None = None,
    phase_timings: dict | None = None,
    failure: dict | None = None,
) -> dict:
    previous = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    manifest = {
        "schema": 1,
        "created_at": previous.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "status": phase,
        **recipe,
    }
    if calibration is not None:
        manifest["calibration"] = calibration
        ledger = _get(calibration, "metadata", "read_ledger")
        if ledger is not None:
            manifest["source_reads"] = ledger
    elif "calibration" in previous:
        manifest["calibration"] = previous["calibration"]
        if "source_reads" in previous:
            manifest["source_reads"] = previous["source_reads"]
    if calibration_audit is not None:
        manifest["calibration_audit"] = calibration_audit
    elif calibration is None and "calibration_audit" in previous:
        manifest["calibration_audit"] = previous["calibration_audit"]
    if storage_preflight is not None:
        manifest["pass_two_storage_preflight"] = storage_preflight
    elif "pass_two_storage_preflight" in previous:
        manifest["pass_two_storage_preflight"] = previous["pass_two_storage_preflight"]
    if quantized is not None:
        manifest["quantized"] = quantized
    elif "quantized" in previous:
        manifest["quantized"] = previous["quantized"]
    if calibration_execution is not None:
        manifest["calibration_execution"] = _merge_calibration_execution(
            previous.get("calibration_execution"), calibration_execution
        )
    elif "calibration_execution" in previous:
        manifest["calibration_execution"] = previous["calibration_execution"]
    if phase_timings is not None:
        manifest["phase_timings"] = {
            **previous.get("phase_timings", {}),
            **phase_timings,
        }
    elif "phase_timings" in previous:
        manifest["phase_timings"] = previous["phase_timings"]
    if failure is not None:
        manifest["failure"] = failure

    calibration_value = calibration or manifest.get("calibration")
    quantized_value = quantized or manifest.get("quantized")
    fingerprints = {
        "calibration_artifact": _get(calibration_value, "artifact_fingerprint"),
        "calibration_engine_build": _get(calibration_value, "metadata", "engine_build"),
        "calibration_run": _get(calibration_value, "metadata", "run_fingerprint"),
        "source": _get(calibration_value, "metadata", "source_manifest", "fingerprint"),
        "samples": _get(calibration_value, "metadata", "job", "samples", "fingerprint"),
        "quantized_artifact": _get(quantized_value, "artifact_fingerprint"),
        "quantized_payload": _get(quantized_value, "metadata", "quantization_hash", "value"),
    }
    manifest["fingerprints"] = {key: value for key, value in fingerprints.items() if value is not None}
    _atomic_json(path, manifest)
    return manifest


def accumulate_attempt_timing(
    manifest: dict,
    *,
    phase_name: str,
    elapsed_seconds: float,
) -> dict:
    """Return cumulative and last-attempt timing fields for one workflow phase."""

    timings = manifest.get("phase_timings", {})
    prior_seconds = timings.get(f"{phase_name}_seconds", 0.0)
    elapsed_seconds = round(float(elapsed_seconds), 6)
    return {
        f"{phase_name}_seconds": round(float(prior_seconds) + elapsed_seconds, 6),
        f"last_{phase_name}_attempt_seconds": elapsed_seconds,
    }


def inspect_artifact(coexistence: str, path: Path) -> dict:
    result = subprocess.run(
        [coexistence, "artifact", "inspect", "--input", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def calibration_audit_command(coexistence: str, path: Path) -> list[str]:
    return [coexistence, "artifact", "audit-calibration", "--input", str(path)]


def audit_calibration_artifact(coexistence: str, path: Path) -> dict:
    result = subprocess.run(
        calibration_audit_command(coexistence, path),
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def calibration_validation_command(collect_cmd: list[str]) -> list[str]:
    return [*collect_cmd, "--dry-run"]


def inspect_calibration_plan(collect_cmd: list[str]) -> dict:
    result = subprocess.run(
        calibration_validation_command(collect_cmd),
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def calibration_boundary_checkpoint(calib: Path) -> Path:
    return calib.with_name(f".{calib.name}.boundary") / "calibration-boundary.json"


def _read_calibration_boundary(calib: Path) -> dict | None:
    path = calibration_boundary_checkpoint(calib)
    if not path.is_file():
        return None
    try:
        checkpoint = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid calibration boundary checkpoint {path}: {error}") from error
    for field in ("completed_layers", "total_layers"):
        if not isinstance(checkpoint.get(field), int) or checkpoint[field] < 0:
            raise RuntimeError(f"invalid calibration boundary {field}: {checkpoint.get(field)!r}")
    if not isinstance(checkpoint.get("artifact_complete"), bool):
        raise RuntimeError("invalid calibration boundary artifact_complete")
    return checkpoint


def run_calibration_pass(
    collect_exec: list[str],
    *,
    calib: Path,
    total_layers: int,
    segment_layers: int,
    runner=subprocess.run,
    release_seconds: int = DEFAULT_CALIBRATION_SEGMENT_RELEASE_SECONDS,
    progress=None,
) -> dict:
    """Run the native teacher once or in resumable process-sized segments.

    Process segmentation is deliberately outside the semantic recipe. The
    native boundary checkpoint remains authoritative for progress and source
    reads, while process exit releases source mappings before the next segment.
    """

    if total_layers < 1:
        raise RuntimeError(f"invalid native calibration layer count {total_layers}")
    if segment_layers < 0:
        raise RuntimeError("calibration segment layers must be nonnegative")
    if release_seconds < 0:
        raise RuntimeError("calibration segment release seconds must be nonnegative")
    if "--pause-after-layers" in collect_exec:
        raise RuntimeError("calibration command already contains --pause-after-layers")

    execution = {
        "mode": "segmented" if segment_layers else "single_process",
        "process_segment_layers": segment_layers,
        "release_seconds": release_seconds if segment_layers else 0,
        "segments": [],
    }
    if segment_layers == 0:
        runner(collect_exec, check=True)
        checkpoint = _read_calibration_boundary(calib)
        if checkpoint is not None:
            execution.update(
                completed_layers=checkpoint["completed_layers"],
                total_layers=checkpoint["total_layers"],
                artifact_complete=checkpoint["artifact_complete"],
            )
        return execution

    while True:
        before = _read_calibration_boundary(calib)
        completed = before["completed_layers"] if before is not None else 0
        checkpoint_total = before["total_layers"] if before is not None else total_layers
        if checkpoint_total != total_layers:
            raise RuntimeError(
                f"calibration boundary total layer mismatch: checkpoint={checkpoint_total}, plan={total_layers}"
            )
        if before is not None and before["artifact_complete"]:
            execution.update(
                completed_layers=completed,
                total_layers=checkpoint_total,
                artifact_complete=True,
            )
            return execution
        if completed > total_layers:
            raise RuntimeError(
                f"calibration boundary completed layer count {completed} exceeds plan {total_layers}"
            )

        pause_after = min(completed + segment_layers, total_layers)
        final_process = pause_after == total_layers
        command = list(collect_exec)
        if not final_process:
            command.extend(["--pause-after-layers", str(pause_after)])
        runner(command, check=True)

        after = _read_calibration_boundary(calib)
        if after is None:
            raise RuntimeError(
                f"calibration process returned success without boundary checkpoint {calibration_boundary_checkpoint(calib)}"
            )
        if after["total_layers"] != total_layers:
            raise RuntimeError(
                f"calibration boundary total layer mismatch: checkpoint={after['total_layers']}, plan={total_layers}"
            )
        if after["completed_layers"] <= completed:
            raise RuntimeError(
                f"calibration process did not advance durable progress beyond layer {completed}"
            )
        if final_process:
            if after["completed_layers"] != total_layers or not after["artifact_complete"]:
                raise RuntimeError(
                    "final calibration process returned success without a complete artifact"
                )
        elif after["completed_layers"] != pause_after or after["artifact_complete"]:
            raise RuntimeError(
                f"segmented calibration stopped at invalid boundary {after['completed_layers']} (expected {pause_after})"
            )
        execution["segments"].append(
            {
                "started_after_layer": completed,
                "pause_after_layer": None if final_process else pause_after,
                "completed_layers": after["completed_layers"],
                "artifact_complete": after["artifact_complete"],
            }
        )
        if after["artifact_complete"]:
            execution.update(
                completed_layers=after["completed_layers"],
                total_layers=after["total_layers"],
                artifact_complete=True,
            )
            if progress is not None:
                progress(execution)
            return execution
        if progress is not None:
            progress(execution)
        if release_seconds:
            time.sleep(release_seconds)


def _phase_failure(phase_name: str, error: BaseException) -> tuple[str, dict]:
    failure = {
        "recorded_at": _utc_now(),
        "kind": "exception",
        "message": str(error),
    }
    signal_number = None
    if isinstance(error, KeyboardInterrupt):
        signal_number = 2
    elif isinstance(error, subprocess.CalledProcessError):
        returncode = error.returncode
        failure["returncode"] = returncode
        if returncode < 0:
            signal_number = -returncode
        elif 129 <= returncode <= 192:
            signal_number = returncode - 128
        else:
            failure["kind"] = "process_error"
    if signal_number is not None:
        failure["kind"] = "signal"
        failure["signal"] = signal_number
        return f"{phase_name}_interrupted", failure
    return f"{phase_name}_failed", failure


def _quantization_failure(error: BaseException) -> tuple[str, dict]:
    return _phase_failure("quantization", error)


def run_calibration_attempt(
    collect_exec: list[str],
    *,
    calib: Path,
    total_layers: int,
    segment_layers: int,
    runner=subprocess.run,
    release_seconds: int = DEFAULT_CALIBRATION_SEGMENT_RELEASE_SECONDS,
    progress=None,
    on_failure=None,
) -> tuple[dict, float]:
    """Run pass one and durably report its wall time before propagating failure."""

    started = time.monotonic()
    try:
        execution = run_calibration_pass(
            collect_exec,
            calib=calib,
            total_layers=total_layers,
            segment_layers=segment_layers,
            runner=runner,
            release_seconds=release_seconds,
            progress=progress,
        )
    except BaseException as error:
        elapsed = round(time.monotonic() - started, 6)
        phase, failure = _phase_failure("calibration", error)
        if on_failure is not None:
            on_failure(phase, elapsed, failure)
        raise
    return execution, round(time.monotonic() - started, 6)


def run_quantization_pass(
    command: list[str],
    *,
    runner=subprocess.run,
    on_failure=None,
) -> float:
    """Run pass two and durably report interruption before propagating it."""

    started = time.monotonic()
    try:
        runner(command, check=True)
    except BaseException as error:
        elapsed = round(time.monotonic() - started, 6)
        phase, failure = _quantization_failure(error)
        if on_failure is not None:
            on_failure(phase, elapsed, failure)
        raise
    return round(time.monotonic() - started, 6)


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"reused calibration {label} mismatch: artifact={actual!r}, requested={expected!r}")


def validate_reusable_calibration(inspection: dict, expected: dict) -> None:
    """Bind --skip-calib to the native producer's exact semantic recipe."""

    metadata = inspection.get("metadata", {})
    job = metadata.get("job", {})
    options = job.get("options", {})
    samples = job.get("samples", {})
    source_manifest = metadata.get("source_manifest", {})
    expected_model = expected.get("model", {})
    expected_source = expected.get("source_plan", {})
    expected_corpus = expected.get("corpus", {})
    expected_geometry = expected.get("microbatch", {})
    expected_expert = expected.get("expert_capture", {})
    expected_kldref = expected.get("kldref", {})

    _require_equal(
        "run fingerprint",
        metadata.get("run_fingerprint"),
        expected.get("run_fingerprint"),
    )
    _require_equal("family", metadata.get("family"), expected_model.get("family"))
    _require_equal(
        "adapter_version",
        metadata.get("adapter_version"),
        expected_model.get("adapter_version"),
    )
    _require_equal("arch_id", metadata.get("arch_id"), expected_model.get("arch_id"))
    _require_equal(
        "source fingerprint",
        source_manifest.get("fingerprint"),
        expected_source.get("source_fingerprint"),
    )
    _require_equal(
        "source job fingerprint",
        job.get("source_fingerprint"),
        expected_source.get("source_fingerprint"),
    )
    _require_equal("source shards", source_manifest.get("shards"), expected_source.get("shards"))
    _require_equal(
        "tokenizer fingerprint",
        job.get("tokenizer_fingerprint"),
        expected_source.get("tokenizer_fingerprint"),
    )
    _require_equal(
        "corpus fingerprint",
        job.get("corpus_fingerprint"),
        expected_corpus.get("corpus_fingerprint"),
    )
    _require_equal(
        "sample fingerprint",
        samples.get("fingerprint"),
        expected_corpus.get("sample_fingerprint"),
    )
    sample_rows = sum(len(sample.get("tokens", [])) for sample in samples.get("samples", []))
    _require_equal("sample count", len(samples.get("samples", [])), expected_corpus.get("sequences"))
    _require_equal("sample context", samples.get("context_len"), expected_corpus.get("context"))
    _require_equal("sample rows", sample_rows, expected_corpus.get("rows"))

    geometry = metadata.get("microbatch_geometry", {})
    _require_equal(
        "geometry sequence_batch",
        geometry.get("sequence_batch"),
        expected_geometry.get("sequence_batch"),
    )
    _require_equal("geometry time_tile", geometry.get("time_tile"), expected_geometry.get("time_tile"))
    _require_equal("geometry row_budget", geometry.get("row_budget"), expected_geometry.get("max_rows"))
    _require_equal("job sequence_batch", options.get("sequence_batch"), expected_geometry.get("sequence_batch"))
    _require_equal("job time_tile", options.get("time_tile"), expected_geometry.get("time_tile"))
    _require_equal("job max_rows", options.get("max_rows"), expected_geometry.get("max_rows"))
    _require_equal("boundary precision", options.get("boundary_precision"), "f32")

    quota = options.get("expert_quota", {})
    _require_equal("minimum_rows", quota.get("min_rows"), expected_expert.get("minimum_rows"))
    _require_equal("target_rows", quota.get("target_rows"), expected_expert.get("target_rows"))
    _require_equal("tile_rows", quota.get("tile_rows"), expected_expert.get("tile_rows"))
    _require_equal("sampling", quota.get("sampling"), expected_expert.get("sampling"))
    _require_equal(
        "required_fraction",
        options.get("required_expert_fraction"),
        expected_expert.get("required_fraction"),
    )
    _require_equal(
        "coverage_policy",
        options.get("expert_coverage_policy"),
        expected_expert.get("coverage_policy"),
    )
    _require_equal("KLDREF enabled", options.get("kldref"), expected_kldref.get("enabled"))
    _require_equal("KLDREF top_k", options.get("kldref_top_k"), expected_kldref.get("top_k"))


def validate_calibration_inspection(inspection: dict) -> None:
    metadata = inspection.get("metadata", {})
    if metadata.get("artifact_kind") != "calibration":
        raise RuntimeError("native pass produced an artifact without artifact_kind=calibration")
    ledger = metadata.get("read_ledger")
    if not isinstance(ledger, dict):
        raise RuntimeError("native calibration artifact has no read_ledger")
    if ledger.get("missing_logical"):
        raise RuntimeError(f"native calibration read ledger has missing tensors: {ledger['missing_logical']}")
    if ledger.get("duplicate_logical"):
        raise RuntimeError(f"native calibration read ledger has duplicate reads: {ledger['duplicate_logical']}")


def validate_calibration_audit(audit: dict, inspection: dict) -> None:
    if audit.get("schema") != "hipfire.calibration_audit.v1" or audit.get("valid") is not True:
        raise RuntimeError("native calibration artifact did not pass the structural audit")
    if audit.get("errors"):
        raise RuntimeError(f"native calibration structural audit reports errors: {audit['errors']}")
    if audit.get("artifact_fingerprint") != inspection.get("artifact_fingerprint"):
        raise RuntimeError("native calibration structural audit fingerprint differs from inspection")
    if audit.get("index_only") is not True or audit.get("payload_values_checked") is not False:
        raise RuntimeError("native calibration structural audit has an unknown evidence scope")


def validate_quantized_inspection(inspection: dict) -> None:
    metadata = inspection.get("metadata", {})
    if not _get(metadata, "quantization_hash", "value"):
        raise RuntimeError("quantized artifact has no embedded quantization_hash")
    if not isinstance(metadata.get("calibration"), dict):
        raise RuntimeError("quantized artifact has no embedded calibration provenance")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streamed calib+KLDREF, then quantize from safetensors.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path, help="Output ending in .calib.hfq.")
    parser.add_argument("--output", required=True, type=Path, help="Canonical quantized .hfq output.")
    parser.add_argument("--manifest", type=Path, help="Atomic two-pass provenance manifest output.")
    parser.add_argument(
        "--format",
        dest="quant_format",
        default=DEFAULT_QUANT_FORMAT,
        help=f"Target weight format (default: {DEFAULT_QUANT_FORMAT}).",
    )
    parser.add_argument("--corpus", default="wikitext")
    parser.add_argument("--n-sequences", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-tile", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument(
        "--layer-prefetch-bytes",
        type=int,
        default=DEFAULT_LAYER_PREFETCH_BYTES,
        help=f"Bounded next-layer source lookahead (default: {DEFAULT_LAYER_PREFETCH_BYTES}; 0 disables).",
    )
    parser.add_argument(
        "--calibration-segment-layers",
        type=int,
        default=0,
        help=(
            "Restart the native calibrator after this many additional durable layers "
            "to release source mappings (default: 0, uninterrupted)."
        ),
    )
    parser.add_argument("--kldref-topk", type=int, default=64)
    parser.add_argument(
        "--min-expert-activations",
        type=int,
        default=DEFAULT_MIN_EXPERT_ACTIVATIONS,
    )
    parser.add_argument(
        "--expert-capture-target",
        type=int,
        default=DEFAULT_EXPERT_CAPTURE_TARGET,
    )
    parser.add_argument(
        "--expert-capture-tile-rows",
        type=int,
        default=DEFAULT_EXPERT_CAPTURE_TILE_ROWS,
    )
    parser.add_argument(
        "--required-expert-fraction",
        type=float,
        default=DEFAULT_REQUIRED_EXPERT_FRACTION,
    )
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument(
        "--expert-coverage-policy",
        choices=("strict", "preserve-undercovered"),
        default=DEFAULT_EXPERT_COVERAGE_POLICY,
    )
    parser.add_argument(
        "--coexistence",
        default="target/release/hipfire-coexistence",
        help="Native compatibility/calibration tool used for pass 1.",
    )
    parser.add_argument("--quantizer", default="target/release/hipfire-quantize")
    parser.add_argument("--hipfire", default="target/release/hipfire", help="CLI used for the scoped GPU lock.")
    parser.add_argument("--skip-calib", action="store_true", help="Reuse an existing calibration artifact.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "quant_args",
        nargs=argparse.REMAINDER,
        help="Additional hipfire-quantize flags after `--`, e.g. `-- --awq --ldlq`.",
    )
    args = parser.parse_args()
    quant_args = args.quant_args[1:] if args.quant_args[:1] == ["--"] else args.quant_args
    if not args.calib.name.endswith(".calib.hfq"):
        parser.error("--calib must end in .calib.hfq")
    if not args.output.name.endswith(".hfq"):
        parser.error("--output must end in .hfq")
    if min(
        args.n_sequences,
        args.ctx_len,
        args.batch_size,
        args.time_tile,
        args.max_rows,
        args.kldref_topk,
        args.min_expert_activations,
        args.expert_capture_target,
        args.expert_capture_tile_rows,
    ) < 1:
        parser.error("sequence, context, geometry, and top-k values must be positive")
    if args.batch_size * args.time_tile > args.max_rows:
        parser.error("--batch-size * --time-tile must not exceed --max-rows")
    if args.layer_prefetch_bytes < 0:
        parser.error("--layer-prefetch-bytes must be nonnegative")
    if args.calibration_segment_layers < 0:
        parser.error("--calibration-segment-layers must be nonnegative")
    if args.expert_capture_target < args.min_expert_activations:
        parser.error("--expert-capture-target must be at least --min-expert-activations")
    if not 0.0 < args.required_expert_fraction <= 1.0:
        parser.error("--required-expert-fraction must be in (0, 1]")
    if args.sampling_seed < 0:
        parser.error("--sampling-seed must be nonnegative")
    manifest_path = args.manifest or args.output.with_suffix(".two-pass.json")
    corpus = Path(args.corpus)
    recipe = recipe_manifest(
        model=args.model,
        calib=args.calib,
        output=args.output,
        quant_format=args.quant_format,
        corpus=corpus,
        n_sequences=args.n_sequences,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        time_tile=args.time_tile,
        max_rows=args.max_rows,
        layer_prefetch_bytes=args.layer_prefetch_bytes,
        kldref_topk=args.kldref_topk,
        min_expert_activations=args.min_expert_activations,
        expert_capture_target=args.expert_capture_target,
        expert_capture_tile_rows=args.expert_capture_tile_rows,
        required_expert_fraction=args.required_expert_fraction,
        sampling_seed=args.sampling_seed,
        expert_coverage_policy=args.expert_coverage_policy,
        quant_args=quant_args,
    )

    collect_cmd, quant_cmd = build_commands(
        coexistence=args.coexistence,
        quantizer=args.quantizer,
        model=args.model,
        calib=args.calib,
        output=args.output,
        quant_format=args.quant_format,
        corpus=args.corpus,
        n_sequences=args.n_sequences,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        time_tile=args.time_tile,
        max_rows=args.max_rows,
        layer_prefetch_bytes=args.layer_prefetch_bytes,
        kldref_topk=args.kldref_topk,
        min_expert_activations=args.min_expert_activations,
        expert_capture_target=args.expert_capture_target,
        expert_capture_tile_rows=args.expert_capture_tile_rows,
        required_expert_fraction=args.required_expert_fraction,
        sampling_seed=args.sampling_seed,
        expert_coverage_policy=args.expert_coverage_policy,
        quant_args=quant_args,
    )
    collect_exec, quant_exec = scope_gpu_commands(args.hipfire, collect_cmd, quant_cmd)
    if not args.skip_calib:
        _print_command("pass 1/2", collect_exec)
    else:
        if not args.calib.is_file() and not args.dry_run:
            parser.error(f"--skip-calib requires an existing artifact: {args.calib}")
        print(f"pass 1/2: reusing {args.calib}", flush=True)
    _print_command("pass 2/2", quant_exec)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "calibration_execution": {
                        "mode": "segmented" if args.calibration_segment_layers else "single_process",
                        "process_segment_layers": args.calibration_segment_layers,
                    },
                    **recipe,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    previous = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    prior_recipe = previous.get("recipe_fingerprint")
    if prior_recipe and prior_recipe != recipe["recipe_fingerprint"]:
        raise RuntimeError(
            f"two-pass manifest recipe mismatch: existing {prior_recipe}, requested {recipe['recipe_fingerprint']}"
        )
    expected_calibration = None
    if args.skip_calib:
        expected_calibration = inspect_calibration_plan(collect_exec)
        update_manifest(
            manifest_path,
            recipe=recipe,
            phase="calibration_validating",
            calibration_execution={"mode": "reused", "process_segment_layers": 0},
        )
    else:
        expected_calibration = inspect_calibration_plan(collect_exec)
        update_manifest(
            manifest_path,
            recipe=recipe,
            phase="calibration_running",
            calibration_execution={
                "mode": "segmented" if args.calibration_segment_layers else "single_process",
                "process_segment_layers": args.calibration_segment_layers,
                "release_seconds": (
                    DEFAULT_CALIBRATION_SEGMENT_RELEASE_SECONDS
                    if args.calibration_segment_layers
                    else 0
                ),
            },
        )
    if not args.skip_calib:
        model_plan = expected_calibration.get("model", {})
        total_layers = model_plan.get("layers")
        if not isinstance(total_layers, int) or total_layers < 1:
            raise RuntimeError(f"native calibration plan has invalid model.layers: {total_layers!r}")

        def calibration_attempt_timings(elapsed: float) -> dict:
            current_manifest = json.loads(manifest_path.read_text())
            return accumulate_attempt_timing(
                current_manifest,
                phase_name="calibration",
                elapsed_seconds=elapsed,
            )

        def record_calibration_failure(phase: str, elapsed: float, failure: dict) -> None:
            update_manifest(
                manifest_path,
                recipe=recipe,
                phase=phase,
                failure=failure,
                phase_timings=calibration_attempt_timings(elapsed),
            )

        calibration_execution, calibration_attempt_seconds = run_calibration_attempt(
            collect_exec,
            calib=args.calib,
            total_layers=total_layers,
            segment_layers=args.calibration_segment_layers,
            progress=lambda execution: update_manifest(
                manifest_path,
                recipe=recipe,
                phase="calibration_running",
                calibration_execution=execution,
            ),
            on_failure=record_calibration_failure,
        )
        update_manifest(
            manifest_path,
            recipe=recipe,
            phase="calibration_validating",
            calibration_execution=calibration_execution,
            phase_timings=calibration_attempt_timings(calibration_attempt_seconds),
        )
    calibration = inspect_artifact(args.coexistence, args.calib)
    validate_calibration_inspection(calibration)
    calibration_audit = audit_calibration_artifact(args.coexistence, args.calib)
    validate_calibration_audit(calibration_audit, calibration)
    if expected_calibration is not None:
        validate_reusable_calibration(calibration, expected_calibration)
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="calibration_complete",
        calibration=calibration,
        calibration_audit=calibration_audit,
    )
    storage_preflight = pass_two_storage_preflight(
        model=args.model,
        output=args.output,
        quant_format=args.quant_format,
        calibration=calibration,
    )
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase=(
            "quantization_ready"
            if storage_preflight["filesystem"]["sufficient"]
            else "quantization_refused_storage"
        ),
        calibration=calibration,
        calibration_audit=calibration_audit,
        storage_preflight=storage_preflight,
    )
    require_pass_two_storage(storage_preflight)
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="quantization_running",
        calibration=calibration,
        calibration_audit=calibration_audit,
        storage_preflight=storage_preflight,
    )

    def quantization_attempt_timings(elapsed: float) -> dict:
        current_manifest = json.loads(manifest_path.read_text())
        return accumulate_attempt_timing(
            current_manifest,
            phase_name="quantization",
            elapsed_seconds=elapsed,
        )

    def record_quantization_failure(phase: str, elapsed: float, failure: dict) -> None:
        update_manifest(
            manifest_path,
            recipe=recipe,
            phase=phase,
            calibration=calibration,
            calibration_audit=calibration_audit,
            storage_preflight=storage_preflight,
            failure=failure,
            phase_timings=quantization_attempt_timings(elapsed),
        )

    quantization_seconds = run_quantization_pass(
        quant_exec,
        on_failure=record_quantization_failure,
    )
    quantized = inspect_artifact(args.coexistence, args.output)
    validate_quantized_inspection(quantized)
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="complete",
        calibration=calibration,
        calibration_audit=calibration_audit,
        storage_preflight=storage_preflight,
        quantized=quantized,
        phase_timings=quantization_attempt_timings(quantization_seconds),
    )


if __name__ == "__main__":
    main()
