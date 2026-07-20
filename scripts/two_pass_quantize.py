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
import shlex
import subprocess
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


def update_manifest(
    path: Path,
    *,
    recipe: dict,
    phase: str,
    calibration: dict | None = None,
    quantized: dict | None = None,
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
    if quantized is not None:
        manifest["quantized"] = quantized
    elif "quantized" in previous:
        manifest["quantized"] = previous["quantized"]

    calibration_value = calibration or manifest.get("calibration")
    quantized_value = quantized or manifest.get("quantized")
    fingerprints = {
        "calibration_artifact": _get(calibration_value, "artifact_fingerprint"),
        "calibration_run": _get(calibration_value, "metadata", "run_fingerprint"),
        "source": _get(calibration_value, "metadata", "source_manifest", "fingerprint"),
        "samples": _get(calibration_value, "metadata", "job", "samples", "fingerprint"),
        "quantized_artifact": _get(quantized_value, "artifact_fingerprint"),
        "quantized_payload": _get(quantized_value, "metadata", "quantization_hash", "value"),
    }
    manifest["fingerprints"] = {key: value for key, value in fingerprints.items() if value is not None}
    _atomic_json(path, manifest)
    return manifest


def inspect_artifact(coexistence: str, path: Path) -> dict:
    result = subprocess.run(
        [coexistence, "artifact", "inspect", "--input", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


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
        print(json.dumps({"manifest": str(manifest_path), **recipe}, indent=2), flush=True)
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
    update_manifest(manifest_path, recipe=recipe, phase="calibration_running")
    if not args.skip_calib:
        subprocess.run(collect_exec, check=True)
    calibration = inspect_artifact(args.coexistence, args.calib)
    validate_calibration_inspection(calibration)
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="calibration_complete",
        calibration=calibration,
    )
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="quantization_running",
        calibration=calibration,
    )
    subprocess.run(quant_exec, check=True)
    quantized = inspect_artifact(args.coexistence, args.output)
    validate_quantized_inspection(quantized)
    update_manifest(
        manifest_path,
        recipe=recipe,
        phase="complete",
        calibration=calibration,
        quantized=quantized,
    )


if __name__ == "__main__":
    main()
