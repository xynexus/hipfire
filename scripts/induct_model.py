#!/usr/bin/env python3
"""Resumable model induction: DFlash, calib+KLDREF, quant, and TriAttention.

This is offline orchestration only. It composes the existing converters and
GPU tools, records each completed stage in a manifest, and resumes from valid
artifacts without rereading a hundreds-of-GiB source checkpoint unnecessarily.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path("/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B")
DEFAULT_DRAFT = Path("/srv/huggingface/models--z-lab--Qwen3.5-397B-A17B-DFlash")
DEFAULT_CORPUS = Path("benchmarks/calib/calib-5m.txt")
DEFAULT_QUANT_FORMAT = "oq4.25++"
DEFAULT_DFLASH_FORMATS = ("bf16", "f16")
DEFAULT_LAYER_PREFETCH_BYTES = 16 * 1024**3
DEFAULT_MIN_EXPERT_ACTIVATIONS = 2048
DEFAULT_EXPERT_CAPTURE_TARGET = 4096
DEFAULT_EXPERT_CAPTURE_TILE_ROWS = 256
DEFAULT_REQUIRED_EXPERT_FRACTION = 1.0
DEFAULT_SAMPLING_SEED = 1
DEFAULT_EXPERT_COVERAGE_POLICY = "preserve-undercovered"
STAGES = ("dflash", "target", "triattn")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_hf_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        candidate = snapshots / main_ref.read_text().strip()
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    if snapshots.is_dir():
        candidates = sorted(
            (candidate for candidate in snapshots.iterdir() if (candidate / "config.json").is_file()),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        if candidates:
            return candidates[-1].resolve()
    raise FileNotFoundError(f"no Hugging Face snapshot/config.json under {path}")


def _read_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def _integer(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"config field {key!r} must be an integer, got {value!r}")
    return value


def _source_summary(path: Path) -> dict:
    config_bytes = (path / "config.json").read_bytes()
    safetensors = sorted(path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"no safetensors files under {path}")
    return {
        "snapshot": str(path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "safetensors_files": len(safetensors),
        "safetensors_bytes": sum(file.stat().st_size for file in safetensors),
    }


def preflight_sources(target: Path, draft: Path) -> dict:
    target = resolve_hf_snapshot(target)
    draft = resolve_hf_snapshot(draft)
    target_root = _read_config(target)
    target_text = target_root.get("text_config") or target_root
    draft_config = _read_config(draft)
    if "DFlashDraftModel" not in draft_config.get("architectures", []):
        raise ValueError("DFlash source config does not declare DFlashDraftModel")
    dflash_config = draft_config.get("dflash_config")
    if not isinstance(dflash_config, dict):
        raise ValueError("DFlash source config is missing dflash_config")
    block_size = draft_config.get("block_size") or dflash_config.get("block_size")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("DFlash block_size must be a positive integer")

    fields = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
    mismatches = []
    # The draft is a separate six-layer Qwen3 model and intentionally has a
    # different attention geometry from the target (published 397B pair:
    # target 32x2x256, draft 32x8x128). The shared runtime contract is the
    # residual width, vocabulary, target-layer count, and extraction indices.
    for field in ("hidden_size", "vocab_size"):
        target_value = _integer(target_text, field)
        draft_value = _integer(draft_config, field)
        if target_value != draft_value:
            mismatches.append(f"{field}: target={target_value}, draft={draft_value}")
    target_layers = _integer(target_text, "num_hidden_layers")
    draft_target_layers = _integer(draft_config, "num_target_layers")
    if target_layers != draft_target_layers:
        mismatches.append(f"num_hidden_layers/num_target_layers: target={target_layers}, draft={draft_target_layers}")
    target_layer_ids = dflash_config.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not all(isinstance(layer, int) for layer in target_layer_ids):
        raise ValueError("DFlash target_layer_ids must be an integer list")
    if any(layer < 0 or layer >= target_layers for layer in target_layer_ids):
        mismatches.append(f"target_layer_ids outside target range 0..{target_layers - 1}")
    mask_token_id = _integer(dflash_config, "mask_token_id")
    if not 0 <= mask_token_id < _integer(target_text, "vocab_size"):
        mismatches.append(f"mask_token_id {mask_token_id} is outside the target vocabulary")
    if mismatches:
        raise ValueError("target/DFlash incompatibility: " + "; ".join(mismatches))

    return {
        "target": {
            **_source_summary(target),
            **{field: _integer(target_text, field) for field in fields},
            "num_hidden_layers": target_layers,
        },
        "draft": {
            **_source_summary(draft),
            **{field: _integer(draft_config, field) for field in fields},
            "num_hidden_layers": _integer(draft_config, "num_hidden_layers"),
            "num_target_layers": draft_target_layers,
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
        },
        "compatibility": "compatible",
    }


def artifact_layout(
    root: Path,
    model_name: str,
    quant_format: str,
    dflash_formats: list[str] | tuple[str, ...],
) -> dict[str, Path]:
    primary_stem = f"{model_name}.{quant_format}"
    paths = {
        "model": root / "models" / f"{primary_stem}.hfq",
        "triattn": root / "triattn" / f"{model_name}.triattn.hfq",
        "calib": root / "calib" / f"{model_name}.calib.hfq",
        "manifest": root / "induction" / primary_stem / "manifest.json",
        "two_pass_manifest": root / "induction" / primary_stem / "two-pass.json",
    }
    for dflash_format in dict.fromkeys(dflash_formats):
        _dflash_format_args(dflash_format)
        paths[f"dflash_{dflash_format}"] = (
            root / "drafts" / f"{model_name}-{dflash_format.upper()}.dflash.hfq"
        )
    if not any(key.startswith("dflash_") for key in paths):
        raise ValueError("at least one DFlash format is required")
    return paths


def _dflash_format_args(dflash_format: str) -> list[str]:
    return {
        "bf16": [],
        "f16": ["--f16"],
        "f32": ["--keep-f32"],
        "mq3": ["--mq3"],
        "mq4": ["--mq4"],
        "mq6": ["--mq6"],
    }[dflash_format]


def default_quant_args(quant_format: str) -> list[str]:
    if quant_format.endswith("++"):
        return ["--awq", "--ldlq"]
    if quant_format.endswith("+"):
        return ["--awq"]
    return []


def build_stage_commands(
    *,
    target: Path,
    draft: Path,
    corpus: Path,
    paths: dict[str, Path],
    quant_format: str,
    dflash_formats: list[str],
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
    triattn_max_tokens: int,
    triattn_chunk_len: int,
    python: str,
    hipfire: str,
    coexistence: str,
    quantizer: str,
    dflash_converter: str,
    triattn_bin: str,
    quant_args: list[str],
    reuse_calibration: bool = False,
    calibration_segment_layers: int = 0,
) -> dict[str, list[list[str]]]:
    dflash_commands = [
        [
            dflash_converter,
            *_dflash_format_args(dflash_format),
            "--input",
            str(draft),
            "--output",
            str(paths[f"dflash_{dflash_format}"]),
        ]
        for dflash_format in dflash_formats
    ]
    target_command = [
        python,
        "scripts/two_pass_quantize.py",
        "--model",
        str(target),
        "--calib",
        str(paths["calib"]),
        "--output",
        str(paths["model"]),
        "--format",
        quant_format,
        "--corpus",
        str(corpus),
        "--n-sequences",
        str(n_sequences),
        "--ctx-len",
        str(ctx_len),
        "--batch-size",
        str(batch_size),
        "--time-tile",
        str(time_tile),
        "--max-rows",
        str(max_rows),
        "--layer-prefetch-bytes",
        str(layer_prefetch_bytes),
        "--calibration-segment-layers",
        str(calibration_segment_layers),
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
        "--manifest",
        str(paths["two_pass_manifest"]),
        "--quantizer",
        quantizer,
        "--coexistence",
        coexistence,
        "--hipfire",
        hipfire,
        *(["--skip-calib"] if reuse_calibration else []),
        "--",
        *quant_args,
    ]
    triattn_command = [
        hipfire,
        "lock",
        "run",
        "induct-triattn",
        "--",
        triattn_bin,
        str(paths["model"]),
        "--sidecar",
        str(paths["triattn"]),
        "--corpus",
        str(corpus),
        "--max-tokens",
        str(triattn_max_tokens),
        "--chunk-len",
        str(triattn_chunk_len),
        "--gpu-calib",
    ]
    return {
        "dflash": dflash_commands,
        "target": [target_command],
        "triattn": [triattn_command],
    }


def artifact_is_valid(path: Path, magic: bytes) -> bool:
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open("rb") as file:
        return file.read(len(magic)) == magic


def should_reuse_calibration(paths: dict[str, Path], *, force: bool) -> bool:
    return not force and artifact_is_valid(paths["calib"], b"HFQM")


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _stage_failure_status(stage: str, paths: dict[str, Path], error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if stage != "target":
        return "failed"
    try:
        two_pass = json.loads(paths["two_pass_manifest"].read_text())
    except (KeyError, OSError, json.JSONDecodeError):
        return "failed"
    status = two_pass.get("status")
    return (
        "interrupted"
        if isinstance(status, str) and status.endswith("_interrupted")
        else "failed"
    )


def _required_outputs(stage: str, paths: dict[str, Path]) -> list[tuple[Path, bytes]]:
    if stage == "dflash":
        return [
            (path, b"HFQM")
            for key, path in paths.items()
            if key.startswith("dflash_")
        ]
    if stage == "target":
        return [(paths["calib"], b"HFQM"), (paths["model"], b"HFQM")]
    if stage == "triattn":
        return [(paths["triattn"], b"TRIA")]
    raise ValueError(f"unknown induction stage {stage}")


def target_stage_complete(paths: dict[str, Path], recipe_fingerprint: str) -> bool:
    if not all(artifact_is_valid(path, magic) for path, magic in _required_outputs("target", paths)):
        return False
    try:
        manifest = json.loads(paths["two_pass_manifest"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    ledger = manifest.get("source_reads")
    fingerprints = manifest.get("fingerprints")
    audit = manifest.get("calibration_audit")
    return (
        manifest.get("status") == "complete"
        and manifest.get("recipe_fingerprint") == recipe_fingerprint
        and isinstance(ledger, dict)
        and not ledger.get("missing_logical")
        and not ledger.get("duplicate_logical")
        and isinstance(fingerprints, dict)
        and bool(fingerprints.get("calibration_artifact"))
        and bool(fingerprints.get("quantized_artifact"))
        and isinstance(audit, dict)
        and audit.get("schema") == "hipfire.calibration_audit.v1"
        and audit.get("valid") is True
        and not audit.get("errors")
        and audit.get("artifact_fingerprint") == fingerprints.get("calibration_artifact")
    )


def _stage_complete(
    stage: str,
    paths: dict[str, Path],
    target_recipe_fingerprint: str | None = None,
) -> bool:
    if stage == "target":
        return bool(target_recipe_fingerprint) and target_stage_complete(paths, target_recipe_fingerprint)
    return all(artifact_is_valid(path, magic) for path, magic in _required_outputs(stage, paths))


def _target_recipe_fingerprint(
    *,
    target: Path,
    corpus: Path,
    paths: dict[str, Path],
    quant_format: str,
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
) -> str:
    corpus_digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    recipe = {
        "model": str(target.resolve()),
        "calibration_artifact": str(paths["calib"].resolve()),
        "quantized_artifact": str(paths["model"].resolve()),
        "quant_format": quant_format,
        "corpus": str(corpus.resolve()),
        "corpus_sha256": f"sha256:{corpus_digest}",
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
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def tool_needs_build(binary: Path, sources: list[Path]) -> bool:
    """Return true when a repo-built tool is absent or older than its sources."""
    binary = binary if binary.is_absolute() else REPO_ROOT / binary
    if not binary.is_file():
        return True
    binary_mtime = binary.stat().st_mtime_ns
    for source in sources:
        resolved_source = source if source.is_absolute() else REPO_ROOT / source
        if resolved_source.is_file() and resolved_source.stat().st_mtime_ns > binary_mtime:
            return True
        if resolved_source.is_dir() and any(
            candidate.stat().st_mtime_ns > binary_mtime
            for candidate in resolved_source.rglob("*.rs")
            if candidate.is_file()
        ):
            return True
    return False


def _build_commands_for_tools(
    selected: list[str], *, hipfire: Path, coexistence: Path, quantizer: Path, dflash_converter: Path, triattn_bin: Path
) -> list[list[str]]:
    commands = []
    workspace_inputs = [Path("Cargo.lock")]
    quantize_inputs = workspace_inputs + [Path("crates/hipfire-quantize/Cargo.toml"), Path("crates/hipfire-quantize/src")]
    if "dflash" in selected and tool_needs_build(dflash_converter, quantize_inputs):
        commands.append(["cargo", "build", "--release", "-p", "hipfire-quantize", "--bin", "dflash_convert"])
    if "target" in selected and tool_needs_build(quantizer, quantize_inputs):
        commands.append(["cargo", "build", "--release", "-p", "hipfire-quantize", "--bin", "hipfire-quantize"])
    coexistence_inputs = workspace_inputs + [
        Path("crates/hipfire-coexistence/Cargo.toml"),
        Path("crates/hipfire-coexistence/src"),
        Path("crates/hipfire-runtime/src/calibration"),
        Path("crates/hipfire-arch-qwen35/src/calibration_stream.rs"),
        Path("crates/hipfire-arch-gemma3/src/calibration_stream.rs"),
    ]
    if "target" in selected and tool_needs_build(coexistence, coexistence_inputs):
        commands.append(
            ["cargo", "build", "--release", "-p", "hipfire-coexistence", "--bin", "hipfire-coexistence"]
        )
    hipfire_inputs = workspace_inputs + [Path("crates/hipfire-cli/Cargo.toml"), Path("crates/hipfire-cli/src")]
    if any(stage in selected for stage in ("target", "triattn")) and tool_needs_build(hipfire, hipfire_inputs):
        commands.append(["cargo", "build", "--release", "-p", "hipfire-cli", "--bin", "hipfire"])
    triattn_inputs = workspace_inputs + [
        Path("crates/hipfire-runtime/Cargo.toml"),
        Path("crates/hipfire-runtime/examples/triattn_validate.rs"),
    ]
    if "triattn" in selected and tool_needs_build(triattn_bin, triattn_inputs):
        commands.append(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "hipfire-runtime",
                "--features",
                "deltanet",
                "--example",
                "triattn_validate",
            ]
        )
    return commands


def _print_command(label: str, command: list[str]) -> None:
    print(f"{label}: {shlex.join(command)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Induct a target model and its DFlash/CASK support artifacts.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dflash-source", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--model-name", default="Qwen3.5-397B-A17B")
    parser.add_argument("--artifact-root", type=Path, default=Path("~/.hipfire"))
    parser.add_argument(
        "--format",
        dest="quant_format",
        default=DEFAULT_QUANT_FORMAT,
        help=f"Target weight format (default: {DEFAULT_QUANT_FORMAT}).",
    )
    parser.add_argument(
        "--dflash-format",
        action="append",
        dest="dflash_formats",
        choices=["bf16", "f16", "f32", "mq3", "mq4", "mq6"],
        help="DFlash sidecar dtype/format; repeat to select multiple (default: bf16 and f16).",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
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
        help="Restart calibration after this many additional durable layers (default: 0).",
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
    parser.add_argument("--triattn-max-tokens", type=int, default=100_000)
    parser.add_argument("--triattn-chunk-len", type=int, default=1024)
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages")
    parser.add_argument("--force", action="store_true", help="Rerun selected stages even when valid artifacts exist.")
    parser.add_argument("--no-auto-build", action="store_true", help="Fail instead of building missing release tools.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--hipfire", type=Path, default=Path("target/release/hipfire"))
    parser.add_argument(
        "--coexistence",
        type=Path,
        default=Path("target/release/hipfire-coexistence"),
    )
    parser.add_argument("--quantizer", type=Path, default=Path("target/release/hipfire-quantize"))
    parser.add_argument("--dflash-converter", type=Path, default=Path("target/release/dflash_convert"))
    parser.add_argument("--triattn-bin", type=Path, default=Path("target/release/examples/triattn_validate"))
    parser.add_argument(
        "quant_args",
        nargs=argparse.REMAINDER,
        help="Additional hipfire-quantize flags after `--`; defaults from the format suffix.",
    )
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.stages or STAGES))
    corpus = args.corpus.expanduser()
    if not corpus.is_absolute():
        corpus = (REPO_ROOT / corpus).resolve()
    if not corpus.is_file():
        parser.error(f"calibration corpus does not exist: {corpus}")
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
        args.triattn_max_tokens,
        args.triattn_chunk_len,
    ) < 1:
        parser.error("token, sequence, batch, and top-k values must be positive")
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

    target = resolve_hf_snapshot(args.target)
    draft = resolve_hf_snapshot(args.dflash_source)
    preflight = preflight_sources(target, draft)
    artifact_root = args.artifact_root.expanduser().resolve()
    dflash_formats = list(dict.fromkeys(args.dflash_formats or DEFAULT_DFLASH_FORMATS))
    paths = artifact_layout(artifact_root, args.model_name, args.quant_format, dflash_formats)
    supplied_quant_args = args.quant_args[1:] if args.quant_args[:1] == ["--"] else args.quant_args
    quant_args = supplied_quant_args or default_quant_args(args.quant_format)
    commands = build_stage_commands(
        target=target,
        draft=draft,
        corpus=corpus,
        paths=paths,
        quant_format=args.quant_format,
        dflash_formats=dflash_formats,
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
        triattn_max_tokens=args.triattn_max_tokens,
        triattn_chunk_len=args.triattn_chunk_len,
        python=args.python,
        hipfire=str(args.hipfire),
        coexistence=str(args.coexistence),
        quantizer=str(args.quantizer),
        dflash_converter=str(args.dflash_converter),
        triattn_bin=str(args.triattn_bin),
        quant_args=quant_args,
        reuse_calibration=should_reuse_calibration(paths, force=args.force),
        calibration_segment_layers=args.calibration_segment_layers,
    )
    target_recipe_fingerprint = _target_recipe_fingerprint(
        target=target,
        corpus=corpus,
        paths=paths,
        quant_format=args.quant_format,
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

    print(json.dumps({"preflight": preflight, "artifacts": {key: str(value) for key, value in paths.items()}}, indent=2))
    stages_to_run = [
        stage
        for stage in selected
        if args.force or not _stage_complete(stage, paths, target_recipe_fingerprint)
    ]
    build_commands = _build_commands_for_tools(
        stages_to_run,
        hipfire=args.hipfire,
        coexistence=args.coexistence,
        quantizer=args.quantizer,
        dflash_converter=args.dflash_converter,
        triattn_bin=args.triattn_bin,
    )
    for command in build_commands:
        _print_command("build", command)
    for stage in selected:
        if _stage_complete(stage, paths, target_recipe_fingerprint) and not args.force:
            print(f"{stage}: reuse valid artifact(s)")
        else:
            for index, command in enumerate(commands[stage], start=1):
                suffix = f"[{index}/{len(commands[stage])}]" if len(commands[stage]) > 1 else ""
                _print_command(f"{stage}{suffix}", command)
    if args.dry_run:
        return
    if build_commands and args.no_auto_build:
        parser.error("required release tools are missing or stale; rerun without --no-auto-build or build them manually")
    for command in build_commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    previous_manifest = {}
    if paths["manifest"].is_file():
        try:
            previous_manifest = json.loads(paths["manifest"].read_text())
        except (json.JSONDecodeError, OSError):
            previous_manifest = {}
    manifest = {
        "schema": 1,
        "created_at": previous_manifest.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "model_name": args.model_name,
        "quant_format": args.quant_format,
        "dflash_formats": dflash_formats,
        "corpus": str(corpus),
        "sources": preflight,
        "artifacts": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "stages": previous_manifest.get("stages", {}),
        "admission": {
            "status": "pending",
            "required_evidence": [
                "finite-logit and coherence smoke",
                "KLD/PPL against BF16 or an accepted higher-precision reference",
                "DFlash acceptance/tau and decoded-output checks",
                "TriAttention/CASK long-context recall",
                "combined DFlash plus CASK coherence and recall",
                "Kernel Atlas AR and DFlash performance rows",
            ],
        },
    }
    _write_manifest(paths["manifest"], manifest)

    for stage in selected:
        if _stage_complete(stage, paths, target_recipe_fingerprint) and not args.force:
            manifest["stages"][stage] = {"status": "reused", "completed_at": utc_now()}
            if stage == "target":
                two_pass = json.loads(paths["two_pass_manifest"].read_text())
                manifest["two_pass"] = two_pass
                manifest["source_reads"] = two_pass["source_reads"]
                manifest["fingerprints"] = two_pass["fingerprints"]
            manifest["updated_at"] = utc_now()
            _write_manifest(paths["manifest"], manifest)
            continue
        if stage == "triattn" and not artifact_is_valid(paths["model"], b"HFQM"):
            raise RuntimeError("TriAttention requires the completed target HFQ; run the target stage first")
        for output, _magic in _required_outputs(stage, paths):
            output.parent.mkdir(parents=True, exist_ok=True)
        manifest["stages"][stage] = {
            "status": "running",
            "started_at": utc_now(),
            "commands": commands[stage],
        }
        manifest["updated_at"] = utc_now()
        _write_manifest(paths["manifest"], manifest)
        try:
            for command in commands[stage]:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            if not _stage_complete(stage, paths, target_recipe_fingerprint):
                raise RuntimeError(f"{stage} command returned success but its output artifact is invalid")
        except BaseException as error:
            failure_status = _stage_failure_status(stage, paths, error)
            manifest["stages"][stage].update(
                {
                    "status": failure_status,
                    f"{failure_status}_at": utc_now(),
                    "error": str(error),
                }
            )
            manifest["updated_at"] = utc_now()
            _write_manifest(paths["manifest"], manifest)
            raise
        manifest["stages"][stage].update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "outputs": [
                    {"path": str(output), "bytes": output.stat().st_size}
                    for output, _magic in _required_outputs(stage, paths)
                ],
            }
        )
        if stage == "target":
            two_pass = json.loads(paths["two_pass_manifest"].read_text())
            manifest["two_pass"] = two_pass
            manifest["source_reads"] = two_pass["source_reads"]
            manifest["fingerprints"] = two_pass["fingerprints"]
        manifest["updated_at"] = utc_now()
        _write_manifest(paths["manifest"], manifest)

    print(f"induction artifacts complete; admission remains pending: {paths['manifest']}")


if __name__ == "__main__":
    main()
