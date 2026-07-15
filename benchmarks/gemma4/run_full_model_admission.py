#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Run the committed Gemma 4 BF16-oracle/OQ8 full-model admission plan.

Invoke this script through `hipfire lock run`; it deliberately does not invent a
second GPU lock. Each case uses the same committed exact-token fixture for the
pinned Transformers oracle and Hipfire candidate, then writes comparison and
run-manifest JSON suitable for later ingestion as hipfire-eval evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from compare_bf16_captures import compare_capture


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).with_name("phase5-oq8-capture-plan.json"),
    )
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--oracle-device-map", choices=("single", "auto"), default="single"
    )
    parser.add_argument("--oracle-gpu-max-memory")
    parser.add_argument("--oracle-cpu-max-memory")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--compare-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != "hipfire.gemma4.full-model-admission-plan.v1":
        raise ValueError("admission plan has the wrong schema")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("admission plan has no cases")
    covered = {
        prompt_class
        for case in cases
        for prompt_class in case.get("classes", [])
    }
    missing = set(plan.get("required_prompt_classes", [])) - covered
    if missing:
        raise ValueError(f"admission plan does not cover prompt classes {sorted(missing)}")
    for case in cases:
        oracle_mode = case.get("oracle_mode", "resident")
        if oracle_mode not in {"resident", "streaming"}:
            raise ValueError(
                f"case {case['id']} has unsupported oracle mode {oracle_mode}"
            )
        if oracle_mode == "streaming" and case["max_new_tokens"] != 1:
            raise ValueError(
                f"streaming oracle case {case['id']} must request one token"
            )


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    thresholds = resolve_repo_path(plan["thresholds"])
    selected = set(args.cases or [])
    cases = [
        case for case in plan["cases"] if not selected or case["id"] in selected
    ]
    unknown = selected - {case["id"] for case in cases}
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "hipfire.gemma4.full-model-admission-run.v1",
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256(args.plan),
        "thresholds": str(thresholds.resolve()),
        "thresholds_sha256": sha256(thresholds),
        "oracle_model": str(args.oracle_model.resolve()),
        "candidate_model": str(args.candidate_model.resolve()),
        "candidate_contract": plan["candidate"],
        "oracle_contract": plan["oracle"],
        "tokenizer_sha256": plan["tokenizer_sha256"],
        "compare_only": args.compare_only,
        "cases": [],
        "status": "running",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("schema") == manifest["schema"]
            and previous.get("plan_sha256") == manifest["plan_sha256"]
        ):
            replaced = {case["id"] for case in cases}
            manifest["cases"] = [
                record
                for record in previous.get("cases", [])
                if record.get("id") not in replaced
            ]

    try:
        for case in cases:
            case_id = case["id"]
            case_root = args.output / case_id
            oracle_dir = case_root / "oracle"
            candidate_dir = case_root / "candidate"
            input_path = resolve_repo_path(case["input"])
            layers = ",".join(str(layer) for layer in case["layers"])
            max_new = str(case["max_new_tokens"])
            record: dict[str, Any] = {
                "id": case_id,
                "classes": case["classes"],
                "input": str(input_path.resolve()),
                "input_sha256": sha256(input_path),
                "layers": case["layers"],
                "max_new_tokens": case["max_new_tokens"],
                "require_lifecycle": case["require_lifecycle"],
                "oracle_mode": case.get("oracle_mode", "resident"),
                "status": "running",
            }
            manifest["cases"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            oracle_mode = case.get("oracle_mode", "resident")
            if oracle_mode == "streaming":
                oracle_command = [
                    sys.executable,
                    "benchmarks/gemma4/capture_transformers_streaming_reference.py",
                    "--model",
                    str(args.oracle_model),
                    "--input-ids",
                    str(input_path),
                    "--output",
                    str(oracle_dir),
                    "--layers",
                    layers,
                    "--max-new-tokens",
                    max_new,
                    "--device",
                    args.device,
                ]
            else:
                oracle_command = [
                    sys.executable,
                    "benchmarks/gemma4/capture_transformers_reference.py",
                    "--model",
                    str(args.oracle_model),
                    "--input-ids",
                    str(input_path),
                    "--output",
                    str(oracle_dir),
                    "--layers",
                    layers,
                    "--max-new-tokens",
                    max_new,
                    "--device",
                    args.device,
                    "--device-map",
                    args.oracle_device_map,
                ]
                if args.oracle_gpu_max_memory is not None:
                    oracle_command.extend(
                        ["--gpu-max-memory", args.oracle_gpu_max_memory]
                    )
                if args.oracle_cpu_max_memory is not None:
                    oracle_command.extend(
                        ["--cpu-max-memory", args.oracle_cpu_max_memory]
                    )
                if args.oracle_device_map == "auto":
                    oracle_command.extend(
                        ["--offload-folder", str(case_root / "oracle-offload")]
                    )
            candidate_command = [
                "cargo",
                "run",
                "-q",
                "-p",
                "hipfire-arch-gemma4",
                "--release",
                "--example",
                "gemma4_capture",
                "--",
                str(args.candidate_model),
                str(input_path),
                str(candidate_dir),
                layers,
                max_new,
            ]
            record["oracle_command"] = command_text(oracle_command)
            record["candidate_command"] = command_text(candidate_command)
            if not args.compare_only:
                oracle_env = os.environ.copy()
                oracle_env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
                record["oracle_environment"] = {
                    "HF_DEACTIVATE_ASYNC_LOAD": "1"
                }
                run(oracle_command, env=oracle_env)
                candidate_env = os.environ.copy()
                if case["require_lifecycle"]:
                    candidate_env["HIPFIRE_GEMMA4_CAPTURE_LIFECYCLE"] = "1"
                run(candidate_command, env=candidate_env)

            comparison = compare_capture(oracle_dir, candidate_dir, thresholds)
            candidate_metadata = json.loads(
                candidate_dir.joinpath("capture.json").read_text()
            )
            if case["require_lifecycle"]:
                lifecycle = candidate_metadata.get("lifecycle", {})
                if lifecycle.get("reset_exact_match") is not True:
                    comparison["failures"].append("reset rerun is not an exact match")
                if lifecycle.get("unload_reload_exact_match") is not True:
                    comparison["failures"].append(
                        "unload/reload rerun is not an exact match"
                    )
                comparison["status"] = (
                    "pass" if not comparison["failures"] else "fail"
                )
            comparison_path = case_root / "comparison.json"
            comparison_path.write_text(
                json.dumps(comparison, indent=2, sort_keys=True) + "\n"
            )
            record["comparison"] = str(comparison_path.resolve())
            record["comparison_sha256"] = sha256(comparison_path)
            record["status"] = comparison["status"]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            if comparison["status"] != "pass":
                raise RuntimeError(
                    f"{case_id} failed the frozen gate: {comparison['failures']}"
                )

        passed = {
            record["id"]
            for record in manifest["cases"]
            if record.get("status") == "pass"
        }
        required = {case["id"] for case in plan["cases"]}
        manifest["status"] = "pass" if passed == required else "partial_pass"
    except Exception as error:
        manifest["status"] = "fail"
        manifest["failure"] = str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"Gemma 4 admission: {manifest['status'].upper()} "
        f"({len(cases)} cases evaluated this run)"
    )


if __name__ == "__main__":
    main()
