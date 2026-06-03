#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Run bounded BF16-referenced KLD comparisons for gfx1151 quant readiness.

This wraps the canonical `eval_hipfire` + `kld_reduce.py` pipeline and records
the exact commands, evaluator md5, BF16 reference sha256, reducer outputs, and
per-case status in one JSON/Markdown evidence pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "release" / "examples" / "eval_hipfire"
DEFAULT_MODELS_DIR = Path(os.environ.get("HIPFIRE_MODELS_DIR", Path.home() / ".hipfire" / "models"))
DEFAULT_REF = ROOT / "benchmarks" / "quality-baselines" / "refs" / "qwen3.5-9b-bf16.kldref.bin"
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
REDUCE = ROOT / "benchmarks" / "quality-baselines" / "harness" / "kld_reduce.py"


@dataclass(frozen=True)
class Case:
    id: str
    model_file: str
    format_id: str
    family: str
    ref_name_tokens: tuple[str, ...] = ()


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq4", "qwen3.5-9b.mq4", "mq4", "dense", ("qwen3.5-9b", "qwen35-9b")),
    Case("qwen35-9b-mq6", "qwen3.5-9b.mq6", "mq6", "dense", ("qwen3.5-9b", "qwen35-9b")),
)

MQ3_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq3", "qwen3.5-9b.mq3", "mq3", "dense", ("qwen3.5-9b", "qwen35-9b")),
)

MQ3_4B_CASES: tuple[Case, ...] = (
    Case("qwen35-4b-mq4", "qwen3.5-4b.mq4", "mq4", "dense", ("qwen3.5-4b", "qwen35-4b")),
    Case("qwen35-4b-mq3", "qwen3.5-4b.mq3", "mq3", "dense", ("qwen3.5-4b", "qwen35-4b")),
)

MQ3_A3B_CASES: tuple[Case, ...] = (
    Case(
        "qwen35-a3b-mq4",
        "qwen3.5-35b-a3b.mq4",
        "mq4",
        "a3b",
        ("qwen3.5-35b-a3b", "qwen35-35b-a3b"),
    ),
    Case(
        "qwen35-a3b-mq3",
        "qwen3.5-35b-a3b.mq3",
        "mq3",
        "a3b",
        ("qwen3.5-35b-a3b", "qwen35-35b-a3b"),
    ),
    Case(
        "qwen36-a3b-mq4",
        "qwen3.6-35b-a3b.mq4",
        "mq4",
        "a3b",
        ("qwen3.6-35b-a3b", "qwen36-35b-a3b"),
    ),
    Case(
        "qwen36-a3b-mq3",
        "qwen3.6-35b-a3b.mq3",
        "mq3",
        "a3b",
        ("qwen3.6-35b-a3b", "qwen36-35b-a3b"),
    ),
)

MQ3_LLOYD_CASES: tuple[Case, ...] = (
    Case("qwen35-4b-mq3-lloyd", "qwen3.5-4b.mq3-lloyd", "mq3-lloyd", "dense", ("qwen3.5-4b", "qwen35-4b")),
    Case("qwen35-9b-mq3-lloyd", "qwen3.5-9b.mq3-lloyd", "mq3-lloyd", "dense", ("qwen3.5-9b", "qwen35-9b")),
    Case("qwen35-27b-mq3-lloyd", "qwen3.5-27b.mq3-lloyd", "mq3-lloyd", "dense", ("qwen3.5-27b", "qwen35-27b")),
    Case(
        "qwen35-a3b-mq3-lloyd",
        "qwen3.5-35b-a3b.mq3-lloyd",
        "mq3-lloyd",
        "a3b",
        ("qwen3.5-35b-a3b", "qwen35-35b-a3b"),
    ),
)

MQ4_LLOYD_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq4-lloyd", "qwen3.5-9b.mq4-lloyd", "mq4-lloyd", "dense", ("qwen3.5-9b", "qwen35-9b")),
    Case("qwen35-27b-mq4-lloyd", "qwen3.5-27b.mq4-lloyd", "mq4-lloyd", "dense", ("qwen3.5-27b", "qwen35-27b")),
    Case(
        "qwen35-a3b-mq4-lloyd",
        "qwen3.5-35b-a3b.mq4-lloyd",
        "mq4-lloyd",
        "a3b",
        ("qwen3.5-35b-a3b", "qwen35-35b-a3b"),
    ),
)

ALL_CASES: tuple[Case, ...] = (
    DEFAULT_CASES
    + MQ3_CASES
    + MQ3_4B_CASES
    + MQ3_A3B_CASES
    + MQ3_LLOYD_CASES
    + MQ4_LLOYD_CASES
)


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def detect_arch() -> str:
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:") and "gfx" in stripped:
            return stripped.split()[-1]
    return "unknown"


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def case_variant(case: Case, kv_mode: str, max_chunks: int | None) -> str:
    suffix = f"kv{kv_mode}"
    if max_chunks is not None:
        suffix += f"-c{max_chunks}"
    return f"{case.model_file}-{suffix}"


def ref_name_matches_case(ref_path: Path, case: Case) -> bool:
    if not case.ref_name_tokens:
        return True
    ref_name = ref_path.name.lower().replace("_", "-")
    return any(token.lower().replace("_", "-") in ref_name for token in case.ref_name_tokens)


def reference_mismatches(ref_path: Path, cases: list[Case]) -> list[str]:
    return [case.id for case in cases if not ref_name_matches_case(ref_path, case)]


def run_case(
    exe: Path,
    models_dir: Path,
    ref_path: Path,
    per_seq_dir: Path,
    case: Case,
    kv_mode: str,
    scoring_mode: str,
    max_chunks: int | None,
    timeout: int,
    hash_models: bool,
) -> dict[str, Any]:
    model_path = models_dir / case.model_file
    variant = case_variant(case, kv_mode, max_chunks)
    output = per_seq_dir / f"{variant}__gfx1151__{scoring_mode}.kldseq"
    payload: dict[str, Any] = {
        **asdict(case),
        "model_path": str(model_path),
        "model_present": model_path.exists(),
        "variant": variant,
        "output": str(output),
    }
    if not model_path.exists():
        payload["status"] = "missing"
        return payload
    if hash_models:
        payload["model_md5"] = digest_file(model_path, "md5")

    command = [
        str(exe),
        "--model",
        str(model_path),
        "--ref",
        str(ref_path),
        "--output",
        str(output),
        "--kv-mode",
        kv_mode,
        "--scoring-mode",
        scoring_mode,
    ]
    if max_chunks is not None:
        command.extend(["--max-chunks", str(max_chunks)])

    print(case.id, file=sys.stderr, flush=True)
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=os.environ.copy(),
    )
    wall_s = time.monotonic() - started
    status = "ok" if proc.returncode == 0 and output.exists() and output.stat().st_size > 0 else "hard_error"
    payload.update(
        {
            "status": status,
            "exit_code": proc.returncode,
            "command": command,
            "wall_s": round(wall_s, 3),
            "output_size_bytes": output.stat().st_size if output.exists() else 0,
            "log_tail": proc.stdout[-4000:],
        }
    )
    return payload


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload.get("reduced_rows", [])
    lines = [
        "# Quant BF16-Referenced KLD",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- eval_hipfire md5: {payload['eval_hipfire']['md5']}",
        f"- bf16 ref sha256: {payload['reference']['sha256']}",
        f"- kv_mode: {payload['params']['kv_mode']}",
        f"- scoring_mode: {payload['params']['scoring_mode']}",
        f"- max_chunks: {payload['params']['max_chunks']}",
        "",
        "| variant | mode | chunks | mean KLD | CI lo | CI hi | p99 KLD | PPL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {mode} | {chunks} | {mean} | {lo} | {hi} | {p99} | {ppl} |".format(
                variant=row["variant"],
                mode=row["scoring_mode"],
                chunks=row["n_chunks"],
                mean=fmt_metric(row["mean_kld"]),
                lo=fmt_metric(row["mean_kld_ci_lo"]),
                hi=fmt_metric(row["mean_kld_ci_hi"]),
                p99=fmt_metric(row["p99_kld"]),
                ppl=fmt_metric(row.get("ppl")),
            )
        )
    lines.extend(["", "Case status:"])
    for case in payload["cases"]:
        lines.append(f"- {case['id']}: {case['status']} ({case.get('output_size_bytes', 0)} bytes)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--ref", default=str(DEFAULT_REF))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq6-kld.json")
    parser.add_argument("--kv-mode", default="q8", choices=("q8", "asym2", "asym3", "asym4"))
    parser.add_argument("--scoring-mode", default="prefill", choices=("prefill", "per-token"))
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument(
        "--allow-ref-mismatch",
        action="store_true",
        help="Allow selected cases to run against a reference whose filename does not match the expected fixture.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in ALL_CASES],
        help="Limit to a specific case id; repeatable. Defaults to MQ4 control plus MQ6 candidate.",
    )
    args = parser.parse_args()

    if args.max_chunks is not None and args.max_chunks < 1:
        raise SystemExit("--max-chunks must be >= 1")

    exe = Path(args.exe)
    ref_path = Path(args.ref)
    models_dir = Path(args.models_dir)
    if not exe.exists():
        raise SystemExit(f"eval_hipfire binary not found: {exe}")
    if not ref_path.exists():
        raise SystemExit(f"BF16 KLD reference not found: {ref_path}")
    if not REDUCE.exists():
        raise SystemExit(f"kld_reduce.py not found: {REDUCE}")

    if args.out:
        out = Path(args.out)
    else:
        date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq6-kld.json"
    run_dir = out.with_suffix("")
    per_seq_dir = run_dir / "per-seq"
    per_seq_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.case)
    cases = [case for case in ALL_CASES if selected and case.id in selected]
    if not selected:
        cases = list(DEFAULT_CASES)

    mismatches = reference_mismatches(ref_path, cases)
    if mismatches and not args.allow_ref_mismatch:
        joined = ", ".join(mismatches)
        raise SystemExit(
            f"reference filename {ref_path.name!r} does not match selected case(s): {joined}; "
            "pass the matching HFKLDR ref or --allow-ref-mismatch for an explicitly documented experiment"
        )

    payload: dict[str, Any] = {
        "schema": "hipfire.quant_kld.gfx1151.v0",
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": detect_arch(),
        "params": {
            "kv_mode": args.kv_mode,
            "scoring_mode": args.scoring_mode,
            "max_chunks": args.max_chunks,
        },
        "eval_hipfire": {
            "path": str(exe),
            "md5": digest_file(exe, "md5"),
        },
        "reference": {
            "path": str(ref_path),
            "sha256": digest_file(ref_path, "sha256"),
            "size_bytes": ref_path.stat().st_size,
        },
        "models_dir": str(models_dir),
        "run_dir": str(run_dir),
        "per_seq_dir": str(per_seq_dir),
        "cases": [],
    }

    for case in cases:
        payload["cases"].append(
            run_case(
                exe,
                models_dir,
                ref_path,
                per_seq_dir,
                case,
                args.kv_mode,
                args.scoring_mode,
                args.max_chunks,
                args.timeout,
                args.hash_models,
            )
        )

    missing = [case["model_path"] for case in payload["cases"] if case["status"] == "missing"]
    hard_errors = [case["id"] for case in payload["cases"] if case["status"] == "hard_error"]
    if not missing and not hard_errors:
        reduce_json = run_dir / "result-data.json"
        reduce_md = run_dir / "result-table.md"
        command = [
            "python3",
            str(REDUCE),
            "--result-dir",
            str(per_seq_dir),
            "--out-md",
            str(reduce_md),
            "--out-json",
            str(reduce_json),
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        payload["reduce"] = {
            "status": "ok" if proc.returncode == 0 and reduce_json.exists() else "hard_error",
            "exit_code": proc.returncode,
            "command": command,
            "result_json": str(reduce_json),
            "result_md": str(reduce_md),
            "log_tail": proc.stdout[-4000:],
        }
        if reduce_json.exists():
            payload["reduced_rows"] = json.loads(reduce_json.read_text(encoding="utf-8"))
    else:
        payload["reduce"] = {"status": "skipped"}
        payload["reduced_rows"] = []

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, out.with_suffix(".md"))
    print(out)

    if missing and args.fail_on_missing:
        for item in missing:
            print(f"missing model: {item}", file=sys.stderr)
        return 2
    if hard_errors or payload.get("reduce", {}).get("status") == "hard_error":
        for item in hard_errors:
            print(f"hard error: {item}", file=sys.stderr)
        if payload.get("reduce", {}).get("status") == "hard_error":
            print("hard error: kld_reduce", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
