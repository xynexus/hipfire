#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Run bounded MQ-family PPL comparisons for gfx1151 readiness.

The underlying `perplexity` example is a single-window NLL/PPL evaluator.  This
wrapper keeps the reproducibility metadata and exact command line next to
results so MQ-family quality evidence is not confused with coherence-only
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "release" / "examples" / "perplexity"
DEFAULT_MODELS_DIR = Path(os.environ.get("HIPFIRE_MODELS_DIR", Path.home() / ".hipfire" / "models"))
DEFAULT_CORPUS = ROOT / "benchmarks" / "quality-baselines" / "slice" / "wikitext2-1024s-2048ctx.txt"
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"


@dataclass(frozen=True)
class Case:
    id: str
    model_file: str
    format_id: str
    family: str


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq4", "qwen3.5-9b.mq4", "mq4", "dense"),
    Case("qwen35-9b-mq6", "qwen3.5-9b.mq6", "mq6", "dense"),
)

MQ3_DENSE_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq3", "qwen3.5-9b.mq3", "mq3", "dense"),
    Case("qwen35-27b-mq4", "qwen3.5-27b.mq4", "mq4", "dense"),
    Case("qwen35-27b-mq3", "qwen3.5-27b.mq3", "mq3", "dense"),
)

A3B_CASES: tuple[Case, ...] = (
    Case("qwen35-a3b-mq4", "qwen3.5-35b-a3b.mq4", "mq4", "moe"),
    Case("qwen35-a3b-mq6", "qwen3.5-35b-a3b.mq6", "mq6", "moe"),
    Case("qwen36-a3b-mq4", "qwen3.6-35b-a3b.mq4", "mq4", "moe"),
    Case("qwen36-a3b-mq6", "qwen3.6-35b-a3b.mq6", "mq6", "moe"),
)

MQ3_A3B_CASES: tuple[Case, ...] = (
    Case("qwen35-a3b-mq3", "qwen3.5-35b-a3b.mq3", "mq3", "moe"),
    Case("qwen36-a3b-mq3", "qwen3.6-35b-a3b.mq3", "mq3", "moe"),
)

ALL_CASES: tuple[Case, ...] = DEFAULT_CASES + MQ3_DENSE_CASES + A3B_CASES + MQ3_A3B_CASES


RESULT_PATTERNS = {
    "scored": re.compile(r"^Scored:\s+(\d+)", re.MULTILINE),
    "nll_tok": re.compile(r"^NLL/tok:\s+([0-9.eE+-]+)", re.MULTILINE),
    "ppl": re.compile(r"^PPL:\s+([0-9.eE+-]+)", re.MULTILINE),
    "elapsed": re.compile(r"^Elapsed:\s+([0-9.eE+-]+)s\s+\(([0-9.eE+-]+) tok/s\)", re.MULTILINE),
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
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


def parse_result(output: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, pattern in RESULT_PATTERNS.items():
        match = pattern.search(output)
        if not match:
            continue
        if key == "elapsed":
            payload["elapsed_s"] = float(match.group(1))
            payload["scored_tok_s"] = float(match.group(2))
        elif key == "scored":
            payload[key] = int(match.group(1))
        else:
            payload[key] = float(match.group(1))
    return payload


def run_case(
    exe: Path,
    corpus: Path,
    models_dir: Path,
    case: Case,
    ctx: int,
    warmup: int,
    offset: int,
    kv_mode: str,
    timeout: int,
    hash_models: bool,
) -> dict[str, Any]:
    model_path = models_dir / case.model_file
    payload: dict[str, Any] = {
        **asdict(case),
        "model_path": str(model_path),
        "model_present": model_path.exists(),
    }
    if not model_path.exists():
        payload["status"] = "missing"
        return payload
    if hash_models:
        payload["model_md5"] = md5_file(model_path)
    command = [
        str(exe),
        str(model_path),
        str(corpus),
        "--ctx",
        str(ctx),
        "--warmup",
        str(warmup),
        "--offset",
        str(offset),
        "--kv-mode",
        kv_mode,
    ]
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
    parsed = parse_result(proc.stdout)
    status = "ok" if proc.returncode == 0 and parsed.get("scored", 0) > 0 else "hard_error"
    payload.update(
        {
            "status": status,
            "exit_code": proc.returncode,
            "command": command,
            "wall_s": round(wall_s, 3),
            "result": parsed,
            "log_tail": proc.stdout[-4000:],
        }
    )
    return payload


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Quant PPL Compare",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- perplexity md5: {payload['perplexity']['md5']}",
        f"- corpus md5: {payload['corpus']['md5']}",
        f"- ctx: {payload['params']['ctx']}",
        f"- warmup: {payload['params']['warmup']}",
        f"- offset: {payload['params']['offset']}",
        f"- kv_mode: {payload['params']['kv_mode']}",
        "",
        "| case | format | family | status | scored | nll/tok | ppl | eval tok/s |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        result = case.get("result", {})
        lines.append(
            "| {id} | {fmt} | {family} | {status} | {scored} | {nll} | {ppl} | {tok_s} |".format(
                id=case["id"],
                fmt=case["format_id"],
                family=case["family"],
                status=case["status"],
                scored=fmt_metric(result.get("scored")),
                nll=fmt_metric(result.get("nll_tok")),
                ppl=fmt_metric(result.get("ppl")),
                tok_s=fmt_metric(result.get("scored_tok_s")),
            )
        )
    lines.extend(["", "Hard errors:"])
    hard = [f"- {case['id']}" for case in payload["cases"] if case["status"] == "hard_error"]
    lines.extend(hard or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq6-ppl.json")
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--include-a3b", action="store_true")
    parser.add_argument("--include-mq3", action="store_true", help="Include dense MQ3 rows in the default case set")
    parser.add_argument("--include-mq3-a3b", action="store_true", help="Include Qwen3.5/Qwen3.6 A3B MQ3 rows")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in ALL_CASES],
        help="Limit to a specific case id; repeatable. Case selection can include A3B rows without --include-a3b.",
    )
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    exe = Path(args.exe)
    corpus = Path(args.corpus)
    models_dir = Path(args.models_dir)
    if not exe.exists():
        raise SystemExit(f"perplexity binary not found: {exe}")
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")

    selected = set(args.case)
    if selected:
        cases = [case for case in ALL_CASES if case.id in selected]
    else:
        cases = list(DEFAULT_CASES)
    if args.include_mq3 and not selected:
        cases.extend(MQ3_DENSE_CASES)
    if args.include_a3b and not selected:
        cases.extend(A3B_CASES)
    if args.include_mq3_a3b and not selected:
        cases.extend(MQ3_A3B_CASES)

    payload: dict[str, Any] = {
        "schema": "hipfire.quant_ppl.gfx1151.v0",
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": detect_arch(),
        "params": {
            "ctx": args.ctx,
            "warmup": args.warmup,
            "offset": args.offset,
            "kv_mode": args.kv_mode,
        },
        "perplexity": {
            "path": str(exe),
            "md5": md5_file(exe),
        },
        "corpus": {
            "path": str(corpus),
            "md5": md5_file(corpus),
            "size_bytes": corpus.stat().st_size,
        },
        "models_dir": str(models_dir),
        "cases": [],
    }

    for case in cases:
        payload["cases"].append(
            run_case(
                exe,
                corpus,
                models_dir,
                case,
                args.ctx,
                args.warmup,
                args.offset,
                args.kv_mode,
                args.timeout,
                args.hash_models,
            )
        )

    if args.out:
        out = Path(args.out)
    else:
        date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq6-ppl.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, out.with_suffix(".md"))
    print(out)

    missing = [case["model_path"] for case in payload["cases"] if case["status"] == "missing"]
    if missing and args.fail_on_missing:
        for path in missing:
            print(f"missing model: {path}", file=sys.stderr)
        return 2
    hard_errors = [case["id"] for case in payload["cases"] if case["status"] == "hard_error"]
    if hard_errors:
        for case_id in hard_errors:
            print(f"hard error: {case_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
