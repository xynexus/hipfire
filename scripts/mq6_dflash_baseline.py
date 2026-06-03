#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Collect DFlash/spec rows for MQ6 gfx1151 readiness.

This helper is scoped to the target-side MQ6 path.  The current DFlash draft
loader accepts F16/F32/MQ4/MQ3 draft matrices, so the evidence here compares
the MQ4 control target and the MQ6 candidate target while using the same MQ4
DFlash draft.  It records prompt md5, binary md5, exact command lines, parsed
bench metrics, and a token-attractor check for every fresh-process run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "release" / "examples" / "dflash_spec_demo"
DEFAULT_MODELS_DIR = Path(os.environ.get("HIPFIRE_MODELS_DIR", Path.home() / ".hipfire" / "models"))
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
DEFAULT_DRAFT = "qwen35-27b-dflash-mq4.hfq"

PROSE_PROMPT = (
    "The Roman Empire, at its height, stretched from the windswept moors of "
    "northern Britain to the sands of the Arabian peninsula. Its decline was "
    "not a single event but a long slow unraveling that took centuries. "
    "Several factors contributed to this gradual collapse. The first and "
    "perhaps most important was"
)

CODE_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''


@dataclass(frozen=True)
class Case:
    id: str
    target_file: str
    target_format: str
    draft_file: str
    prompt_id: str
    prompt: str
    max_tokens: int


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-27b-mq4-dflash-prose", "qwen3.5-27b.mq4", "mq4", DEFAULT_DRAFT, "prose", PROSE_PROMPT, 192),
    Case("qwen35-27b-mq6-dflash-prose", "qwen3.5-27b.mq6", "mq6", DEFAULT_DRAFT, "prose", PROSE_PROMPT, 192),
    Case("qwen35-27b-mq4-dflash-code", "qwen3.5-27b.mq4", "mq4", DEFAULT_DRAFT, "code", CODE_PROMPT, 128),
    Case("qwen35-27b-mq6-dflash-code", "qwen3.5-27b.mq6", "mq6", DEFAULT_DRAFT, "code", CODE_PROMPT, 128),
)

METRIC_PATTERNS = {
    "prompt_tokens": re.compile(r"^prompt_tokens:\s+(\d+)", re.MULTILINE),
    "prefill_secs": re.compile(r"^prefill_secs:\s+([0-9.eE+-]+)", re.MULTILINE),
    "prefill_tok_s": re.compile(r"^prefill_tok_s:\s+([0-9.eE+-]+)", re.MULTILINE),
    "ttft_ms": re.compile(r"^ttft_ms:\s+([0-9.eE+-]+)", re.MULTILINE),
    "decode_tokens_emitted": re.compile(r"^decode_tokens_emitted:\s+(\d+)", re.MULTILINE),
    "decode_secs": re.compile(r"^decode_secs:\s+([0-9.eE+-]+)", re.MULTILINE),
    "decode_tok_s": re.compile(r"^decode_tok_s:\s+([0-9.eE+-]+)", re.MULTILINE),
    "decode_tau": re.compile(r"^decode_tau:\s+([0-9.eE+-]+)", re.MULTILINE),
    "decode_accept_rate": re.compile(r"^decode_accept_rate:\s+([0-9.eE+-]+)", re.MULTILINE),
    "vram_used_mb": re.compile(r"^vram_used_mb:\s+(\d+)", re.MULTILINE),
    "vram_total_mb": re.compile(r"^vram_total_mb:\s+(\d+)", re.MULTILINE),
}

TOKEN_LINE = re.compile(r"DFlash tokens:\s+\[([^\]]*)\]")
EOT_IDS = {248044, 248046}


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


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


def parse_metrics(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    int_keys = {"prompt_tokens", "decode_tokens_emitted", "vram_used_mb", "vram_total_mb"}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if not match:
            continue
        value = match.group(1)
        parsed[key] = int(value) if key in int_keys else float(value)
    return parsed


def parse_tokens(output: str) -> list[int]:
    match = TOKEN_LINE.search(output)
    if not match:
        return []
    tokens = []
    for raw in match.group(1).split(","):
        item = raw.strip()
        if item:
            tokens.append(int(item))
    return tokens


def token_attractor_check(tokens: list[int]) -> dict[str, Any]:
    if not tokens:
        return {"ok": False, "reason": "zero_tokens"}
    trimmed = tokens
    for idx, token in enumerate(tokens):
        if token in EOT_IDS:
            trimmed = tokens[:idx]
            break
    window = trimmed[:128]
    if len(window) < 16:
        return {"ok": True, "total": len(window), "reason": "short_window_ok"}
    counter = collections.Counter(window)
    unique = len(counter)
    total = len(window)
    max_tok, max_count = counter.most_common(1)[0]
    unique_ratio = unique / total
    max_freq = max_count / total
    hard_fail = max_freq > 0.50 or unique_ratio < 0.15
    soft_warn = (max_freq > 0.40 or unique_ratio < 0.30) and not hard_fail
    return {
        "ok": not hard_fail,
        "soft_warn": soft_warn,
        "total": total,
        "unique": unique,
        "unique_ratio": round(unique_ratio, 3),
        "max_freq": round(max_freq, 3),
        "max_tok": max_tok,
        "max_count": max_count,
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [run for run in runs if run["status"] == "ok"]
    summary: dict[str, Any] = {
        "ok_runs": len(ok_runs),
        "total_runs": len(runs),
    }
    for key in (
        "prefill_tok_s",
        "ttft_ms",
        "decode_tokens_emitted",
        "decode_secs",
        "decode_tok_s",
        "decode_tau",
        "decode_accept_rate",
        "vram_used_mb",
    ):
        values = [float(run["metrics"][key]) for run in ok_runs if key in run.get("metrics", {})]
        summary[f"median_{key}"] = median(values)
    return summary


def run_case(
    exe: Path,
    models_dir: Path,
    case: Case,
    run_idx: int,
    ctx: int,
    kv_mode: str,
    timeout: int,
) -> dict[str, Any]:
    target_path = models_dir / case.target_file
    draft_path = models_dir / case.draft_file
    command = [
        str(exe),
        "--target",
        str(target_path),
        "--draft",
        str(draft_path),
        "--prompt",
        case.prompt,
        "--max",
        str(case.max_tokens),
        "--ctx",
        str(ctx),
        "--kv-mode",
        kv_mode,
        "--no-adaptive-b",
        "--no-chatml",
    ]
    print(f"{case.id} run {run_idx}", file=sys.stderr, flush=True)
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
    metrics = parse_metrics(proc.stdout)
    tokens = parse_tokens(proc.stdout)
    attractor = token_attractor_check(tokens)
    panic = next(
        (
            line
            for line in proc.stdout.splitlines()
            if "panicked" in line or "FATAL" in line or "error: " in line
        ),
        "",
    )
    status = "ok"
    if (
        proc.returncode != 0
        or int(metrics.get("decode_tokens_emitted", 0)) <= 0
        or not attractor.get("ok", False)
        or panic
    ):
        status = "hard_error"
    return {
        "run": run_idx,
        "status": status,
        "exit_code": proc.returncode,
        "wall_s": round(wall_s, 3),
        "command": command,
        "metrics": metrics,
        "token_attractor": attractor,
        "panic": panic,
        "log_tail": proc.stdout[-4000:],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ6 DFlash Baseline",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- dflash_spec_demo md5: {payload['dflash_spec_demo']['md5']}",
        f"- kv_mode: {payload['params']['kv_mode']}",
        f"- ctx: {payload['params']['ctx']}",
        "",
        "| case | format | prompt md5 | ok runs | median decode tok/s | median tau | median accept | median prefill tok/s |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        summary = case.get("summary", {})
        lines.append(
            "| {id} | {fmt} | {prompt_md5} | {ok}/{total} | {tok_s} | {tau} | {accept} | {prefill} |".format(
                id=case["id"],
                fmt=case["target_format"],
                prompt_md5=case["prompt_md5"],
                ok=summary.get("ok_runs", 0),
                total=summary.get("total_runs", 0),
                tok_s=_fmt(summary.get("median_decode_tok_s")),
                tau=_fmt(summary.get("median_decode_tau")),
                accept=_fmt(summary.get("median_decode_accept_rate")),
                prefill=_fmt(summary.get("median_prefill_tok_s")),
            )
        )
    lines.extend(["", "Hard errors:"])
    hard = [
        f"- {case['id']} run {run['run']}: {run['panic'] or run['token_attractor']}"
        for case in payload["cases"]
        for run in case.get("runs", [])
        if run["status"] != "ok"
    ]
    lines.extend(hard or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq6-dflash.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--pretty", action="store_true", help="Accepted for refresh-plan CLI consistency")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in DEFAULT_CASES],
        help="Limit to a case id; repeatable. Defaults to the full MQ4/MQ6 DFlash set.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"dflash_spec_demo binary not found: {exe}")

    models_dir = Path(args.models_dir)
    selected = set(args.case)
    cases = [case for case in DEFAULT_CASES if not selected or case.id in selected]

    payload: dict[str, Any] = {
        "schema": "hipfire.mq6_dflash.gfx1151.v0",
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": detect_arch(),
        "params": {
            "runs_per_case": args.runs,
            "ctx": args.ctx,
            "kv_mode": args.kv_mode,
            "chatml": False,
            "adaptive_b": False,
        },
        "dflash_spec_demo": {
            "path": str(exe),
            "md5": md5_file(exe),
        },
        "models_dir": str(models_dir),
        "cases": [],
    }

    missing = []
    for case in cases:
        target_path = models_dir / case.target_file
        draft_path = models_dir / case.draft_file
        case_payload: dict[str, Any] = {
            **asdict(case),
            "target_path": str(target_path),
            "draft_path": str(draft_path),
            "target_present": target_path.exists(),
            "draft_present": draft_path.exists(),
            "prompt_md5": md5_bytes(case.prompt.encode("utf-8")),
            "runs": [],
        }
        if args.hash_models and target_path.exists():
            case_payload["target_md5"] = md5_file(target_path)
        if args.hash_models and draft_path.exists():
            case_payload["draft_md5"] = md5_file(draft_path)
        if not target_path.exists() or not draft_path.exists():
            if not target_path.exists():
                missing.append(str(target_path))
            if not draft_path.exists():
                missing.append(str(draft_path))
            payload["cases"].append(case_payload)
            continue
        for run_idx in range(1, args.runs + 1):
            case_payload["runs"].append(
                run_case(exe, models_dir, case, run_idx, args.ctx, args.kv_mode, args.timeout)
            )
        case_payload["summary"] = summarize_runs(case_payload["runs"])
        payload["cases"].append(case_payload)

    if args.out:
        out = Path(args.out)
    else:
        date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq6-dflash.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, out.with_suffix(".md"))
    print(out)

    if missing and args.fail_on_missing:
        for item in missing:
            print(f"missing artifact: {item}", file=sys.stderr)
        return 2
    hard_errors = [
        f"{case['id']} run {run['run']}"
        for case in payload["cases"]
        for run in case.get("runs", [])
        if run["status"] != "ok"
    ]
    if hard_errors:
        for item in hard_errors:
            print(f"hard error: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
