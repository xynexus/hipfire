#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Collect fresh-process AR perf rows for gfx1151 MQ6 readiness.

This is a benchmark evidence helper, not a promotion gate.  It runs the daemon
once per case/run, records prompt md5 and daemon binary md5, and emits a JSON
artifact with per-run stats plus medians.  The default case set compares MQ6
against the MQ4 control on the same prompts used by the coherence matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "release" / "examples" / "daemon"
DEFAULT_MODELS_DIR = Path(os.environ.get("HIPFIRE_MODELS_DIR", Path.home() / ".hipfire" / "models"))
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"

SHEEP_PROMPT = (
    "A farmer has 17 sheep. All but 9 die. How many are left? "
    "Show brief reasoning then state the final number."
)
CAPITAL_PROMPT = "What is the capital of France? Answer in one short sentence."


@dataclass(frozen=True)
class Case:
    id: str
    model_file: str
    format_id: str
    family: str
    prompt: str
    max_tokens: int


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-9b-mq4-sheep", "qwen3.5-9b.mq4", "mq4", "dense", SHEEP_PROMPT, 300),
    Case("qwen35-9b-mq6-sheep", "qwen3.5-9b.mq6", "mq6", "dense", SHEEP_PROMPT, 300),
    Case("qwen35-27b-mq4-cap", "qwen3.5-27b.mq4", "mq4", "dense", CAPITAL_PROMPT, 80),
    Case("qwen35-27b-mq6-cap", "qwen3.5-27b.mq6", "mq6", "dense", CAPITAL_PROMPT, 80),
    Case("qwen35-a3b-mq4-sheep", "qwen3.5-35b-a3b.mq4", "mq4", "moe", SHEEP_PROMPT, 500),
    Case("qwen35-a3b-mq6-sheep", "qwen3.5-35b-a3b.mq6", "mq6", "moe", SHEEP_PROMPT, 500),
    Case("qwen36-a3b-mq4-sheep", "qwen3.6-35b-a3b.mq4", "mq4", "moe", SHEEP_PROMPT, 800),
    Case("qwen36-a3b-mq6-sheep", "qwen3.6-35b-a3b.mq6", "mq6", "moe", SHEEP_PROMPT, 800),
)


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


def parse_daemon_output(raw: str) -> tuple[dict[str, Any] | None, str]:
    done = None
    tokens: list[str] = []
    for line in raw.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") == "done":
            done = item
        elif item.get("type") == "token":
            tokens.append(str(item.get("text", "")))
    return done, "".join(tokens)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [run for run in runs if run["status"] == "ok" and run.get("done")]
    summary: dict[str, Any] = {
        "ok_runs": len(ok_runs),
        "total_runs": len(runs),
    }
    for key in ("tok_s", "prefill_tok_s", "decode_tok_s", "ttft_ms", "prefill_ms", "tokens"):
        values = [float(run["done"][key]) for run in ok_runs if key in run["done"]]
        summary[f"median_{key}"] = median(values)
    return summary


def run_case(exe: Path, model_path: Path, case: Case, run_idx: int, timeout: int) -> dict[str, Any]:
    request = "\n".join(
        (
            json.dumps({"type": "load", "model": str(model_path), "params": {"max_seq": 4096}}),
            json.dumps(
                {
                    "type": "generate",
                    "id": f"{case.id}-run{run_idx}",
                    "prompt": case.prompt,
                    "temperature": 0.0,
                    "max_tokens": case.max_tokens,
                    "repeat_penalty": 1.05,
                }
            ),
            json.dumps({"type": "unload"}),
            "",
        )
    )
    started = time.monotonic()
    proc = subprocess.run(
        [str(exe)],
        cwd=ROOT,
        input=request,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=os.environ.copy(),
    )
    wall_s = time.monotonic() - started
    done, text = parse_daemon_output(proc.stdout)
    panic = next(
        (
            line
            for line in proc.stdout.splitlines()
            if "panicked" in line or "FATAL" in line or "error: " in line
        ),
        "",
    )
    status = "ok"
    if proc.returncode != 0 or not done or int(done.get("tokens", 0)) <= 0 or panic:
        status = "hard_error"
    return {
        "run": run_idx,
        "status": status,
        "exit_code": proc.returncode,
        "wall_s": round(wall_s, 3),
        "done": done,
        "panic": panic,
        "output_prefix": text[:500],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ6 AR Perf Baseline",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- daemon md5: {payload['daemon']['md5']}",
        f"- runs per case: {payload['runs_per_case']}",
        "",
        "| case | format | family | prompt md5 | ok runs | median prefill tok/s | median decode tok/s | median tok/s | median tokens |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        summary = case.get("summary", {})
        lines.append(
            "| {id} | {fmt} | {family} | {prompt_md5} | {ok}/{total} | {prefill} | {decode} | {tok_s} | {tokens} |".format(
                id=case["id"],
                fmt=case["format_id"],
                family=case["family"],
                prompt_md5=case["prompt_md5"],
                ok=summary.get("ok_runs", 0),
                total=summary.get("total_runs", 0),
                prefill=_fmt(summary.get("median_prefill_tok_s")),
                decode=_fmt(summary.get("median_decode_tok_s")),
                tok_s=_fmt(summary.get("median_tok_s")),
                tokens=_fmt(summary.get("median_tokens")),
            )
        )
    lines.extend(["", "Hard errors:"])
    hard = [
        f"- {case['id']} run {run['run']}: {run['panic'] or run.get('output_prefix', '')[:120]}"
        for case in payload["cases"]
        for run in case.get("runs", [])
        if run["status"] != "ok"
    ]
    lines.extend(hard or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq6-ar-perf.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--pretty", action="store_true", help="Accepted for refresh-plan CLI consistency")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in DEFAULT_CASES],
        help="Limit to a case id; repeatable. Defaults to the full MQ4/MQ6 set.",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"daemon binary not found: {exe}")
    models_dir = Path(args.models_dir)
    selected_ids = set(args.case)
    cases = [case for case in DEFAULT_CASES if not selected_ids or case.id in selected_ids]

    payload: dict[str, Any] = {
        "schema": "hipfire.mq6_ar_perf.gfx1151.v0",
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": detect_arch(),
        "runs_per_case": args.runs,
        "models_dir": str(models_dir),
        "daemon": {
            "path": str(exe),
            "md5": md5_file(exe),
        },
        "cases": [],
    }

    missing = []
    for case in cases:
        model_path = models_dir / case.model_file
        case_payload: dict[str, Any] = {
            **asdict(case),
            "model_path": str(model_path),
            "model_present": model_path.exists(),
            "prompt_md5": md5_bytes(case.prompt.encode("utf-8")),
            "runs": [],
        }
        if not model_path.exists():
            missing.append(str(model_path))
            payload["cases"].append(case_payload)
            continue
        for run_idx in range(1, args.runs + 1):
            print(f"{case.id} run {run_idx}/{args.runs}", file=sys.stderr, flush=True)
            case_payload["runs"].append(run_case(exe, model_path, case, run_idx, args.timeout))
        case_payload["summary"] = summarize_runs(case_payload["runs"])
        payload["cases"].append(case_payload)

    if args.out:
        out = Path(args.out)
    else:
        date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq6-ar-perf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, out.with_suffix(".md"))
    print(out)

    if missing and args.fail_on_missing:
        for item in missing:
            print(f"missing model: {item}", file=sys.stderr)
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
