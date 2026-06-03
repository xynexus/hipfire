#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Collect fresh-process AR perf rows for gfx1151 MQ3 readiness.

This helper records prompt md5, daemon binary md5, exact per-run output
summaries, and medians for the MQ3 size-gated fixtures that have current
coherence evidence. It compares MQ3 against the MQ4 control and is evidence for
the speed-baseline lane only, not a standalone promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mq6_ar_perf_baseline import (
    CAPITAL_PROMPT,
    DEFAULT_EXE,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT_DIR,
    SHEEP_PROMPT,
    Case,
    detect_arch,
    git_value,
    md5_bytes,
    md5_file,
    run_case,
    summarize_runs,
)


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-27b-mq4-cap", "qwen3.5-27b.mq4", "mq4", "dense", CAPITAL_PROMPT, 80),
    Case("qwen35-27b-mq3-cap", "qwen3.5-27b.mq3", "mq3", "dense", CAPITAL_PROMPT, 80),
    Case("qwen35-a3b-mq4-sheep", "qwen3.5-35b-a3b.mq4", "mq4", "moe", SHEEP_PROMPT, 500),
    Case("qwen35-a3b-mq3-sheep", "qwen3.5-35b-a3b.mq3", "mq3", "moe", SHEEP_PROMPT, 500),
    Case("qwen36-a3b-mq4-sheep", "qwen3.6-35b-a3b.mq4", "mq4", "moe", SHEEP_PROMPT, 800),
    Case("qwen36-a3b-mq3-sheep", "qwen3.6-35b-a3b.mq3", "mq3", "moe", SHEEP_PROMPT, 800),
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ3 AR Perf Baseline",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- daemon md5: {payload['daemon']['md5']}",
        f"- runs per case: {payload['runs_per_case']}",
        "",
        "| case | format | family | prompt md5 | ok runs | median tok/s | median decode tok/s | median prefill tok/s | median ttft ms | median tokens |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        summary = case.get("summary", {})
        lines.append(
            "| {id} | {fmt} | {family} | {prompt_md5} | {ok}/{total} | {tok_s} | {decode} | {prefill} | {ttft} | {tokens} |".format(
                id=case["id"],
                fmt=case["format_id"],
                family=case["family"],
                prompt_md5=case["prompt_md5"],
                ok=summary.get("ok_runs", 0),
                total=summary.get("total_runs", 0),
                tok_s=_fmt(summary.get("median_tok_s")),
                decode=_fmt(summary.get("median_decode_tok_s")),
                prefill=_fmt(summary.get("median_prefill_tok_s")),
                ttft=_fmt(summary.get("median_ttft_ms")),
                tokens=_fmt(summary.get("median_tokens")),
            )
        )
    lines.extend(["", "Hard errors:"])
    hard = [
        f"- {case['id']} run {run['run']}: {run['panic'] or 'missing done'}"
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
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq3-ar-perf.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in DEFAULT_CASES],
        help="Limit to a case id; repeatable. Defaults to the full MQ4/MQ3 AR set.",
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
        "schema": "hipfire.mq3_ar_perf.gfx1151.v0",
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
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq3-ar-perf.json"
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
