#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Collect 27B DFlash/spec rows for gfx1151 MQ3 readiness.

The current dense MQ3 size gate leaves 27B as the only dense fixture with clean
single-prompt coherence. This helper compares the MQ3 target against the MQ4
control while using the same MQ4 DFlash draft, and records prompt md5, binary
md5, exact commands, parsed metrics, and token-attractor checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mq6_dflash_baseline import (
    CODE_PROMPT,
    DEFAULT_DRAFT,
    DEFAULT_EXE,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT_DIR,
    PROSE_PROMPT,
    Case,
    detect_arch,
    git_value,
    md5_bytes,
    md5_file,
    run_case,
    summarize_runs,
)


DEFAULT_CASES: tuple[Case, ...] = (
    Case("qwen35-27b-mq4-dflash-prose", "qwen3.5-27b.mq4", "mq4", DEFAULT_DRAFT, "prose", PROSE_PROMPT, 192),
    Case("qwen35-27b-mq3-dflash-prose", "qwen3.5-27b.mq3", "mq3", DEFAULT_DRAFT, "prose", PROSE_PROMPT, 192),
    Case("qwen35-27b-mq4-dflash-code", "qwen3.5-27b.mq4", "mq4", DEFAULT_DRAFT, "code", CODE_PROMPT, 128),
    Case("qwen35-27b-mq3-dflash-code", "qwen3.5-27b.mq3", "mq3", DEFAULT_DRAFT, "code", CODE_PROMPT, 128),
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ3 DFlash Baseline",
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
        "|---|---|---|---:|---:|---:|---:|---:|",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq3-dflash.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in DEFAULT_CASES],
        help="Limit to a case id; repeatable. Defaults to the full MQ4/MQ3 DFlash set.",
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
        "schema": "hipfire.mq3_dflash.gfx1151.v0",
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
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq3-dflash.json"
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
