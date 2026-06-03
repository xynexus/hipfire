#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Run broader MQ3 A3B coherence prompts against MQ4 controls on gfx1151.

This is quality evidence, not a promotion gate.  It extends the single sheep
prompt in `coherence-gate.sh --full` with committed prompt fixtures and records
daemon md5, prompt md5s, raw outputs, and hard-error status in one artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mq6_ar_perf_baseline import (
    DEFAULT_EXE,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT_DIR,
    detect_arch,
    git_value,
    md5_file,
    parse_daemon_output,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "benchmarks" / "prompts"


@dataclass(frozen=True)
class ModelCase:
    id: str
    model_file: str
    format_id: str
    family: str


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt_file: str
    max_tokens: int


@dataclass(frozen=True)
class Case:
    id: str
    model: ModelCase
    prompt: PromptCase


MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase("qwen35-a3b-mq4", "qwen3.5-35b-a3b.mq4", "mq4", "moe"),
    ModelCase("qwen35-a3b-mq3", "qwen3.5-35b-a3b.mq3", "mq3", "moe"),
    ModelCase("qwen36-a3b-mq4", "qwen3.6-35b-a3b.mq4", "mq4", "moe"),
    ModelCase("qwen36-a3b-mq3", "qwen3.6-35b-a3b.mq3", "mq3", "moe"),
)

PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase("trains", "trains-meet.txt", 260),
    PromptCase("humaneval3", "humaneval_3_below_zero.txt", 260),
    PromptCase("long-lru", "coherence_lloyd_long.txt", 220),
)

DEFAULT_CASES: tuple[Case, ...] = tuple(
    Case(f"{model.id}-{prompt.id}", model, prompt)
    for prompt in PROMPT_CASES
    for model in MODEL_CASES
)


def load_prompt(prompt: PromptCase) -> tuple[Path, str]:
    path = PROMPT_DIR / prompt.prompt_file
    return path, path.read_text(encoding="utf-8")


def run_case(exe: Path, models_dir: Path, case: Case, timeout: int) -> dict[str, Any]:
    model_path = models_dir / case.model.model_file
    prompt_path, prompt = load_prompt(case.prompt)
    payload: dict[str, Any] = {
        "id": case.id,
        "model": asdict(case.model),
        "prompt": asdict(case.prompt),
        "model_path": str(model_path),
        "model_present": model_path.exists(),
        "prompt_path": str(prompt_path),
        "prompt_present": prompt_path.exists(),
        "prompt_md5": md5_file(prompt_path) if prompt_path.exists() else None,
    }
    if not model_path.exists():
        payload["status"] = "missing_model"
        return payload
    if not prompt_path.exists():
        payload["status"] = "missing_prompt"
        return payload

    request = "\n".join(
        (
            json.dumps({"type": "load", "model": str(model_path), "params": {"max_seq": 4096}}),
            json.dumps(
                {
                    "type": "generate",
                    "id": case.id,
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": case.prompt.max_tokens,
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
    payload.update(
        {
            "status": status,
            "exit_code": proc.returncode,
            "wall_s": round(wall_s, 3),
            "done": done,
            "hit_max_tokens": bool(done and int(done.get("tokens", 0)) >= case.prompt.max_tokens),
            "panic": panic,
            "output": text,
            "log_tail": proc.stdout[-4000:],
        }
    )
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ3 A3B Broader Coherence",
        "",
        f"- date: {payload['date']}",
        f"- commit: {payload['commit']}",
        f"- branch: {payload['branch']}",
        f"- arch: {payload['arch']}",
        f"- daemon md5: {payload['daemon']['md5']}",
        "",
        "| case | format | prompt | status | tokens | tok/s | prefill tok/s | decode tok/s | prompt md5 | hit cap |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for case in payload["cases"]:
        done = case.get("done") or {}
        lines.append(
            "| {id} | {fmt} | {prompt} | {status} | {tokens} | {tok_s} | {prefill} | {decode} | {prompt_md5} | {hit_cap} |".format(
                id=case["id"],
                fmt=case["model"]["format_id"],
                prompt=case["prompt"]["id"],
                status=case["status"],
                tokens=_fmt(done.get("tokens")),
                tok_s=_fmt(done.get("tok_s")),
                prefill=_fmt(done.get("prefill_tok_s")),
                decode=_fmt(done.get("decode_tok_s")),
                prompt_md5=case.get("prompt_md5") or "",
                hit_cap="yes" if case.get("hit_max_tokens") else "no",
            )
        )
    lines.extend(["", "Hard errors:"])
    hard = [f"- {case['id']}: {case.get('panic') or case['status']}" for case in payload["cases"] if case["status"] == "hard_error"]
    lines.extend(hard or ["- none"])
    lines.append("")
    lines.append("Outputs:")
    for case in payload["cases"]:
        lines.extend(
            [
                "",
                f"## {case['id']}",
                "",
                f"- model: `{case['model']['model_file']}`",
                f"- prompt: `@{case['prompt']['prompt_file']}`",
                "",
                "```text",
                case.get("output", ""),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--out", help="Output JSON path; defaults to benchmarks/results/gfx1151-quant-readiness/<utc-date>-mq3-a3b-broader-coherence.json")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.id for case in DEFAULT_CASES],
        help="Limit to a case id; repeatable. Defaults to all MQ4/MQ3 A3B prompt pairs.",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"daemon binary not found: {exe}")
    models_dir = Path(args.models_dir)
    selected_ids = set(args.case)
    cases = [case for case in DEFAULT_CASES if not selected_ids or case.id in selected_ids]

    payload: dict[str, Any] = {
        "schema": "hipfire.mq3_a3b_coherence.gfx1151.v0",
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": detect_arch(),
        "models_dir": str(models_dir),
        "daemon": {
            "path": str(exe),
            "md5": md5_file(exe),
        },
        "cases": [],
    }

    for case in cases:
        print(case.id, file=sys.stderr, flush=True)
        payload["cases"].append(run_case(exe, models_dir, case, args.timeout))

    if args.out:
        out = Path(args.out)
    else:
        date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DEFAULT_OUT_DIR / f"{date_slug}-mq3-a3b-broader-coherence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, out.with_suffix(".md"))
    print(out)

    missing = [case for case in payload["cases"] if case["status"].startswith("missing_")]
    hard_errors = [case["id"] for case in payload["cases"] if case["status"] == "hard_error"]
    if missing and args.fail_on_missing:
        for item in missing:
            print(f"missing: {item['id']} {item['status']}", file=sys.stderr)
        return 2
    if hard_errors:
        for item in hard_errors:
            print(f"hard error: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
