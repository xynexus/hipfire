#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize MQ2 dense-readiness status for gfx1151.

Plain MQ2 is allowed to exist as an ablation and fallback/admission smoke, but
the current dense Qwen sweep is quality-rejection evidence.  This helper parses
that sweep and the candidate provenance artifact so the rejection is explicit
and machine-readable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
SCHEMA = "hipfire.mq2_status.gfx1151.v0"
DEFAULT_PROVENANCE = RESULTS_DIR / "2026-06-03-mq2-artifact-provenance.json"
DEFAULT_SWEEP = RESULTS_DIR / "2026-06-03-mq2-sweep.md"
DEFAULT_OUT = RESULTS_DIR / "2026-06-03-mq2-status.json"

REQUIRED_DENSE_ARTIFACTS = (
    "qwen3.5-0.8b-mq2.hfq",
    "qwen3.5-4b-mq2.hfq",
    "qwen3.5-9b-mq2.hfq",
)

PROMPT_EXPECTATIONS = {
    "cap": ("paris",),
    "code": ("def square", "return"),
    "reason": ("sheep", "9"),
    "longform": ("deep learning", "traditional"),
}


SECTION_RE = re.compile(r"^## (?P<model>qwen3\.5-[^\s]+\.mq2)\s*$")
PROMPT_RE = re.compile(r"^### (?P<prompt>[a-z0-9_-]+)\s*$")
STATUS_RE = re.compile(r"- wall: .* status: \*\*(?P<status>[^*]+)\*\*")
STATS_RE = re.compile(r"- stats: `(?P<json>\{.*\})`")
PROMPT_FILE_RE = re.compile(r"- prompt-file: `(?P<path>[^`]+)`\s+md5=`(?P<md5>[0-9a-f]+)`")
MANIFEST_RE = re.compile(r"- `(?P<name>[^`]+)` md5=`(?P<md5>[0-9a-f]+)` size=(?P<size>\d+)B")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    artifacts = []
    for item in payload.get("formats", []):
        if item.get("id") != "mq2":
            continue
        artifacts = item.get("candidate_artifacts", {}).get("artifacts", [])
        break
    names = {artifact.get("name") for artifact in artifacts}
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_artifacts": list(REQUIRED_DENSE_ARTIFACTS),
        "required_artifacts_present": sorted(names & set(REQUIRED_DENSE_ARTIFACTS)),
        "required_artifacts_missing": sorted(set(REQUIRED_DENSE_ARTIFACTS) - names),
        "canonical_dense_artifacts_present": all(name in names for name in REQUIRED_DENSE_ARTIFACTS),
    }


def prompt_manifest(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        match = MANIFEST_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "name": match.group("name"),
                    "md5": match.group("md5"),
                    "size_bytes": int(match.group("size")),
                }
            )
    return rows


def output_block(lines: list[str], start_idx: int) -> tuple[str, int]:
    idx = start_idx
    while idx < len(lines) and lines[idx].strip() != "```":
        idx += 1
    if idx >= len(lines):
        return "", idx
    idx += 1
    collected = []
    while idx < len(lines) and lines[idx].strip() != "```":
        collected.append(lines[idx])
        idx += 1
    return "\n".join(collected).strip(), idx


def expectation_passed(prompt_id: str, output: str) -> bool:
    expected = PROMPT_EXPECTATIONS.get(prompt_id, ())
    lowered = output.lower()
    return bool(expected) and all(token in lowered for token in expected)


def parse_sweep(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows = []
    current_model = None
    current_prompt = None
    current: dict[str, Any] | None = None
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        section = SECTION_RE.match(line)
        if section:
            current_model = section.group("model")
            current_prompt = None
            current = None
            idx += 1
            continue
        prompt = PROMPT_RE.match(line)
        if prompt and current_model:
            current_prompt = prompt.group("prompt")
            current = {
                "model": current_model,
                "prompt_id": current_prompt,
                "status": None,
                "stats": {},
                "prompt_file": None,
                "prompt_md5": None,
                "output": "",
                "expectation_passed": False,
                "collapsed": True,
            }
            rows.append(current)
            idx += 1
            continue
        if current is not None:
            status = STATUS_RE.match(line)
            if status:
                current["status"] = status.group("status")
                idx += 1
                continue
            stats = STATS_RE.match(line)
            if stats:
                try:
                    current["stats"] = json.loads(stats.group("json"))
                except json.JSONDecodeError:
                    current["stats"] = {}
                idx += 1
                continue
            prompt_file = PROMPT_FILE_RE.match(line)
            if prompt_file:
                current["prompt_file"] = prompt_file.group("path")
                current["prompt_md5"] = prompt_file.group("md5")
                idx += 1
                continue
            if line == "**Output:**":
                output, next_idx = output_block(lines, idx + 1)
                current["output"] = output
                current["expectation_passed"] = expectation_passed(str(current_prompt), output)
                current["collapsed"] = not current["expectation_passed"]
                idx = next_idx + 1
                continue
        idx += 1

    hard_errors = [row for row in rows if row["status"] != "OK"]
    collapsed = [row for row in rows if row["collapsed"]]
    passed = [row for row in rows if row["expectation_passed"]]
    return {
        "path": str(path),
        "prompt_manifest": prompt_manifest(text),
        "model_count": len({row["model"] for row in rows}),
        "prompt_count": len({row["prompt_id"] for row in rows}),
        "row_count": len(rows),
        "hard_error_count": len(hard_errors),
        "collapsed_row_count": len(collapsed),
        "quality_pass_row_count": len(passed),
        "all_rows_runtime_ok": not hard_errors,
        "all_rows_collapsed": bool(rows) and len(collapsed) == len(rows),
        "rows": [
            {
                "model": row["model"],
                "prompt_id": row["prompt_id"],
                "status": row["status"],
                "tokens": row["stats"].get("tokens"),
                "prefill_tok_s": row["stats"].get("prefill_tok_s"),
                "decode_tok_s": row["stats"].get("decode_tok_s"),
                "prompt_md5": row["prompt_md5"],
                "expectation_passed": row["expectation_passed"],
                "collapsed": row["collapsed"],
                "output_preview": row["output"][:160],
            }
            for row in rows
        ],
    }


def fixture_quality_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    outcomes = []
    for model, model_rows in sorted(by_model.items()):
        collapsed = sum(1 for row in model_rows if row["collapsed"])
        passed = sum(1 for row in model_rows if row["expectation_passed"])
        total = len(model_rows)
        outcomes.append(
            {
                "model": model,
                "row_count": total,
                "collapsed_row_count": collapsed,
                "quality_pass_row_count": passed,
                "status": (
                    "rejected"
                    if total > 0 and collapsed == total
                    else "passed"
                    if total > 0 and passed == total
                    else "incomplete"
                ),
            }
        )
    return outcomes


def quality_rejection_boundary(
    *,
    gates: dict[str, bool],
    sweep_summary: dict[str, Any],
) -> dict[str, Any]:
    outcomes = fixture_quality_outcomes(sweep_summary["rows"])
    by_model = {item["model"]: item for item in outcomes}
    sub_9b_models = ("qwen3.5-0.8b.mq2", "qwen3.5-4b.mq2")
    sub_9b_failures = all(
        by_model.get(model, {}).get("status") == "rejected" for model in sub_9b_models
    )
    dense_9b_rejected = by_model.get("qwen3.5-9b.mq2", {}).get("status") == "rejected"
    return {
        "dense_text_status": (
            "rejected_all_current_rows_collapsed"
            if gates["all_dense_rows_collapsed"]
            else "quality_clean"
            if gates["dense_quality_clean"]
            else "quality_incomplete"
        ),
        "runtime_smoke_interpretation": (
            "runtime_ok_not_quality_evidence"
            if gates["runtime_smoke_no_hard_errors"]
            else "runtime_smoke_has_hard_errors"
        ),
        "artifact_scope": "dense_0_8b_4b_9b",
        "fixture_quality_outcomes": outcomes,
        "quality_override_required": not gates["dense_quality_clean"],
        "override_requires": [
            "new_mq2_calibration_or_producer_change",
            "bounded_dense_quality_gate_clean",
            "kld_ppl_evidence_against_mq4_or_q8",
            "then_gfx1151_perf_baseline",
        ],
        "sub_9b_failures_explicit": sub_9b_failures,
        "dense_9b_rejected": dense_9b_rejected,
        "promotion_perf_collection_allowed": (
            gates["dense_quality_clean"] and gates["kld_ppl_evidence_present"]
        ),
        "performance_rows_promotable": (
            gates["dense_quality_clean"]
            and gates["kld_ppl_evidence_present"]
            and gates["perf_promotion_evidence_present"]
        ),
        "next_unblocked_step": (
            "produce_new_mq2_calibration_that_passes_bounded_dense_quality"
            if not gates["dense_quality_clean"]
            else "run_kld_ppl_evidence_against_mq4_or_q8"
            if not gates["kld_ppl_evidence_present"]
            else "run_gfx1151_perf_baseline"
            if not gates["perf_promotion_evidence_present"]
            else "verify_readiness_matrix_before_promotion_claim"
        ),
    }


def build_status(provenance: Path = DEFAULT_PROVENANCE, sweep: Path = DEFAULT_SWEEP) -> dict[str, Any]:
    artifacts = artifact_summary(provenance)
    sweep_summary = parse_sweep(sweep)
    dense_quality_clean = (
        sweep_summary["row_count"] > 0
        and sweep_summary["quality_pass_row_count"] == sweep_summary["row_count"]
    )
    gates = {
        "canonical_dense_artifacts_present": artifacts["canonical_dense_artifacts_present"],
        "runtime_smoke_no_hard_errors": sweep_summary["all_rows_runtime_ok"],
        "dense_quality_clean": dense_quality_clean,
        "all_dense_rows_collapsed": sweep_summary["all_rows_collapsed"],
        "kld_ppl_evidence_present": False,
        "perf_promotion_evidence_present": False,
    }
    boundary = quality_rejection_boundary(gates=gates, sweep_summary=sweep_summary)
    promotion_allowed = (
        gates["canonical_dense_artifacts_present"]
        and gates["runtime_smoke_no_hard_errors"]
        and gates["dense_quality_clean"]
        and gates["kld_ppl_evidence_present"]
        and gates["perf_promotion_evidence_present"]
    )
    blockers = []
    if not gates["canonical_dense_artifacts_present"]:
        blockers.append("canonical 0.8B/4B/9B MQ2 artifacts are incomplete")
    if gates["all_dense_rows_collapsed"]:
        blockers.append("all current dense MQ2 sweep rows are quality-collapsed")
    if not gates["kld_ppl_evidence_present"]:
        blockers.append("no MQ2 KLD/PPL evidence clears the dense quality gate")
    if not gates["perf_promotion_evidence_present"]:
        blockers.append("tok/s rows are diagnostic only until quality clears")
    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "format": "mq2",
        "status": "quality-rejected",
        "promotion_allowed": promotion_allowed,
        "artifact_provenance": artifacts,
        "sweep": sweep_summary,
        "gates": gates,
        "quality_rejection_boundary": boundary,
        "blockers": blockers,
        "decision": (
            "keep MQ2 dense-text rejected; runtime OK rows are fallback/admission "
            "smoke only because every bounded dense prompt collapsed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_status(Path(args.provenance), Path(args.sweep))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty) + "\n",
        encoding="utf-8",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
