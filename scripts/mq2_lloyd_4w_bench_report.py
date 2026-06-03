#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Convert MQ2-Lloyd 4-warp benchmark stdout into readiness artifacts.

This helper does not run the GPU benchmark.  It parses stdout from:

    cargo run --release -p rdna-compute --example bench_mq2g256_lloyd_moe_4w

and emits a machine-readable JSON payload plus an optional Markdown summary.
The verdict is intentionally conservative: synthetic kernel A/B evidence can
keep an opt-in path alive, but it cannot promote a model format without
artifact-backed coherence and model-level perf evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
DEFAULT_MODEL_ROOTS = (
    Path("/home/sadara/Models"),
    Path("/home/sadara/.hipfire/models"),
)
SCHEMA = "hipfire.mq2_lloyd_4w_bench.gfx1151.v0"
COMMAND = "cargo run --release -p rdna-compute --example bench_mq2g256_lloyd_moe_4w"
SPECIALTY_SCOPE = {
    "dense_text_status": "rejected_by_mq2_dense_quality_gate",
    "routed_expert_status": "specialty-research-only",
    "default_kernel_change_allowed": False,
    "model_level_evidence_present": False,
    "bandwidth_bottleneck_proven": False,
    "effective_bandwidth_caveat": (
        "Model-level effective-bandwidth shortcuts overcount inactive experts; "
        "the current decode bottleneck is not proven to be pure DDR bandwidth."
    ),
    "likely_model_level_blockers": (
        "launch_overhead",
        "scalar_gemv_occupancy",
        "codebook_unpack_lookup_cost",
        "non_mq2_decode_work",
    ),
    "next_experiments": (
        "batch_selected_experts_across_tokens",
        "reduce_launches_or_fuse_small_decode_steps",
        "profile_scalar_gemv_occupancy_codebook_and_non_mq2_phases",
    ),
}
ORIGIN_RELEVANT_COMMITS = (
    {
        "commit": "89f42e4bb4a4317da586d37741bb23ea90ef2e24",
        "short": "89f42e4b",
        "subject": "fix(forward): flatten nosync if-else chain",
        "remote_ref": "origin/master",
        "impact": "restores reachable mmqload/lloyd_4w/base paths after nosync routing",
    },
    {
        "commit": "d5985c3e51197c70fa804f84cd694abbcd38f0d7",
        "short": "d5985c3e",
        "subject": "fix(stragglers): 4 GPU leaks/dead-doc + GGUF Promote6 Mq4Lloyd",
        "remote_ref": "origin/master",
        "impact": "records current Lloyd-family upstream cleanup context",
    },
    {
        "commit": "edf922db7a4ccd97f40e0afc2e8171984b886fc7",
        "short": "edf922db",
        "subject": "fix(deepseek4): revert MMQLOAD-default -- long-context attractor on mq2lloyd",
        "remote_ref": "origin/minimax/m2.7-impl",
        "impact": "keeps mq2lloyd preload and model-level perf collection behind coherence evidence",
    },
    {
        "commit": "fb36f5395d17b48f2d539f7ba4adc6a6c18b808e",
        "short": "fb36f539",
        "subject": "fix(engine): hunt-3 batch - 16 engine bugs across pp / sampling+MoE / daemon / bun-CLI",
        "remote_ref": "origin/fix/hunt3-engine-bugs",
        "impact": (
            "requires refreshing future MQ2-Lloyd model-backed coherence/perf evidence "
            "against the selected engine branch"
        ),
    },
)

HEADER_RE = re.compile(
    r"^===\s+(?P<label>.*?)\s+\|\s+M=(?P<m>\d+)\s+K=(?P<k>\d+)\s+"
    r"batch=(?P<batch>\d+)\s+m_total=(?P<m_total>\d+)\s+===$"
)
CORRECTNESS_RE = re.compile(
    r"correctness:\s+max_abs=(?P<max_abs>[0-9.eE+-]+)\s+"
    r"max_rel=(?P<max_rel>[0-9.eE+-]+)\s+bad=(?P<bad>\d+)/(?P<total>\d+)\s+"
    r"nan=(?P<nan>\d+)\s+(?P<status>OK|FAIL)"
)
PERF_RE = re.compile(
    r"ref\s+\(1w16x16\):\s+(?P<baseline_us>[0-9.eE+-]+)\s+.*?"
    r"4w\s+\(64x16\):\s+(?P<candidate_us>[0-9.eE+-]+)\s+.*?"
    r"speedup:\s+(?P<speedup>[0-9.eE+-]+)"
)
ARCH_RE = re.compile(r"^Arch:\s+(?P<arch>\S+)\s*$")


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def commit_on_ref(commit: str, ref: str) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, ref],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def origin_context() -> dict[str, Any]:
    commits = []
    for item in ORIGIN_RELEVANT_COMMITS:
        commit = item["commit"]
        remote_ref = item["remote_ref"]
        commits.append(
            {
                **item,
                "present_on_remote_ref": commit_on_ref(commit, remote_ref),
                "present_on_origin_master": commit_on_ref(commit, "origin/master"),
                "present_on_head": commit_on_ref(commit, "HEAD"),
            }
        )
    origin_master_missing = [
        commit["short"]
        for commit in commits
        if commit["present_on_origin_master"] is True and commit["present_on_head"] is False
    ]
    remote_branch_missing = [
        commit["short"]
        for commit in commits
        if commit["present_on_remote_ref"] is True
        and commit["present_on_origin_master"] is False
        and commit["present_on_head"] is False
    ]
    return {
        "checked_remote": "origin",
        "origin_master_commit": git_value(["rev-parse", "origin/master"]),
        "relevant_upstream_commits": commits,
        "origin_master_refresh_required": bool(origin_master_missing),
        "remote_branch_refresh_required": bool(remote_branch_missing),
        "model_evidence_refresh_required": bool(origin_master_missing or remote_branch_missing),
        "origin_master_missing_commit_shorts": origin_master_missing,
        "remote_branch_missing_commit_shorts": remote_branch_missing,
    }


def md5_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".locks", "__pycache__"}]
        current = Path(dirpath)
        for filename in filenames:
            yield current / filename


def is_mq2_lloyd_candidate(path: Path) -> bool:
    name = path.name.lower()
    return "mq2lloyd" in name or ("mq2" in name and "lloyd" in name)


def is_current_a3b_or_deepseek_candidate(path: Path) -> bool:
    name = path.name.lower()
    return is_mq2_lloyd_candidate(path) and ("a3b" in name or "deepseek" in name)


def artifact_inventory(roots: list[Path] | tuple[Path, ...] = DEFAULT_MODEL_ROOTS) -> dict[str, Any]:
    searched = []
    matches = []
    current_matches = []
    for root in roots:
        root = root.expanduser()
        searched.append({"root": str(root), "exists": root.exists()})
        for path in iter_files(root):
            if not path.is_file() or not is_mq2_lloyd_candidate(path):
                continue
            record = {
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "current_a3b_or_deepseek": is_current_a3b_or_deepseek_candidate(path),
            }
            matches.append(record)
            if record["current_a3b_or_deepseek"]:
                current_matches.append(record)
    matches = sorted(matches, key=lambda item: item["path"])
    current_matches = sorted(current_matches, key=lambda item: item["path"])
    return {
        "searched_roots": searched,
        "mq2_lloyd_artifacts": matches,
        "current_a3b_or_deepseek_artifacts": current_matches,
        "mq2_lloyd_artifact_present": bool(matches),
        "current_a3b_or_deepseek_artifact_present": bool(current_matches),
    }


def phase_for_label(label: str) -> str:
    lowered = label.lower()
    if "gate/up" in lowered or "gate_up" in lowered:
        return "gate_up"
    if lowered.startswith("down") or " down" in lowered:
        return "down"
    return "unknown"


def normalize_id(label: str, batch: int) -> str:
    return f"{phase_for_label(label)}_b{batch}"


def parse_benchmark_output(raw: str) -> tuple[str | None, list[dict[str, Any]]]:
    arch = None
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        arch_match = ARCH_RE.match(stripped)
        if arch_match:
            arch = arch_match.group("arch")
            continue

        header_match = HEADER_RE.match(stripped)
        if header_match:
            if current:
                rows.append(current)
            label = header_match.group("label")
            batch = int(header_match.group("batch"))
            current = {
                "id": normalize_id(label, batch),
                "label": label,
                "phase": phase_for_label(label),
                "m": int(header_match.group("m")),
                "k": int(header_match.group("k")),
                "batch": batch,
                "m_total": int(header_match.group("m_total")),
            }
            continue

        if current is None:
            continue

        correctness_match = CORRECTNESS_RE.search(stripped)
        if correctness_match:
            current["correctness"] = {
                "status": "ok" if correctness_match.group("status") == "OK" else "fail",
                "max_abs": float(correctness_match.group("max_abs")),
                "max_rel": float(correctness_match.group("max_rel")),
                "bad": int(correctness_match.group("bad")),
                "total": int(correctness_match.group("total")),
                "nan": int(correctness_match.group("nan")),
            }
            continue

        perf_match = PERF_RE.search(stripped)
        if perf_match:
            baseline_us = float(perf_match.group("baseline_us"))
            candidate_us = float(perf_match.group("candidate_us"))
            current["baseline_us"] = baseline_us
            current["candidate_4w_us"] = candidate_us
            current["speedup"] = round(baseline_us / candidate_us, 4)
            current["reported_speedup"] = float(perf_match.group("speedup"))

    if current:
        rows.append(current)
    return arch, rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row["speedup"]) for row in rows if "speedup" in row]
    all_correct = all(
        row.get("correctness", {}).get("status") == "ok"
        and row.get("correctness", {}).get("bad") == 0
        and row.get("correctness", {}).get("nan") == 0
        for row in rows
    )
    all_slower = bool(speedups) and all(speedup < 1.0 for speedup in speedups)
    return {
        "row_count": len(rows),
        "all_correct": all_correct,
        "all_candidate_slower_than_baseline": all_slower,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "promote_4w_default": False,
        "default_kernel_change_allowed": False,
        "model_level_promotion_allowed": False,
        "specialty_research_allowed": all_correct,
        "bandwidth_bottleneck_proven": False,
        "recommended_next_lever": "batch_selected_experts_across_tokens_before_more_packing",
        "decision": (
            "keep HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W opt-in/research; synthetic "
            "kernel evidence does not prove model-level readiness"
        ),
    }


def specialty_boundary(
    summary: dict[str, Any],
    scope: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    origin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not summary["all_correct"]:
        synthetic_result = "correctness_failed"
    elif summary["all_candidate_slower_than_baseline"]:
        synthetic_result = "correct_but_slower_than_k2"
    else:
        synthetic_result = "correct_and_faster_than_k2"

    artifacts = artifacts or artifact_inventory()
    origin = origin or origin_context()
    model_evidence_refresh_required = bool(origin["model_evidence_refresh_required"])
    remote_moe_fix_refresh_required = "fb36f539" in origin["remote_branch_missing_commit_shorts"]
    current_artifact_present = bool(artifacts["current_a3b_or_deepseek_artifact_present"])
    routed_coherence_present = False
    model_perf_present = bool(scope["model_level_evidence_present"])
    perf_collection_allowed = (
        current_artifact_present
        and routed_coherence_present
        and model_perf_present
        and synthetic_result == "correct_and_faster_than_k2"
        and not model_evidence_refresh_required
    )
    if not current_artifact_present:
        status = "specialty_blocked_no_model_artifact"
        next_step = "generate_or_locate_a3b_or_deepseek_mq2_lloyd_artifact"
    elif not routed_coherence_present:
        status = "specialty_blocked_no_routed_coherence"
        next_step = "run_routed_expert_coherence"
    elif not model_perf_present:
        status = "specialty_blocked_no_model_perf"
        next_step = "run_model_level_perf_vs_k2"
    elif synthetic_result != "correct_and_faster_than_k2":
        status = "specialty_blocked_no_model_backed_win"
        next_step = "produce_model_backed_speedup_that_preserves_coherence"
    elif model_evidence_refresh_required:
        status = "specialty_blocked_origin_refresh"
        next_step = "reconcile_remote_engine_fixes_before_model_perf_claims"
    else:
        status = "specialty_perf_collection_ready"
        next_step = "evaluate_readiness_matrix_before_specialty_promotion"

    return {
        "status": status,
        "dense_text_status": scope["dense_text_status"],
        "routed_expert_status": scope["routed_expert_status"],
        "synthetic_kernel_result": synthetic_result,
        "current_a3b_or_deepseek_artifact_present": current_artifact_present,
        "mq2_lloyd_artifact_present": bool(artifacts["mq2_lloyd_artifact_present"]),
        "artifact_search_roots": artifacts["searched_roots"],
        "routed_expert_coherence_present": routed_coherence_present,
        "model_level_perf_evidence_present": model_perf_present,
        "model_level_promotion_allowed": summary["model_level_promotion_allowed"],
        "default_kernel_change_allowed": summary["default_kernel_change_allowed"],
        "bandwidth_bottleneck_proven": summary["bandwidth_bottleneck_proven"],
        "requires_model_backed_win_over_k2": True,
        "origin_master_refresh_required": bool(origin["origin_master_refresh_required"]),
        "remote_branch_refresh_required": bool(origin["remote_branch_refresh_required"]),
        "remote_moe_fix_refresh_required": remote_moe_fix_refresh_required,
        "model_evidence_refresh_required": model_evidence_refresh_required,
        "perf_collection_allowed": perf_collection_allowed,
        "synthetic_kernel_promotes_default": summary["promote_4w_default"],
        "next_unblocked_step": next_step,
        "required_next_gates": [
            "reconcile_remote_engine_fixes_before_model_perf_claims",
            "current_a3b_or_deepseek_mq2_lloyd_artifact",
            "routed_expert_coherence",
            "model_level_perf_vs_k2",
            "model_backed_speedup_preserves_coherence",
        ],
        "likely_model_level_blockers": list(scope["likely_model_level_blockers"]),
        "origin_evidence_summary": (
            "origin contains lloyd_4w dispatch reachability cleanup and an mq2lloyd "
            "long-context attractor revert; origin/fix/hunt3-engine-bugs adds "
            "sampling+MoE engine fixes that require refreshed model-backed evidence. "
            "Keep research paths gated by coherence."
        ),
    }


def normalize_payload(payload: dict[str, Any], model_roots: list[Path] | None = None) -> dict[str, Any]:
    summary = payload["summary"]
    scope = payload["specialty_scope"]
    artifacts = artifact_inventory(model_roots or list(DEFAULT_MODEL_ROOTS))
    origin = origin_context()
    payload["model_artifact_inventory"] = artifacts
    payload["origin_context"] = origin
    payload["specialty_boundary"] = specialty_boundary(summary, scope, artifacts, origin)
    payload["status"] = payload["specialty_boundary"]["status"]
    payload["promotion_allowed"] = payload["specialty_boundary"]["model_level_promotion_allowed"]
    return payload


def build_payload(
    raw: str,
    *,
    arch: str | None = None,
    benchmark_binary: Path | None = None,
    model_roots: list[Path] | None = None,
) -> dict[str, Any]:
    parsed_arch, rows = parse_benchmark_output(raw)
    if not rows:
        raise ValueError("no benchmark rows parsed")
    detected_arch = arch or parsed_arch or "unknown"
    summary = summarize(rows)
    scope = dict(SPECIALTY_SCOPE)
    artifacts = artifact_inventory(model_roots or list(DEFAULT_MODEL_ROOTS))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "date": datetime.now(timezone.utc).isoformat(),
        "arch": detected_arch,
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "command": COMMAND,
        "scope": "synthetic DeepSeek V4 MoE hot-path shapes; kernel-only evidence",
        "baseline_kernel": "gemm_mq2g256_lloyd_moe_grouped_wmma_k2",
        "candidate_kernel": "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2",
        "rows": rows,
        "specialty_scope": scope,
        "summary": summary,
        "model_artifact_inventory": artifacts,
    }
    origin = origin_context()
    payload["origin_context"] = origin
    payload["specialty_boundary"] = specialty_boundary(summary, scope, artifacts, origin)
    payload["status"] = payload["specialty_boundary"]["status"]
    payload["promotion_allowed"] = payload["specialty_boundary"]["model_level_promotion_allowed"]
    if benchmark_binary:
        payload["benchmark_binary"] = {
            "path": str(benchmark_binary),
            "md5": md5_file(benchmark_binary),
        }
    return payload


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MQ2-Lloyd 4-Warp Kernel A/B",
        "",
        f"- schema: `{payload['schema']}`",
        f"- arch: `{payload['arch']}`",
        f"- command: `{payload['command']}`",
        f"- baseline: `{payload['baseline_kernel']}`",
        f"- candidate: `{payload['candidate_kernel']}`",
        f"- decision: {payload['summary']['decision']}",
        f"- routed-expert scope: `{payload['specialty_scope']['routed_expert_status']}`",
        f"- specialty boundary: `{payload['specialty_boundary']['status']}`",
        f"- next unblocked step: `{payload['specialty_boundary']['next_unblocked_step']}`",
        f"- next lever: `{payload['summary']['recommended_next_lever']}`",
        "",
        "| row | correctness | baseline us | 4-warp us | speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        correctness = row.get("correctness", {})
        status = correctness.get("status", "missing")
        if correctness:
            status = (
                f"{status}, max_abs={correctness['max_abs']:.3e}, "
                f"bad={correctness['bad']}, nan={correctness['nan']}"
            )
        lines.append(
            "| {label} | {status} | {baseline:.1f} | {candidate:.1f} | {speedup:.4f} |".format(
                label=row["label"],
                status=status,
                baseline=row["baseline_us"],
                candidate=row["candidate_4w_us"],
                speedup=row["speedup"],
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Raw stdout captured from the Rust benchmark")
    source.add_argument("--input-json", help="Existing JSON artifact to normalize without rerunning benchmark")
    parser.add_argument("--out-json", help="Output JSON path")
    parser.add_argument("--out-md", help="Optional Markdown output path")
    parser.add_argument("--arch", help="Override detected arch")
    parser.add_argument("--model-root", action="append", default=[], help="Model root to scan for MQ2-Lloyd artifacts; repeatable")
    parser.add_argument(
        "--benchmark-binary",
        default=str(ROOT / "target" / "release" / "examples" / "bench_mq2g256_lloyd_moe_4w"),
    )
    args = parser.parse_args()

    model_roots = [Path(path) for path in args.model_root] if args.model_root else list(DEFAULT_MODEL_ROOTS)
    if args.input_json:
        payload = normalize_payload(
            json.loads(Path(args.input_json).read_text(encoding="utf-8")),
            model_roots=model_roots,
        )
    else:
        raw = Path(args.input).read_text(encoding="utf-8")
        binary = Path(args.benchmark_binary) if args.benchmark_binary else None
        payload = build_payload(raw, arch=args.arch, benchmark_binary=binary, model_roots=model_roots)

    out_json = Path(args.out_json) if args.out_json else DEFAULT_OUT_DIR / "mq2-lloyd-4w-bench.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.out_md:
        write_markdown(payload, Path(args.out_md))
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
