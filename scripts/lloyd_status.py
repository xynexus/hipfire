#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize Lloyd-format readiness decisions for gfx1151.

MQ3-Lloyd and MQ4-Lloyd have very different current failure modes.  MQ3-Lloyd
has coherent 9B smoke output but loses the current BF16-referenced KLD gate.
MQ4-Lloyd is container-valid, but the current 9B artifact emits an obvious
token attractor and should not receive KLD or perf time until recalibrated.
This helper keeps those decisions machine-readable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
SCHEMA = "hipfire.lloyd_status.gfx1151.v0"
DEFAULT_OUT = RESULTS_DIR / "2026-06-03-lloyd-status.json"
DEFAULT_MODEL_ROOTS = (
    Path("/home/sadara/Models"),
    Path("/home/sadara/.hipfire/models"),
)

DEFAULT_PATHS = {
    "mq3_provenance": RESULTS_DIR / "2026-06-03-mq3-lloyd-artifact-provenance.json",
    "mq4_provenance": RESULTS_DIR / "2026-06-03-mq4-lloyd-artifact-provenance.json",
    "mq3_coherence": RESULTS_DIR / "2026-06-03-coherence-full-after-mq3-lloyd.md",
    "mq4_coherence": RESULTS_DIR / "2026-06-03-coherence-full-after-mq4-lloyd.md",
    "mq3_kld": RESULTS_DIR / "2026-06-03-mq3-lloyd-9b-kld.json",
    "mq6_c512_kld": RESULTS_DIR / "2026-06-03-mq6-kld-c512.json",
    "mq4_current_kld": RESULTS_DIR / "2026-06-03-mq4-lloyd-9b-kld-c20.json",
    "mq3_historical_kld": RESULTS_DIR / "2026-06-03-lloyd-historical-2026-05-08-kld.json",
    "mq4_historical_kld": RESULTS_DIR / "2026-06-03-lloyd-historical-2026-05-13-kld.json",
    "mq4_container_audit": RESULTS_DIR / "2026-06-03-mq4-lloyd-container-audit.md",
}

ORIGIN_RELEVANT_COMMITS = (
    {
        "commit": "d5985c3e51197c70fa804f84cd694abbcd38f0d7",
        "short": "d5985c3e",
        "subject": "fix(stragglers): 4 GPU leaks/dead-doc + GGUF Promote6 Mq4Lloyd",
        "formats": ["mq4-lloyd", "mq6"],
        "impact": (
            "fixes GGUF --kmap-promote 6 for Mq4Lloyd so Promote6 emits "
            "MQ6G256 instead of silently staying MQ4G256Lloyd"
        ),
        "promotion_effect": "producer-flow fix only; does not repair the current coherence-rejected MQ4-Lloyd artifact",
    },
    {
        "commit": "5d09c2ee8a08ab99417ee12cff24ec04b1ba6a90",
        "short": "5d09c2ee",
        "subject": "perf(gate_up): barrier-free MQ4-Lloyd WMMA gate_up variants",
        "formats": ["mq4-lloyd"],
        "impact": "adds opt-in MQ4-Lloyd gate_up perf kernels for post-quality validation",
        "promotion_effect": "perf context only; speed rows remain blocked until coherence passes",
    },
    {
        "commit": "46898ecd8b1a96cef295924998f114f8b11748d0",
        "short": "46898ecd",
        "subject": "perf(gate_up): barrier-free MQ3-Lloyd WMMA gate_up variants",
        "formats": ["mq3-lloyd"],
        "impact": "adds opt-in MQ3-Lloyd gate_up perf kernels for post-quality validation",
        "promotion_effect": "perf context only; current MQ3-Lloyd KLD still loses to MQ4",
    },
    {
        "commit": "3663934fa9a78afb28ead03ef4362cffae344c35",
        "short": "3663934f",
        "subject": "fix(gate-up): make nosync variants buildable and opt-in",
        "formats": ["mq3-lloyd", "mq4-lloyd"],
        "impact": "wires parked Lloyd nosync kernels behind HIPFIRE_GATE_UP_NOSYNC=1",
        "promotion_effect": "opt-in perf lever only; not promoted evidence",
    },
    {
        "commit": "aa781d742435c8dddcb0d3d212bdaeb5636d910a",
        "short": "aa781d74",
        "subject": "fix(gate_up): arch-gate default variant",
        "formats": ["mq3-lloyd", "mq4-lloyd"],
        "impact": "defaults gfx1151 gate_up routing to nosync variants while keeping ldscoop elsewhere",
        "promotion_effect": "post-quality perf context only; current Lloyd artifacts remain quality-gated",
    },
    {
        "commit": "89f42e4bb4a4317da586d37741bb23ea90ef2e24",
        "short": "89f42e4b",
        "subject": "fix(forward): flatten nosync if-else chain",
        "formats": ["mq2-lloyd", "mq3-lloyd", "mq4-lloyd"],
        "impact": "restores reachable mmqload/lloyd_4w/base paths after nosync routing",
        "promotion_effect": "runtime routing cleanup; current quality gates still decide promotion",
    },
)

MQ4_LLOYD_PROMOTION_REQUIRED_GATES = (
    "canonical_9b_artifact_present",
    "non_9b_artifact_present",
    "container_qtype_named",
    "container_bounded",
    "coherence_9b_clean",
    "current_same_run_kld_present",
    "current_same_run_candidate_valid",
    "current_same_run_beats_mq4",
    "current_same_run_beats_mq6",
    "perf_evidence_present",
)
MQ4_LLOYD_PROMOTION_FORBIDDEN_GATES = (
    "current_same_run_invalid_zero_kld_no_ppl",
    "origin_refresh_required",
)


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


def git_ahead_behind(base: str = "HEAD", upstream: str = "origin/master") -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", f"{base}...{upstream}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        local_only, upstream_only = [int(part) for part in output.split()]
    except Exception:
        local_only = None
        upstream_only = None
    return {
        "base": base,
        "upstream": upstream,
        "local_only_commits": local_only,
        "upstream_only_commits": upstream_only,
        "upstream_reconciliation_required": bool(local_only or upstream_only),
        "origin_master_commit": git_value(["rev-parse", "origin/master"]),
    }


def commit_on_ref(commit: str, ref: str = "origin/master") -> bool | None:
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
    upstream = git_ahead_behind()
    commits = []
    refresh_by_format = {
        "mq3-lloyd": False,
        "mq4-lloyd": False,
    }
    for item in ORIGIN_RELEVANT_COMMITS:
        present_on_origin = commit_on_ref(item["commit"])
        present_on_head = commit_on_ref(item["commit"], "HEAD")
        if present_on_origin is True and present_on_head is False:
            for format_id in item["formats"]:
                if format_id in refresh_by_format:
                    refresh_by_format[format_id] = True
        commits.append(
            {
                **item,
                "present_on_origin_master": present_on_origin,
                "present_on_head": present_on_head,
            }
        )
    return {
        **upstream,
        "relevant_upstream_commits": commits,
        "format_refresh_required": refresh_by_format,
        "lloyd_evidence_refresh_required": any(refresh_by_format.values()),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


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


def lloyd_format_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if "lloyd" not in name:
        return None
    if "mq3" in name:
        return "mq3-lloyd"
    if "mq4" in name:
        return "mq4-lloyd"
    return None


def artifact_scope_flags(artifacts: list[dict[str, Any]]) -> dict[str, bool]:
    names = [str(artifact.get("name", "")).lower() for artifact in artifacts]
    return {
        "dense_4b_artifact_present": any("4b" in name for name in names),
        "dense_9b_artifact_present": any("9b" in name for name in names),
        "dense_27b_artifact_present": any("27b" in name for name in names),
        "a3b_artifact_present": any("a3b" in name for name in names),
        "only_9b_artifact_present": bool(names)
        and any("9b" in name for name in names)
        and not any(("4b" in name or "27b" in name or "a3b" in name) for name in names),
    }


def lloyd_artifact_inventory(roots: list[Path] | tuple[Path, ...] = DEFAULT_MODEL_ROOTS) -> dict[str, Any]:
    searched_roots = []
    formats = {
        "mq3-lloyd": {"artifacts": []},
        "mq4-lloyd": {"artifacts": []},
    }
    for root in roots:
        root = root.expanduser()
        searched_roots.append({"root": str(root), "exists": root.exists()})
        for path in iter_files(root):
            if not path.is_file():
                continue
            format_id = lloyd_format_from_name(path)
            if format_id is None:
                continue
            formats[format_id]["artifacts"].append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                }
            )
    for record in formats.values():
        record["artifacts"] = sorted(record["artifacts"], key=lambda item: item["path"])
        record["artifact_count"] = len(record["artifacts"])
        record.update(artifact_scope_flags(record["artifacts"]))
    return {
        "searched_roots": searched_roots,
        "formats": formats,
    }


def format_record(provenance: dict[str, Any], format_id: str) -> dict[str, Any]:
    for item in provenance.get("formats", []):
        if item.get("id") == format_id:
            artifacts = item.get("candidate_artifacts", {}).get("artifacts", [])
            return {
                "status": item.get("status"),
                "artifacts": artifacts,
                "artifact_count": len(artifacts),
                "canonical_9b_artifact_present": any("qwen3.5-9b" in artifact.get("name", "") for artifact in artifacts),
                "non_9b_artifact_present": any("qwen3.5-9b" not in artifact.get("name", "") for artifact in artifacts),
            }
    return {
        "status": "missing",
        "artifacts": [],
        "artifact_count": 0,
        "canonical_9b_artifact_present": False,
        "non_9b_artifact_present": False,
    }


def rows_from_kld(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("reduced_rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def row_matching(rows: list[dict[str, Any]], *tokens: str) -> dict[str, Any] | None:
    for row in rows:
        variant = str(row.get("variant", ""))
        if all(token in variant for token in tokens):
            return row
    return None


def row_matching_without(rows: list[dict[str, Any]], *, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> dict[str, Any] | None:
    for row in rows:
        variant = str(row.get("variant", ""))
        if all(token in variant for token in include) and not any(token in variant for token in exclude):
            return row
    return None


def kld_digest(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "variant": row.get("variant"),
        "arch": row.get("arch"),
        "scoring_mode": row.get("scoring_mode"),
        "n_chunks": row.get("n_chunks"),
        "mean_kld": row.get("mean_kld"),
        "p99_kld": row.get("p99_kld"),
        "ppl": row.get("ppl"),
    }


def mean_kld(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("mean_kld")
    return float(value) if value is not None else None


def finite_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def current_row_valid(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return finite_positive_number(row.get("mean_kld")) and finite_positive_number(row.get("ppl"))


def artifact_name_contains(provenance: dict[str, Any], *tokens: str) -> bool:
    artifacts = provenance.get("artifacts", [])
    for artifact in artifacts:
        name = str(artifact.get("name", "")).lower()
        if all(token.lower() in name for token in tokens):
            return True
    return False


def current_kld_summary(path: Path, candidate_token: str, control_token: str = ".mq4-") -> dict[str, Any]:
    rows = rows_from_kld(read_json(path))
    candidate = row_matching(rows, candidate_token)
    control = row_matching(rows, control_token)
    candidate_kld = mean_kld(candidate)
    control_kld = mean_kld(control)
    ratio = candidate_kld / control_kld if candidate_kld is not None and control_kld else None
    return {
        "path": str(path),
        "candidate": kld_digest(candidate),
        "control": kld_digest(control),
        "candidate_to_control_mean_kld_ratio": ratio,
        "candidate_beats_control": bool(candidate_kld is not None and control_kld is not None and candidate_kld < control_kld),
    }


def current_mq4_lloyd_same_run_summary(path: Path) -> dict[str, Any]:
    rows = rows_from_kld(read_json(path))
    candidate = row_matching(rows, "mq4-lloyd")
    mq4 = row_matching_without(rows, include=(".mq4-",), exclude=("lloyd",))
    mq6 = row_matching_without(rows, include=(".mq6-",), exclude=("lloyd",))
    candidate_kld = mean_kld(candidate)
    mq4_kld = mean_kld(mq4)
    mq6_kld = mean_kld(mq6)
    candidate_valid = current_row_valid(candidate)
    invalid_zero_no_ppl = bool(
        candidate is not None
        and candidate.get("mean_kld") in (0, 0.0)
        and candidate.get("ppl") is None
    )
    return {
        "path": str(path),
        "candidate": kld_digest(candidate),
        "mq4_control": kld_digest(mq4),
        "mq6_control": kld_digest(mq6),
        "same_run_present": bool(candidate and mq4 and mq6),
        "candidate_valid": candidate_valid,
        "candidate_invalid_zero_kld_no_ppl": invalid_zero_no_ppl,
        "candidate_to_mq4_mean_kld_ratio": candidate_kld / mq4_kld if candidate_valid and mq4_kld else None,
        "candidate_to_mq6_mean_kld_ratio": candidate_kld / mq6_kld if candidate_valid and mq6_kld else None,
        "candidate_beats_mq4": bool(candidate_valid and mq4_kld is not None and candidate_kld is not None and candidate_kld < mq4_kld),
        "candidate_beats_mq6": bool(candidate_valid and mq6_kld is not None and candidate_kld is not None and candidate_kld < mq6_kld),
    }


def historical_mq3_summary(path: Path) -> dict[str, Any]:
    rows = rows_from_kld(read_json(path))
    candidate = row_matching(rows, "mq3-lloyd")
    mq3 = row_matching_without(rows, include=(".mq3",), exclude=("lloyd",))
    mq4 = row_matching_without(rows, include=(".mq4",), exclude=("lloyd",))
    mq6 = row_matching_without(rows, include=(".mq6",), exclude=("lloyd",))
    candidate_kld = mean_kld(candidate)
    mq3_kld = mean_kld(mq3)
    mq4_kld = mean_kld(mq4)
    mq6_kld = mean_kld(mq6)
    return {
        "path": str(path),
        "candidate": kld_digest(candidate),
        "mq3_control": kld_digest(mq3),
        "mq4_control": kld_digest(mq4),
        "mq6_control": kld_digest(mq6),
        "beats_uniform_mq3": bool(candidate_kld is not None and mq3_kld is not None and candidate_kld < mq3_kld),
        "beats_mq4": bool(candidate_kld is not None and mq4_kld is not None and candidate_kld < mq4_kld),
        "beats_mq6": bool(candidate_kld is not None and mq6_kld is not None and candidate_kld < mq6_kld),
    }


def historical_mq4_summary(path: Path, current_control: dict[str, Any]) -> dict[str, Any]:
    rows = rows_from_kld(read_json(path))
    lloyd_rows = [row for row in rows if "mq4-lloyd" in str(row.get("variant", ""))]
    c512_rows = [row for row in lloyd_rows if "c512" in str(row.get("variant", ""))]
    if c512_rows:
        lloyd_rows = c512_rows
    best = min(lloyd_rows, key=lambda row: float(row.get("mean_kld", float("inf")))) if lloyd_rows else None
    mq6 = row_matching(rows, ".mq6")
    best_kld = mean_kld(best)
    current_mq4_kld = mean_kld(current_control)
    mq6_kld = mean_kld(mq6)
    return {
        "path": str(path),
        "best_lloyd_row": kld_digest(best),
        "historical_mq6_row": kld_digest(mq6),
        "best_beats_current_mq4": bool(best_kld is not None and current_mq4_kld is not None and best_kld < current_mq4_kld),
        "best_beats_historical_mq6": bool(best_kld is not None and mq6_kld is not None and best_kld < mq6_kld),
    }


def coherence_summary(path: Path, *, model: str, expected_tokens: tuple[str, ...], forbidden_tokens: tuple[str, ...]) -> dict[str, Any]:
    text = read_text(path)
    section = model_section(text, model)
    has_row = bool(section)
    expected_present = all(token in section for token in expected_tokens)
    forbidden_present = [token for token in forbidden_tokens if token in section]
    return {
        "path": str(path),
        "has_row": has_row,
        "expected_tokens_present": expected_present,
        "forbidden_tokens_present": forbidden_present,
        "qualitatively_clean": has_row and expected_present and not forbidden_present,
    }


def container_summary(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "names_mq4_lloyd_qtype": "MQ4G256_LLOYD" in text,
        "names_mq3_lloyd_qtype": "MQ3G256_LLOYD" in text,
        "data_end_matches_file_size": "data_end" in text and "equals the file size" in text,
        "records_attractor_reject": "!!!!!!!!!!!" in text,
    }


def mq3_lloyd_size_scope(
    provenance: dict[str, Any],
    live_inventory: dict[str, Any],
    current_kld: dict[str, Any],
    coherence: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    current_kld_present = bool(current_kld["candidate"] and current_kld["control"])
    current_kld_finite = bool(
        current_row_valid(current_kld["candidate"])
        and current_row_valid(current_kld["control"])
    )
    current_kld_loses = bool(
        current_kld_present
        and current_kld_finite
        and not current_kld["candidate_beats_control"]
    )
    live_scope = artifact_scope_flags(live_inventory.get("artifacts", []))
    return {
        "only_9b_artifact_present": (
            provenance["canonical_9b_artifact_present"]
            and not provenance["non_9b_artifact_present"]
        ),
        "live_only_9b_artifact_present": live_scope["only_9b_artifact_present"],
        "live_artifact_count": live_inventory.get("artifact_count", 0),
        "dense_4b_artifact_present": artifact_name_contains(provenance, "4b"),
        "live_dense_4b_artifact_present": live_scope["dense_4b_artifact_present"],
        "dense_9b_artifact_present": provenance["canonical_9b_artifact_present"],
        "live_dense_9b_artifact_present": live_scope["dense_9b_artifact_present"],
        "dense_27b_artifact_present": artifact_name_contains(provenance, "27b"),
        "live_dense_27b_artifact_present": live_scope["dense_27b_artifact_present"],
        "a3b_artifact_present": artifact_name_contains(provenance, "a3b"),
        "live_a3b_artifact_present": live_scope["a3b_artifact_present"],
        "current_9b_coherence_clean": coherence["qualitatively_clean"],
        "current_9b_kld_present": current_kld_present,
        "current_9b_kld_finite": current_kld_finite,
        "current_9b_kld_loses_to_mq4": current_kld_loses,
        "current_9b_candidate_to_mq4_kld_ratio": current_kld["candidate_to_control_mean_kld_ratio"],
        "historical_only_beats_uniform_mq3": (
            historical["beats_uniform_mq3"]
            and not historical["beats_mq4"]
            and not historical["beats_mq6"]
        ),
        "perf_evidence_allowed": False,
    }


def mq4_lloyd_value_boundary(*, gates: dict[str, bool]) -> dict[str, Any]:
    coherence_ok = gates["coherence_9b_clean"]
    same_run_valid = gates["current_same_run_candidate_valid"]
    beats_mq4 = gates["current_same_run_beats_mq4"]
    beats_mq6 = gates["current_same_run_beats_mq6"]
    origin_refresh_required = gates.get("origin_refresh_required", False)
    perf_allowed = coherence_ok and same_run_valid and beats_mq4 and beats_mq6 and not origin_refresh_required
    if not coherence_ok:
        next_step = "produce_new_coherent_mq4_lloyd_artifact"
    elif not same_run_valid:
        next_step = "rerun_same_run_finite_kld_ppl_against_mq4_and_mq6"
    elif not beats_mq4:
        next_step = "prove_same_run_kld_beats_mq4"
    elif not beats_mq6:
        next_step = "prove_same_run_kld_beats_mq6"
    elif origin_refresh_required:
        next_step = "reconcile_origin_master_and_rerun_lloyd_evidence"
    elif not gates["perf_evidence_present"]:
        next_step = "run_fresh_process_gfx1151_perf_baselines"
    else:
        next_step = "verify_readiness_matrix_before_promotion_claim"
    return {
        "bytes_per_group": 160,
        "mq4_control_bytes_per_group": 136,
        "mq6_comparator_bytes_per_group": 200,
        "status": "quality_value_candidate" if perf_allowed else "value_not_justified_current_artifact",
        "container_valid": gates["container_qtype_named"] and gates["container_bounded"],
        "coherence_gate_passed": coherence_ok,
        "same_run_kld_ppl_valid": same_run_valid,
        "invalid_zero_kld_no_ppl_blocks_quality": gates["current_same_run_invalid_zero_kld_no_ppl"],
        "beats_mq4_same_run": beats_mq4,
        "beats_mq6_same_run": beats_mq6,
        "historical_rows_justify_value": (
            gates["historical_best_beats_current_mq4"]
            and gates["historical_best_beats_mq6"]
        ),
        "historical_rows_are_promotion_evidence": False,
        "origin_refresh_required": origin_refresh_required,
        "branch_reconciled_for_promotion": not origin_refresh_required,
        "perf_collection_allowed": perf_allowed,
        "performance_rows_promotable": perf_allowed and gates["perf_evidence_present"],
        "requires_new_artifact_hash": not coherence_ok,
        "next_unblocked_step": next_step,
    }


def mq4_lloyd_promotion_allowed(gates: dict[str, bool]) -> bool:
    required_ok = all(bool(gates.get(key)) for key in MQ4_LLOYD_PROMOTION_REQUIRED_GATES)
    forbidden_clear = not any(
        bool(gates.get(key)) for key in MQ4_LLOYD_PROMOTION_FORBIDDEN_GATES
    )
    return required_ok and forbidden_clear


def mq4_lloyd_promotion_contract(gates: dict[str, bool]) -> dict[str, Any]:
    return {
        "required_gates": list(MQ4_LLOYD_PROMOTION_REQUIRED_GATES),
        "forbidden_gates": list(MQ4_LLOYD_PROMOTION_FORBIDDEN_GATES),
        "required_satisfied": {
            key: bool(gates.get(key)) for key in MQ4_LLOYD_PROMOTION_REQUIRED_GATES
        },
        "forbidden_clear": {
            key: not bool(gates.get(key)) for key in MQ4_LLOYD_PROMOTION_FORBIDDEN_GATES
        },
        "current_same_run_invalid_zero_kld_no_ppl_must_be_false": True,
        "historical_rows_required_for_promotion": False,
        "historical_rows_role": "context_only_when_current_same_run_mq4_mq6_cohort_exists",
        "promotion_allowed": mq4_lloyd_promotion_allowed(gates),
    }


def mq4_lloyd_producer_repair_plan(
    *,
    gates: dict[str, bool],
    value_boundary: dict[str, Any],
    origin: dict[str, Any],
) -> dict[str, Any]:
    origin_commits = []
    for commit in origin.get("relevant_upstream_commits", []):
        if "mq4-lloyd" not in commit.get("formats", []):
            continue
        if commit.get("present_on_origin_master") is True and commit.get("present_on_head") is False:
            origin_commits.append(str(commit.get("short")))
    same_run_ready = bool(
        gates["current_same_run_candidate_valid"]
        and gates["current_same_run_beats_mq4"]
        and gates["current_same_run_beats_mq6"]
    )
    return {
        "status": (
            "ready_for_perf"
            if value_boundary["perf_collection_allowed"]
            else "blocked_before_perf"
        ),
        "current_artifact_status": (
            "coherence_rejected"
            if not gates["coherence_9b_clean"]
            else "quality_gated"
        ),
        "requires_new_artifact_hash": bool(value_boundary["requires_new_artifact_hash"]),
        "requires_origin_reconciliation": bool(origin_commits),
        "origin_commits_to_reconcile": origin_commits,
        "same_run_quality_ready": same_run_ready,
        "invalid_zero_kld_no_ppl_blocks_quality": gates["current_same_run_invalid_zero_kld_no_ppl"],
        "perf_collection_allowed": bool(value_boundary["perf_collection_allowed"]),
        "rerun_commands": [
            "python3 scripts/lloyd_status.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-lloyd-status.json",
            "./scripts/coherence-gate.sh --full",
            "rerun current MQ4/MQ4-Lloyd/MQ6 KLD cohort after a coherent MQ4-Lloyd artifact exists",
        ],
        "validation_commands": [
            "python3 scripts/test_lloyd_status.py",
            "python3 scripts/test_quant_readiness.py",
            "python3 scripts/test_quant_readiness_integrity.py",
        ],
        "next_unblocked_step": value_boundary["next_unblocked_step"],
    }


def model_section(text: str, model: str) -> str:
    marker = f"## {model}"
    sections = []
    start = text.find(marker)
    while start >= 0:
        next_start = text.find("\n## ", start + len(marker))
        if next_start < 0:
            sections.append(text[start:])
            break
        sections.append(text[start:next_start])
        start = text.find(marker, next_start + 1)
    return "\n".join(sections)


def build_status(
    paths: dict[str, Path] | None = None,
    model_roots: list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    paths = {**DEFAULT_PATHS, **(paths or {})}
    origin = origin_context()
    artifact_inventory = lloyd_artifact_inventory(model_roots or DEFAULT_MODEL_ROOTS)

    mq3_provenance = format_record(read_json(paths["mq3_provenance"]), "mq3-lloyd")
    mq4_provenance = format_record(read_json(paths["mq4_provenance"]), "mq4-lloyd")
    mq3_live_inventory = artifact_inventory["formats"]["mq3-lloyd"]
    mq4_live_inventory = artifact_inventory["formats"]["mq4-lloyd"]
    mq3_current_kld = current_kld_summary(paths["mq3_kld"], "mq3-lloyd")
    current_c512 = current_kld_summary(paths["mq6_c512_kld"], ".mq6-", control_token=".mq4-")
    mq4_current_kld = current_mq4_lloyd_same_run_summary(paths["mq4_current_kld"])
    mq3_historical = historical_mq3_summary(paths["mq3_historical_kld"])
    mq4_historical = historical_mq4_summary(paths["mq4_historical_kld"], current_c512["control"] or {})
    mq3_coherence = coherence_summary(
        paths["mq3_coherence"],
        model="qwen3.5-9b.mq3-lloyd",
        expected_tokens=("Final Number", "9<|im_end|>", "O(1)"),
        forbidden_tokens=("!!!!!!!!!!!",),
    )
    mq4_coherence = coherence_summary(
        paths["mq4_coherence"],
        model="qwen3.5-9b.mq4-lloyd",
        expected_tokens=(),
        forbidden_tokens=("!!!!!!!!!!!",),
    )
    container = container_summary(paths["mq4_container_audit"])
    mq3_size_scope = mq3_lloyd_size_scope(
        mq3_provenance,
        mq3_live_inventory,
        mq3_current_kld,
        mq3_coherence,
        mq3_historical,
    )

    mq3_gates = {
        "canonical_9b_artifact_present": mq3_provenance["canonical_9b_artifact_present"],
        "non_9b_artifact_present": mq3_provenance["non_9b_artifact_present"],
        "coherence_9b_clean": mq3_coherence["qualitatively_clean"],
        "current_kld_present": mq3_size_scope["current_9b_kld_present"],
        "current_kld_finite": mq3_size_scope["current_9b_kld_finite"],
        "current_kld_beats_mq4": mq3_current_kld["candidate_beats_control"],
        "current_kld_loses_to_mq4": mq3_size_scope["current_9b_kld_loses_to_mq4"],
        "historical_beats_uniform_mq3": mq3_historical["beats_uniform_mq3"],
        "historical_beats_mq4": mq3_historical["beats_mq4"],
        "historical_beats_mq6": mq3_historical["beats_mq6"],
        "perf_evidence_present": False,
        "origin_refresh_required": origin["format_refresh_required"]["mq3-lloyd"],
    }
    mq3_promotion_gate_keys = tuple(key for key in mq3_gates if key != "current_kld_loses_to_mq4")
    mq3_allowed = (
        all(mq3_gates[key] for key in mq3_promotion_gate_keys if key != "origin_refresh_required")
        and not mq3_gates["origin_refresh_required"]
    )
    mq3_blockers = []
    if not mq3_gates["current_kld_present"]:
        mq3_blockers.append("current 9B MQ3-Lloyd KLD row is absent")
    elif not mq3_gates["current_kld_finite"]:
        mq3_blockers.append("current 9B MQ3-Lloyd KLD/PPL row is not finite")
    if not mq3_gates["current_kld_beats_mq4"]:
        ratio = mq3_size_scope["current_9b_candidate_to_mq4_kld_ratio"]
        if ratio is None:
            mq3_blockers.append("current 9B MQ3-Lloyd KLD does not beat MQ4 control")
        else:
            mq3_blockers.append(f"current 9B MQ3-Lloyd KLD is {ratio:.2f}x MQ4 control")
    if not mq3_gates["non_9b_artifact_present"]:
        mq3_blockers.append("only the 9B MQ3-Lloyd artifact exists; 4B, 27B, and A3B are absent")
    if mq3_size_scope["live_only_9b_artifact_present"]:
        mq3_blockers.append("live local scan also found only 9B MQ3-Lloyd artifacts")
    if not mq3_gates["historical_beats_mq4"]:
        mq3_blockers.append("historical MQ3-Lloyd KLD still lagged MQ4")
    if not mq3_gates["perf_evidence_present"]:
        mq3_blockers.append("no fresh-process gfx1151 perf baseline exists")
    if mq3_gates["origin_refresh_required"]:
        mq3_blockers.append("origin/master has MQ3-Lloyd-relevant commits missing from HEAD; reconcile and rerun evidence")

    mq4_live_scope = artifact_scope_flags(mq4_live_inventory["artifacts"])
    mq4_gates = {
        "canonical_9b_artifact_present": mq4_provenance["canonical_9b_artifact_present"],
        "non_9b_artifact_present": mq4_provenance["non_9b_artifact_present"],
        "container_qtype_named": container["names_mq4_lloyd_qtype"],
        "container_bounded": container["data_end_matches_file_size"],
        "coherence_9b_clean": mq4_coherence["qualitatively_clean"],
        "historical_best_beats_current_mq4": mq4_historical["best_beats_current_mq4"],
        "historical_best_beats_mq6": mq4_historical["best_beats_historical_mq6"],
        "current_same_run_kld_present": mq4_current_kld["same_run_present"],
        "current_same_run_candidate_valid": mq4_current_kld["candidate_valid"],
        "current_same_run_invalid_zero_kld_no_ppl": mq4_current_kld["candidate_invalid_zero_kld_no_ppl"],
        "current_same_run_beats_mq4": mq4_current_kld["candidate_beats_mq4"],
        "current_same_run_beats_mq6": mq4_current_kld["candidate_beats_mq6"],
        "perf_evidence_present": False,
        "origin_refresh_required": origin["format_refresh_required"]["mq4-lloyd"],
    }
    mq4_promotion_contract = mq4_lloyd_promotion_contract(mq4_gates)
    mq4_allowed = mq4_promotion_contract["promotion_allowed"]
    mq4_blockers = []
    if not mq4_gates["coherence_9b_clean"]:
        mq4_blockers.append("current 9B MQ4-Lloyd coherence row emits token attractor")
    if not mq4_gates["historical_best_beats_current_mq4"]:
        mq4_blockers.append("historical best MQ4-Lloyd KLD does not beat current MQ4 context")
    if not mq4_gates["historical_best_beats_mq6"]:
        mq4_blockers.append("historical best MQ4-Lloyd KLD does not beat MQ6 context")
    if not mq4_gates["current_same_run_kld_present"]:
        mq4_blockers.append("no current same-run MQ4/MQ4-Lloyd/MQ6 KLD result exists")
    elif not mq4_gates["current_same_run_candidate_valid"]:
        mq4_blockers.append("current same-run MQ4-Lloyd KLD row is invalid zero-KLD/NaN-PPL evidence")
    else:
        if not mq4_gates["current_same_run_beats_mq4"]:
            mq4_blockers.append("current same-run MQ4-Lloyd KLD does not beat MQ4")
        if not mq4_gates["current_same_run_beats_mq6"]:
            mq4_blockers.append("current same-run MQ4-Lloyd KLD does not beat MQ6")
    if not mq4_gates["perf_evidence_present"]:
        mq4_blockers.append("no fresh-process gfx1151 perf baseline exists")
    if mq4_live_scope["only_9b_artifact_present"]:
        mq4_blockers.append("live local scan found only 9B MQ4-Lloyd artifacts")
    if mq4_gates["origin_refresh_required"]:
        mq4_blockers.append("origin/master has MQ4-Lloyd-relevant commits missing from HEAD; reconcile and rerun evidence")
    mq4_value_boundary = mq4_lloyd_value_boundary(gates=mq4_gates)
    mq4_repair_plan = mq4_lloyd_producer_repair_plan(
        gates=mq4_gates,
        value_boundary=mq4_value_boundary,
        origin=origin,
    )

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "formats": {
            "mq3-lloyd": {
                "status": "dense-quality-rejected-current-9b",
                "promotion_allowed": mq3_allowed,
                "artifact_provenance": mq3_provenance,
                "live_artifact_inventory": mq3_live_inventory,
                "coherence": mq3_coherence,
                "current_kld": mq3_current_kld,
                "historical_kld": mq3_historical,
                "size_scope": mq3_size_scope,
                "gates": mq3_gates,
                "promotion_gate_keys": list(mq3_promotion_gate_keys),
                "blockers": mq3_blockers,
                "decision": (
                    "keep MQ3-Lloyd research-gated; coherent 9B smoke is not enough "
                    "because current BF16-referenced KLD loses to MQ4"
                ),
            },
            "mq4-lloyd": {
                "status": "coherence-rejected-current-9b",
                "promotion_allowed": mq4_allowed,
                "artifact_provenance": mq4_provenance,
                "live_artifact_inventory": mq4_live_inventory,
                "artifact_scope": mq4_live_scope,
                "coherence": mq4_coherence,
                "container": container,
                "current_mq4_mq6_context": current_c512,
                "current_same_run_kld": mq4_current_kld,
                "historical_kld": mq4_historical,
                "value_boundary": mq4_value_boundary,
                "promotion_contract": mq4_promotion_contract,
                "producer_repair_plan": mq4_repair_plan,
                "gates": mq4_gates,
                "blockers": mq4_blockers,
                "next_work": [
                    "Produce a new coherent MQ4-Lloyd 9B artifact hash before collecting perf.",
                    "Reconcile origin/master Lloyd producer/runtime commits, including d5985c3e, before rerunning evidence.",
                    "Rerun full coherence for the new artifact and reject any token-attractor row.",
                    "Rerun the current MQ4/MQ4-Lloyd/MQ6 KLD cohort and require finite PPL plus wins over MQ4 and MQ6.",
                    "Only collect fresh-process gfx1151 perf after coherence and same-run KLD/PPL are valid.",
                ],
                "decision": (
                    "keep MQ4-Lloyd research-gated; current 9B artifact is "
                    "container-valid but coherence-rejected and current same-run "
                    "KLD produced invalid zero-KLD/NaN-PPL evidence"
                ),
            },
        },
        "artifact_inventory": artifact_inventory,
        "origin_context": origin,
        "summary": {
            "promotion_allowed": False,
            "decisions": {
                "mq3-lloyd": "reject current 9B dense promotion; require new artifact hash before more perf",
                "mq4-lloyd": "reject current 9B artifact on coherence; require producer/calibration fix",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model-root", action="append", default=[], help="Model root to scan for Lloyd artifacts; repeatable")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    model_roots = [Path(path) for path in args.model_root] if args.model_root else list(DEFAULT_MODEL_ROOTS)
    payload = build_status(model_roots=model_roots)
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
