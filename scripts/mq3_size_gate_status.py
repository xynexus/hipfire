#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize MQ3 size-gated readiness status for gfx1151.

Plain MQ3 is not one global promote/reject decision.  The current evidence
rejects 4B and 9B dense text, keeps 27B dense incomplete until a matching KLD
reference exists, and keeps A3B/MoE research-scoped until matching KLD refs and
draft sidecars exist.  This helper turns that split into a structured artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
SCHEMA = "hipfire.mq3_size_gate_status.gfx1151.v0"
DEFAULT_OUT = RESULTS_DIR / "2026-06-03-mq3-size-gate-status.json"
DEFAULT_MODEL_ROOTS = (
    Path("/home/sadara/Models"),
    Path("/home/sadara/.hipfire/models"),
)
QWEN35_SOURCE = ROOT / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs"
MQ3_MOE_DECODE_BOUNDARY_TEST = (
    "cargo test -p hipfire-arch-qwen35 --lib "
    "mq3_a3b_prefill_path2_but_moe_decode_lacks_indexed_route"
)
MQ3_LONG_PREFILL_PATH2_TEST = "cargo test -p hipfire-arch-qwen35 --lib moe_prefill"
MQ3_LONG_PREFILL_PATH2_TEST_NAME = (
    "qwen35::tests::moe_prefill_mq3_long_prefill_path2_shape_is_production_shaped"
)

DEFAULT_PATHS = {
    "provenance": RESULTS_DIR / "2026-06-03-mq3-artifact-provenance.json",
    "size_audit": RESULTS_DIR / "2026-06-03-mq3-size-gate-audit.md",
    "kld_4b": RESULTS_DIR / "2026-06-03-mq3-4b-kld.json",
    "kld_9b": RESULTS_DIR / "2026-06-03-mq3-9b-kld.json",
    "ppl_9b": RESULTS_DIR / "2026-06-03-mq3-9b-ppl.json",
    "ppl_27b": RESULTS_DIR / "2026-06-03-mq3-27b-ppl.json",
    "ppl_a3b": RESULTS_DIR / "2026-06-03-mq3-a3b-ppl.json",
    "kld_refs": RESULTS_DIR / "2026-06-03-mq3-kld-reference-inventory.json",
    "a3b_coherence": RESULTS_DIR / "2026-06-03-mq3-a3b-broader-coherence.json",
    "a3b_dflash_audit": RESULTS_DIR / "2026-06-03-mq3-a3b-dflash-fixture-audit.md",
    "long_prefill_audit": RESULTS_DIR / "2026-06-03-mq3-long-prefill-path2-shape-audit.md",
    "ar_perf": RESULTS_DIR / "2026-06-03-mq3-ar-perf.json",
    "dflash_perf": RESULTS_DIR / "2026-06-03-mq3-dflash.json",
}

REQUIRED_MQ3_ARTIFACTS = (
    "qwen3.5-4b-mq3.hfq",
    "qwen3.5-9b-mq3.hfq",
    "qwen3.5-27b-mq3.hfq",
    "qwen3.5-35b-a3b-mq3.hfq",
    "qwen3.6-35b-a3b-mq3.hfq",
)

REQUIRED_BY_ROLE = {
    "qwen3.5-4b": "qwen3.5-4b-mq3.hfq",
    "qwen3.5-9b": "qwen3.5-9b-mq3.hfq",
    "qwen3.5-27b": "qwen3.5-27b-mq3.hfq",
    "qwen3.5-35b-a3b": "qwen3.5-35b-a3b-mq3.hfq",
    "qwen3.6-35b-a3b": "qwen3.6-35b-a3b-mq3.hfq",
}


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def artifact_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    artifacts = []
    for item in payload.get("formats", []):
        if item.get("id") == "mq3":
            artifacts = item.get("candidate_artifacts", {}).get("artifacts", [])
            break
    names = {artifact.get("name") for artifact in artifacts}
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_artifacts": list(REQUIRED_MQ3_ARTIFACTS),
        "required_artifacts_present": sorted(names & set(REQUIRED_MQ3_ARTIFACTS)),
        "required_artifacts_missing": sorted(set(REQUIRED_MQ3_ARTIFACTS) - names),
        "canonical_mq3_artifacts_present": all(name in names for name in REQUIRED_MQ3_ARTIFACTS),
    }


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            yield Path(dirpath) / filename


def is_mq3_candidate(path: Path) -> bool:
    name = path.name.lower()
    if "mq3" not in name:
        return False
    if "lloyd" in name or "dflash" in name or "draft" in name:
        return False
    return path.is_file() or path.is_symlink()


def mq3_role(path: Path) -> str:
    name = path.name.lower()
    if "qwen3.6-35b-a3b" in name:
        return "qwen3.6-35b-a3b"
    if "qwen3.6-27b" in name:
        return "qwen3.6-27b"
    if "qwen3.5-122b-a10b" in name:
        return "qwen3.5-122b-a10b"
    if "qwen3.5-35b-a3b" in name:
        return "qwen3.5-35b-a3b"
    if "qwen3.5-27b" in name:
        return "qwen3.5-27b"
    if "qwen3.5-9b" in name:
        return "qwen3.5-9b"
    if "qwen3.5-4b" in name:
        return "qwen3.5-4b"
    if "qwen3.5-2b" in name:
        return "qwen3.5-2b"
    if "qwen3.5-0.8b" in name:
        return "qwen3.5-0.8b"
    return "unknown"


def canonical_mq3_name(path: Path) -> str | None:
    role = mq3_role(path)
    if "awq-mtp" in path.name.lower():
        return None
    if path.name.lower().endswith(".mq3"):
        return REQUIRED_BY_ROLE.get(role)
    return path.name if path.name in REQUIRED_MQ3_ARTIFACTS else None


def live_artifact_inventory(model_roots: Iterable[Path] = DEFAULT_MODEL_ROOTS) -> dict[str, Any]:
    roots = [Path(root).expanduser() for root in model_roots]
    artifacts = []
    for root in roots:
        for path in iter_files(root):
            if not is_mq3_candidate(path):
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None
            canonical = canonical_mq3_name(path)
            lower_name = path.name.lower()
            artifacts.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "root": str(root),
                    "size_bytes": size_bytes,
                    "role": mq3_role(path),
                    "canonical_required_name": canonical,
                    "canonical_required": canonical in REQUIRED_MQ3_ARTIFACTS,
                    "awq_mtp": "awq-mtp" in lower_name,
                    "cache_artifact": root == Path("/home/sadara/.hipfire/models"),
                    "candidate_root_artifact": "hipfire-candidates/gfx1151-readiness" in str(path),
                    "is_symlink": path.is_symlink(),
                    "symlink_target": os.readlink(path) if path.is_symlink() else None,
                }
            )
    canonical_present = sorted(
        {
            item["canonical_required_name"]
            for item in artifacts
            if item.get("canonical_required_name") in REQUIRED_MQ3_ARTIFACTS
        }
    )
    missing = sorted(set(REQUIRED_MQ3_ARTIFACTS) - set(canonical_present))
    return {
        "searched_roots": [{"root": str(root), "exists": root.exists()} for root in roots],
        "artifact_count": len(artifacts),
        "candidate_root_artifact_count": sum(1 for item in artifacts if item["candidate_root_artifact"]),
        "cache_artifact_count": sum(1 for item in artifacts if item["cache_artifact"]),
        "awq_mtp_artifact_count": sum(1 for item in artifacts if item["awq_mtp"]),
        "canonical_required_present": canonical_present,
        "canonical_required_missing": missing,
        "canonical_required_complete": not missing,
        "extra_installed_roles": sorted(
            {
                item["role"]
                for item in artifacts
                if not item["canonical_required"] and not item["awq_mtp"] and item["role"] != "unknown"
            }
        ),
        "artifacts": sorted(artifacts, key=lambda item: (item["role"], item["name"], item["path"])),
    }


def rows_from_kld(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("reduced_rows", []) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def row_containing(rows: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    for row in rows:
        if token in str(row.get("variant", "")):
            return row
    return None


def metric_digest(row: dict[str, Any] | None) -> dict[str, Any] | None:
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


def row_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


def kld_comparison(path: Path, candidate_token: str, control_token: str) -> dict[str, Any]:
    rows = rows_from_kld(path)
    candidate = row_containing(rows, candidate_token)
    control = row_containing(rows, control_token)
    candidate_kld = row_float(candidate, "mean_kld")
    control_kld = row_float(control, "mean_kld")
    ratio = candidate_kld / control_kld if candidate_kld is not None and control_kld else None
    return {
        "path": str(path),
        "candidate": metric_digest(candidate),
        "control": metric_digest(control),
        "candidate_to_control_mean_kld_ratio": ratio,
        "candidate_beats_control": bool(candidate_kld is not None and control_kld is not None and candidate_kld < control_kld),
    }


def ppl_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    result = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        row = {
            "id": case.get("id"),
            "family": case.get("family"),
            "format_id": case.get("format_id"),
            "exit_code": case.get("exit_code"),
            "result": case.get("result", {}),
        }
        result[str(case.get("id"))] = row
    return result


def ppl_comparison(path: Path, candidate_id: str, control_id: str) -> dict[str, Any]:
    by_id = ppl_cases(path)
    candidate = by_id.get(candidate_id)
    control = by_id.get(control_id)
    candidate_ppl = None if candidate is None else candidate.get("result", {}).get("ppl")
    control_ppl = None if control is None else control.get("result", {}).get("ppl")
    ratio = float(candidate_ppl) / float(control_ppl) if candidate_ppl is not None and control_ppl else None
    return {
        "path": str(path),
        "candidate": candidate,
        "control": control,
        "candidate_to_control_ppl_ratio": ratio,
        "candidate_beats_control": bool(candidate_ppl is not None and control_ppl is not None and float(candidate_ppl) < float(control_ppl)),
    }


def reference_status(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("expected_missing_or_required", []) if isinstance(payload, dict) else []
    by_fixture = {row.get("fixture"): row for row in rows if isinstance(row, dict)}
    return {
        "path": str(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "qwen35_27b_present": bool(by_fixture.get("qwen3.5-27b", {}).get("local_manifest_ref_sha256_ok")),
        "qwen35_a3b_present": bool(by_fixture.get("qwen3.5-35b-a3b", {}).get("local_manifest_ref_sha256_ok")),
        "qwen36_a3b_present": bool(by_fixture.get("qwen3.6-35b-a3b", {}).get("local_manifest_ref_sha256_ok")),
        "expected": by_fixture,
    }


def run_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    ok = 0
    for case in cases:
        runs = case.get("runs", [])
        if case.get("status") == "ok":
            ok += 1
        elif runs and all(run.get("status") == "ok" for run in runs):
            ok += 1
    hit_max = sum(1 for case in cases if case.get("hit_max_tokens"))
    return {
        "case_count": total,
        "ok_case_count": ok,
        "hard_error_count": total - ok,
        "hit_max_tokens_count": hit_max,
        "capped_rows_present": hit_max > 0,
        "all_cases_uncapped": total > 0 and hit_max == 0,
        "no_hard_errors": total > 0 and ok == total,
        "promotion_grade_no_hard_errors_and_uncapped": total > 0 and ok == total and hit_max == 0,
    }


def a3b_coherence_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    summary = run_summary([case for case in cases if isinstance(case, dict)])
    return {
        "path": str(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        **summary,
    }


def perf_summary(path: Path, *, format_token: str) -> dict[str, Any]:
    payload = read_json(path)
    cases = [case for case in payload.get("cases", []) if isinstance(case, dict)] if isinstance(payload, dict) else []
    matching = [case for case in cases if format_token in str(case.get("id", ""))]
    ok_cases = []
    token_attr_case_count = 0
    token_attr_clean_case_count = 0
    reached_cap_case_count = 0
    for case in matching:
        summary = case.get("summary", {})
        if summary.get("ok_runs") == summary.get("total_runs") and summary.get("total_runs", 0) > 0:
            ok_cases.append(case)
        runs = [run for run in case.get("runs", []) if isinstance(run, dict)]
        token_checks = [
            run.get("token_attractor", {}).get("ok")
            for run in runs
            if isinstance(run.get("token_attractor"), dict)
        ]
        if token_checks:
            token_attr_case_count += 1
            if all(check is True for check in token_checks):
                token_attr_clean_case_count += 1
        max_tokens = case.get("max_tokens")
        median_tokens = summary.get("median_tokens")
        if isinstance(max_tokens, (int, float)) and isinstance(median_tokens, (int, float)):
            if float(median_tokens) >= float(max_tokens):
                reached_cap_case_count += 1
    return {
        "path": str(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "matching_case_count": len(matching),
        "ok_case_count": len(ok_cases),
        "all_matching_cases_ok": bool(matching) and len(matching) == len(ok_cases),
        "token_attractor_case_count": token_attr_case_count,
        "token_attractor_clean_case_count": token_attr_clean_case_count,
        "all_matching_token_attractor_clean": bool(matching) and token_attr_case_count == len(matching) and token_attr_clean_case_count == len(matching),
        "reached_token_cap_case_count": reached_cap_case_count,
        "all_matching_cases_reached_token_cap": bool(matching) and reached_cap_case_count == len(matching),
        "cases": [
            {
                "id": case.get("id"),
                "max_tokens": case.get("max_tokens"),
                "prompt_md5": case.get("prompt_md5"),
                "summary": case.get("summary", {}),
            }
            for case in matching
        ],
    }


def dflash_sidecar_summary(path: Path) -> dict[str, Any]:
    text = read_text(path)
    no_matches = "Result: no matches" in text
    return {
        "path": str(path),
        "audit_present": bool(text),
        "paired_a3b_dflash_sidecars_present": bool(text) and not no_matches,
        "records_no_matches": no_matches,
    }


def _extract_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def long_prefill_shape_contract(path: Path) -> dict[str, Any]:
    text = read_text(path)
    lower = text.lower()
    n_tokens = _extract_int(text, r"N=(\d+)")
    k_top = _extract_int(text, r"K_TOP=(\d+)")
    num_experts = _extract_int(text, r"num_experts=(\d+)")
    total_slots = _extract_int(text, r"total_slots\s*=\s*(\d+)")
    m_total_bound = _extract_int(text, r"m_total_bound\s*=\s*(\d+)")
    gate_up_x_row_div = _extract_int(text, r"Gate/up grouped GEMM.*?x_row_div\s*=\s*(\d+)")
    down_x_row_div = _extract_int(text, r"Down grouped GEMM.*?x_row_div\s*=\s*(\d+)")
    phrase_checks = {
        "scope": "no-GPU contract for MQ3 A3B grouped MoE long-prefill routing" in text,
        "test_command": MQ3_LONG_PREFILL_PATH2_TEST in text,
        "test_name": MQ3_LONG_PREFILL_PATH2_TEST_NAME in text,
        "admission_gfx1151": "MQ3 routed experts remain admitted for `gfx1151`" in text,
        "path2_forced_without_indexed_fallback": all(
            phrase in text
            for phrase in (
                "MQ3 forces grouped path2",
                "HIPFIRE_MOE_GROUPED_GEMM=0",
                "no\n  indexed MQ3 fallback is wired",
            )
        ),
        "full_chunk_shape": all(
            value is not None
            for value in (n_tokens, k_top, num_experts, total_slots, m_total_bound)
        ),
        "m_total_bound_multiple_of_16": "m_total_bound % 16 == 0" in text,
        "gate_up_row_div": gate_up_x_row_div == 8,
        "down_row_div": down_x_row_div == 1,
        "records_no_promotion_claim": all(
            phrase in lower
            for phrase in (
                "does not claim artifact-backed promotion evidence",
                "not a promotion-grade runtime result",
                "still needs\nartifact-backed long-prefill coherence/perf",
            )
        ),
    }
    all_phrase_checks_present = all(phrase_checks.values())
    records_no_promotion_claim = phrase_checks["records_no_promotion_claim"]
    artifact_backed_long_prefill_evidence_present = bool(text) and not records_no_promotion_claim
    status = "missing_audit"
    if text and all_phrase_checks_present and records_no_promotion_claim:
        status = "covered_no_gpu_not_runtime"
    elif text and all_phrase_checks_present:
        status = "covered_unclassified"
    elif text:
        status = "shape_contract_unverified"
    return {
        "path": str(path),
        "audit_present": bool(text),
        "test_command": MQ3_LONG_PREFILL_PATH2_TEST,
        "test_name": MQ3_LONG_PREFILL_PATH2_TEST_NAME,
        "phrase_checks": phrase_checks,
        "all_phrase_checks_present": all_phrase_checks_present,
        "status": status,
        "promotion_grade_runtime_result": artifact_backed_long_prefill_evidence_present,
        "artifact_backed_long_prefill_evidence_present": artifact_backed_long_prefill_evidence_present,
        "invariants": {
            "n_tokens": n_tokens,
            "k_top": k_top,
            "num_experts": num_experts,
            "total_slots": total_slots,
            "m_total_bound": m_total_bound,
            "m_total_bound_multiple_of_16": (
                bool(m_total_bound and m_total_bound % 16 == 0)
                and phrase_checks["m_total_bound_multiple_of_16"]
            ),
            "gate_up_x_row_div": gate_up_x_row_div,
            "down_x_row_div": down_x_row_div,
        },
    }


def mq3_moe_decode_boundary_evidence(source: Path = QWEN35_SOURCE) -> dict[str, Any]:
    text = read_text(source)
    phrase_checks = {
        "test_name": "mq3_a3b_prefill_path2_but_moe_decode_lacks_indexed_route" in text,
        "prefill_admission": all(
            phrase in text
            for phrase in (
                "moe_ffn_batched_admissible_for_dtypes(",
                "&dtypes, false, \"gfx1151\"",
            )
        ),
        "path2_required": "moe_grouped_gemm_path2_required_for_dtype(DType::MQ3G256)" in text,
        "path2_forced_on_gfx1151": all(
            phrase in text
            for phrase in (
                "moe_grouped_gemm_path2_eligible_for_dtype(",
                "DType::MQ3G256",
                "\"gfx1151\"",
                "false",
            )
        ),
        "decode_records_no_indexed_route": "flags.routed_path, MoeDecodeIndexedRoutedPath::None" in text,
        "decode_gpu_topk_blocked": "!flags.use_gpu_topk" in text,
        "decode_no_rotation_scratch": "!flags.needs_x_rot_local" in text,
    }
    indexed_route_supported = "MoeDecodeIndexedRoutedPath::Mq3" in text
    boundary_recorded = all(phrase_checks.values()) and not indexed_route_supported
    return {
        "path": str(source),
        "test_command": MQ3_MOE_DECODE_BOUNDARY_TEST,
        "phrase_checks": phrase_checks,
        "all_phrase_checks_present": all(phrase_checks.values()),
        "indexed_route_supported": indexed_route_supported,
        "boundary_recorded": boundary_recorded,
        "status": (
            "decode_indexed_route_missing_recorded"
            if boundary_recorded
            else "decode_boundary_unverified"
        ),
    }


def size_gate_boundary(
    *,
    promotion_allowed: bool,
    gates: dict[str, Any],
    long_prefill: dict[str, Any],
    a3b_moe_decode_boundary: dict[str, Any],
) -> dict[str, Any]:
    if gates["a3b_broader_coherence_promotion_grade"]:
        a3b_coherence_status = "promotion_grade"
    elif gates["a3b_broader_coherence_no_hard_errors"] and gates[
        "a3b_broader_coherence_capped_rows_present"
    ]:
        a3b_coherence_status = "no_hard_errors_but_capped_rows_not_promotion_grade"
    elif gates["a3b_broader_coherence_no_hard_errors"]:
        a3b_coherence_status = "no_hard_errors_unclassified"
    else:
        a3b_coherence_status = "hard_error_or_missing"
    return {
        "status": "size_gated_incomplete",
        "promotion_allowed": promotion_allowed,
        "dense_4b": {
            "status": "rejected" if gates["dense_4b_rejected"] else "candidate",
            "next_unblocked_step": "new_calibration_that_beats_mq4_on_4b_kld_and_coherence",
        },
        "dense_9b": {
            "status": "rejected" if gates["dense_9b_rejected"] else "candidate",
            "next_unblocked_step": "new_calibration_that_beats_mq4_on_9b_ppl_kld_and_coherence",
        },
        "dense_27b": {
            "status": (
                "candidate_incomplete_kld_reference"
                if not gates["dense_27b_kld_reference_present"]
                else "candidate_incomplete_kld_row"
            ),
            "next_unblocked_step": "manifest_pin_qwen35_27b_bf16_or_q8_kld_reference",
        },
        "a3b": {
            "status": "research_blocked",
            "coherence_status": a3b_coherence_status,
            "broader_coherence_no_hard_errors": gates["a3b_broader_coherence_no_hard_errors"],
            "broader_coherence_capped_rows_present": gates[
                "a3b_broader_coherence_capped_rows_present"
            ],
            "broader_coherence_case_count": gates["a3b_broader_coherence_case_count"],
            "broader_coherence_hit_max_tokens_count": gates[
                "a3b_broader_coherence_hit_max_tokens_count"
            ],
            "promotion_grade_coherence_present": gates["a3b_broader_coherence_promotion_grade"],
            "next_unblocked_step": "add_a3b_kld_refs_and_paired_mq3_dflash_sidecars",
        },
        "long_prefill_no_gpu_contract": {
            "status": long_prefill["status"],
            "test_command": long_prefill["test_command"],
            "test_name": long_prefill["test_name"],
            "all_phrase_checks_present": long_prefill["all_phrase_checks_present"],
            "artifact_backed_long_prefill_evidence_present": (
                long_prefill["artifact_backed_long_prefill_evidence_present"]
            ),
            "invariants": long_prefill["invariants"],
            "next_unblocked_step": "run_artifact_backed_mq3_a3b_long_prefill_coherence_and_perf",
        },
        "moe_decode": {
            "status": a3b_moe_decode_boundary["status"],
            "indexed_route_supported": a3b_moe_decode_boundary["indexed_route_supported"],
            "next_unblocked_step": "wire_indexed_mq3_a3b_moe_decode_route_or_keep_decode_out_of_scope",
        },
        "next_unblocked_step": (
            "manifest-pin qwen3.5-27B KLD reference or generate a new small-model MQ3 calibration"
        ),
    }


def build_status(
    paths: dict[str, Path] | None = None,
    model_roots: Iterable[Path] | None = None,
) -> dict[str, Any]:
    if paths is None:
        paths = DEFAULT_PATHS
    if model_roots is None:
        model_roots = DEFAULT_MODEL_ROOTS
    artifacts = artifact_summary(paths["provenance"])
    live_artifacts = live_artifact_inventory(model_roots)
    refs = reference_status(paths["kld_refs"])
    kld_4b = kld_comparison(paths["kld_4b"], "qwen3.5-4b.mq3", "qwen3.5-4b.mq4")
    kld_9b = kld_comparison(paths["kld_9b"], "qwen3.5-9b.mq3", "qwen3.5-9b.mq4")
    ppl_9b = ppl_comparison(paths["ppl_9b"], "qwen35-9b-mq3", "qwen35-9b-mq4")
    ppl_27b = ppl_comparison(paths["ppl_27b"], "qwen35-27b-mq3", "qwen35-27b-mq4")
    ppl_qwen35_a3b = ppl_comparison(paths["ppl_a3b"], "qwen35-a3b-mq3", "qwen35-a3b-mq4")
    ppl_qwen36_a3b = ppl_comparison(paths["ppl_a3b"], "qwen36-a3b-mq3", "qwen36-a3b-mq4")
    a3b_coherence = a3b_coherence_summary(paths["a3b_coherence"])
    a3b_dflash = dflash_sidecar_summary(paths["a3b_dflash_audit"])
    ar_perf = perf_summary(paths["ar_perf"], format_token="qwen35-27b-mq3")
    dflash_perf = perf_summary(paths["dflash_perf"], format_token="qwen35-27b-mq3")
    a3b_moe_decode_boundary = mq3_moe_decode_boundary_evidence()
    long_prefill = long_prefill_shape_contract(paths["long_prefill_audit"])
    size_audit_present = paths["size_audit"].exists()

    gates = {
        "canonical_mq3_artifacts_present": artifacts["canonical_mq3_artifacts_present"],
        "live_canonical_mq3_artifacts_present": live_artifacts["canonical_required_complete"],
        "dense_4b_rejected": not kld_4b["candidate_beats_control"],
        "dense_9b_rejected": not kld_9b["candidate_beats_control"] and not ppl_9b["candidate_beats_control"],
        "dense_27b_ppl_beats_mq4": ppl_27b["candidate_beats_control"],
        "dense_27b_kld_reference_present": refs["qwen35_27b_present"],
        "dense_27b_kld_evidence_present": False,
        "dense_27b_ar_perf_present": ar_perf["all_matching_cases_ok"],
        "dense_27b_ar_reached_token_cap": ar_perf["all_matching_cases_reached_token_cap"],
        "dense_27b_dflash_perf_present": dflash_perf["all_matching_cases_ok"],
        "dense_27b_dflash_token_attractor_clean": dflash_perf["all_matching_token_attractor_clean"],
        "a3b_qwen35_ppl_beats_mq4": ppl_qwen35_a3b["candidate_beats_control"],
        "a3b_qwen36_ppl_beats_mq4": ppl_qwen36_a3b["candidate_beats_control"],
        "a3b_broader_coherence_no_hard_errors": a3b_coherence["no_hard_errors"],
        "a3b_broader_coherence_case_count": a3b_coherence["case_count"],
        "a3b_broader_coherence_hit_max_tokens_count": a3b_coherence["hit_max_tokens_count"],
        "a3b_broader_coherence_capped_rows_present": a3b_coherence["capped_rows_present"],
        "a3b_broader_coherence_promotion_grade": a3b_coherence[
            "promotion_grade_no_hard_errors_and_uncapped"
        ],
        "a3b_kld_references_present": refs["qwen35_a3b_present"] and refs["qwen36_a3b_present"],
        "a3b_dflash_sidecars_present": a3b_dflash["paired_a3b_dflash_sidecars_present"],
        "a3b_moe_decode_boundary_recorded": a3b_moe_decode_boundary["boundary_recorded"],
        "a3b_moe_decode_indexed_route_supported": a3b_moe_decode_boundary["indexed_route_supported"],
        "long_prefill_shape_audit_present": long_prefill["audit_present"],
        "long_prefill_shape_contract_covered": long_prefill["all_phrase_checks_present"],
        "long_prefill_runtime_evidence_present": long_prefill["artifact_backed_long_prefill_evidence_present"],
        "size_gate_audit_present": size_audit_present,
    }
    promotion_allowed = (
        gates["canonical_mq3_artifacts_present"]
        and gates["live_canonical_mq3_artifacts_present"]
        and not gates["dense_4b_rejected"]
        and not gates["dense_9b_rejected"]
        and gates["dense_27b_ppl_beats_mq4"]
        and gates["dense_27b_kld_reference_present"]
        and gates["dense_27b_kld_evidence_present"]
        and gates["dense_27b_ar_perf_present"]
        and gates["dense_27b_ar_reached_token_cap"]
        and gates["dense_27b_dflash_perf_present"]
        and gates["dense_27b_dflash_token_attractor_clean"]
        and gates["a3b_qwen35_ppl_beats_mq4"]
        and gates["a3b_qwen36_ppl_beats_mq4"]
        and gates["a3b_broader_coherence_promotion_grade"]
        and gates["a3b_kld_references_present"]
        and gates["a3b_dflash_sidecars_present"]
        and gates["a3b_moe_decode_indexed_route_supported"]
        and gates["long_prefill_shape_audit_present"]
        and gates["long_prefill_shape_contract_covered"]
        and gates["long_prefill_runtime_evidence_present"]
    )
    blockers = []
    if not gates["live_canonical_mq3_artifacts_present"]:
        blockers.append(
            "Live local scan is missing canonical MQ3 artifacts: "
            + ", ".join(live_artifacts["canonical_required_missing"])
        )
    if gates["dense_4b_rejected"]:
        blockers.append("Qwen3.5 4B MQ3 is rejected by current coherence/KLD evidence")
    if gates["dense_9b_rejected"]:
        blockers.append("Qwen3.5 9B MQ3 is rejected by current coherence/PPL/KLD evidence")
    if not gates["dense_27b_kld_reference_present"]:
        blockers.append("Qwen3.5 27B MQ3 lacks a comparable BF16/Q8 KLD reference")
    if not gates["dense_27b_kld_evidence_present"]:
        blockers.append("Qwen3.5 27B MQ3 lacks a BF16/Q8-referenced KLD row")
    if gates["dense_27b_ar_perf_present"] and not gates["dense_27b_ar_reached_token_cap"]:
        blockers.append("Qwen3.5 27B MQ3 AR perf row stopped before the max-token cap")
    if gates["dense_27b_dflash_perf_present"] and not gates["dense_27b_dflash_token_attractor_clean"]:
        blockers.append("Qwen3.5 27B MQ3 DFlash/spec rows are not token-attractor clean")
    if not gates["a3b_kld_references_present"]:
        blockers.append("Qwen3.5/Qwen3.6 A3B MQ3 lacks matching KLD references")
    if gates["a3b_broader_coherence_capped_rows_present"]:
        blockers.append(
            "A3B MQ3 broader coherence is no-hard-error smoke only; "
            f"{gates['a3b_broader_coherence_hit_max_tokens_count']}/"
            f"{gates['a3b_broader_coherence_case_count']} rows hit the max-token cap"
        )
    if not gates["a3b_dflash_sidecars_present"]:
        blockers.append("A3B MQ3 DFlash/spec is blocked on missing paired draft sidecars")
    if not gates["a3b_moe_decode_boundary_recorded"]:
        blockers.append("MQ3 A3B MoE decode boundary lacks a no-GPU source test")
    elif not gates["a3b_moe_decode_indexed_route_supported"]:
        blockers.append(
            "MQ3 A3B MoE prefill has grouped path2 coverage, but MoE decode "
            "lacks an indexed routed-expert path"
        )
    if gates["long_prefill_shape_contract_covered"] and not gates["long_prefill_runtime_evidence_present"]:
        blockers.append(
            "MQ3 A3B long-prefill has no-GPU shape coverage but lacks "
            "artifact-backed coherence/perf evidence"
        )
    if not gates["a3b_qwen36_ppl_beats_mq4"]:
        blockers.append("Qwen3.6 A3B MQ3 currently regresses PPL versus MQ4")

    boundary = size_gate_boundary(
        promotion_allowed=promotion_allowed,
        gates=gates,
        long_prefill=long_prefill,
        a3b_moe_decode_boundary=a3b_moe_decode_boundary,
    )
    next_work = [
        "Keep 4B and 9B rejected unless a new MQ3 calibration overturns current KLD/PPL/coherence evidence.",
        "Manifest-pin a comparable qwen3.5-27B BF16/Q8 KLD reference and rerun MQ3/MQ4 KLD.",
        "Add matching Qwen3.5/Qwen3.6 A3B KLD references and paired MQ3 A3B DFlash sidecars before MoE spec claims.",
        "Run artifact-backed MQ3 A3B long-prefill coherence/perf; the current audit is no-GPU shape coverage only.",
        "Wire an indexed MQ3 A3B MoE decode route or keep MoE decode outside promotion scope.",
    ]

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "format": "mq3",
        "status": "candidate-size-gated-incomplete",
        "promotion_allowed": promotion_allowed,
        "artifact_provenance": artifacts,
        "live_artifact_inventory": live_artifacts,
        "quality": {
            "kld_4b": kld_4b,
            "kld_9b": kld_9b,
            "ppl_9b": ppl_9b,
            "ppl_27b": ppl_27b,
            "ppl_qwen35_a3b": ppl_qwen35_a3b,
            "ppl_qwen36_a3b": ppl_qwen36_a3b,
            "kld_references": refs,
            "a3b_coherence": a3b_coherence,
        },
        "perf": {
            "dense_27b_ar": ar_perf,
            "dense_27b_dflash": dflash_perf,
            "a3b_dflash_sidecars": a3b_dflash,
        },
        "runtime": {
            "a3b_moe_decode_boundary": a3b_moe_decode_boundary,
            "long_prefill_shape_contract": long_prefill,
        },
        "gates": gates,
        "size_gate_boundary": boundary,
        "quality_boundary": {
            "dense_small_model_status": "4b_and_9b_rejected_current_calibration",
            "dense_27b_status": boundary["dense_27b"]["status"],
            "a3b_status": boundary["a3b"]["status"],
            "a3b_coherence_status": boundary["a3b"]["coherence_status"],
            "promotion_requires_new_quality_evidence": True,
        },
        "blockers": blockers,
        "next_work": next_work,
        "fixture_decisions": {
            "qwen3.5-4b": "reject-boundary-weakness",
            "qwen3.5-9b": "reject-quality-risk",
            "qwen3.5-27b": "candidate-incomplete-kld-reference",
            "qwen3.5-35b-a3b": "research-candidate-missing-kld-and-dflash-fixtures",
            "qwen3.6-35b-a3b": "research-candidate-ppl-regressed-missing-kld-and-dflash-fixtures",
        },
        "decision": (
            "keep MQ3 size-gated: 4B and 9B are rejects, 27B dense remains "
            "KLD-reference blocked, and A3B remains research-scoped until "
            "matching KLD refs plus paired draft sidecars exist"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--model-root",
        action="append",
        default=[],
        help="Model/artifact root to scan for live MQ3 artifacts; repeatable",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    model_roots = tuple(Path(root) for root in args.model_root) if args.model_root else None
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
