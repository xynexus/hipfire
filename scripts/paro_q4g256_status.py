#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize ParoQ4G256 prototype status.

ParoQ4G256 is intentionally prototype-only until a true group_size=256
producer/export exists.  This helper joins the source inventory, contract audit,
and CPU probe surface into one machine-readable gate report.
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
SCHEMA = "hipfire.paro_q4g256_status.gfx1151.v0"
DEFAULT_SOURCE_INVENTORY = RESULTS_DIR / "2026-06-03-paro-q4g256-source-inventory.json"
DEFAULT_CONTRACT_AUDIT = RESULTS_DIR / "2026-06-03-paro-q4g256-contract-audit.md"
DEFAULT_PROBE_SCRIPT = ROOT / "scripts" / "paroquant_g256_probe.py"
DEFAULT_UPSTREAM_SUMMARY = ROOT / "docs" / "investigations" / "paro-g256-perfmax" / "SUMMARY.md"
DEFAULT_PROBE_RESULTS = (
    ROOT / "docs" / "investigations" / "paro-g256-perfmax" / "g256-probe-0.8b.json",
    ROOT / "docs" / "investigations" / "paro-g256-perfmax" / "g256-probe-9b.json",
)
DEFAULT_OUT = RESULTS_DIR / "2026-06-03-paro-q4g256-status.json"
EVIDENCE_TARGETS = {
    "source_inventory": DEFAULT_SOURCE_INVENTORY,
    "contract_audit": DEFAULT_CONTRACT_AUDIT,
    "status": DEFAULT_OUT,
    "cpu_g256_probe": RESULTS_DIR / "2026-06-03-paro-q4g256-cpu-probe.json",
    "paro_oracle_comparison": RESULTS_DIR / "2026-06-03-paro-q4g256-oracle.json",
    "payload_ratio": RESULTS_DIR / "2026-06-03-paro-q4g256-payload-ratio.json",
    "kld_ppl": RESULTS_DIR / "2026-06-03-paro-q4g256-kld-ppl.json",
    "runtime_dtype_audit": RESULTS_DIR / "2026-06-03-paro-q4g256-runtime-dtype-audit.md",
}
PAYLOAD_RATIO_TARGET = 1.03
KLD_PPL_TOLERANCE_RATIO = 1.05
REQUIRED_PARO_SUFFIXES = (
    "qweight",
    "qzeros",
    "scales",
    "pairs",
    "theta",
    "channel_scales",
)

CONTRACT_STAGES = (
    "true_group_size_256_source",
    "cpu_g256_probe",
    "paro_oracle_comparison",
    "payload_ratio_lte_1_03",
    "kld_ppl_within_5_percent_of_paro_q4g128",
    "runtime_dtype_container_work",
)

ORIGIN_RELEVANT_COMMITS = (
    {
        "commit": "f1e2cef815e504a8df1d9142910c3f442dd5d526",
        "short": "f1e2cef8",
        "subject": "docs(paro-g256-perfmax): Phase 3 Lever 2 + gfx12 asymptote + final SUMMARY",
        "impact": (
            "records Exit B for the G256 perfmax thread: G256 is opt-in "
            "research, while G128 levers are the shipped runtime path"
        ),
    },
    {
        "commit": "907cbb2f93c5641bd8969ff8235bb91c729df037",
        "short": "907cbb2f",
        "subject": "docs(paro-g256-perfmax): Phase 4 — A3B-PARO 60+ tok/s on gfx1201 + Lever 4 NaN fix",
        "impact": (
            "captures ParoQ4G128 A3B runtime context via PR #319; useful for "
            "ranking Paro work, but not a true PARO4G256 source/export"
        ),
    },
    {
        "commit": "79d30777f7d77abcc0b33d54aa0bd4c7dfddf6e3",
        "short": "79d30777",
        "subject": "research(paroquant): add G256 milestone probes",
        "impact": (
            "adds probe/milestone scaffolding for Paro G256; it is not a "
            "native group_size=256 source checkpoint or runtime DType path"
        ),
    },
    {
        "commit": "020ed5e124f3d7f2632e95ea8199b98c417c845a",
        "short": "020ed5e1",
        "subject": "Revert \"test(paro-la-gates): failing stub for MQ4G256 codec\"",
        "impact": (
            "confirms the older MQ4G256 codec stub was reverted; do not infer "
            "runtime PARO4G256/PARO4G256_MQ support from that history"
        ),
    },
)

TABLE_ROW_RE = re.compile(
    r"^\|\s*`(?P<variant>PARO4G256(?:_MQ)?)`(?P<label>.*?)\|\s*"
    r"(?P<avg_nrmse>[0-9.]+)\s*\|\s*"
    r"(?P<worst_nrmse>[0-9.]+)\s*\|\s*"
    r"(?P<payload_ratio>[0-9.]+)x\s*\|"
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
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
    commits = []
    for item in ORIGIN_RELEVANT_COMMITS:
        commits.append(
            {
                **item,
                "present_on_origin_master": commit_on_ref(item["commit"]),
                "present_on_head": commit_on_ref(item["commit"], "HEAD"),
            }
        )
    return {
        **git_ahead_behind(),
        "relevant_upstream_commits": commits,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_inventory(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    native = payload.get("native_paro", {})
    decision = payload.get("decision", {})
    complete_module_count = int(native.get("complete_module_count", 0))
    incomplete_module_count = int(native.get("incomplete_module_count", 0))
    filename_hit_count = int(payload.get("filename_hit_count", 0))
    scan_errors = payload.get("scan_errors", [])
    if complete_module_count:
        search_result = "native_paro_modules_found"
    elif incomplete_module_count:
        search_result = "incomplete_paro_suffix_sets_found"
    elif filename_hit_count or scan_errors:
        search_result = "paro_hints_without_complete_modules"
    else:
        search_result = "no_paro_suffix_or_filename_hits"
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "roots": payload.get("roots", []),
        "files_seen": payload.get("files_seen"),
        "filename_hit_count": filename_hit_count,
        "safetensor_dirs_scanned": payload.get("safetensor_dirs_scanned"),
        "safetensor_files_scanned": payload.get("safetensor_files_scanned"),
        "scan_errors": scan_errors,
        "complete_module_count": complete_module_count,
        "incomplete_module_count": incomplete_module_count,
        "g128_complete_module_count": int(native.get("g128_complete_module_count", 0)),
        "g256_complete_module_count": int(native.get("g256_complete_module_count", 0)),
        "native_paro_source_found": bool(decision.get("native_paro_source_found")),
        "native_paro_g256_source_found": bool(decision.get("native_paro_g256_source_found")),
        "quality_state": decision.get("quality_state"),
        "source_search_result": search_result,
    }


def parse_format_loss_table(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    variants: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        variant = match.group("variant")
        variants[variant] = {
            "label": match.group("label").strip(),
            "avg_output_nrmse": float(match.group("avg_nrmse")),
            "worst_output_nrmse": float(match.group("worst_nrmse")),
            "avg_payload_ratio_vs_source": float(match.group("payload_ratio")),
            "source": "regrouped G128 format-loss probe recorded in contract audit",
            "true_g256_quality_evidence": False,
        }
    mq = variants.get("PARO4G256_MQ", {})
    true_payload_variants = [
        item
        for item in variants.values()
        if item.get("true_g256_quality_evidence")
        and item.get("avg_payload_ratio_vs_source") is not None
    ]
    true_payload_gate = bool(true_payload_variants) and all(
        float(item["avg_payload_ratio_vs_source"]) <= PAYLOAD_RATIO_TARGET
        for item in true_payload_variants
    )
    return {
        "path": str(path),
        "variants": variants,
        "payload_ratio_target": PAYLOAD_RATIO_TARGET,
        "regrouped_mq_payload_ratio_gate_passed": bool(mq)
        and float(mq["avg_payload_ratio_vs_source"]) <= PAYLOAD_RATIO_TARGET,
        "mq_payload_ratio_gate_passed": true_payload_gate,
        "current_true_g256_payload_ratio_evidence_present": bool(true_payload_variants),
        "current_true_g256_payload_ratio_gate_passed": true_payload_gate,
        "true_g256_quality_evidence_present": any(
            bool(item.get("true_g256_quality_evidence")) for item in variants.values()
        ),
    }


def probe_surface(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_local_only_flag": "--local-only" in text,
        "has_schema": "hipfire.astrea.paro_g256_probe.v0" in text,
        "records_regrouped_g128_caveat": "not a true G256 ParoQuant calibration run" in text,
    }


def parse_upstream_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    exit_decision = "B" if "Exit decision: B" in text else None
    opt_in_research = "G256 quality-viable as opt-in research, not default" in text
    return {
        "path": str(path),
        "exists": path.exists(),
        "exit_decision": exit_decision,
        "g256_opt_in_research_not_default": opt_in_research,
        "mentions_pr319_paro_g128": "PR #319" in text and "A3B-PARO" in text,
        "mentions_gfx1151_conditional_ports": "gfx1151" in text
        and "conditional ports" in text.lower(),
        "g256_default_ready": False if opt_in_research else None,
    }


def summarize_probe_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "schema": None,
            "records_regrouped_g128_caveat": False,
        }
    payload = read_json(path)
    summary = payload.get("summary", {})
    payload_ratios = summary.get("avg_payload_ratio_vs_source", {})
    output_nrmse = summary.get("avg_output_nrmse", {})
    worst_nrmse = summary.get("worst_output_nrmse", {})
    caveat = str(payload.get("caveat", ""))
    return {
        "path": str(path),
        "exists": True,
        "schema": payload.get("schema"),
        "model": payload.get("model"),
        "modules_probed": payload.get("modules_probed"),
        "records_regrouped_g128_caveat": "not a true G256 ParoQuant calibration run" in caveat,
        "avg_payload_ratio_vs_source": payload_ratios,
        "avg_output_nrmse": output_nrmse,
        "worst_output_nrmse": worst_nrmse,
    }


def upstream_g256_evidence(summary_path: Path, probe_paths: tuple[Path, ...]) -> dict[str, Any]:
    summary = parse_upstream_summary(summary_path)
    probes = [summarize_probe_result(path) for path in probe_paths]
    present_probes = [probe for probe in probes if probe.get("exists")]
    all_probe_caveats = bool(probes) and all(
        bool(probe.get("records_regrouped_g128_caveat")) for probe in probes
    )
    mq_body_ratios = [
        float(probe.get("avg_payload_ratio_vs_source", {}).get("paro4g256_mq_body_plus_side"))
        for probe in present_probes
        if probe.get("avg_payload_ratio_vs_source", {}).get("paro4g256_mq_body_plus_side") is not None
    ]
    awq_ratios = [
        float(probe.get("avg_payload_ratio_vs_source", {}).get("paro4g256_awq"))
        for probe in present_probes
        if probe.get("avg_payload_ratio_vs_source", {}).get("paro4g256_awq") is not None
    ]
    research_evidence = (
        summary["exists"]
        and summary["exit_decision"] == "B"
        and bool(summary["g256_opt_in_research_not_default"])
        and len(present_probes) == len(probes)
        and all_probe_caveats
    )
    return {
        "summary": summary,
        "probes": {
            "paths": [str(path) for path in probe_paths],
            "items": probes,
            "probe_count": len(present_probes),
            "all_probe_results_present": len(present_probes) == len(probes),
            "all_probe_caveats_regrouped_g128": all_probe_caveats,
            "max_mq_body_payload_ratio_vs_source": max(mq_body_ratios)
            if mq_body_ratios
            else None,
            "max_awq_payload_ratio_vs_source": max(awq_ratios) if awq_ratios else None,
        },
        "research_evidence_present": research_evidence,
        "evidence_class": "regrouped_g128_opt_in_research" if research_evidence else "missing",
        "true_g256_source_evidence_present": False,
        "runtime_dtype_container_unblocked_by_upstream": False,
        "promotion_proof": False,
    }


def runtime_surface(source_root: Path) -> dict[str, Any]:
    hfq = source_root / "crates" / "hipfire-runtime" / "src" / "hfq.rs"
    qwen35 = source_root / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs"
    hfq_text = hfq.read_text(encoding="utf-8") if hfq.exists() else ""
    qwen35_text = qwen35.read_text(encoding="utf-8") if qwen35.exists() else ""
    # Avoid counting comments in planning docs; this is intentionally scoped to
    # current runtime source files.
    dtype_tokens = ("PARO4G256", "PARO4G256_MQ")
    return {
        "source_root": str(source_root),
        "hfq_runtime_dtype_present": any(token in hfq_text for token in dtype_tokens),
        "qwen35_runtime_dtype_present": any(token in qwen35_text for token in dtype_tokens),
    }


def contract_state(
    *,
    inventory: dict[str, Any],
    gates: dict[str, bool],
    format_loss: dict[str, Any],
    upstream_evidence: dict[str, Any],
) -> dict[str, Any]:
    true_source = gates["native_g256_source_found"]
    probe_runnable = gates["cpu_probe_runnable_now"]
    quality_state = inventory.get("quality_state") or "UNKNOWN"
    runtime_blockers = []
    if not true_source:
        runtime_blockers.append("true_group_size_256_source")
    if not probe_runnable:
        runtime_blockers.append("cpu_g256_probe")
    if not gates["true_g256_quality_evidence_present"]:
        runtime_blockers.append("paro_oracle_comparison")
    if not gates["quality_comparable_to_paro_q4g128"]:
        runtime_blockers.append("kld_ppl_within_5_percent_of_paro_q4g128")
    return {
        "required_stage_order": list(CONTRACT_STAGES),
        "producer_contract": {
            "required_tensor_families": list(REQUIRED_PARO_SUFFIXES),
            "required_group_size": 256,
            "complete_native_paro_module_count": inventory["complete_module_count"],
            "complete_g128_module_count": inventory["g128_complete_module_count"],
            "complete_g256_module_count": inventory["g256_complete_module_count"],
            "incomplete_module_count": inventory["incomplete_module_count"],
            "filename_hit_count": inventory["filename_hit_count"],
            "source_search_result": inventory["source_search_result"],
            "quality_verifiable_only_with_true_g256_source": True,
        },
        "upstream_g256_evidence": upstream_evidence,
        "current_stage": (
            "blocked_before_true_group_size_256_source"
            if not true_source
            else "blocked_before_cpu_g256_probe"
            if not probe_runnable
            else "blocked_before_paro_oracle_and_kld_ppl"
        ),
        "quality_state": quality_state,
        "quality_marked_unverifiable": quality_state.upper() == "UNVERIFIABLE",
        "evidence_class": (
            "regrouped_g128_format_loss_only"
            if format_loss["variants"] and not gates["true_g256_quality_evidence_present"]
            else "true_g256_quality"
            if gates["true_g256_quality_evidence_present"]
            else "missing"
        ),
        "payload_ratio_target": PAYLOAD_RATIO_TARGET,
        "kld_ppl_tolerance_ratio": KLD_PPL_TOLERANCE_RATIO,
        "runtime_work_allowed": (
            true_source
            and probe_runnable
            and gates["payload_ratio_gate_passed"]
            and gates["current_true_g256_payload_ratio_gate_passed"]
            and gates["true_g256_quality_evidence_present"]
            and gates["quality_comparable_to_paro_q4g128"]
        ),
        "runtime_work_blocked_by": runtime_blockers,
    }


def next_unblocked_step(*, gates: dict[str, bool], contract: dict[str, Any]) -> str:
    if not gates["native_g256_source_found"]:
        return "locate_or_generate_true_group_size_256_paro_source"
    if not gates["cpu_probe_runnable_now"]:
        return "rerun_cpu_g256_probe_against_native_paro_source"
    if not gates["true_g256_quality_evidence_present"]:
        return "run_paro_oracle_against_true_g256_export"
    if not gates["quality_comparable_to_paro_q4g128"]:
        return "run_kld_ppl_within_5_percent_of_paro_q4g128"
    if not contract["runtime_work_allowed"]:
        return "wire_runtime_dtype_container_after_contract_gates"
    return "verify_runtime_dtype_container_and_readiness_matrix"


def prototype_boundary(
    *,
    inventory: dict[str, Any],
    gates: dict[str, bool],
    contract: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    payload_ratio_only = (
        gates["regrouped_payload_ratio_gate_passed"]
        and not gates["true_g256_quality_evidence_present"]
    )
    return {
        "current_stage": contract["current_stage"],
        "true_group_size_256_source_required": True,
        "true_group_size_256_source_found": gates["native_g256_source_found"],
        "producer_contract": contract["producer_contract"],
        "source_inventory_quality_state": inventory.get("quality_state"),
        "quality_marked_unverifiable": contract["quality_marked_unverifiable"],
        "format_loss_ratio_target": PAYLOAD_RATIO_TARGET,
        "format_loss_ratio_passed": gates["regrouped_payload_ratio_gate_passed"],
        "regrouped_payload_ratio_gate_passed": gates["regrouped_payload_ratio_gate_passed"],
        "current_true_g256_payload_ratio_evidence_present": gates[
            "current_true_g256_payload_ratio_evidence_present"
        ],
        "current_true_g256_payload_ratio_gate_passed": gates[
            "current_true_g256_payload_ratio_gate_passed"
        ],
        "format_loss_evidence_class": contract["evidence_class"],
        "payload_ratio_only_not_promotion_evidence": payload_ratio_only,
        "true_g256_quality_evidence_present": gates["true_g256_quality_evidence_present"],
        "kld_ppl_within_5_percent_of_paro_q4g128_required": True,
        "kld_ppl_within_5_percent_of_paro_q4g128_proven": gates[
            "quality_comparable_to_paro_q4g128"
        ],
        "runtime_dtype_container_work_allowed": contract["runtime_work_allowed"],
        "runtime_dtype_container_work_blocked_by": contract["runtime_work_blocked_by"],
        "runtime_dtypes_present": {
            "hfq_runtime_dtype_present": runtime["hfq_runtime_dtype_present"],
            "qwen35_runtime_dtype_present": runtime["qwen35_runtime_dtype_present"],
        },
        "upstream_g256_research_evidence_present": gates[
            "upstream_g256_research_evidence_present"
        ],
        "upstream_g256_evidence_class": contract["upstream_g256_evidence"]["evidence_class"],
        "upstream_g256_evidence_is_promotion_proof": contract["upstream_g256_evidence"][
            "promotion_proof"
        ],
        "artifact_generation_without_true_source_allowed": False,
        "next_unblocked_step": next_unblocked_step(gates=gates, contract=contract),
    }


def prototype_plan(
    *,
    gates: dict[str, bool],
    contract: dict[str, Any],
    boundary: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    native_source = "<true-group-size-256-paro-safetensors-dir-or-hf-repo>"
    imported_hfq = "<imported-paro-q4g256.hfq>"
    next_step = next_unblocked_step(gates=gates, contract=contract)
    stages = {
        "true_group_size_256_source": {
            "satisfied": gates["native_g256_source_found"],
            "required_group_size": 256,
            "required_tensor_families": list(REQUIRED_PARO_SUFFIXES),
            "current_result": inventory["source_search_result"],
            "complete_g256_module_count": inventory["g256_complete_module_count"],
            "command": (
                "python3 scripts/paroquant_inventory.py --pretty --out "
                f"{repo_path(EVIDENCE_TARGETS['source_inventory'])}"
            ),
            "artifact": repo_path(EVIDENCE_TARGETS["source_inventory"]),
            "blocks": ["cpu_g256_probe"],
        },
        "cpu_g256_probe": {
            "satisfied": gates["cpu_probe_runnable_now"],
            "blocked_by": []
            if gates["native_g256_source_found"]
            else ["true_group_size_256_source"],
            "command_template": (
                "python3 scripts/paroquant_g256_probe.py --model "
                f"{native_source} --local-only --pretty > "
                f"{repo_path(EVIDENCE_TARGETS['cpu_g256_probe'])}"
            ),
            "artifact": repo_path(EVIDENCE_TARGETS["cpu_g256_probe"]),
            "records_regrouped_g128_caveat_required": False,
        },
        "paro_oracle_comparison": {
            "satisfied": gates["true_g256_quality_evidence_present"],
            "blocked_by": ["cpu_g256_probe"],
            "command_template": (
                "python3 scripts/astrea.py paro-oracle --source "
                f"{native_source} --hfq {imported_hfq} --pretty --out "
                f"{repo_path(EVIDENCE_TARGETS['paro_oracle_comparison'])}"
            ),
            "artifact": repo_path(EVIDENCE_TARGETS["paro_oracle_comparison"]),
            "requirement": "compare imported true-G256 bytes against a source oracle or equivalent true-G256 CPU probe",
        },
        "payload_ratio_lte_1_03": {
            "satisfied": gates["current_true_g256_payload_ratio_gate_passed"],
            "blocked_by": ["cpu_g256_probe"],
            "target_ratio": PAYLOAD_RATIO_TARGET,
            "artifact": repo_path(EVIDENCE_TARGETS["payload_ratio"]),
            "regrouped_g128_payload_ratio_is_promotable": False,
        },
        "kld_ppl_within_5_percent_of_paro_q4g128": {
            "satisfied": gates["quality_comparable_to_paro_q4g128"],
            "blocked_by": ["paro_oracle_comparison", "payload_ratio_lte_1_03"],
            "tolerance_ratio": KLD_PPL_TOLERANCE_RATIO,
            "artifact": repo_path(EVIDENCE_TARGETS["kld_ppl"]),
        },
        "runtime_dtype_container_work": {
            "satisfied": gates["runtime_dtype_container_ready"] and contract["runtime_work_allowed"],
            "allowed": contract["runtime_work_allowed"],
            "blocked_by": contract["runtime_work_blocked_by"],
            "artifact": repo_path(EVIDENCE_TARGETS["runtime_dtype_audit"]),
            "requirement": "only wire PARO4G256/PARO4G256_MQ runtime DType/container support after every producer and quality gate passes",
        },
    }
    return {
        "status": contract["current_stage"],
        "next_unblocked_step": next_step,
        "stage_order": list(CONTRACT_STAGES),
        "stages": stages,
        "evidence_artifact_targets": {
            name: repo_path(path) for name, path in EVIDENCE_TARGETS.items()
        },
        "quality_unverifiable_until_true_g256_source": boundary["quality_marked_unverifiable"],
        "runtime_dtype_container_work_allowed": boundary["runtime_dtype_container_work_allowed"],
        "upstream_research_evidence_is_promotion_proof": boundary[
            "upstream_g256_evidence_is_promotion_proof"
        ],
        "next_work": [
            "Locate or produce a true group_size=256 Paro source with qweight/qzeros/scales/pairs/theta/channel_scales.",
            "Rerun scripts/paroquant_g256_probe.py --local-only against that true G256 source; regrouped-G128 probes remain research context only.",
            "Run a true-G256 oracle comparison and payload-ratio artifact, requiring <=1.03x ParoQ4G128 payload.",
            "Run same-run KLD/PPL versus ParoQ4G128 and require <=1.05x before runtime DType/container work.",
            "Keep PARO4G256/PARO4G256_MQ runtime DType/container work blocked until prototype_plan.runtime_dtype_container_work_allowed=true.",
        ],
    }


def build_status(
    *,
    source_inventory: Path = DEFAULT_SOURCE_INVENTORY,
    contract_audit: Path = DEFAULT_CONTRACT_AUDIT,
    probe_script: Path = DEFAULT_PROBE_SCRIPT,
    source_root: Path = ROOT,
    upstream_summary: Path = DEFAULT_UPSTREAM_SUMMARY,
    probe_results: tuple[Path, ...] = DEFAULT_PROBE_RESULTS,
) -> dict[str, Any]:
    inventory = summarize_inventory(source_inventory)
    format_loss = parse_format_loss_table(contract_audit)
    probe = probe_surface(probe_script)
    runtime = runtime_surface(source_root)
    upstream_evidence = upstream_g256_evidence(upstream_summary, tuple(probe_results))
    probe_available = (
        probe["exists"]
        and probe["has_local_only_flag"]
        and probe["has_schema"]
        and probe["records_regrouped_g128_caveat"]
    )
    runtime_ready = runtime["hfq_runtime_dtype_present"] and runtime["qwen35_runtime_dtype_present"]
    native_g256 = inventory["native_paro_g256_source_found"]
    gates = {
        "producer_contract_explicit": bool(format_loss["variants"]) and probe_available,
        "native_g256_source_found": native_g256,
        "cpu_probe_available": probe_available,
        "cpu_probe_runnable_now": probe_available and native_g256,
        "regrouped_payload_ratio_gate_passed": bool(format_loss["regrouped_mq_payload_ratio_gate_passed"]),
        "current_true_g256_payload_ratio_evidence_present": bool(
            format_loss["current_true_g256_payload_ratio_evidence_present"]
        ),
        "current_true_g256_payload_ratio_gate_passed": bool(
            format_loss["current_true_g256_payload_ratio_gate_passed"]
        ),
        "payload_ratio_gate_passed": bool(format_loss["mq_payload_ratio_gate_passed"]),
        "true_g256_quality_evidence_present": bool(format_loss["true_g256_quality_evidence_present"]),
        "quality_comparable_to_paro_q4g128": False,
        "runtime_dtype_container_ready": runtime_ready,
        "upstream_g256_research_evidence_present": bool(
            upstream_evidence["research_evidence_present"]
        ),
    }
    contract = contract_state(
        inventory=inventory,
        gates=gates,
        format_loss=format_loss,
        upstream_evidence=upstream_evidence,
    )
    boundary = prototype_boundary(
        inventory=inventory,
        gates=gates,
        contract=contract,
        runtime=runtime,
    )
    plan = prototype_plan(
        gates=gates,
        contract=contract,
        boundary=boundary,
        inventory=inventory,
    )
    promotion_allowed = all(gates.values())
    blockers = []
    if not gates["native_g256_source_found"]:
        blockers.append("no true group_size=256 Paro source checkpoint was found")
    if not gates["cpu_probe_runnable_now"]:
        blockers.append("CPU G256 probe cannot be rerun locally without a native Paro source")
    if not gates["true_g256_quality_evidence_present"]:
        blockers.append("only regrouped-G128 format-loss evidence exists; true G256 quality is unverifiable")
    if not gates["current_true_g256_payload_ratio_gate_passed"]:
        blockers.append("current true-G256 payload ratio <=1.03x has not been proven")
    if not gates["quality_comparable_to_paro_q4g128"]:
        blockers.append("KLD/PPL within 5% of ParoQ4G128 has not been proven")
    if not gates["runtime_dtype_container_ready"]:
        blockers.append("PARO4G256/PARO4G256_MQ runtime DType/container support is not wired")
    if not gates["upstream_g256_research_evidence_present"]:
        blockers.append("origin G256 perfmax summary/probe evidence is missing")

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "format": "paro-q4g256",
        "status": "promotion-ready" if promotion_allowed else "prototype-only",
        "promotion_allowed": promotion_allowed,
        "source_inventory": inventory,
        "format_loss_evidence": format_loss,
        "probe_surface": probe,
        "runtime_surface": runtime,
        "contract_state": contract,
        "prototype_boundary": boundary,
        "prototype_plan": plan,
        "origin_context": origin_context(),
        "gates": gates,
        "blockers": blockers,
        "next_unblocked_step": plan["next_unblocked_step"],
        "next_work": plan["next_work"],
        "decision": (
            "keep ParoQ4G256 prototype-only; do not start runtime DType/container "
            "work from regrouped-G128 evidence"
            if not promotion_allowed
            else "all ParoQ4G256 gates are present; verify readiness matrix before promotion claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", default=str(DEFAULT_SOURCE_INVENTORY))
    parser.add_argument("--contract-audit", default=str(DEFAULT_CONTRACT_AUDIT))
    parser.add_argument("--probe-script", default=str(DEFAULT_PROBE_SCRIPT))
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--upstream-summary", default=str(DEFAULT_UPSTREAM_SUMMARY))
    parser.add_argument(
        "--probe-result",
        action="append",
        dest="probe_results",
        help="Paro G256 probe result JSON; may be repeated",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_status(
        source_inventory=Path(args.source_inventory),
        contract_audit=Path(args.contract_audit),
        probe_script=Path(args.probe_script),
        source_root=Path(args.source_root),
        upstream_summary=Path(args.upstream_summary),
        probe_results=tuple(Path(path) for path in args.probe_results)
        if args.probe_results
        else DEFAULT_PROBE_RESULTS,
    )
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
