#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize ParoQ4G128 productization status from current evidence.

This joins the Paro source probes and Astrea bundle-plan artifact into one
machine-readable status report.  It does not import Paro weights and it does
not make quality claims.  Its purpose is to keep the productization boundary
explicit until a native Paro checkpoint, imported HFQ, oracle, quality, and
gfx1151 perf evidence exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
SCHEMA = "hipfire.paro_q4g128_productization_status.gfx1151.v0"

DEFAULT_PROBES = (
    DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-qwen35-9b-source-probe.json",
    DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-qwen35-a3b-source-probe.json",
    DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-qwen36-a3b-source-probe.json",
)
DEFAULT_BUNDLE_PLAN = DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-astrea-bundle-plan.json"
DEFAULT_SOURCE_INVENTORY = DEFAULT_RESULTS_DIR / "2026-06-03-paro-source-inventory.json"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-productization-status.json"

REQUIRED_PRODUCT_ENV = {
    "HIPFIRE_PARO_BATCHED",
    "HIPFIRE_MOE_PARO_I8",
    "HIPFIRE_MOE_PARO_I8_K8",
}
REQUIRED_RESEARCH_KNOBS = {
    "HIPFIRE_PARO_PREROTATE",
    "HIPFIRE_PARO_FUSE_RMSNORM",
    "HIPFIRE_PARO_FUSED_PACK2",
}
EXPECTED_SOURCE_PROBES = {
    "qwen3.5-9b",
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
}
REQUIRED_PARO_SUFFIXES = (
    "qweight",
    "qzeros",
    "scales",
    "pairs",
    "theta",
    "channel_scales",
)

ORIGIN_RELEVANT_COMMITS = (
    {
        "commit": "676338a47dff1d8bcb923cc3b636845bd684ca2f",
        "short": "676338a4",
        "subject": "fix(qwen35): batched RoPE ignored compact_offset -> phase skew after eviction (H4)",
        "impact": (
            "requires refreshed long-context, eviction, MTP, and DFlash evidence "
            "after branch reconciliation; it does not provide a ParoQ4G128 source "
            "or promotion artifact"
        ),
        "action": "rerun Paro coherence/spec evidence after importing a native source on the reconciled branch",
    },
    {
        "commit": "d5985c3e51197c70fa804f84cd694abbcd38f0d7",
        "short": "d5985c3e",
        "subject": "fix(stragglers): 4 GPU leaks/dead-doc + GGUF Promote6 Mq4Lloyd",
        "impact": (
            "contains producer/runtime cleanup relevant to the wider quant matrix; "
            "it does not satisfy the ParoQ4G128 producer, oracle, quality, or perf gates"
        ),
        "action": "refresh producer and leak-sensitive runtime rows after reconciliation",
    },
    {
        "commit": "b4adca1f3f6fc97d08a3c9a4bab98ead64a5ef99",
        "short": "b4adca1f",
        "subject": "fix(qwen35,daemon): free leaked GPU scratch - Path-2 MoE prefill buffers",
        "impact": (
            "affects A3B/MoE prefill stability, including the ParoQ4G128 productization lane; "
            "it is a runtime refresh requirement, not source or quality evidence"
        ),
        "action": "rerun A3B/MoE prefill stability and perf evidence once Paro import exists",
    },
    {
        "commit": "8de4545596e5434564d186bfe495eea785297216",
        "short": "8de45455",
        "subject": "refactor(paro): remove stale env-gated experiments",
        "impact": (
            "confirms stale Paro experiment knobs were removed upstream; it does "
            "not provide a native ParoQ4G128 checkpoint, imported HFQ artifact, "
            "oracle row, quality row, or gfx1151 perf row"
        ),
        "action": (
            "keep the promoted ParoQ4G128 path on typed source/import/oracle/"
            "coherence/KLD/perf artifacts after branch reconciliation"
        ),
    },
)

EVIDENCE_TARGETS = {
    "source_inventory": DEFAULT_SOURCE_INVENTORY,
    "source_probe_qwen35_9b": DEFAULT_PROBES[0],
    "source_probe_qwen35_a3b": DEFAULT_PROBES[1],
    "source_probe_qwen36_a3b": DEFAULT_PROBES[2],
    "bundle_plan": DEFAULT_BUNDLE_PLAN,
    "status": DEFAULT_OUT,
    "import_report": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-import.json",
    "oracle": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-oracle.json",
    "coherence": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-coherence.md",
    "finite_logit_nan": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-finite-logit-nan.json",
    "quality": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-kld-ppl.json",
    "gfx1151_perf": DEFAULT_RESULTS_DIR / "2026-06-03-paro-q4g128-gfx1151-perf.json",
}


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
    upstream = git_ahead_behind()
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
        **upstream,
        "relevant_upstream_commits": commits,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_id(path: Path, payload: dict[str, Any]) -> str:
    model = str(payload.get("model", ""))
    if "Qwen3.6-35B-A3B" in model:
        return "qwen3.6-35b-a3b"
    if "Qwen3.5-35B-A3B" in model:
        return "qwen3.5-35b-a3b"
    if "Qwen3.5-9B" in model:
        return "qwen3.5-9b"
    return path.stem


def summarize_probe(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    paro = payload.get("paro", {})
    runtime_contract = payload.get("runtime_contract", {})
    complete_modules = int(paro.get("complete_module_count", 0))
    suffix_counts = {str(key): int(value) for key, value in paro.get("suffix_counts", {}).items()}
    required_suffix_counts = {
        suffix: suffix_counts.get(suffix, 0)
        for suffix in REQUIRED_PARO_SUFFIXES
    }
    if complete_modules > 0:
        required_suffixes_present = list(REQUIRED_PARO_SUFFIXES)
        required_suffixes_absent: list[str] = []
    else:
        required_suffixes_present = [
            suffix for suffix, count in required_suffix_counts.items() if count > 0
        ]
        required_suffixes_absent = [
            suffix for suffix, count in required_suffix_counts.items() if count == 0
        ]
    return {
        "id": probe_id(path, payload),
        "path": str(path),
        "schema": payload.get("schema"),
        "model": payload.get("model"),
        "resolved_path": payload.get("resolved_path"),
        "architecture": payload.get("architecture"),
        "tensor_count": payload.get("tensor_count"),
        "complete_module_count": complete_modules,
        "incomplete_module_count": int(paro.get("incomplete_module_count", 0)),
        "suffix_counts": suffix_counts,
        "required_suffix_counts": required_suffix_counts,
        "required_suffixes_present": required_suffixes_present,
        "required_suffixes_absent": required_suffixes_absent,
        "group_size": runtime_contract.get("group_size"),
        "hfq_quant_type": runtime_contract.get("hfq_quant_type"),
        "hfq_quant_type_name": runtime_contract.get("hfq_quant_type_name"),
        "native_paro_source": complete_modules > 0,
    }


def extract_bundle_status(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    transform = payload.get("sections", {}).get("transform.paro", {})
    boundary = transform.get("runtime_env_boundary", {})
    product_env = [item.get("name") for item in boundary.get("productization_candidate", [])]
    research_only = list(boundary.get("research_only", []))
    requirements = list(boundary.get("promotion_report_requirements", []))
    weights_source = payload.get("sections", {}).get("weights", {}).get("source", {})
    product_env_set = {name for name in product_env if name}
    research_set = set(research_only)
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "bundle_id": payload.get("bundle_id"),
        "container_format": payload.get("container", {}).get("format"),
        "external_sidecars_target": bool(payload.get("external_sidecars_target")),
        "transform_runtime_status": transform.get("runtime_status"),
        "weights_source": weights_source,
        "weights_source_exists": bool(weights_source.get("exists")) and bool(weights_source.get("is_file")),
        "runtime_env_boundary": {
            "productization_candidate": product_env,
            "research_only": research_only,
            "promotion_report_requirements": requirements,
            "required_product_env_present": sorted(REQUIRED_PRODUCT_ENV & product_env_set),
            "required_product_env_missing": sorted(REQUIRED_PRODUCT_ENV - product_env_set),
            "required_research_knobs_present": sorted(REQUIRED_RESEARCH_KNOBS & research_set),
            "required_research_knobs_missing": sorted(REQUIRED_RESEARCH_KNOBS - research_set),
        },
    }


def promotion_requirements_covered(requirements: list[str]) -> bool:
    joined = " ".join(requirements).lower()
    return all(term in joined for term in ("oracle", "coherence", "kld/ppl", "perf"))


def artifact_record(path: str) -> dict[str, Any]:
    raw_path = Path(path)
    resolved = raw_path if raw_path.is_absolute() else ROOT / raw_path
    return {
        "path": path,
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
    }


def artifact_records(paths: list[str]) -> list[dict[str, Any]]:
    return [artifact_record(path) for path in paths]


def source_inventory_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "present": False,
            "schema": None,
            "schema_ok": False,
            "native_paro_source_found": None,
            "native_paro_g128_source_found": None,
            "complete_module_count": None,
            "g128_complete_module_count": None,
            "g256_complete_module_count": None,
            "quality_state": None,
            "roots": [],
        }
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
            "schema": None,
            "schema_ok": False,
            "native_paro_source_found": None,
            "native_paro_g128_source_found": None,
            "complete_module_count": None,
            "g128_complete_module_count": None,
            "g256_complete_module_count": None,
            "quality_state": None,
            "roots": [],
        }

    payload = read_json(path)
    native = payload.get("native_paro", {})
    decision = payload.get("decision", {})
    g128_count = int(native.get("g128_complete_module_count", 0))
    return {
        "path": str(path),
        "present": True,
        "schema": payload.get("schema"),
        "schema_ok": payload.get("schema") == "hipfire.astrea.paro_source_inventory.v0",
        "files_seen": payload.get("files_seen"),
        "safetensor_dirs_scanned": payload.get("safetensor_dirs_scanned"),
        "safetensor_files_scanned": payload.get("safetensor_files_scanned"),
        "filename_hit_count": payload.get("filename_hit_count"),
        "complete_module_count": int(native.get("complete_module_count", 0)),
        "incomplete_module_count": int(native.get("incomplete_module_count", 0)),
        "g128_complete_module_count": g128_count,
        "g256_complete_module_count": int(native.get("g256_complete_module_count", 0)),
        "native_paro_source_found": bool(decision.get("native_paro_source_found")),
        "native_paro_g128_source_found": g128_count > 0,
        "quality_state": decision.get("quality_state"),
        "roots": payload.get("roots", []),
    }


def all_records_exist(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(record["exists"] and record["is_file"] for record in records)


def evidence_summary(
    *,
    oracle_artifacts: list[str] | None = None,
    coherence_artifacts: list[str] | None = None,
    nan_artifacts: list[str] | None = None,
    quality_artifacts: list[str] | None = None,
    perf_artifacts: list[str] | None = None,
    research_env_artifacts: list[str] | None = None,
    generic_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    oracle_artifacts = oracle_artifacts or []
    coherence_artifacts = coherence_artifacts or []
    nan_artifacts = nan_artifacts or []
    quality_artifacts = quality_artifacts or []
    perf_artifacts = perf_artifacts or []
    research_env_artifacts = research_env_artifacts or []
    generic_artifacts = generic_artifacts or []
    oracle_records = artifact_records(oracle_artifacts)
    coherence_records = artifact_records(coherence_artifacts)
    nan_records = artifact_records(nan_artifacts)
    quality_records = artifact_records(quality_artifacts)
    perf_records = artifact_records(perf_artifacts)
    research_env_records = artifact_records(research_env_artifacts)
    generic_records = artifact_records(generic_artifacts)
    return {
        "oracle_artifacts": oracle_artifacts,
        "coherence_artifacts": coherence_artifacts,
        "finite_logit_nan_artifacts": nan_artifacts,
        "quality_artifacts": quality_artifacts,
        "gfx1151_perf_artifacts": perf_artifacts,
        "research_only_env_artifacts": research_env_artifacts,
        "generic_artifacts": generic_artifacts,
        "artifact_records": {
            "oracle": oracle_records,
            "coherence": coherence_records,
            "finite_logit_nan": nan_records,
            "quality": quality_records,
            "gfx1151_perf": perf_records,
            "research_only_env": research_env_records,
            "generic": generic_records,
        },
        "typed_artifact_existence": {
            "oracle": all_records_exist(oracle_records),
            "coherence": all_records_exist(coherence_records),
            "finite_logit_nan": all_records_exist(nan_records),
            "quality": all_records_exist(quality_records),
            "gfx1151_perf": all_records_exist(perf_records),
            "all_positive_typed_evidence_files_exist": all(
                all_records_exist(records)
                for records in (
                    oracle_records,
                    coherence_records,
                    nan_records,
                    quality_records,
                    perf_records,
                )
            ),
        },
        "oracle_evidence_present": all_records_exist(oracle_records),
        "coherence_evidence_present": all_records_exist(coherence_records),
        "finite_logit_nan_evidence_present": all_records_exist(nan_records),
        "quality_evidence_present": all_records_exist(quality_records),
        "gfx1151_perf_evidence_present": all_records_exist(perf_records),
        "research_only_env_evidence_present": bool(research_env_artifacts),
    }


def source_probe_coverage(source_probes: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {probe["id"] for probe in source_probes}
    expected_present = sorted(EXPECTED_SOURCE_PROBES & ids)
    expected_missing = sorted(EXPECTED_SOURCE_PROBES - ids)
    contract_bad = [
        probe["id"]
        for probe in source_probes
        if probe.get("id") in EXPECTED_SOURCE_PROBES
        and (probe.get("group_size") != 128 or probe.get("hfq_quant_type_name") != "PARO4G128")
    ]
    expected_probe_absence = {
        probe["id"]: probe.get("required_suffixes_absent", [])
        for probe in source_probes
        if probe.get("id") in EXPECTED_SOURCE_PROBES
    }
    aggregate_required_suffix_counts = {
        suffix: sum(
            int(probe.get("required_suffix_counts", {}).get(suffix, 0))
            for probe in source_probes
            if probe.get("id") in EXPECTED_SOURCE_PROBES
        )
        for suffix in REQUIRED_PARO_SUFFIXES
    }
    return {
        "expected_probe_ids": sorted(EXPECTED_SOURCE_PROBES),
        "present_probe_ids": sorted(ids),
        "expected_probe_ids_present": expected_present,
        "expected_probe_ids_missing": expected_missing,
        "contract_mismatch_probe_ids": sorted(contract_bad),
        "required_native_tensor_families": list(REQUIRED_PARO_SUFFIXES),
        "required_suffix_absence_by_probe": expected_probe_absence,
        "aggregate_required_suffix_counts": aggregate_required_suffix_counts,
        "all_required_suffixes_absent_across_expected_probes": all(
            count == 0 for count in aggregate_required_suffix_counts.values()
        ),
        "complete": not expected_missing and not contract_bad,
    }


def dependency_graph(gates: dict[str, bool]) -> dict[str, Any]:
    nodes = {
        "source_probe": {
            "satisfied": gates["source_probe_coverage_complete"],
            "blocks": ["native_source"],
        },
        "native_source": {
            "satisfied": gates["native_paro_source_found"],
            "blocks": ["imported_hfq"],
        },
        "imported_hfq": {
            "satisfied": gates["imported_hfq_exists"],
            "blocked_by": ["native_source"],
            "blocks": ["paro_oracle", "coherence", "finite_logit_nan", "quality", "gfx1151_perf"],
        },
        "paro_oracle": {
            "satisfied": gates["oracle_evidence_present"],
            "blocked_by": ["imported_hfq"],
        },
        "coherence": {
            "satisfied": gates["coherence_evidence_present"],
            "blocked_by": ["imported_hfq"],
        },
        "finite_logit_nan": {
            "satisfied": gates["finite_logit_nan_evidence_present"],
            "blocked_by": ["imported_hfq"],
        },
        "quality": {
            "satisfied": gates["quality_evidence_present"],
            "blocked_by": ["imported_hfq"],
        },
        "gfx1151_perf": {
            "satisfied": gates["gfx1151_perf_evidence_present"],
            "blocked_by": ["imported_hfq"],
        },
        "typed_evidence_files": {
            "satisfied": gates["typed_evidence_files_exist"],
            "blocked_by": ["imported_hfq"],
            "requires": [
                "paro_oracle",
                "coherence",
                "finite_logit_nan",
                "quality",
                "gfx1151_perf",
            ],
        },
        "runtime_env_boundary": {
            "satisfied": gates["runtime_env_boundary_recorded"],
        },
        "package_contract": {
            "satisfied": gates["package_contract_present"],
            "blocked_by": ["imported_hfq"],
        },
        "main_path_clean": {
            "satisfied": gates["promotion_main_path_clean"],
            "blocked_by": ["runtime_env_boundary"],
        },
        "source_inventory": {
            "satisfied": gates["source_inventory_present"],
            "blocks": ["native_source"],
        },
        "source_inventory_consistency": {
            "satisfied": gates["source_inventory_consistent"],
            "blocked_by": ["source_inventory", "source_probe"],
        },
    }
    source_absent = not gates["native_paro_source_found"]
    import_absent = not gates["imported_hfq_exists"]
    return {
        "nodes": nodes,
        "edges": [
            ["source_probe", "native_source"],
            ["native_source", "imported_hfq"],
            ["imported_hfq", "paro_oracle"],
            ["imported_hfq", "coherence"],
            ["imported_hfq", "finite_logit_nan"],
            ["imported_hfq", "quality"],
            ["imported_hfq", "gfx1151_perf"],
        ],
        "native_source_absent": source_absent,
        "import_blocked_by_source_absence": source_absent and import_absent,
        "oracle_blocked_by_import_absence": import_absent and not gates["oracle_evidence_present"],
        "quality_blocked_by_import_absence": import_absent and not gates["quality_evidence_present"],
        "perf_blocked_by_import_absence": import_absent and not gates["gfx1151_perf_evidence_present"],
        "next_unblocked_step": (
            "evaluate_readiness_matrix_for_promotion"
            if all(
                value
                for key, value in gates.items()
                if key != "source_inventory_native_paro_g128_found"
            )
            else "locate_or_generate_native_paro_q4g128_checkpoint"
            if source_absent
            else "run_paro_import"
            if import_absent
            else "run_paro_oracle_coherence_nan_kld_perf"
        ),
    }


def productization_plan(
    *,
    gates: dict[str, bool],
    bundle: dict[str, Any],
    evidence: dict[str, Any],
    dependencies: dict[str, Any],
    promotion_allowed: bool,
) -> dict[str, Any]:
    if promotion_allowed:
        status = "promotion_ready"
    elif not gates["native_paro_source_found"]:
        status = "blocked_before_native_source_import"
    elif not gates["imported_hfq_exists"]:
        status = "ready_for_paro_import"
    else:
        status = "blocked_before_typed_oracle_quality_perf"

    imported_hfq = str(bundle["weights_source"].get("path") or "<imported-paro-q4g128.hfq>")
    native_source = "<native-paro-q4g128-safetensors-dir-or-hf-repo>"
    package_hfq = "<packaged-paro-q4g128.hfq>"
    source_inventory_path = repo_path(EVIDENCE_TARGETS["source_inventory"])
    import_report_path = repo_path(EVIDENCE_TARGETS["import_report"])
    oracle_path = repo_path(EVIDENCE_TARGETS["oracle"])
    bundle_plan_path = repo_path(EVIDENCE_TARGETS["bundle_plan"])

    stage_order = [
        "source_inventory",
        "native_source",
        "source_probe",
        "imported_hfq",
        "bundle_plan_with_weights",
        "paro_oracle",
        "coherence",
        "finite_logit_nan",
        "quality_kld_ppl",
        "gfx1151_perf",
        "promotion_review",
    ]
    stages = {
        "source_inventory": {
            "satisfied": gates["source_inventory_present"] and gates["source_inventory_consistent"],
            "command": f"python3 scripts/paroquant_inventory.py --pretty --out {source_inventory_path}",
            "artifact": source_inventory_path,
            "native_paro_g128_source_found": dependencies["source_inventory"].get(
                "native_paro_g128_source_found"
            ),
            "g128_complete_module_count": dependencies["source_inventory"].get(
                "g128_complete_module_count"
            ),
        },
        "native_source": {
            "satisfied": gates["native_paro_source_found"],
            "blocked_by": [] if gates["source_inventory_present"] else ["source_inventory"],
            "required_tensor_families": list(REQUIRED_PARO_SUFFIXES),
            "current_result": "missing" if not gates["native_paro_source_found"] else "present",
            "blocks": ["source_probe", "imported_hfq"],
        },
        "source_probe": {
            "satisfied": gates["source_probe_coverage_complete"],
            "blocked_by": ["native_source"],
            "command_template": (
                "python3 scripts/astrea.py paro-probe --model "
                f"{native_source} --pretty --out <paro-source-probe.json>"
            ),
            "artifacts": [repo_path(path) for path in DEFAULT_PROBES],
        },
        "imported_hfq": {
            "satisfied": gates["imported_hfq_exists"],
            "blocked_by": ["native_source"],
            "command_template": (
                "python3 scripts/astrea.py paro-import --model "
                f"{native_source} --output {imported_hfq} --layout native --copy-floats f16 "
                f"--pretty --out {import_report_path}"
            ),
            "artifact": imported_hfq,
            "import_report_artifact": import_report_path,
        },
        "bundle_plan_with_weights": {
            "satisfied": gates["package_contract_present"] and gates["imported_hfq_exists"],
            "blocked_by": ["imported_hfq"],
            "command_template": (
                "python3 scripts/astrea.py bundle-plan --model "
                f"{imported_hfq} --output {package_hfq} --include weights --include paro "
                f"--pretty --out {bundle_plan_path}"
            ),
            "artifact": bundle_plan_path,
        },
        "paro_oracle": {
            "satisfied": gates["oracle_evidence_present"],
            "blocked_by": ["imported_hfq"],
            "command_template": (
                "python3 scripts/astrea.py paro-oracle --source "
                f"{native_source} --hfq {imported_hfq} --pretty --out {oracle_path}"
            ),
            "artifact": oracle_path,
        },
        "coherence": {
            "satisfied": gates["coherence_evidence_present"],
            "blocked_by": ["imported_hfq"],
            "command_template": "./scripts/coherence-gate.sh --full",
            "artifact": repo_path(EVIDENCE_TARGETS["coherence"]),
        },
        "finite_logit_nan": {
            "satisfied": gates["finite_logit_nan_evidence_present"],
            "blocked_by": ["imported_hfq"],
            "artifact": repo_path(EVIDENCE_TARGETS["finite_logit_nan"]),
            "requirement": "finite logits and no NaN/Inf rows for dense and A3B Paro runs",
        },
        "quality_kld_ppl": {
            "satisfied": gates["quality_evidence_present"],
            "blocked_by": ["imported_hfq", "paro_oracle"],
            "artifact": repo_path(EVIDENCE_TARGETS["quality"]),
            "requirement": "same-run KLD/PPL comparison against the MQ4 control before promotion",
        },
        "gfx1151_perf": {
            "satisfied": gates["gfx1151_perf_evidence_present"],
            "blocked_by": ["imported_hfq", "coherence", "quality_kld_ppl"],
            "artifact": repo_path(EVIDENCE_TARGETS["gfx1151_perf"]),
            "requirement": "fresh-process gfx1151 dense and A3B perf rows with prompt and binary hashes",
        },
        "promotion_review": {
            "satisfied": promotion_allowed,
            "blocked_by": [
                "native_source",
                "imported_hfq",
                "paro_oracle",
                "coherence",
                "finite_logit_nan",
                "quality_kld_ppl",
                "gfx1151_perf",
            ],
            "requirement": "matrix review only after every typed evidence stage is satisfied",
        },
    }
    return {
        "status": status,
        "next_unblocked_step": dependencies["next_unblocked_step"],
        "stage_order": stage_order,
        "stages": stages,
        "evidence_artifact_targets": {
            name: repo_path(path)
            for name, path in EVIDENCE_TARGETS.items()
        },
        "typed_evidence_complete": gates["typed_evidence_files_exist"]
        and gates["oracle_quality_perf_evidence_present"],
        "refresh_required_after_origin_reconciliation": any(
            item.get("present_on_origin_master") and not item.get("present_on_head")
            for item in origin_context()["relevant_upstream_commits"]
        ),
        "next_work": [
            "Locate or produce a native ParoQ4G128 checkpoint with qweight/qzeros/scales/pairs/theta/channel_scales.",
            "Run paro-probe, paro-import, and paro-oracle against the native source and imported HFQ.",
            "Regenerate the Astrea bundle plan with the imported HFQ as the weights source.",
            "Run coherence, finite-logit/NaN, KLD/PPL, and gfx1151 perf evidence before any promotion claim.",
            "Refresh the evidence after reconciling origin commits that touch Paro, MoE, RoPE, DFlash, or runtime leaks.",
        ],
        "current_typed_evidence": evidence["typed_artifact_existence"],
    }


def productization_boundary(
    *,
    gates: dict[str, bool],
    bundle: dict[str, Any],
    evidence: dict[str, Any],
    dependencies: dict[str, Any],
    promotion_allowed: bool,
) -> dict[str, Any]:
    if promotion_allowed:
        current_stage = "promotion_ready"
    elif not gates["native_paro_source_found"] or not gates["imported_hfq_exists"]:
        current_stage = "blocked_before_native_source_import"
    else:
        current_stage = "blocked_before_typed_oracle_quality_perf"

    if not gates["native_paro_source_found"]:
        importer_state = "blocked_no_native_paro_source"
    elif not gates["imported_hfq_exists"]:
        importer_state = "ready_for_paro_import"
    else:
        importer_state = "imported_hfq_present"

    if not gates["package_contract_present"]:
        package_state = "missing_package_contract"
    elif not gates["imported_hfq_exists"]:
        package_state = "contract_only_missing_weights"
    else:
        package_state = "package_contract_with_weights"

    if not gates["runtime_env_boundary_recorded"]:
        main_path_state = "runtime_env_boundary_incomplete"
    elif evidence["research_only_env_evidence_present"]:
        main_path_state = "blocked_by_research_only_env_evidence"
    elif promotion_allowed:
        main_path_state = "promotion_candidate_clean"
    else:
        main_path_state = "clean_but_unpromoted"

    return {
        "current_stage": current_stage,
        "importer_state": importer_state,
        "package_state": package_state,
        "astrea_state": "bundle_plan_contract_only_not_model_writer",
        "main_path_state": main_path_state,
        "artifact_generation_without_native_source_allowed": gates["native_paro_source_found"],
        "hidden_research_env_dependency_present": evidence["research_only_env_evidence_present"],
        "research_only_env_blocks_promotion": True,
        "productization_candidate_env_documented": gates["runtime_env_boundary_recorded"],
        "promoted_main_path_requires_typed_evidence": True,
        "typed_evidence_complete": gates["oracle_quality_perf_evidence_present"],
        "native_source_contract": {
            "required_tensor_families": list(REQUIRED_PARO_SUFFIXES),
            "all_required_suffixes_absent_across_expected_probes": dependencies[
                "all_required_suffixes_absent_across_expected_probes"
            ],
            "required_suffix_absence_by_probe": dependencies[
                "required_suffix_absence_by_probe"
            ],
            "broader_source_inventory": dependencies["source_inventory"],
        },
        "next_unblocked_step": dependencies["next_unblocked_step"],
        "weights_source": bundle["weights_source"],
    }


def build_status(
    probes: list[Path],
    bundle_plan: Path,
    *,
    source_inventory: Path | None = None,
    evidence_artifacts: list[str] | None = None,
    oracle_artifacts: list[str] | None = None,
    coherence_artifacts: list[str] | None = None,
    nan_artifacts: list[str] | None = None,
    quality_artifacts: list[str] | None = None,
    perf_artifacts: list[str] | None = None,
    research_env_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    source_probes = [summarize_probe(path) for path in probes]
    probe_coverage = source_probe_coverage(source_probes)
    inventory = source_inventory_summary(source_inventory)
    bundle = extract_bundle_status(bundle_plan)
    boundary = bundle["runtime_env_boundary"]
    requirements = boundary["promotion_report_requirements"]
    evidence = evidence_summary(
        oracle_artifacts=oracle_artifacts,
        coherence_artifacts=coherence_artifacts,
        nan_artifacts=nan_artifacts,
        quality_artifacts=quality_artifacts,
        perf_artifacts=perf_artifacts,
        research_env_artifacts=research_env_artifacts,
        generic_artifacts=evidence_artifacts,
    )

    native_source_found = any(probe["native_paro_source"] for probe in source_probes)
    env_boundary_recorded = (
        not boundary["required_product_env_missing"]
        and not boundary["required_research_knobs_missing"]
        and promotion_requirements_covered(requirements)
    )
    package_contract_present = (
        bundle["schema"] == "hipfire.astrea.bundle_plan.v0"
        and bundle["container_format"] == "hfq-package-v0"
        and bundle["external_sidecars_target"] is False
        and bundle["transform_runtime_status"] == "deferred_until_loader_and_fused_kernel_exist"
    )
    quality_perf_evidence_present = (
        evidence["oracle_evidence_present"]
        and evidence["coherence_evidence_present"]
        and evidence["finite_logit_nan_evidence_present"]
        and evidence["quality_evidence_present"]
        and evidence["gfx1151_perf_evidence_present"]
    )
    typed_evidence_files_exist = evidence["typed_artifact_existence"][
        "all_positive_typed_evidence_files_exist"
    ]
    research_only_env_evidence_absent = not evidence["research_only_env_evidence_present"]
    promotion_main_path_clean = env_boundary_recorded and research_only_env_evidence_absent
    source_inventory_present = source_inventory is None or (inventory["present"] and inventory["schema_ok"])
    source_inventory_consistent = source_inventory is None or (
        inventory["native_paro_source_found"] == native_source_found
    )

    gates = {
        "source_probe_coverage_complete": probe_coverage["complete"],
        "source_inventory_present": source_inventory_present,
        "source_inventory_consistent": source_inventory_consistent,
        "source_inventory_native_paro_g128_found": bool(inventory["native_paro_g128_source_found"]),
        "native_paro_source_found": native_source_found,
        "imported_hfq_exists": bool(bundle["weights_source_exists"]),
        "runtime_env_boundary_recorded": env_boundary_recorded,
        "research_only_env_evidence_absent": research_only_env_evidence_absent,
        "promotion_main_path_clean": promotion_main_path_clean,
        "package_contract_present": package_contract_present,
        "oracle_evidence_present": evidence["oracle_evidence_present"],
        "coherence_evidence_present": evidence["coherence_evidence_present"],
        "finite_logit_nan_evidence_present": evidence["finite_logit_nan_evidence_present"],
        "quality_evidence_present": evidence["quality_evidence_present"],
        "gfx1151_perf_evidence_present": evidence["gfx1151_perf_evidence_present"],
        "typed_evidence_files_exist": typed_evidence_files_exist,
        "oracle_quality_perf_evidence_present": quality_perf_evidence_present,
    }
    dependencies = dependency_graph(gates)
    dependencies["required_suffix_absence_by_probe"] = probe_coverage[
        "required_suffix_absence_by_probe"
    ]
    dependencies["all_required_suffixes_absent_across_expected_probes"] = probe_coverage[
        "all_required_suffixes_absent_across_expected_probes"
    ]
    dependencies["source_inventory"] = inventory
    blockers = []
    if not gates["source_probe_coverage_complete"]:
        blockers.append("ParoQ4G128 source probe coverage is incomplete or has a contract mismatch")
    if not gates["source_inventory_present"]:
        blockers.append("broader local Paro source inventory is missing or has the wrong schema")
    if not gates["source_inventory_consistent"]:
        blockers.append("broader local Paro source inventory contradicts the named source probes")
    if source_inventory is not None and not gates["source_inventory_native_paro_g128_found"]:
        blockers.append("broader local Paro source inventory found zero native ParoQ4G128 modules")
    if not gates["native_paro_source_found"]:
        blockers.append(
            "native ParoQ4G128 source checkpoint with qweight/qzeros/scales/"
            "pairs/theta/channel_scales companion tensors is missing"
        )
    if not gates["imported_hfq_exists"]:
        blockers.append("imported Paro HFQ weights source is missing")
    if not gates["oracle_evidence_present"]:
        blockers.append("paro-oracle evidence is missing")
    if not gates["coherence_evidence_present"]:
        blockers.append("dense/A3B coherence evidence is missing")
    if not gates["finite_logit_nan_evidence_present"]:
        blockers.append("finite-logit/NaN stability evidence is missing")
    if not gates["quality_evidence_present"]:
        blockers.append("KLD/PPL quality evidence is missing")
    if not gates["gfx1151_perf_evidence_present"]:
        blockers.append("gfx1151 perf evidence is missing")
    if not gates["typed_evidence_files_exist"]:
        missing_positive = []
        for category, records in evidence["artifact_records"].items():
            if category in {"research_only_env", "generic"}:
                continue
            for record in records:
                if not (record["exists"] and record["is_file"]):
                    missing_positive.append(f"{category}:{record['path']}")
        if missing_positive:
            blockers.append(
                "typed positive evidence artifact paths do not exist: "
                + ", ".join(missing_positive)
            )
    if not gates["runtime_env_boundary_recorded"]:
        blockers.append("runtime env boundary is incomplete")
    if not gates["research_only_env_evidence_absent"]:
        blockers.append("research-only Paro env evidence cannot satisfy promoted main-path gates")
    if not gates["package_contract_present"]:
        blockers.append("Astrea package contract is incomplete")

    promotion_gate_keys = tuple(
        key for key in gates if key != "source_inventory_native_paro_g128_found"
    )
    promotion_allowed = all(gates[key] for key in promotion_gate_keys)
    boundary_summary = productization_boundary(
        gates=gates,
        bundle=bundle,
        evidence=evidence,
        dependencies=dependencies,
        promotion_allowed=promotion_allowed,
    )
    plan = productization_plan(
        gates=gates,
        bundle=bundle,
        evidence=evidence,
        dependencies=dependencies,
        promotion_allowed=promotion_allowed,
    )
    origin = origin_context()
    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "format": "paro-q4g128",
        "status": "candidate-productization" if not promotion_allowed else "promotion-ready",
        "promotion_allowed": promotion_allowed,
        "source_probes": source_probes,
        "source_probe_coverage": probe_coverage,
        "source_inventory": inventory,
        "bundle_plan": bundle,
        "evidence": evidence,
        "gates": gates,
        "promotion_gate_keys": list(promotion_gate_keys),
        "dependency_graph": dependencies,
        "productization_plan": plan,
        "productization_boundary": boundary_summary,
        "origin_context": origin,
        "blockers": blockers,
        "next_work": plan["next_work"],
        "decision": (
            "blocked at producer/import/oracle evidence; keep ParoQ4G128 out of "
            "promoted main path"
            if not promotion_allowed
            else "all productization gates present; evaluate readiness matrix before promotion claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", default=[], help="Paro source probe JSON; repeatable")
    parser.add_argument("--bundle-plan", default=str(DEFAULT_BUNDLE_PLAN))
    parser.add_argument("--source-inventory", default=str(DEFAULT_SOURCE_INVENTORY), help="Broader Paro source inventory JSON")
    parser.add_argument("--evidence-artifact", action="append", default=[], help="Legacy generic artifact reference; does not satisfy a promotion evidence gate")
    parser.add_argument("--oracle-artifact", action="append", default=[], help="Paro oracle artifact; repeatable")
    parser.add_argument("--coherence-artifact", action="append", default=[], help="Dense/A3B coherence artifact; repeatable")
    parser.add_argument("--nan-artifact", action="append", default=[], help="Finite-logit/NaN stability artifact; repeatable")
    parser.add_argument("--quality-artifact", action="append", default=[], help="KLD/PPL quality artifact; repeatable")
    parser.add_argument("--perf-artifact", action="append", default=[], help="gfx1151 perf artifact; repeatable")
    parser.add_argument("--research-env-artifact", action="append", default=[], help="Evidence that used research-only Paro env knobs; blocks promotion")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    probes = [Path(path) for path in args.probe] if args.probe else list(DEFAULT_PROBES)
    payload = build_status(
        probes,
        Path(args.bundle_plan),
        source_inventory=Path(args.source_inventory) if args.source_inventory else None,
        evidence_artifacts=list(args.evidence_artifact),
        oracle_artifacts=list(args.oracle_artifact),
        coherence_artifacts=list(args.coherence_artifact),
        nan_artifacts=list(args.nan_artifact),
        quality_artifacts=list(args.quality_artifact),
        perf_artifacts=list(args.perf_artifact),
        research_env_artifacts=list(args.research_env_artifact),
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
