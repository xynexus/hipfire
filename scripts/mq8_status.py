#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize MQ8 gfx1151 readiness status.

MQ8 has runtime substrate, but the readiness contract requires a product role,
canonical artifact, quality evidence, and gfx1151 perf before promotion work
can restart.  This helper makes those gates explicit and machine-readable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hipfire.mq8_status.gfx1151.v0"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "2026-06-03-mq8-status.json"
DEFAULT_MODEL_ROOTS = (
    Path("/home/sadara/Models"),
    Path("/home/sadara/.hipfire/models"),
)
DEFAULT_CANDIDATE_ROOT = Path("/home/sadara/Models/hipfire-candidates/gfx1151-readiness")
DEFAULT_COMPILED_ROOT = Path.home() / ".hipfire" / "bin" / "kernels" / "compiled" / "gfx1151"
REQUIRED_BLOBS = (
    "attention_hfq8_kv.hsaco",
    "gemv_hfq8g256.hsaco",
    "gemv_mq8g256.hsaco",
    "kv_cache_write_hfq8.hsaco",
)
BYTES_PER_GROUP = {
    "mq4": 136,
    "mq6": 200,
    "mq8": 258,
}
REOPEN_REQUIREMENTS = (
    "product_role_over_q8_or_mq6",
    "canonical_qwen3_5_9b_mq8_artifact",
    "mq8_example_or_benchmark_harness",
    "coherence_or_kld_ppl_quality_evidence",
    "gfx1151_ar_and_dflash_perf_baselines",
)
LOCAL_RUNTIME_CONTEXT_COMMITS = (
    {
        "commit": "282cc9c5ba2035d5901fcb74848cbd58b02c0c9d",
        "short": "282cc9c5",
        "subject": "Expand qwen35 MoE quant admission and bf16 tooling",
        "impact": "adds the broader MQ-family MoE admission matrix that includes MQ8 as gfx1151 bring-up only",
    },
    {
        "commit": "94794414251516f71093c04c5258cb0d32f5fbf1",
        "short": "94794414",
        "subject": "Add gfx1151 qwen35 MoE MQ bring-up",
        "impact": "adds gfx1151 MQ runtime bring-up substrate covering MQ8 scalar paths",
    },
    {
        "commit": "e038b9868c601db9e2c5c804be79ac311f91758d",
        "short": "e038b986",
        "subject": "Fix qwen35 MoE MQ8 artifact and gfx1151 GEMV",
        "impact": "fixes local MQ8 artifact/runtime details but does not add product evidence",
    },
    {
        "commit": "6bec9f715d8ec5503600ccc1cd6207e78311edee",
        "short": "6bec9f71",
        "subject": "Fix gfx1151 MQ8 MoE batched numerics",
        "impact": "fixes local MQ8 MoE batched numerics but does not add product evidence",
    },
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def upstream_mq8_commits(base: str = "HEAD", upstream: str = "origin/master") -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "log",
                "--format=%H%x00%s",
                "--regexp-ignore-case",
                "--extended-regexp",
                "--grep=mq8|hfq8",
                f"{base}..{upstream}",
            ],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    commits = []
    for line in output.splitlines():
        if "\0" not in line:
            continue
        commit, subject = line.split("\0", 1)
        commits.append({"commit": commit, "short": commit[:8], "subject": subject})
    return commits


def origin_context() -> dict[str, Any]:
    local_commits = []
    for item in LOCAL_RUNTIME_CONTEXT_COMMITS:
        local_commits.append(
            {
                **item,
                "present_on_head": commit_on_ref(item["commit"], "HEAD"),
                "present_on_origin_master": commit_on_ref(item["commit"], "origin/master"),
            }
        )
    upstream_commits = upstream_mq8_commits()
    product_terms = ("product", "promot", "canonical", "quality", "perf")
    return {
        **git_ahead_behind(),
        "local_runtime_context_commits": local_commits,
        "upstream_mq8_commits": upstream_commits,
        "upstream_mq8_product_role_commit_found": any(
            any(term in commit["subject"].lower() for term in product_terms)
            for commit in upstream_commits
        ),
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def iter_files(root: Path) -> Iterable[Path]:
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


def is_model_artifact(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith((".hfq", ".mq8", ".q8f16"))


def artifact_summary(roots: Iterable[Path], candidate_root: Path) -> dict[str, Any]:
    searched = []
    mq8_artifacts = []
    q8f16_artifacts = []
    for root in [*roots, candidate_root]:
        root = root.expanduser()
        searched.append({"root": str(root), "exists": root.exists()})
        for path in iter_files(root):
            if not is_model_artifact(path):
                continue
            lowered = path.name.lower()
            if "mq8" in lowered:
                mq8_artifacts.append(str(path))
            elif "q8f16" in lowered:
                q8f16_artifacts.append(str(path))
    return {
        "searched_roots": searched,
        "mq8_artifacts": sorted(dict.fromkeys(mq8_artifacts)),
        "q8f16_artifacts": sorted(dict.fromkeys(q8f16_artifacts)),
    }


def source_hook_summary(source_root: Path) -> dict[str, Any]:
    quantize = read_text(source_root / "crates" / "hipfire-quantize" / "src" / "main.rs")
    hfq = read_text(source_root / "crates" / "hipfire-runtime" / "src" / "hfq.rs")
    llama = read_text(source_root / "crates" / "hipfire-runtime" / "src" / "llama.rs")
    qwen35 = read_text(source_root / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs")
    dispatch = read_text(source_root / "crates" / "rdna-compute" / "src" / "dispatch.rs")
    kernels = read_text(source_root / "crates" / "rdna-compute" / "src" / "kernels.rs")

    producer_checks = {
        "format_flag": "use_mq8g256" in quantize and '"mq8"' in quantize and '"mq8g256"' in quantize,
        "quantizer": "quantize_mq8g256" in quantize,
        "quant_type": "MQ8G256 = 14" in quantize,
        "q8f16_fallback_note": "Q8F16" in quantize and "use_mq8g256" in quantize,
    }
    runtime_checks = {
        "hfq_loader_dtype": "DType::MQ8G256" in hfq,
        "dense_decode_gemv": "gemv_mq8g256_with_rotate" in llama,
        "rotate_quantize": "rotate_quantize_x_mq8" in llama and "rotate_quantize_x_mq8" in dispatch,
        "prerotated_gemv": "gemv_mq8g256_prerotated" in llama and "gemv_mq8g256_prerotated" in dispatch,
        "qwen35_qtype_mapping": "14 => Some(DType::MQ8G256)" in qwen35,
        "qwen35_moe_scalar_paths": all(
            token in qwen35
            for token in (
                "gemm_gate_up_hfq8g256",
                "gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched",
                "gemv_hfq8g256_moe_gate_up_k8_indexed_batched",
                "gemv_hfq8g256_moe_down_k8_indexed_batched_expanded",
            )
        ),
        "kernel_source_registered": "GEMV_MQ8G256_SRC" in kernels,
    }
    no_gpu_checks = {
        "gfx12_grouped_reject_matrix": "moe_prefill_quant_matrix_documents_mq2_mq3_mq4_mq6_mq8" in qwen35,
        "gfx1151_scalar_bringup": "moe_prefill_admits_gfx1151_scalar_bringup_families" in qwen35,
        "mixed_routed_family_reject": "moe_prefill_rejects_mixed_routed_family_without_grouped_gemm" in qwen35,
    }
    return {
        "producer_checks": producer_checks,
        "runtime_checks": runtime_checks,
        "no_gpu_admission_checks": no_gpu_checks,
        "producer_surface_present": all(producer_checks.values()),
        "runtime_surface_present": all(runtime_checks.values()),
        "no_gpu_admission_covered": all(no_gpu_checks.values()),
    }


def compiled_blob_summary(compiled_root: Path) -> dict[str, Any]:
    blobs = []
    for name in REQUIRED_BLOBS:
        path = compiled_root / name
        blobs.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return {
        "root": str(compiled_root),
        "required": list(REQUIRED_BLOBS),
        "blobs": blobs,
        "all_present": all(blob["exists"] for blob in blobs),
    }


def harness_summary(source_root: Path) -> dict[str, Any]:
    roots = (
        source_root / "crates" / "rdna-compute" / "examples",
        source_root / "crates" / "hipfire-runtime" / "examples",
        source_root / "benchmarks",
    )
    matches = []
    for root in roots:
        for path in iter_files(root):
            if not path.is_file() or path.suffix not in {".rs", ".py", ".sh"}:
                continue
            if "results" in path.relative_to(source_root).parts:
                continue
            lowered = path.name.lower()
            text = read_text(path).lower()
            if "mq8" in lowered or "hfq8" in lowered or "mq8" in text or "hfq8" in text:
                matches.append(str(path))
    return {
        "searched_roots": [{"root": str(root), "exists": root.exists()} for root in roots],
        "matches": sorted(dict.fromkeys(matches)),
        "present": bool(matches),
    }


def byte_tradeoff_summary() -> dict[str, Any]:
    mq8 = BYTES_PER_GROUP["mq8"]
    mq6 = BYTES_PER_GROUP["mq6"]
    mq4 = BYTES_PER_GROUP["mq4"]
    return {
        "bytes_per_group": dict(BYTES_PER_GROUP),
        "mq8_to_mq6_ratio": mq8 / mq6,
        "mq8_to_mq4_ratio": mq8 / mq4,
        "requires_role_over_mq6": mq8 > mq6,
        "requires_role_over_mq4_control": mq8 > mq4,
    }


def purpose_decision(gates: dict[str, bool], promotion_allowed: bool, product_role: str | None) -> dict[str, Any]:
    return {
        "classification": "promotion-ready" if promotion_allowed else "permanent-runtime-research",
        "promotion_backlog_closed": not promotion_allowed,
        "runtime_substrate_maintenance_allowed": bool(
            gates["producer_surface_present"]
            or gates["runtime_surface_present"]
            or gates["gfx1151_compiled_blobs_present"]
        ),
        "artifact_generation_without_product_role_allowed": bool(product_role),
        "reopen_requires_all_contract_gates": True,
        "closed_reason": None if promotion_allowed else "no_product_role_artifact_quality_or_perf",
    }


def next_unblocked_step(gates: dict[str, bool]) -> str:
    ordered_requirements = (
        ("product_role_defined", "define_product_role_over_q8_or_mq6"),
        ("canonical_mq8_artifact_present", "generate_canonical_qwen3_5_9b_mq8_artifact"),
        ("candidate_model_benchmark_harness_present", "add_candidate_model_mq8_harness"),
        ("quality_evidence_present", "run_coherence_or_kld_ppl_quality_gate"),
        ("perf_evidence_present", "run_gfx1151_ar_and_dflash_perf_baselines"),
        ("producer_surface_present", "restore_mq8_producer_surface"),
        ("runtime_surface_present", "restore_mq8_runtime_surface"),
        ("gfx1151_compiled_blobs_present", "compile_gfx1151_mq8_hfq8_blobs"),
        ("no_gpu_admission_covered", "restore_no_gpu_mq8_admission_tests"),
    )
    for gate, step in ordered_requirements:
        if not gates[gate]:
            return step
    return "verify_readiness_matrix_before_promotion_claim"


def promotion_boundary(
    *,
    gates: dict[str, bool],
    promotion_allowed: bool,
    product_role: str | None,
    byte_tradeoff: dict[str, Any],
) -> dict[str, Any]:
    reopen_satisfied = all(
        gates[gate]
        for gate in (
            "product_role_defined",
            "canonical_mq8_artifact_present",
            "candidate_model_benchmark_harness_present",
            "quality_evidence_present",
            "perf_evidence_present",
        )
    )
    missing = []
    if not gates["product_role_defined"]:
        missing.append("product_role_over_q8_or_mq6")
    if not gates["canonical_mq8_artifact_present"]:
        missing.append("canonical_qwen3_5_9b_mq8_artifact")
    if not gates["example_or_benchmark_harness_present"]:
        missing.append("mq8_example_or_benchmark_harness")
    elif not gates["candidate_model_benchmark_harness_present"]:
        missing.append("mq8_example_or_benchmark_harness")
    if not gates["quality_evidence_present"]:
        missing.append("coherence_or_kld_ppl_quality_evidence")
    if not gates["perf_evidence_present"]:
        missing.append("gfx1151_ar_and_dflash_perf_baselines")
    candidate_model_baseline = (
        gates["canonical_mq8_artifact_present"]
        and gates["quality_evidence_present"]
        and gates["perf_evidence_present"]
    )
    return {
        "promotion_lane_state": (
            "open_for_promotion_review" if promotion_allowed else "closed_until_reopen_contract"
        ),
        "artifact_generation_state": (
            "allowed_for_reopen_candidate"
            if product_role
            else "blocked_missing_product_role"
        ),
        "runtime_substrate_maintenance_allowed": (
            gates["producer_surface_present"]
            or gates["runtime_surface_present"]
            or gates["gfx1151_compiled_blobs_present"]
        ),
        "candidate_model_baseline_available": candidate_model_baseline,
        "reopen_contract_satisfied": reopen_satisfied,
        "missing_reopen_requirements": missing,
        "high_bit_justification_required": True,
        "bytes_per_group": byte_tradeoff["bytes_per_group"],
        "mq8_to_mq6_ratio": byte_tradeoff["mq8_to_mq6_ratio"],
        "mq8_to_mq4_ratio": byte_tradeoff["mq8_to_mq4_ratio"],
        "next_unblocked_step": next_unblocked_step(gates),
    }


def research_closure(
    *,
    gates: dict[str, bool],
    promotion_allowed: bool,
    boundary: dict[str, Any],
    byte_tradeoff: dict[str, Any],
) -> dict[str, Any]:
    mq8_to_mq6 = byte_tradeoff["mq8_to_mq6_ratio"]
    mq8_to_mq4 = byte_tradeoff["mq8_to_mq4_ratio"]
    if promotion_allowed:
        rationale = (
            "all MQ8 reopen gates are present; verify the readiness matrix "
            "before making a promotion claim"
        )
    else:
        rationale = (
            f"MQ8 costs {mq8_to_mq6:.2f}x MQ6 and {mq8_to_mq4:.2f}x MQ4 "
            "per group, but has no product role, canonical artifact, quality "
            "evidence, or gfx1151 perf baseline"
        )
    return {
        "classification": "promotion-ready" if promotion_allowed else "permanent-runtime-research",
        "closed_without_product_role": not gates["product_role_defined"],
        "artifact_generation_blocked": not gates["product_role_defined"],
        "perf_collection_blocked": not (
            gates["canonical_mq8_artifact_present"] and gates["quality_evidence_present"]
        ),
        "runtime_substrate_maintenance_allowed": boundary["runtime_substrate_maintenance_allowed"],
        "candidate_artifact_generation_requires_product_role": True,
        "candidate_perf_collection_requires_artifact_and_quality": True,
        "high_bit_value_must_exceed_mq6_or_q8": True,
        "minimum_reopen_requirements": list(REOPEN_REQUIREMENTS),
        "missing_reopen_requirements": boundary["missing_reopen_requirements"],
        "next_unblocked_step": boundary["next_unblocked_step"],
        "rationale": rationale,
    }


def build_status(
    *,
    source_root: Path = ROOT,
    model_roots: Iterable[Path] = DEFAULT_MODEL_ROOTS,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    compiled_root: Path = DEFAULT_COMPILED_ROOT,
    product_role: str | None = None,
    quality_artifacts: list[str] | None = None,
    perf_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    quality_artifacts = quality_artifacts or []
    perf_artifacts = perf_artifacts or []
    artifacts = artifact_summary(model_roots, candidate_root)
    hooks = source_hook_summary(source_root)
    blobs = compiled_blob_summary(compiled_root)
    harness = harness_summary(source_root)
    byte_tradeoff = byte_tradeoff_summary()

    gates = {
        "product_role_defined": bool(product_role),
        "canonical_mq8_artifact_present": bool(artifacts["mq8_artifacts"]),
        "example_or_benchmark_harness_present": harness["present"],
        "candidate_model_benchmark_harness_present": bool(artifacts["mq8_artifacts"])
        and harness["present"],
        "producer_surface_present": hooks["producer_surface_present"],
        "runtime_surface_present": hooks["runtime_surface_present"],
        "gfx1151_compiled_blobs_present": blobs["all_present"],
        "no_gpu_admission_covered": hooks["no_gpu_admission_covered"],
        "quality_evidence_present": bool(quality_artifacts),
        "perf_evidence_present": bool(perf_artifacts),
    }
    promotion_allowed = all(gates.values())
    boundary = promotion_boundary(
        gates=gates,
        promotion_allowed=promotion_allowed,
        product_role=product_role,
        byte_tradeoff=byte_tradeoff,
    )
    closure = research_closure(
        gates=gates,
        promotion_allowed=promotion_allowed,
        boundary=boundary,
        byte_tradeoff=byte_tradeoff,
    )
    blockers = []
    if not gates["product_role_defined"]:
        blockers.append("no MQ8 product role has been proposed over Q8/MQ6")
    if not gates["canonical_mq8_artifact_present"]:
        blockers.append("no canonical MQ8 candidate artifact was found")
    if not gates["example_or_benchmark_harness_present"]:
        blockers.append("no MQ8 example or benchmark harness was found")
    elif not gates["candidate_model_benchmark_harness_present"]:
        blockers.append(
            "MQ8 benchmark harness is generic substrate only; no canonical candidate-model harness is available"
        )
    if not gates["quality_evidence_present"]:
        blockers.append("no MQ8 coherence/KLD/PPL evidence was provided")
    if not gates["perf_evidence_present"]:
        blockers.append("no MQ8 gfx1151 AR/DFlash perf baseline was provided")
    if not gates["producer_surface_present"]:
        blockers.append("MQ8 producer source hooks are incomplete")
    if not gates["runtime_surface_present"]:
        blockers.append("MQ8 runtime source hooks are incomplete")
    if not gates["gfx1151_compiled_blobs_present"]:
        blockers.append("required gfx1151 MQ8/HFQ8 compiled blobs are missing")
    if not gates["no_gpu_admission_covered"]:
        blockers.append("MQ8 no-GPU admission tests are incomplete")

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "arch": "gfx1151",
        "format": "mq8",
        "status": "promotion-ready" if promotion_allowed else "permanent-runtime-research",
        "promotion_allowed": promotion_allowed,
        "product_role": product_role,
        "active_promotion_backlog": bool(promotion_allowed),
        "purpose_decision": purpose_decision(gates, promotion_allowed, product_role),
        "promotion_boundary": boundary,
        "research_closure": closure,
        "maintenance_scope": [
            "dense decode substrate",
            "gfx1151 scalar MoE bring-up admission",
            "compiled kernel cache health",
        ],
        "reopen_contract": {
            "requirements": list(REOPEN_REQUIREMENTS),
            "satisfied": {
                "product_role_over_q8_or_mq6": gates["product_role_defined"],
                "canonical_qwen3_5_9b_mq8_artifact": gates["canonical_mq8_artifact_present"],
                "mq8_example_or_benchmark_harness": gates["candidate_model_benchmark_harness_present"],
                "coherence_or_kld_ppl_quality_evidence": gates["quality_evidence_present"],
                "gfx1151_ar_and_dflash_perf_baselines": gates["perf_evidence_present"],
            },
        },
        "byte_tradeoff": byte_tradeoff,
        "artifact_inventory": artifacts,
        "source_hooks": hooks,
        "compiled_blobs": blobs,
        "harness": {
            **harness,
            "candidate_model_harness_present": gates["candidate_model_benchmark_harness_present"],
            "generic_substrate_harness_present": gates["example_or_benchmark_harness_present"],
            "candidate_harness_requires_canonical_artifact": True,
        },
        "quality_artifacts": quality_artifacts,
        "perf_artifacts": perf_artifacts,
        "gates": gates,
        "blockers": blockers,
        "origin_context": origin_context(),
        "decision": (
            "keep MQ8 permanent-runtime-research; reopen only with a product role, "
            "canonical artifact, quality evidence, and gfx1151 perf baselines"
            if not promotion_allowed
            else "all MQ8 gates are present; verify readiness matrix before promotion claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--model-root", action="append", default=[], help="Model root to scan; repeatable")
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--compiled-root", default=str(DEFAULT_COMPILED_ROOT))
    parser.add_argument("--product-role")
    parser.add_argument("--quality-artifact", action="append", default=[])
    parser.add_argument("--perf-artifact", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    model_roots = [Path(item) for item in args.model_root] if args.model_root else list(DEFAULT_MODEL_ROOTS)
    payload = build_status(
        source_root=Path(args.source_root),
        model_roots=model_roots,
        candidate_root=Path(args.candidate_root),
        compiled_root=Path(args.compiled_root),
        product_role=args.product_role,
        quality_artifacts=list(args.quality_artifact),
        perf_artifacts=list(args.perf_artifact),
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
