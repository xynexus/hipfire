#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Summarize MQ6 gfx1151 readiness status.

MQ6 is the closest non-MQ4 candidate in the current matrix, but the evidence is
split: quality is favorable and A3B/MoE runtime perf is promising, while dense
AR and dense DFlash throughput still lag MQ4 materially.  This helper converts
that split into a structured artifact so the readiness matrix cannot silently
turn "good evidence exists" into an unscoped broad-promotion claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = Path("/home/sadara/Models")
CANDIDATE_SUBDIR = Path("hipfire-candidates/gfx1151-readiness")
RESULTS_DIR = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness"
SCHEMA = "hipfire.mq6_status.gfx1151.v0"
DEFAULT_OUT = RESULTS_DIR / "2026-06-03-mq6-status.json"
DEFAULT_REPO_KERNEL_ROOT = ROOT / "kernels" / "compiled" / "gfx1151"
DEFAULT_INSTALL_KERNEL_ROOT = Path.home() / ".hipfire" / "bin" / "kernels" / "compiled" / "gfx1151"
DEFAULT_QUANTIZE_SOURCE = ROOT / "crates" / "hipfire-quantize" / "src" / "main.rs"
DEFAULT_DAEMON_SOURCE = ROOT / "crates" / "hipfire-runtime" / "examples" / "daemon.rs"
DENSE_PROMOTION_RATIO = 0.90
A3B_PREFILL_SPEEDUP_RATIO = 1.50
A3B_DECODE_CLOSE_RATIO = 0.90
MQ6_PRODUCER_ORIGIN_FIX_SHORT = "d5985c3e"
MQ6_PRODUCER_ORIGIN_FIX_COMMIT = "d5985c3e51197c70fa804f84cd694abbcd38f0d7"
MQ6_ROPE_COMPACT_OFFSET_ORIGIN_FIX_SHORT = "676338a4"
MQ6_ROPE_COMPACT_OFFSET_ORIGIN_FIX_COMMIT = "676338a47dff1d8bcb923cc3b636845bd684ca2f"
MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_SHORT = "d5bc3f6"
MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_COMMIT = "d5bc3f6af03d70e50b19b9b7564571b7f87b1f8c"
MQ6_DDTREE_EVICTION_TARGET_HIDDEN_HOST_FIX_SHORT = "d12b3ad4"
MQ6_DDTREE_EVICTION_TARGET_HIDDEN_HOST_FIX_COMMIT = "d12b3ad49fbd2464410a8f4ac75a0d3be699c21b"
MQ6_MOE_PREFILL_GROUPED_SCRATCH_FREE_FIX_SHORT = "b4adca1f"
MQ6_MOE_PREFILL_GROUPED_SCRATCH_FREE_FIX_COMMIT = "b4adca1f3f6fc97d08a3c9a4bab98ead64a5ef99"
MQ6_HUNT2_ENGINE_FIX_SHORT = "a7dcfb0d"
MQ6_HUNT2_ENGINE_FIX_COMMIT = "a7dcfb0dc3e94981b7e0f926790656a4bf616cbe"
MQ6_QWEN36_A3B_RUNTIME_FIX_SHORT = "379484ba"
MQ6_QWEN36_A3B_RUNTIME_FIX_COMMIT = "379484ba828152b63f4b22b7748261d51e705502"

DEFAULT_PATHS = {
    "provenance": RESULTS_DIR / "2026-06-02-mq6-artifact-provenance.json",
    "ppl_dense": RESULTS_DIR / "2026-06-03-mq6-ppl.json",
    "ppl_qwen35_a3b": RESULTS_DIR / "2026-06-03-mq6-a3b-ppl.json",
    "ppl_qwen36_a3b": RESULTS_DIR / "2026-06-03-mq6-qwen36-a3b-ppl.json",
    "kld_c512": RESULTS_DIR / "2026-06-03-mq6-kld-c512.json",
    "ar_perf": RESULTS_DIR / "2026-06-02-mq6-ar-perf.json",
    "dflash": RESULTS_DIR / "2026-06-03-mq6-dflash-r3.json",
    "readiness_audit": RESULTS_DIR / "2026-06-03-mq6-readiness-audit.md",
    "i8_grouped_decision": RESULTS_DIR / "2026-06-03-mq6-i8-grouped-decision.md",
}

REQUIRED_MQ6_ARTIFACTS = (
    "qwen3.5-9b-mq6.hfq",
    "qwen3.5-27b-mq6.hfq",
    "qwen3.5-35b-a3b-mq6.hfq",
    "qwen3.6-35b-a3b-mq6.hfq",
)

REQUIRED_MQ6_DENSE_DECODE_KERNELS = (
    "gemv_mq6g256.hsaco",
    "gemv_hfq6g256.hsaco",
    "gemv_hfq6g256_residual.hsaco",
    "gemv_hfq6g256_residual_sigmoid_scaled.hsaco",
    "fused_rmsnorm_mq_rotate.hsaco",
    "fused_silu_mul_mq_rotate.hsaco",
)

MQ6_DENSE_DECODE_ROUTE_TEST = "cargo test -p hipfire-runtime --lib mq6_dense_decode_routes_through_hfq6_family"
MQ6_MOE_DECODE_ROUTE_TEST = (
    "cargo test -p hipfire-arch-qwen35 --lib moe_decode_routes_mq6_indexed_path_for_k8"
)
MQ6_DFLASH_SPEC_BOUNDARY_CONTRACT_TEST = (
    "python3 scripts/test_mq6_status.py "
    "Mq6StatusTest.test_dflash_spec_boundary_marks_mq4_draft_target_verify_only"
)

MQ6_RELEVANT_UPSTREAM_COMMITS = (
    {
        "commit": "676338a47dff1d8bcb923cc3b636845bd684ca2f",
        "short": "676338a4",
        "subject": "fix(qwen35): batched RoPE ignored compact_offset -> phase skew after eviction (H4)",
        "surfaces": ("batched_rope", "eviction", "mtp", "dflash"),
        "refreshes": ("dense_dflash_perf", "spec_verify", "long_context_or_eviction_rows"),
    },
    {
        "commit": "d5985c3e51197c70fa804f84cd694abbcd38f0d7",
        "short": "d5985c3e",
        "subject": "fix(stragglers): 4 GPU leaks/dead-doc + GGUF Promote6 Mq4Lloyd",
        "surfaces": ("dflash_drafter_unload", "dpm_warmup", "gguf_producer"),
        "refreshes": ("dflash_stability", "producer_context"),
    },
    {
        "commit": "a7dcfb0dc3e94981b7e0f926790656a4bf616cbe",
        "short": "a7dcfb0d",
        "subject": "fix(engine): hunt-2 batch - premature-stop, OOB write, daemon crash, 4 GPU leaks",
        "surfaces": ("a3b", "mtp", "daemon", "kv", "gpu_leaks"),
        "refreshes": (
            "a3b_ar_perf",
            "a3b_ppl_smoke",
            "spec_verify",
            "cask_scratch_teardown",
            "mtp_gpu_teardown",
            "asym4_fwht_fallback",
        ),
    },
    {
        "commit": "b4adca1f3f6fc97d08a3c9a4bab98ead64a5ef99",
        "short": "b4adca1f",
        "subject": "fix(qwen35,daemon): free leaked GPU scratch - Path-2 MoE prefill buffers",
        "surfaces": ("a3b", "moe_prefill", "serve_oom", "deltanet"),
        "refreshes": ("a3b_ar_perf", "long_running_moe_stability", "moe_prefill_scratch_teardown"),
    },
    {
        "commit": "d5bc3f6af03d70e50b19b9b7564571b7f87b1f8c",
        "short": "d5bc3f6",
        "subject": "fix(daemon): AR exact-match DeltaNet double-advance + DFlash first-token EOS",
        "surfaces": ("ar", "dflash", "deltanet", "daemon"),
        "refreshes": (
            "dense_ar_perf",
            "dense_dflash_perf",
            "coherence",
            "daemon_ar_exact_match_delta_replay",
        ),
    },
    {
        "commit": "379484ba828152b63f4b22b7748261d51e705502",
        "short": "379484ba",
        "subject": "fix(qwen3.6-a3b): serving robustness + DeltaNet Q8 dither + OpenAI penalties",
        "surfaces": ("qwen3.6-a3b", "deltanet_q8", "daemon", "sampler"),
        "refreshes": (
            "qwen36_a3b_runtime_source_reconciliation",
            "qwen36_a3b_ppl",
            "qwen36_a3b_ar_perf",
        ),
    },
)

MQ6_RELEVANT_REMOTE_BRANCH_COMMITS = (
    {
        "commit": "fb36f5395d17b48f2d539f7ba4adc6a6c18b808e",
        "short": "fb36f539",
        "subject": "fix(engine): hunt-3 batch - 16 engine bugs across pp / sampling+MoE / daemon / bun-CLI",
        "remote_ref": "origin/fix/hunt3-engine-bugs",
        "surfaces": ("sampling", "moe", "daemon", "dflash", "spec_verify"),
        "refreshes": (
            "a3b_ar_perf",
            "a3b_ppl_smoke",
            "dense_dflash_perf",
            "spec_verify",
            "coherence",
            "hunt3_nan_safe_sampling",
        ),
    },
    {
        "commit": "d12b3ad49fbd2464410a8f4ac75a0d3be699c21b",
        "short": "d12b3ad4",
        "subject": "fix(qwen35): compact target_hidden_host on eviction - DDTree spec_step panic (#272)",
        "remote_ref": "origin/fix/issue-triage-2026-06-03",
        "surfaces": ("ddtree", "cask", "eviction", "dflash", "spec_verify"),
        "refreshes": ("ddtree_eviction_stability",),
    },
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
    origin_missing = []
    for item in MQ6_RELEVANT_UPSTREAM_COMMITS:
        present_on_origin = commit_on_ref(str(item["commit"]))
        present_on_head = commit_on_ref(str(item["commit"]), "HEAD")
        if present_on_origin is True and present_on_head is False:
            origin_missing.append(item["short"])
        commits.append(
            {
                **item,
                "surfaces": list(item["surfaces"]),
                "refreshes": list(item["refreshes"]),
                "present_on_origin_master": present_on_origin,
                "present_on_head": present_on_head,
            }
        )
    remote_commits = []
    remote_missing = []
    for item in MQ6_RELEVANT_REMOTE_BRANCH_COMMITS:
        commit = str(item["commit"])
        remote_ref = str(item["remote_ref"])
        present_on_remote = commit_on_ref(commit, remote_ref)
        present_on_origin = commit_on_ref(commit, "origin/master")
        present_on_head = commit_on_ref(commit, "HEAD")
        if present_on_remote is True and present_on_origin is False and present_on_head is False:
            remote_missing.append(item["short"])
        remote_commits.append(
            {
                **item,
                "surfaces": list(item["surfaces"]),
                "refreshes": list(item["refreshes"]),
                "present_on_remote_ref": present_on_remote,
                "present_on_origin_master": present_on_origin,
                "present_on_head": present_on_head,
            }
        )
    origin_refresh_required = bool(origin_missing)
    remote_refresh_required = bool(remote_missing)
    return {
        **upstream,
        "origin_master_refresh_required": origin_refresh_required,
        "remote_branch_refresh_required": remote_refresh_required,
        "mq6_evidence_refresh_required": bool(origin_refresh_required or remote_refresh_required),
        "origin_master_missing_commit_shorts": origin_missing,
        "remote_branch_missing_commit_shorts": remote_missing,
        "relevant_upstream_commits": commits,
        "relevant_remote_branch_commits": remote_commits,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def ratio(candidate: float | int | None, control: float | int | None) -> float | None:
    if candidate is None or control in (None, 0):
        return None
    return float(candidate) / float(control)


def artifact_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    artifacts = []
    for item in payload.get("formats", []):
        if item.get("id") == "mq6":
            artifacts = item.get("candidate_artifacts", {}).get("artifacts", [])
            break
    names = {artifact.get("name") for artifact in artifacts}
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_artifacts": list(REQUIRED_MQ6_ARTIFACTS),
        "required_artifacts_present": sorted(names & set(REQUIRED_MQ6_ARTIFACTS)),
        "required_artifacts_missing": sorted(set(REQUIRED_MQ6_ARTIFACTS) - names),
        "canonical_mq6_artifacts_present": all(name in names for name in REQUIRED_MQ6_ARTIFACTS),
    }


def describe_live_candidate(candidate_root: Path, path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(path.relative_to(candidate_root)),
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        payload["symlink_target"] = os.readlink(path)
        payload["resolved_path"] = str(path.resolve())
    if path.exists():
        payload["size_bytes"] = path.stat().st_size
    return payload


def live_candidate_inventory(models_root: Path = DEFAULT_MODELS_ROOT) -> dict[str, Any]:
    candidate_root = models_root / CANDIDATE_SUBDIR
    artifacts = []
    for name in REQUIRED_MQ6_ARTIFACTS:
        path = candidate_root / name
        artifacts.append(describe_live_candidate(candidate_root, path))
    present = sorted(artifact["name"] for artifact in artifacts if artifact["exists"])
    missing = sorted(set(REQUIRED_MQ6_ARTIFACTS) - set(present))
    return {
        "models_root": str(models_root),
        "candidate_root": str(candidate_root),
        "candidate_root_exists": candidate_root.exists(),
        "artifact_count": len(present),
        "symlink_count": sum(1 for artifact in artifacts if artifact["is_symlink"]),
        "artifacts": artifacts,
        "required_artifacts": list(REQUIRED_MQ6_ARTIFACTS),
        "required_artifacts_present": present,
        "required_artifacts_missing": missing,
        "canonical_mq6_artifacts_present": not missing,
    }


def mq6_producer_path_evidence(source_path: Path = DEFAULT_QUANTIZE_SOURCE) -> dict[str, Any]:
    text = read_text(source_path)
    promote_anchor = "kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0"
    arm_anchor = "GgufFormat::Mq4Lloyd => {"
    source_present = bool(text)
    promote_idx = text.find(promote_anchor) if source_present else -1
    arm_idx = text.find(arm_anchor, promote_idx) if promote_idx >= 0 else -1
    next_arm_idx = text.find("GgufFormat::F16 | GgufFormat::Bf16", arm_idx) if arm_idx >= 0 else -1
    matched_source = text[arm_idx:next_arm_idx].strip() if next_arm_idx > arm_idx >= 0 else ""
    emits_mq6 = all(
        token in matched_source
        for token in (
            "GgufFormat::Mq4Lloyd",
            "quantize_mq6g256",
            "QuantType::MQ6G256",
            '\"MQ6G256\"',
        )
    )
    emits_mq4_lloyd = any(
        token in matched_source
        for token in (
            "quantize_mq4g256_lloyd",
            "QuantType::MQ4G256Lloyd",
            '\"MQ4G256Lloyd\"',
        )
    )
    return {
        "schema": "hipfire.mq6_producer_path.gfx1151.v0",
        "path": str(source_path),
        "source_present": source_present,
        "origin_fix_short": MQ6_PRODUCER_ORIGIN_FIX_SHORT,
        "origin_fix_commit": MQ6_PRODUCER_ORIGIN_FIX_COMMIT,
        "promote6_anchor_present": promote_idx >= 0,
        "mq4_lloyd_arm_found_in_promote6_path": bool(matched_source),
        "promote6_mq4_lloyd_emits_mq6": emits_mq6 and not emits_mq4_lloyd,
        "origin_fix_reconciled_in_worktree": emits_mq6 and not emits_mq4_lloyd,
        "target_quant_type": "MQ6G256" if emits_mq6 else None,
        "target_group_size": 256 if "256u32" in matched_source and emits_mq6 else None,
        "stale_mq4_lloyd_emit_present": emits_mq4_lloyd,
        "evidence_class": "source_level_producer_path_check_not_artifact_rebuild",
        "matched_source": matched_source,
    }


def batched_rope_zero_offset_callsite_present(text: str) -> bool:
    call = "rope_partial_interleaved_f32_batched("
    idx = text.find(call)
    if idx < 0:
        return False
    end = text.find(")?;", idx)
    window = text[idx:] if end < 0 else text[idx : end + 3]
    return "0," in window


def mq6_batched_rope_compact_offset_evidence(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "interleaved_kernel": root / "kernels" / "src" / "rope_partial_interleaved_batched.hip",
        "halfsplit_kernel": root / "kernels" / "src" / "rope_partial_halfsplit_batched.hip",
        "dispatch": root / "crates" / "rdna-compute" / "src" / "dispatch.rs",
        "qwen35": root / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs",
        "mtp_head": root / "crates" / "hipfire-arch-qwen35" / "src" / "mtp_head.rs",
        "speculative": root / "crates" / "hipfire-arch-qwen35" / "src" / "speculative.rs",
        "no_model_example": root / "crates" / "rdna-compute" / "examples" / "rope_compact_offset_check.rs",
    }
    texts = {name: read_text(path) for name, path in paths.items()}
    qwen35_compact_offset_callsites = texts["qwen35"].count("kv_cache.compact_offset as i32")
    checks = {
        "interleaved_kernel_accepts_pos_offset": "int pos_offset" in texts["interleaved_kernel"],
        "interleaved_kernel_applies_pos_offset": "positions[b] + pos_offset" in texts[
            "interleaved_kernel"
        ],
        "halfsplit_kernel_accepts_pos_offset": "int pos_offset" in texts["halfsplit_kernel"],
        "halfsplit_kernel_applies_pos_offset": "positions[b] + pos_offset" in texts[
            "halfsplit_kernel"
        ],
        "dispatch_accepts_pos_offset": "pos_offset: i32" in texts["dispatch"],
        "dispatch_pushes_pos_offset_kernarg": "let mut po = pos_offset;" in texts["dispatch"]
        and "b.push_i32(po)" in texts["dispatch"],
        "qwen35_batched_fa_passes_compact_offset": qwen35_compact_offset_callsites >= 2,
        "mtp_head_explicit_zero_offset": batched_rope_zero_offset_callsite_present(texts["mtp_head"]),
        "tree_verify_explicit_zero_offset": batched_rope_zero_offset_callsite_present(
            texts["speculative"]
        ),
        "no_model_equivalence_example_present": 'println!("RESULT: {}"' in texts["no_model_example"]
        and '"PASS"' in texts["no_model_example"]
        and "T1 offset equivalence" in texts["no_model_example"]
        and "T2 batched offset equals per-token" in texts["no_model_example"],
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_batched_rope_compact_offset.gfx1151.v0",
        "origin_fix_short": MQ6_ROPE_COMPACT_OFFSET_ORIGIN_FIX_SHORT,
        "origin_fix_commit": MQ6_ROPE_COMPACT_OFFSET_ORIGIN_FIX_COMMIT,
        "paths": {name: str(path) for name, path in paths.items()},
        "evidence_class": "source_level_batched_rope_compact_offset_check_not_perf_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "qwen35_compact_offset_callsite_count": qwen35_compact_offset_callsites,
        "checks": checks,
        "validation_command": (
            "cargo run --release -p rdna-compute --features deltanet "
            "--example rope_compact_offset_check"
        ),
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "long_context_or_eviction_rows_refreshed_by_this_check": False,
    }


def mq6_daemon_dflash_first_token_eos_evidence(
    source_path: Path = DEFAULT_DAEMON_SOURCE,
) -> dict[str, Any]:
    text = read_text(source_path)
    checks = {
        "source_present": bool(text),
        "first_token_guard_declared": "let first_token_is_eos =" in text,
        "checks_model_eos_token": "first_token == target.config.eos_token" in text,
        "checks_chat_im_end_token": "im_end_token == Some(first_token)" in text,
        "checks_tokenizer_terminator": "tokenizer.is_terminator(first_token)" in text,
        "spec_loop_guarded": "while !first_token_is_eos && generated < max_tokens" in text,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_daemon_dflash_first_token_eos.gfx1151.v0",
        "path": str(source_path),
        "origin_fix_short": MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_SHORT,
        "origin_fix_commit": MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_COMMIT,
        "evidence_class": "source_level_daemon_dflash_first_token_eos_check_not_perf_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "coherence_rows_refreshed_by_this_check": False,
    }


def mq6_daemon_ar_exact_match_delta_replay_evidence(
    source_path: Path = DEFAULT_DAEMON_SOURCE,
) -> dict[str, Any]:
    text = read_text(source_path)
    generate_idx = text.find("fn generate(")
    deepseek_idx = text.find("\nfn generate_deepseek4(", generate_idx)
    generate_window = text[generate_idx:deepseek_idx] if generate_idx >= 0 and deepseek_idx > generate_idx else ""
    qwen35_idx = generate_window.find("if m.arch_id == 5 || m.arch_id == 6 {")
    qwen35_end_idx = generate_window.find("// ngram scope", qwen35_idx)
    qwen35_window = (
        generate_window[qwen35_idx:qwen35_end_idx]
        if qwen35_idx >= 0 and qwen35_end_idx > qwen35_idx
        else ""
    )
    legacy_terms = (
        "lcp_adj",
        "lcp == rendered.len()",
        "cached_tokens_count",
        "rendered[lcp_adj..]",
        "prior_len",
    )
    direct_prefill_terms = (
        "qwen35::forward_prefill_batch(",
        "&new_tokens",
        "m.seq_pos",
        "m.seq_pos += new_tokens.len();",
        "m.conversation_tokens.extend_from_slice(&new_tokens);",
    )
    checks = {
        "source_present": bool(text),
        "generate_window_found": bool(generate_window),
        "qwen35_single_gpu_branch_found": bool(qwen35_window),
        "legacy_exact_match_lcp_rewind_absent": not any(term in generate_window for term in legacy_terms),
        "qwen35_prefills_direct_new_tokens": all(term in qwen35_window for term in direct_prefill_terms),
        "qwen35_no_conversation_lcp_scan": "longest_common_prefix" not in qwen35_window
        and "conversation_tokens.iter()" not in qwen35_window,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_daemon_ar_exact_match_delta_replay.gfx1151.v0",
        "path": str(source_path),
        "origin_fix_short": MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_SHORT,
        "origin_fix_commit": MQ6_DAEMON_DFLASH_EOS_ORIGIN_FIX_COMMIT,
        "evidence_class": "source_level_daemon_ar_exact_match_delta_replay_check_not_perf_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "legacy_terms_checked": list(legacy_terms),
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "coherence_rows_refreshed_by_this_check": False,
    }


def mq6_ddtree_eviction_target_hidden_host_evidence(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "speculative": root / "crates" / "hipfire-arch-qwen35" / "src" / "speculative.rs",
        "daemon": root / "crates" / "hipfire-runtime" / "examples" / "daemon.rs",
        "dflash_spec_demo": root / "crates" / "hipfire-runtime" / "examples" / "dflash_spec_demo.rs",
    }
    texts = {name: read_text(path) for name, path in paths.items()}
    helper_name = "compact_target_hidden_host"
    checks = {
        "speculative_source_present": bool(texts["speculative"]),
        "helper_declared": f"pub fn {helper_name}" in texts["speculative"],
        "helper_empty_mask_noop": "if retain_mask.is_empty()" in texts["speculative"],
        "helper_uses_retain_mask_rows": "for &src_idx in retain_mask" in texts["speculative"],
        "helper_replaces_cpu_shadow": "*target_hidden_host = compacted;" in texts["speculative"],
        "unit_test_selected_rows_present": "compact_target_hidden_host_keeps_selected_rows" in texts["speculative"],
        "unit_test_empty_mask_present": "compact_target_hidden_host_empty_mask_is_noop" in texts["speculative"],
        "daemon_post_prefill_and_cycle_calls_present": texts["daemon"].count(helper_name) >= 2
        and texts["daemon"].count("&mut df.target_hidden_host") >= 2,
        "dflash_spec_demo_post_prefill_and_cycle_calls_present": texts["dflash_spec_demo"].count(helper_name) >= 2
        and texts["dflash_spec_demo"].count("&mut target_hidden_host") >= 2,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_ddtree_eviction_target_hidden_host.gfx1151.v0",
        "origin_fix_short": MQ6_DDTREE_EVICTION_TARGET_HIDDEN_HOST_FIX_SHORT,
        "origin_fix_commit": MQ6_DDTREE_EVICTION_TARGET_HIDDEN_HOST_FIX_COMMIT,
        "remote_ref": "origin/fix/issue-triage-2026-06-03",
        "paths": {name: str(path) for name, path in paths.items()},
        "evidence_class": "source_level_ddtree_eviction_cpu_shadow_check_not_model_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "validation_command": "cargo test -p hipfire-arch-qwen35 --lib compact_target_hidden_host",
        "perf_rows_refreshed_by_this_check": False,
        "coherence_rows_refreshed_by_this_check": False,
        "long_context_or_eviction_rows_refreshed_by_this_check": False,
    }


def mq6_moe_prefill_grouped_scratch_free_evidence(
    source_path: Path = ROOT / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs",
) -> dict[str, Any]:
    text = read_text(source_path)
    impl_anchor = "impl PrefillBatchScratch"
    fn_anchor = "pub fn free_gpu(self, gpu: &mut Gpu)"
    fields = (
        "self.moe_expert_token_counts",
        "self.moe_expert_offsets",
        "self.moe_sorted_slot_index",
        "self.moe_inverse_perm",
        "self.moe_expert_tile_ids",
        "self.moe_y_gate_up_grouped",
        "self.moe_y_down_grouped",
    )
    impl_idx = text.find(impl_anchor)
    fn_idx = text.find(fn_anchor, impl_idx) if impl_idx >= 0 else -1
    window = text[fn_idx : text.find("    }\n}", fn_idx)] if fn_idx >= 0 else ""
    checks = {
        "source_present": bool(text),
        "prefill_batch_impl_found": impl_idx >= 0,
        "prefill_batch_free_gpu_found": bool(window),
        "frees_moe_expert_token_counts": fields[0] in window,
        "frees_moe_expert_offsets": fields[1] in window,
        "frees_moe_sorted_slot_index": fields[2] in window,
        "frees_moe_inverse_perm": fields[3] in window,
        "frees_moe_expert_tile_ids": fields[4] in window,
        "frees_moe_y_gate_up_grouped": fields[5] in window,
        "frees_moe_y_down_grouped": fields[6] in window,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_moe_prefill_grouped_scratch_free.gfx1151.v0",
        "path": str(source_path),
        "origin_fix_short": MQ6_MOE_PREFILL_GROUPED_SCRATCH_FREE_FIX_SHORT,
        "origin_fix_commit": MQ6_MOE_PREFILL_GROUPED_SCRATCH_FREE_FIX_COMMIT,
        "evidence_class": "source_level_moe_prefill_grouped_scratch_teardown_check_not_stability_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "required_fields": list(fields),
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "long_running_moe_stability_refreshed_by_this_check": False,
    }


def mq6_cask_scratch_teardown_evidence(
    source_path: Path = ROOT / "crates" / "hipfire-runtime" / "src" / "cask.rs",
) -> dict[str, Any]:
    text = read_text(source_path)
    alloc_indices_idx = text.find("let indices_dev = gpu.alloc_tensor")
    alloc_weights_idx = text.find("let weights_dev = gpu.alloc_tensor")
    closure_idx = text.find("let inner_result = (|| -> HipResult<()>")
    free_indices_idx = text.find("let _ = gpu.free_tensor(indices_dev);", closure_idx)
    free_weights_idx = text.find("let _ = gpu.free_tensor(weights_dev);", closure_idx)
    propagate_idx = text.find("inner_result?;", free_weights_idx)
    checks = {
        "source_present": bool(text),
        "indices_scratch_allocated": alloc_indices_idx >= 0,
        "weights_scratch_allocated": alloc_weights_idx >= 0,
        "fallible_work_wrapped_in_inner_result": closure_idx >= 0,
        "indices_scratch_freed_after_inner_result": free_indices_idx > closure_idx,
        "weights_scratch_freed_after_inner_result": free_weights_idx > closure_idx,
        "inner_result_propagated_after_free": propagate_idx > free_weights_idx,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_cask_scratch_teardown.gfx1151.v0",
        "path": str(source_path),
        "origin_fix_short": MQ6_HUNT2_ENGINE_FIX_SHORT,
        "origin_fix_commit": MQ6_HUNT2_ENGINE_FIX_COMMIT,
        "evidence_class": "source_level_cask_scratch_teardown_check_not_eviction_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "long_context_or_eviction_rows_refreshed_by_this_check": False,
    }


def mq6_mtp_gpu_teardown_evidence(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "mtp_head": root / "crates" / "hipfire-arch-qwen35" / "src" / "mtp_head.rs",
        "mtp_spec": root / "crates" / "hipfire-arch-qwen35" / "src" / "mtp_spec.rs",
        "mtp_compose": root / "crates" / "hipfire-arch-qwen35" / "src" / "mtp_compose.rs",
    }
    texts = {name: read_text(path) for name, path in paths.items()}
    head_idx = texts["mtp_head"].find("impl Qwen35MtpHeadKvCache")
    head_free_idx = texts["mtp_head"].find(
        "pub fn free_gpu(self, gpu: &mut Gpu)",
        head_idx,
    )
    head_window = texts["mtp_head"][head_free_idx : head_free_idx + 240] if head_free_idx >= 0 else ""
    spec_idx = texts["mtp_spec"].find("impl MtpSpecState")
    spec_free_idx = texts["mtp_spec"].find(
        "pub fn free_gpu(self, gpu: &mut Gpu)",
        spec_idx,
    )
    spec_window = texts["mtp_spec"][spec_free_idx : spec_free_idx + 3500] if spec_free_idx >= 0 else ""
    checks = {
        "mtp_head_source_present": bool(texts["mtp_head"]),
        "mtp_spec_source_present": bool(texts["mtp_spec"]),
        "mtp_compose_source_present": bool(texts["mtp_compose"]),
        "mtp_head_kv_cache_free_gpu_found": bool(head_window),
        "mtp_head_kv_cache_frees_inner_gpu_tensors": "self.inner.free_gpu(gpu);" in head_window,
        "mtp_head_no_drop_inner_in_kv_free_gpu": "drop(self.inner)" not in head_window,
        "mtp_spec_free_gpu_found": bool(spec_window),
        "mtp_spec_frees_trunk_snapshot_gpu": "self.trunk_snap.free_gpu(gpu);" in spec_window,
        "mtp_spec_frees_mtp_kv_inner_gpu_tensors": "self.mtp_kv.inner.free_gpu(gpu);" in spec_window,
        "mtp_spec_no_drop_trunk_snapshot": "drop(self.trunk_snap)" not in spec_window,
        "mtp_compose_frees_both_mtp_kv_inner_states": texts["mtp_compose"].count(
            "self.mtp_kv.inner.free_gpu(gpu);"
        )
        >= 2,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_mtp_gpu_teardown.gfx1151.v0",
        "origin_fix_short": MQ6_HUNT2_ENGINE_FIX_SHORT,
        "origin_fix_commit": MQ6_HUNT2_ENGINE_FIX_COMMIT,
        "paths": {name: str(path) for name, path in paths.items()},
        "evidence_class": "source_level_mtp_gpu_teardown_check_not_long_run_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "long_running_mtp_rows_refreshed_by_this_check": False,
    }


def mq6_asym4_fwht_fallback_evidence(
    source_path: Path = ROOT / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs",
) -> dict[str, Any]:
    text = read_text(source_path)
    fn_idx = text.find("fn run_fa_layer_body(")
    next_mode_idx = text.find("} else if kv_cache.quant_asym3 {", fn_idx)
    window = text[fn_idx:next_mode_idx] if fn_idx >= 0 and next_mode_idx > fn_idx else ""
    checks = {
        "source_present": bool(text),
        "run_fa_layer_body_found": fn_idx >= 0,
        "asym4_branch_found": "if kv_cache.quant_asym4 {" in window,
        "fwht_subbranch_found": "if kv_cache.quant_fwht {" in window,
        "fwht_writer_dispatched": "gpu.kv_cache_write_fwht4_fused(" in window,
        "fwht_attention_dispatched": "gpu.attention_flash_fwht4(" in window,
        "fwht4_uses_branch_native_q8_v_signature": "kv_cache.v_mode_bits()" not in window,
        "asym4_fallback_writer_preserved": "gpu.kv_cache_write_asym4_fused(" in window,
        "asym4_fallback_attention_preserved": "gpu.attention_flash_asym4(" in window,
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_asym4_fwht_fallback.gfx1151.v0",
        "path": str(source_path),
        "origin_fix_short": MQ6_HUNT2_ENGINE_FIX_SHORT,
        "origin_fix_commit": MQ6_HUNT2_ENGINE_FIX_COMMIT,
        "evidence_class": "source_level_asym4_fwht_fallback_dispatch_check_not_decode_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "perf_rows_refreshed_by_this_check": False,
        "coherence_rows_refreshed_by_this_check": False,
    }


def mq6_qwen36_a3b_runtime_source_reconciliation_evidence(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "sampler": root / "crates" / "hipfire-runtime" / "src" / "sampler.rs",
        "daemon": root / "crates" / "hipfire-runtime" / "examples" / "daemon.rs",
        "dispatch": root / "crates" / "rdna-compute" / "src" / "dispatch.rs",
        "cli": root / "cli" / "index.ts",
        "sample_top_p_kernel": root / "kernels" / "src" / "sample_top_p.hip",
        "gated_delta_net_q8_kernel": root / "kernels" / "src" / "gated_delta_net_q8.hip",
    }
    texts = {name: read_text(path) for name, path in paths.items()}
    checks = {
        "sources_present": all(bool(text) for text in texts.values()),
        "sampler_config_has_native_penalties": all(
            token in texts["sampler"]
            for token in (
                "pub presence_penalty: f32",
                "pub frequency_penalty: f32",
            )
        ),
        "sampler_gpu_path_threads_penalties": all(
            token in texts["sampler"]
            for token in (
                ".sample_top_p_pf(",
                "cfg.presence_penalty",
                "cfg.frequency_penalty",
            )
        ),
        "sampler_cpu_path_subtracts_additive_penalties": (
            "logits[tok as usize] -= cfg.frequency_penalty * count + cfg.presence_penalty"
            in texts["sampler"]
        ),
        "dispatch_exposes_sample_top_p_pf": all(
            token in texts["dispatch"]
            for token in (
                "pub fn sample_top_p_pf",
                "let mut pp = presence_penalty",
                "let mut fp = frequency_penalty",
            )
        ),
        "sample_top_p_kernel_accepts_and_applies_penalties": all(
            token in texts["sample_top_p_kernel"]
            for token in (
                "float presence_penalty",
                "float frequency_penalty",
                "logits[tok] -= frequency_penalty * (float)count + presence_penalty",
            )
        ),
        "daemon_parses_and_threads_native_penalties": all(
            token in texts["daemon"]
            for token in (
                '.get("presence_penalty")',
                '.get("frequency_penalty")',
                "presence_penalty: f32",
                "frequency_penalty: f32",
            )
        )
        and texts["daemon"].count("presence_penalty,") >= 5
        and texts["daemon"].count("frequency_penalty,") >= 5,
        "daemon_allocates_wide_repeat_buffers": all(
            token in texts["daemon"]
            for token in (
                "Qwen35Scratch::new_with_kv_max(gpu, &config, 2048",
                "Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 2048",
            )
        ),
        "cli_sends_native_penalties_without_repeat_fold": all(
            token in texts["cli"]
            for token in (
                "const presencePenalty = Math.max(0, Number(body.presence_penalty) || 0)",
                "const frequencyPenalty = Math.max(0, Number(body.frequency_penalty) || 0)",
                "presence_penalty: presencePenalty",
                "frequency_penalty: frequencyPenalty",
                "repeat_penalty: body.repeat_penalty ?? effective.repeat_penalty",
            )
        )
        and "oaiPenalty" not in texts["cli"],
        "cli_marks_open_think_for_visible_stripping": all(
            token in texts["cli"]
            for token in (
                'genParams.assistant_prefix = "open_think"',
                'let inThink = genParams.assistant_prefix === "open_think"',
                'content = "<think>" + content',
            )
        ),
        "deltanet_dispatch_uses_per_launch_requant_frame": all(
            token in texts["dispatch"]
            for token in (
                "static GDN_REQUANT_FRAME: AtomicU32",
                "GDN_REQUANT_FRAME.fetch_add(1, Ordering::Relaxed)",
                "b.push_i32(fr)",
            )
        ),
        "deltanet_kernel_uses_frame_independent_dither": all(
            token in texts["gated_delta_net_q8_kernel"]
            for token in (
                "unsigned int frame",
                "frame * 2531011u",
            )
        )
        and "__float_as_uint(my_max) * 2531011u" not in texts["gated_delta_net_q8_kernel"],
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_qwen36_a3b_runtime_source_reconciliation.gfx1151.v0",
        "origin_fix_short": MQ6_QWEN36_A3B_RUNTIME_FIX_SHORT,
        "origin_fix_commit": MQ6_QWEN36_A3B_RUNTIME_FIX_COMMIT,
        "paths": {name: str(path) for name, path in paths.items()},
        "evidence_class": "source_level_qwen36_a3b_runtime_check_not_model_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "validation_commands": [
            "cargo test -p hipfire-runtime --lib sampler",
            "cargo check --workspace --all-targets",
            "bun test cli/index.ts",
        ],
        "ppl_rows_refreshed_by_this_check": False,
        "perf_rows_refreshed_by_this_check": False,
        "coherence_rows_refreshed_by_this_check": False,
    }


def mq6_hunt3_nan_safe_sampling_evidence(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "daemon": root / "crates" / "hipfire-runtime" / "examples" / "daemon.rs",
        "llama": root / "crates" / "hipfire-runtime" / "src" / "llama.rs",
        "mtp_spec": root / "crates" / "hipfire-arch-qwen35" / "src" / "mtp_spec.rs",
        "deepseek_sampling": root
        / "crates"
        / "hipfire-arch-deepseek4"
        / "src"
        / "sampling.rs",
    }
    texts = {name: read_text(path) for name, path in paths.items()}
    checks = {
        "llama_argmax_uses_nan_safe_fold": all(
            token in texts["llama"]
            for token in (
                "fold((0usize, f32::NEG_INFINITY)",
                "if v > best.1",
                "fn argmax_ignores_nan_logits",
            )
        ),
        "llama_cpu_sampler_rng_reset_exported": all(
            token in texts["llama"]
            for token in (
                "pub fn reset_cpu_sampler_rng(seed: u32)",
                "if seed == 0 { 1 } else { seed }",
                "fn cpu_sampler_rng_reset_is_deterministic",
            )
        ),
        "daemon_cpu_sampler_rng_reset_wired": texts["daemon"].count(
            "hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);"
        )
        >= 2
        and "fn generate(" in texts["daemon"]
        and "fn generate_vl(" in texts["daemon"],
        "mtp_sampling_drops_nan_candidates": all(
            token in texts["mtp_spec"]
            for token in (
                ".filter(|(_, &l)| !l.is_nan())",
                "return (0, 0.0);",
                "fn sample_from_logits_drops_nan_candidates",
            )
        )
        and "partial_cmp(&a.1).unwrap()" not in texts["mtp_spec"],
        "deepseek_sampling_drops_nan_candidates": all(
            token in texts["deepseek_sampling"]
            for token in (
                ".filter(|&i| !logits[i].is_nan())",
                "fn sampled_path_drops_nan_logits",
                "fn all_nan_logits_return_zero_without_panic",
            )
        )
        and "partial_cmp(b.1).unwrap()" not in texts["deepseek_sampling"]
        and "partial_cmp(&logits[a]).unwrap()" not in texts["deepseek_sampling"],
    }
    source_reconciled = all(checks.values())
    return {
        "schema": "hipfire.mq6_hunt3_nan_safe_sampling.gfx1151.v0",
        "origin_fix_short": "fb36f539",
        "origin_fix_commit": "fb36f5395d17b48f2d539f7ba4adc6a6c18b808e",
        "remote_ref": "origin/fix/hunt3-engine-bugs",
        "paths": {name: str(path) for name, path in paths.items()},
        "evidence_class": "source_level_hunt3_nan_safe_sampling_check_not_model_rerun",
        "source_reconciled_in_worktree": source_reconciled,
        "checks": checks,
        "unreconciled_by_source_check": sorted(
            name for name, present in checks.items() if not present
        ),
        "validation_commands": [
            "cargo test -p hipfire-runtime --lib argmax",
            "cargo test -p hipfire-arch-qwen35 --lib sample_from_logits",
            "cargo test -p hipfire-arch-deepseek4 --lib sampling",
            "cargo check -p hipfire-runtime --example daemon",
            "cargo check --workspace --all-targets",
        ],
        "coherence_rows_refreshed_by_this_check": False,
        "perf_rows_refreshed_by_this_check": False,
    }


def describe_kernel_root(label: str, root: Path, required: tuple[str, ...]) -> dict[str, Any]:
    kernels = []
    for name in required:
        path = root / name
        entry: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            entry["size_bytes"] = path.stat().st_size
        kernels.append(entry)
    present = sorted(kernel["name"] for kernel in kernels if kernel["exists"])
    missing = sorted(set(required) - set(present))
    return {
        "label": label,
        "root": str(root),
        "root_exists": root.exists(),
        "required_count": len(required),
        "present_count": len(present),
        "missing_count": len(missing),
        "required_kernels_present": not missing,
        "present": present,
        "missing": missing,
        "kernels": kernels,
    }


def kernel_inventory(
    repo_root: Path = DEFAULT_REPO_KERNEL_ROOT,
    install_root: Path = DEFAULT_INSTALL_KERNEL_ROOT,
) -> dict[str, Any]:
    repo = describe_kernel_root("repo", repo_root, REQUIRED_MQ6_DENSE_DECODE_KERNELS)
    install = describe_kernel_root("install", install_root, REQUIRED_MQ6_DENSE_DECODE_KERNELS)
    return {
        "schema": "hipfire.mq6_dense_decode_kernel_inventory.gfx1151.v0",
        "arch": "gfx1151",
        "format": "mq6",
        "evidence_class": "compiled_blob_presence_only_not_dispatch_proof",
        "required_kernels": list(REQUIRED_MQ6_DENSE_DECODE_KERNELS),
        "roots": {
            "repo": repo,
            "install": install,
        },
        "repo_required_kernels_present": bool(repo["required_kernels_present"]),
        "install_required_kernels_present": bool(install["required_kernels_present"]),
        "any_required_kernel_set_present": bool(
            repo["required_kernels_present"] or install["required_kernels_present"]
        ),
    }


def cases_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    return {str(case.get("id")): case for case in cases if isinstance(case, dict)}


def ppl_case_digest(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {
        "id": case.get("id"),
        "family": case.get("family"),
        "format_id": case.get("format_id"),
        "status": case.get("status"),
        "exit_code": case.get("exit_code"),
        "model_file": case.get("model_file"),
        "result": case.get("result", {}),
    }


def ppl_comparison(path: Path, candidate_id: str, control_id: str) -> dict[str, Any]:
    by_id = cases_by_id(path)
    candidate = by_id.get(candidate_id)
    control = by_id.get(control_id)
    candidate_ppl = None if candidate is None else candidate.get("result", {}).get("ppl")
    control_ppl = None if control is None else control.get("result", {}).get("ppl")
    return {
        "path": str(path),
        "candidate": ppl_case_digest(candidate),
        "control": ppl_case_digest(control),
        "candidate_to_control_ppl_ratio": ratio(candidate_ppl, control_ppl),
        "candidate_beats_control": bool(
            candidate_ppl is not None and control_ppl is not None and float(candidate_ppl) < float(control_ppl)
        ),
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


def row_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


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


def kld_comparison(path: Path, candidate_token: str, control_token: str) -> dict[str, Any]:
    rows = rows_from_kld(path)
    candidate = row_containing(rows, candidate_token)
    control = row_containing(rows, control_token)
    candidate_kld = row_float(candidate, "mean_kld")
    control_kld = row_float(control, "mean_kld")
    return {
        "path": str(path),
        "candidate": kld_digest(candidate),
        "control": kld_digest(control),
        "candidate_to_control_mean_kld_ratio": ratio(candidate_kld, control_kld),
        "candidate_beats_control": bool(
            candidate_kld is not None and control_kld is not None and candidate_kld < control_kld
        ),
    }


def runs_ok(case: dict[str, Any] | None) -> bool:
    if case is None:
        return False
    summary = case.get("summary", {})
    total = summary.get("total_runs")
    ok = summary.get("ok_runs")
    if total is not None or ok is not None:
        return bool(total and ok == total)
    runs = case.get("runs", [])
    return bool(runs) and all(run.get("status") == "ok" and run.get("exit_code") == 0 for run in runs)


def token_attractor_ok(case: dict[str, Any] | None) -> bool:
    if case is None:
        return False
    runs = case.get("runs", [])
    checks = [run.get("token_attractor", {}).get("ok") for run in runs if isinstance(run, dict)]
    return bool(checks) and all(check is True for check in checks)


def perf_case_digest(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {
        "id": case.get("id"),
        "family": case.get("family"),
        "format_id": case.get("format_id"),
        "prompt_md5": case.get("prompt_md5"),
        "model_file": case.get("model_file") or case.get("target_file"),
        "target_format": case.get("target_format"),
        "summary": case.get("summary", {}),
        "all_runs_ok": runs_ok(case),
        "token_attractor_ok": token_attractor_ok(case) if case.get("runs") else None,
    }


def perf_pair(path: Path, candidate_id: str, control_id: str) -> dict[str, Any]:
    by_id = cases_by_id(path)
    candidate = by_id.get(candidate_id)
    control = by_id.get(control_id)
    candidate_summary = candidate.get("summary", {}) if candidate else {}
    control_summary = control.get("summary", {}) if control else {}
    candidate_prefill = candidate_summary.get("median_prefill_tok_s")
    control_prefill = control_summary.get("median_prefill_tok_s")
    candidate_decode = candidate_summary.get("median_decode_tok_s")
    control_decode = control_summary.get("median_decode_tok_s")
    return {
        "path": str(path),
        "candidate": perf_case_digest(candidate),
        "control": perf_case_digest(control),
        "prefill_ratio": ratio(candidate_prefill, control_prefill),
        "decode_ratio": ratio(candidate_decode, control_decode),
        "all_runs_ok": runs_ok(candidate) and runs_ok(control),
    }


def dflash_pair(path: Path, candidate_id: str, control_id: str) -> dict[str, Any]:
    pair = perf_pair(path, candidate_id, control_id)
    candidate_summary = pair.get("candidate", {}).get("summary", {}) if pair.get("candidate") else {}
    control_summary = pair.get("control", {}).get("summary", {}) if pair.get("control") else {}
    pair["tau_ratio"] = ratio(candidate_summary.get("median_decode_tau"), control_summary.get("median_decode_tau"))
    pair["accept_rate_ratio"] = ratio(
        candidate_summary.get("median_decode_accept_rate"),
        control_summary.get("median_decode_accept_rate"),
    )
    pair["token_attractor_ok"] = bool(
        pair.get("candidate", {}).get("token_attractor_ok")
        and pair.get("control", {}).get("token_attractor_ok")
    )
    return pair


def format_from_filename(name: str | None) -> str:
    lowered = str(name or "").lower()
    for token in ("mq2lloyd", "mq3lloyd", "mq4lloyd", "mq8", "mq6", "mq4", "mq3", "q8", "f16", "bf16"):
        if token in lowered:
            return token
    return "unknown"


def classify_dflash_spec_evidence(
    *,
    mq6_case_count: int,
    mq4_draft_used: bool,
    mq6_draft_present: bool,
    dense_27b_only: bool,
    a3b_present: bool,
    all_three_run: bool,
    token_clean: bool,
) -> dict[str, Any]:
    if mq6_case_count == 0:
        evidence_class = "missing_mq6_dflash_evidence"
    elif mq6_draft_present and a3b_present:
        evidence_class = "mq6_draft_and_a3b_dflash_evidence_present"
    elif mq6_draft_present:
        evidence_class = "mq6_draft_evidence_present"
    elif a3b_present:
        evidence_class = "a3b_dflash_evidence_present"
    elif mq4_draft_used and dense_27b_only:
        evidence_class = "target_side_verify_only_mq4_draft"
    else:
        evidence_class = "mixed_or_ambiguous_dflash_evidence"

    scope_expanded = evidence_class in {
        "mq6_draft_and_a3b_dflash_evidence_present",
        "mq6_draft_evidence_present",
        "a3b_dflash_evidence_present",
    }
    target_verify_only = evidence_class == "target_side_verify_only_mq4_draft"
    return {
        "evidence_class": evidence_class,
        "target_side_verify_only": target_verify_only,
        "scope_expanded_beyond_target_verify": scope_expanded,
        "contract_ready": bool(mq6_case_count and all_three_run and token_clean),
        "promotion_scope_supported": False,
    }


def dflash_spec_boundary(path: Path, origin: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    mq6_cases = [
        case
        for case in cases
        if isinstance(case, dict) and str(case.get("target_format", "")).lower() == "mq6"
    ]
    run_counts = []
    for case in mq6_cases:
        summary = case.get("summary", {})
        run_count = summary.get("total_runs")
        if run_count is None:
            run_count = len(case.get("runs", []))
        run_counts.append(int(run_count or 0))
    draft_formats = sorted({format_from_filename(case.get("draft_file")) for case in mq6_cases})
    target_files = sorted({str(case.get("target_file")) for case in mq6_cases if case.get("target_file")})
    prompt_md5s = sorted({str(case.get("prompt_md5")) for case in mq6_cases if case.get("prompt_md5")})
    mq4_draft_used = bool(mq6_cases) and draft_formats == ["mq4"]
    mq6_draft_present = any(format_from_filename(case.get("draft_file")) == "mq6" for case in mq6_cases)
    dense_27b_only = bool(mq6_cases) and all(
        "27b" in str(case.get("target_file", "")).lower()
        and "a3b" not in str(case.get("target_file", "")).lower()
        for case in mq6_cases
    )
    a3b_present = any("a3b" in str(case.get("target_file", "")).lower() for case in mq6_cases)
    all_three_run = bool(run_counts) and min(run_counts) >= 3
    token_clean = bool(mq6_cases) and all(token_attractor_ok(case) for case in mq6_cases)
    classification = classify_dflash_spec_evidence(
        mq6_case_count=len(mq6_cases),
        mq4_draft_used=mq4_draft_used,
        mq6_draft_present=mq6_draft_present,
        dense_27b_only=dense_27b_only,
        a3b_present=a3b_present,
        all_three_run=all_three_run,
        token_clean=token_clean,
    )
    refresh_required = bool(origin["mq6_evidence_refresh_required"])
    return {
        "path": str(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        **classification,
        "mq6_case_count": len(mq6_cases),
        "target_files": target_files,
        "draft_formats": draft_formats,
        "prompt_md5s": prompt_md5s,
        "min_runs_per_case": min(run_counts) if run_counts else 0,
        "all_mq6_cases_three_run": all_three_run,
        "token_attractor_clean": token_clean,
        "mq4_draft_used_for_all_mq6_rows": mq4_draft_used,
        "mq6_draft_evidence_present": mq6_draft_present,
        "dense_27b_only": dense_27b_only,
        "a3b_dflash_evidence_present": a3b_present,
        "sampling_or_non_greedy_evidence_present": False,
        "refresh_required": refresh_required,
        "next_unblocked_step": "reconcile_origin_and_remote_fixes_then_rerun_dense_and_a3b_dflash_spec",
    }


def surface_record(
    *,
    status: str,
    runtime_evidence_present: bool,
    quality_evidence_present: bool,
    perf_evidence_present: bool,
    promotion_ready: bool,
    refresh_required: bool,
    evidence: list[str],
    blockers: list[str],
    next_unblocked_step: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "runtime_evidence_present": runtime_evidence_present,
        "quality_evidence_present": quality_evidence_present,
        "perf_evidence_present": perf_evidence_present,
        "promotion_ready": promotion_ready,
        "refresh_required": refresh_required,
        "evidence": evidence,
        "blockers": blockers,
        "next_unblocked_step": next_unblocked_step,
        "details": details or {},
    }


def mq6_surface_coverage(
    *,
    gates: dict[str, Any],
    kernels: dict[str, Any],
    batched_rope_compact_offset: dict[str, Any],
    daemon_dflash_first_token_eos: dict[str, Any],
    daemon_ar_exact_match_delta_replay: dict[str, Any],
    ddtree_eviction_target_hidden_host: dict[str, Any],
    moe_prefill_grouped_scratch_free: dict[str, Any],
    cask_scratch_teardown: dict[str, Any],
    mtp_gpu_teardown: dict[str, Any],
    asym4_fwht_fallback: dict[str, Any],
    dense_decode_route_test: dict[str, Any],
    moe_decode_route_test: dict[str, Any],
    dflash_boundary_contract_test: dict[str, Any],
    dense_ar_pairs: dict[str, dict[str, Any]],
    a3b_ar_pairs: dict[str, dict[str, Any]],
    dflash_boundary: dict[str, Any],
    refresh_required: bool,
) -> dict[str, Any]:
    dense_decode_ratios = {
        key: pair.get("decode_ratio")
        for key, pair in dense_ar_pairs.items()
    }
    dense_prefill_ratios = {
        key: pair.get("prefill_ratio")
        for key, pair in dense_ar_pairs.items()
    }
    a3b_decode_ratios = {
        key: pair.get("decode_ratio")
        for key, pair in a3b_ar_pairs.items()
    }
    a3b_prefill_ratios = {
        key: pair.get("prefill_ratio")
        for key, pair in a3b_ar_pairs.items()
    }
    surfaces = {
        "dense_decode": surface_record(
            status="perf_gated",
            runtime_evidence_present=bool(
                gates["dense_ar_perf_present"] and gates["dense_decode_repo_kernel_inventory_present"]
                and gates["dense_decode_no_gpu_route_test_present"]
            ),
            quality_evidence_present=bool(
                gates["dense_9b_ppl_beats_mq4"] and gates["dense_9b_c512_kld_beats_mq4"]
            ),
            perf_evidence_present=bool(gates["dense_ar_perf_present"]),
            promotion_ready=bool(gates["dense_ar_perf_promotable"] and not refresh_required),
            refresh_required=refresh_required,
            evidence=[
                MQ6_DENSE_DECODE_ROUTE_TEST,
                "kernel_inventory.required_mq6_dense_decode_kernels",
                "2026-06-02-mq6-ar-perf.json",
                "2026-06-03-mq6-ppl.json",
                "2026-06-03-mq6-kld-c512.json",
            ],
            blockers=["dense_decode_ratio_below_0.90x_mq4", "origin_or_remote_refresh_required"],
            next_unblocked_step="fix_dense_decode_throughput_or_scope_dense_out",
            details={
                "decode_ratios": dense_decode_ratios,
                "kernel_inventory": {
                    "evidence_class": kernels["evidence_class"],
                    "repo_required_kernels_present": kernels["repo_required_kernels_present"],
                    "install_required_kernels_present": kernels["install_required_kernels_present"],
                    "required_kernels": kernels["required_kernels"],
                    "repo_missing": kernels["roots"]["repo"]["missing"],
                    "install_missing": kernels["roots"]["install"]["missing"],
                },
                "no_gpu_route_test": {
                    "command": dense_decode_route_test["command"],
                    "present": dense_decode_route_test["present"],
                    "all_phrase_checks_present": dense_decode_route_test["all_phrase_checks_present"],
                },
                "daemon_ar_exact_match_delta_replay": {
                    "origin_fix_short": daemon_ar_exact_match_delta_replay["origin_fix_short"],
                    "evidence_class": daemon_ar_exact_match_delta_replay["evidence_class"],
                    "source_reconciled_in_worktree": daemon_ar_exact_match_delta_replay[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "coherence_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                        "coherence_rows_refreshed_by_this_check"
                    ],
                    "checks": daemon_ar_exact_match_delta_replay["checks"],
                },
            },
        ),
        "dense_prefill": surface_record(
            status="perf_gated",
            runtime_evidence_present=bool(gates["dense_ar_perf_present"]),
            quality_evidence_present=bool(
                gates["dense_9b_ppl_beats_mq4"] and gates["dense_9b_c512_kld_beats_mq4"]
            ),
            perf_evidence_present=bool(gates["dense_ar_perf_present"]),
            promotion_ready=bool(gates["dense_ar_perf_promotable"] and not refresh_required),
            refresh_required=refresh_required,
            evidence=[
                "cargo test -p hipfire-arch-qwen35 --lib dense_prefill_mq6_and_hfq6_are_batchable_in_qwen35",
                "2026-06-02-mq6-ar-perf.json",
                "2026-06-03-mq6-dflash-r3.json",
            ],
            blockers=["dense_prefill_ratio_below_0.90x_mq4", "origin_or_remote_refresh_required"],
            next_unblocked_step="fix_dense_prefill_throughput_or_scope_dense_out",
            details={
                "prefill_ratios": dense_prefill_ratios,
                "daemon_ar_exact_match_delta_replay": {
                    "origin_fix_short": daemon_ar_exact_match_delta_replay["origin_fix_short"],
                    "evidence_class": daemon_ar_exact_match_delta_replay["evidence_class"],
                    "source_reconciled_in_worktree": daemon_ar_exact_match_delta_replay[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "coherence_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                        "coherence_rows_refreshed_by_this_check"
                    ],
                    "checks": daemon_ar_exact_match_delta_replay["checks"],
                },
            },
        ),
        "moe_prefill": surface_record(
            status="a3b_first_candidate_refresh_blocked",
            runtime_evidence_present=True,
            quality_evidence_present=bool(
                gates["qwen35_a3b_ppl_beats_mq4"] and gates["qwen36_a3b_ppl_beats_mq4"]
            ),
            perf_evidence_present=bool(gates["a3b_ar_perf_present"]),
            promotion_ready=bool(gates["a3b_prefill_speedup_present"] and not refresh_required),
            refresh_required=refresh_required,
            evidence=[
                "cargo test -p hipfire-arch-qwen35 --lib moe_prefill",
                "2026-06-02-mq6-ar-perf.json",
                "2026-06-03-mq6-a3b-ppl.json",
                "2026-06-03-mq6-qwen36-a3b-ppl.json",
            ],
            blockers=["origin_or_remote_refresh_required"],
            next_unblocked_step="rerun_a3b_moe_prefill_after_branch_reconciliation",
            details={
                "prefill_ratios": a3b_prefill_ratios,
                "grouped_scratch_free": {
                    "origin_fix_short": moe_prefill_grouped_scratch_free["origin_fix_short"],
                    "evidence_class": moe_prefill_grouped_scratch_free["evidence_class"],
                    "source_reconciled_in_worktree": moe_prefill_grouped_scratch_free[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": moe_prefill_grouped_scratch_free[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "long_running_moe_stability_refreshed_by_this_check": moe_prefill_grouped_scratch_free[
                        "long_running_moe_stability_refreshed_by_this_check"
                    ],
                    "checks": moe_prefill_grouped_scratch_free["checks"],
                },
            },
        ),
        "moe_decode": surface_record(
            status="a3b_first_candidate_refresh_blocked",
            runtime_evidence_present=bool(
                gates["a3b_ar_perf_present"] and gates["moe_decode_no_gpu_route_test_present"]
            ),
            quality_evidence_present=bool(
                gates["qwen35_a3b_ppl_beats_mq4"] and gates["qwen36_a3b_ppl_beats_mq4"]
            ),
            perf_evidence_present=bool(gates["a3b_ar_perf_present"]),
            promotion_ready=bool(gates["a3b_decode_close_to_mq4"] and not refresh_required),
            refresh_required=refresh_required,
            evidence=[
                MQ6_MOE_DECODE_ROUTE_TEST,
                "2026-06-02-mq6-ar-perf.json",
                "2026-06-03-mq6-a3b-ppl.json",
                "2026-06-03-mq6-qwen36-a3b-ppl.json",
            ],
            blockers=["origin_or_remote_refresh_required"],
            next_unblocked_step="rerun_a3b_moe_decode_after_branch_reconciliation",
            details={
                "decode_ratios": a3b_decode_ratios,
                "no_gpu_route_test": {
                    "command": moe_decode_route_test["command"],
                    "present": moe_decode_route_test["present"],
                    "all_phrase_checks_present": moe_decode_route_test[
                        "all_phrase_checks_present"
                    ],
                },
            },
        ),
        "dflash_spec": surface_record(
            status="target_verify_only_refresh_blocked",
            runtime_evidence_present=bool(
                gates["dense_dflash_perf_present"]
                and gates["dflash_spec_boundary_contract_test_present"]
            ),
            quality_evidence_present=bool(gates["dense_dflash_perf_present"]),
            perf_evidence_present=bool(gates["dense_dflash_perf_present"]),
            promotion_ready=False,
            refresh_required=bool(dflash_boundary["refresh_required"]),
            evidence=[
                MQ6_DFLASH_SPEC_BOUNDARY_CONTRACT_TEST,
                "source: mq6_batched_rope_compact_offset",
                batched_rope_compact_offset["validation_command"],
                "2026-06-03-mq6-dflash-r3.json",
            ],
            blockers=[
                "target_side_verify_only_mq4_draft",
                "no_mq6_draft_evidence",
                "no_a3b_dflash_evidence",
                "origin_or_remote_refresh_required",
            ],
            next_unblocked_step=str(dflash_boundary["next_unblocked_step"]),
            details={
                "evidence_class": dflash_boundary["evidence_class"],
                "draft_formats": dflash_boundary["draft_formats"],
                "min_runs_per_case": dflash_boundary["min_runs_per_case"],
                "scope_expanded_beyond_target_verify": dflash_boundary[
                    "scope_expanded_beyond_target_verify"
                ],
                "boundary_contract_test": {
                    "command": dflash_boundary_contract_test["command"],
                    "present": dflash_boundary_contract_test["present"],
                    "all_phrase_checks_present": dflash_boundary_contract_test[
                        "all_phrase_checks_present"
                    ],
                },
                "batched_rope_compact_offset": {
                    "origin_fix_short": batched_rope_compact_offset["origin_fix_short"],
                    "evidence_class": batched_rope_compact_offset["evidence_class"],
                    "source_reconciled_in_worktree": batched_rope_compact_offset[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": batched_rope_compact_offset[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "long_context_or_eviction_rows_refreshed_by_this_check": batched_rope_compact_offset[
                        "long_context_or_eviction_rows_refreshed_by_this_check"
                    ],
                    "validation_command": batched_rope_compact_offset["validation_command"],
                    "checks": batched_rope_compact_offset["checks"],
                },
                "daemon_dflash_first_token_eos": {
                    "origin_fix_short": daemon_dflash_first_token_eos["origin_fix_short"],
                    "evidence_class": daemon_dflash_first_token_eos["evidence_class"],
                    "source_reconciled_in_worktree": daemon_dflash_first_token_eos[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": daemon_dflash_first_token_eos[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "coherence_rows_refreshed_by_this_check": daemon_dflash_first_token_eos[
                        "coherence_rows_refreshed_by_this_check"
                    ],
                    "checks": daemon_dflash_first_token_eos["checks"],
                },
                "ddtree_eviction_target_hidden_host": {
                    "origin_fix_short": ddtree_eviction_target_hidden_host["origin_fix_short"],
                    "remote_ref": ddtree_eviction_target_hidden_host["remote_ref"],
                    "evidence_class": ddtree_eviction_target_hidden_host["evidence_class"],
                    "source_reconciled_in_worktree": ddtree_eviction_target_hidden_host[
                        "source_reconciled_in_worktree"
                    ],
                    "validation_command": ddtree_eviction_target_hidden_host[
                        "validation_command"
                    ],
                    "perf_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "coherence_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                        "coherence_rows_refreshed_by_this_check"
                    ],
                    "long_context_or_eviction_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                        "long_context_or_eviction_rows_refreshed_by_this_check"
                    ],
                    "checks": ddtree_eviction_target_hidden_host["checks"],
                },
                "cask_scratch_teardown": {
                    "origin_fix_short": cask_scratch_teardown["origin_fix_short"],
                    "evidence_class": cask_scratch_teardown["evidence_class"],
                    "source_reconciled_in_worktree": cask_scratch_teardown[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": cask_scratch_teardown[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "long_context_or_eviction_rows_refreshed_by_this_check": cask_scratch_teardown[
                        "long_context_or_eviction_rows_refreshed_by_this_check"
                    ],
                    "checks": cask_scratch_teardown["checks"],
                },
                "mtp_gpu_teardown": {
                    "origin_fix_short": mtp_gpu_teardown["origin_fix_short"],
                    "evidence_class": mtp_gpu_teardown["evidence_class"],
                    "source_reconciled_in_worktree": mtp_gpu_teardown[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": mtp_gpu_teardown[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "long_running_mtp_rows_refreshed_by_this_check": mtp_gpu_teardown[
                        "long_running_mtp_rows_refreshed_by_this_check"
                    ],
                    "checks": mtp_gpu_teardown["checks"],
                },
                "asym4_fwht_fallback": {
                    "origin_fix_short": asym4_fwht_fallback["origin_fix_short"],
                    "evidence_class": asym4_fwht_fallback["evidence_class"],
                    "source_reconciled_in_worktree": asym4_fwht_fallback[
                        "source_reconciled_in_worktree"
                    ],
                    "perf_rows_refreshed_by_this_check": asym4_fwht_fallback[
                        "perf_rows_refreshed_by_this_check"
                    ],
                    "coherence_rows_refreshed_by_this_check": asym4_fwht_fallback[
                        "coherence_rows_refreshed_by_this_check"
                    ],
                    "checks": asym4_fwht_fallback["checks"],
                },
            },
        ),
    }
    return {
        "schema": "hipfire.mq6_surface_coverage.gfx1151.v0",
        "surfaces": surfaces,
        "all_required_surfaces_recorded": set(surfaces) == {
            "dense_decode",
            "dense_prefill",
            "moe_prefill",
            "moe_decode",
            "dflash_spec",
        },
        "promotion_ready_surfaces": sorted(
            name for name, surface in surfaces.items() if surface["promotion_ready"]
        ),
        "blocked_surfaces": sorted(
            name for name, surface in surfaces.items() if not surface["promotion_ready"]
        ),
        "all_surfaces_promotion_ready": all(surface["promotion_ready"] for surface in surfaces.values()),
        "refresh_required": refresh_required,
    }


def evidence_refresh_plan(
    origin: dict[str, Any],
    surface_coverage: dict[str, Any],
    locally_reconciled_refresh_reasons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reason_to_surfaces = {
        "dense_ar_perf": {"dense_decode", "dense_prefill"},
        "dense_dflash_perf": {"dflash_spec"},
        "dflash_stability": {"dflash_spec"},
        "ddtree_eviction_stability": {"dflash_spec"},
        "long_context_or_eviction_rows": {"dflash_spec"},
        "spec_verify": {"dflash_spec"},
        "coherence": {"dense_decode", "dense_prefill", "moe_prefill", "moe_decode", "dflash_spec"},
        "a3b_ar_perf": {"moe_prefill", "moe_decode"},
        "a3b_ppl_smoke": {"moe_prefill", "moe_decode"},
        "qwen36_a3b_ppl": {"moe_prefill", "moe_decode"},
        "qwen36_a3b_ar_perf": {"moe_prefill", "moe_decode"},
        "qwen36_a3b_runtime_source_reconciliation": {"moe_prefill", "moe_decode"},
        "hunt3_nan_safe_sampling": {"dense_decode", "moe_decode", "dflash_spec"},
        "long_running_moe_stability": {"moe_prefill", "moe_decode"},
        "moe_prefill_scratch_teardown": {"moe_prefill"},
        "producer_context": {"dense_decode", "dense_prefill", "moe_prefill", "moe_decode"},
        "cask_scratch_teardown": {"dflash_spec"},
        "mtp_gpu_teardown": {"dflash_spec"},
        "asym4_fwht_fallback": {"dense_decode", "moe_decode", "dflash_spec"},
        "daemon_ar_exact_match_delta_replay": {"dense_decode", "dense_prefill"},
    }
    reason_to_artifacts = {
        "dense_ar_perf": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
        },
        "dense_dflash_perf": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "dflash_stability": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "ddtree_eviction_stability": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "long_context_or_eviction_rows": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "spec_verify": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "coherence": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-coherence-md5.md",
        },
        "a3b_ar_perf": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
        },
        "a3b_ppl_smoke": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-a3b-ppl.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-a3b-ppl.md",
        },
        "qwen36_a3b_ppl": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-qwen36-a3b-ppl.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-qwen36-a3b-ppl.md",
        },
        "qwen36_a3b_ar_perf": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
        },
        "qwen36_a3b_runtime_source_reconciliation": set(),
        "hunt3_nan_safe_sampling": {
            "cargo test -p hipfire-runtime --lib argmax",
            "cargo test -p hipfire-arch-qwen35 --lib sample_from_logits",
            "cargo test -p hipfire-arch-deepseek4 --lib sampling",
            "cargo check --workspace --all-targets",
        },
        "long_running_moe_stability": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
        },
        "moe_prefill_scratch_teardown": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
        },
        "cask_scratch_teardown": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "mtp_gpu_teardown": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
        "asym4_fwht_fallback": {
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.md",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.md",
        },
    }
    reason_to_commands = {
        "dense_ar_perf": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
        },
        "dense_dflash_perf": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "dflash_stability": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "ddtree_eviction_stability": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "long_context_or_eviction_rows": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "spec_verify": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "coherence": {"./scripts/coherence-gate.sh --full"},
        "a3b_ar_perf": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
        },
        "a3b_ppl_smoke": {"python3 scripts/mq6_ppl_compare.py --pretty"},
        "qwen36_a3b_ppl": {"python3 scripts/mq6_ppl_compare.py --pretty"},
        "qwen36_a3b_ar_perf": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
        },
        "qwen36_a3b_runtime_source_reconciliation": {
            "cargo test -p hipfire-runtime --lib sampler",
            "cargo check --workspace --all-targets",
        },
        "long_running_moe_stability": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
        },
        "moe_prefill_scratch_teardown": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
        },
        "cask_scratch_teardown": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "mtp_gpu_teardown": {
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
        "asym4_fwht_fallback": {
            "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
        },
    }
    stale_origin = list(origin.get("origin_master_missing_commit_shorts", []))
    stale_remote = list(origin.get("remote_branch_missing_commit_shorts", []))
    local_reconciliations = locally_reconciled_refresh_reasons or {}
    locally_reconciled = set(local_reconciliations)

    def reconciles_reason_for_commit(reason: str, short: str) -> bool:
        reconciliation = local_reconciliations.get(reason)
        if not isinstance(reconciliation, dict):
            return False
        scoped_short = (
            reconciliation.get("commit_short")
            or reconciliation.get("origin_fix_short")
            or reconciliation.get("short")
        )
        if scoped_short is None:
            return True
        return str(scoped_short) == short

    stale_commits = []
    refresh_reasons: set[str] = set()
    for source, commits in (
        ("origin/master", origin.get("relevant_upstream_commits", [])),
        ("origin/fix/hunt3-engine-bugs", origin.get("relevant_remote_branch_commits", [])),
    ):
        for commit in commits:
            short = str(commit.get("short"))
            if short not in stale_origin and short not in stale_remote:
                continue
            reasons = sorted(str(reason) for reason in commit.get("refreshes", []))
            reconciled_reasons = sorted(
                reason for reason in reasons if reconciles_reason_for_commit(reason, short)
            )
            unreconciled_reasons = sorted(
                reason for reason in reasons if not reconciles_reason_for_commit(reason, short)
            )
            refresh_reasons.update(unreconciled_reasons)
            stale_commits.append(
                {
                    "short": short,
                    "source": commit.get("remote_ref", source),
                    "subject": commit.get("subject"),
                    "refreshes": reasons,
                    "locally_reconciled_refreshes": reconciled_reasons,
                    "unreconciled_refreshes": unreconciled_reasons,
                    "surfaces": list(commit.get("surfaces", [])),
                }
            )

    affected_surfaces: set[str] = set()
    artifacts: set[str] = set()
    commands: set[str] = set()
    for reason in refresh_reasons:
        affected_surfaces.update(reason_to_surfaces.get(reason, set()))
        artifacts.update(reason_to_artifacts.get(reason, set()))
        commands.update(reason_to_commands.get(reason, set()))
    valid_surfaces = set(surface_coverage.get("surfaces", {}))
    affected_surfaces = {surface for surface in affected_surfaces if surface in valid_surfaces}
    surface_next_steps = {
        name: surface_coverage["surfaces"][name]["next_unblocked_step"]
        for name in sorted(affected_surfaces)
    }
    return {
        "refresh_required": bool(origin.get("mq6_evidence_refresh_required")),
        "stale_origin_commit_shorts": stale_origin,
        "stale_remote_commit_shorts": stale_remote,
        "stale_commits": stale_commits,
        "locally_reconciled_refresh_reasons": sorted(locally_reconciled),
        "local_reconciliations": local_reconciliations,
        "refresh_reasons": sorted(refresh_reasons),
        "affected_surfaces": sorted(affected_surfaces),
        "surface_next_steps": surface_next_steps,
        "rerun_commands": sorted(commands),
        "evidence_artifacts_to_refresh": sorted(artifacts),
        "next_unblocked_step": (
            "reconcile_qwen35_native_mtp_with_origin_and_hunt3_then_rerun_mq6_evidence"
            if origin.get("mq6_evidence_refresh_required")
            else "rerun_surface_specific_mq6_evidence_after_next_runtime_change"
        ),
    }


def text_evidence(path: Path, phrases: tuple[str, ...]) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "present": bool(text),
        "phrase_checks": {phrase: phrase in text for phrase in phrases},
    }


def dense_decode_route_test_evidence() -> dict[str, Any]:
    evidence = text_evidence(
        ROOT / "crates" / "hipfire-runtime" / "src" / "llama.rs",
        (
            "mq6_dense_decode_routes_through_hfq6_family",
            "DenseGemvRoute::Mq6RotateThenMq6Prerotated",
            "DensePrerotatedGemvRoute::Mq6Prerotated",
            "DenseResidualRoute::Mq6RotateThenHfq6Residual",
            "DenseResidualRoute::Hfq6ResidualDirect",
            "DenseSwigluResidualRoute::Mq6RotateThenHfq6Residual",
        ),
    )
    return {
        **evidence,
        "command": MQ6_DENSE_DECODE_ROUTE_TEST,
        "all_phrase_checks_present": all(evidence["phrase_checks"].values()),
    }


def moe_decode_route_test_evidence() -> dict[str, Any]:
    evidence = text_evidence(
        ROOT / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs",
        (
            "moe_decode_routes_mq6_indexed_path_for_k8",
            "MoeDecodeIndexedRoutedPath::Mq6",
            "moe_decode_keeps_mq4_control_on_indexed_path",
            "moe_decode_rejects_mismatched_mq6_gate_up_and_down_from_indexed_path",
            "moe_decode_k8_shape_required_for_gpu_topk",
        ),
    )
    return {
        **evidence,
        "command": MQ6_MOE_DECODE_ROUTE_TEST,
        "all_phrase_checks_present": all(evidence["phrase_checks"].values()),
    }


def dflash_boundary_contract_test_evidence() -> dict[str, Any]:
    evidence = text_evidence(
        ROOT / "scripts" / "test_mq6_status.py",
        (
            "test_dflash_spec_boundary_marks_mq4_draft_target_verify_only",
            "test_dflash_spec_boundary_detects_mq6_draft_scope_expansion",
            "test_dflash_spec_boundary_detects_a3b_scope_expansion",
            "target_side_verify_only_mq4_draft",
        ),
    )
    return {
        **evidence,
        "command": MQ6_DFLASH_SPEC_BOUNDARY_CONTRACT_TEST,
        "all_phrase_checks_present": all(evidence["phrase_checks"].values()),
    }


def ratio_at_least(pair: dict[str, Any], key: str, threshold: float) -> bool:
    value = pair.get(key)
    return bool(value is not None and float(value) >= threshold)


def all_pairs_ok(pairs: dict[str, dict[str, Any]]) -> bool:
    return bool(pairs) and all(pair.get("all_runs_ok") for pair in pairs.values())


def build_status(
    paths: dict[str, Path] = DEFAULT_PATHS,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> dict[str, Any]:
    artifacts = artifact_summary(paths["provenance"])
    live_artifacts = live_candidate_inventory(models_root)
    producer_path = mq6_producer_path_evidence()
    batched_rope_compact_offset = mq6_batched_rope_compact_offset_evidence()
    daemon_dflash_first_token_eos = mq6_daemon_dflash_first_token_eos_evidence()
    daemon_ar_exact_match_delta_replay = mq6_daemon_ar_exact_match_delta_replay_evidence()
    ddtree_eviction_target_hidden_host = mq6_ddtree_eviction_target_hidden_host_evidence()
    moe_prefill_grouped_scratch_free = mq6_moe_prefill_grouped_scratch_free_evidence()
    cask_scratch_teardown = mq6_cask_scratch_teardown_evidence()
    mtp_gpu_teardown = mq6_mtp_gpu_teardown_evidence()
    asym4_fwht_fallback = mq6_asym4_fwht_fallback_evidence()
    qwen36_a3b_runtime_source_reconciliation = (
        mq6_qwen36_a3b_runtime_source_reconciliation_evidence()
    )
    hunt3_nan_safe_sampling = mq6_hunt3_nan_safe_sampling_evidence()
    kernels = kernel_inventory()
    dense_decode_route_test = dense_decode_route_test_evidence()
    moe_decode_route_test = moe_decode_route_test_evidence()
    dflash_boundary_contract_test = dflash_boundary_contract_test_evidence()
    dense_ppl = ppl_comparison(paths["ppl_dense"], "qwen35-9b-mq6", "qwen35-9b-mq4")
    qwen35_a3b_ppl = ppl_comparison(paths["ppl_qwen35_a3b"], "qwen35-a3b-mq6", "qwen35-a3b-mq4")
    qwen36_a3b_ppl = ppl_comparison(paths["ppl_qwen36_a3b"], "qwen36-a3b-mq6", "qwen36-a3b-mq4")
    kld_c512 = kld_comparison(paths["kld_c512"], "qwen3.5-9b.mq6", "qwen3.5-9b.mq4")

    ar_pairs = {
        "dense_9b": perf_pair(paths["ar_perf"], "qwen35-9b-mq6-sheep", "qwen35-9b-mq4-sheep"),
        "dense_27b": perf_pair(paths["ar_perf"], "qwen35-27b-mq6-cap", "qwen35-27b-mq4-cap"),
        "qwen35_a3b": perf_pair(paths["ar_perf"], "qwen35-a3b-mq6-sheep", "qwen35-a3b-mq4-sheep"),
        "qwen36_a3b": perf_pair(paths["ar_perf"], "qwen36-a3b-mq6-sheep", "qwen36-a3b-mq4-sheep"),
    }
    dense_ar_pairs = {key: ar_pairs[key] for key in ("dense_9b", "dense_27b")}
    a3b_ar_pairs = {key: ar_pairs[key] for key in ("qwen35_a3b", "qwen36_a3b")}

    dflash_pairs = {
        "prose": dflash_pair(paths["dflash"], "qwen35-27b-mq6-dflash-prose", "qwen35-27b-mq4-dflash-prose"),
        "code": dflash_pair(paths["dflash"], "qwen35-27b-mq6-dflash-code", "qwen35-27b-mq4-dflash-code"),
    }

    upstream = origin_context()
    dflash_boundary = dflash_spec_boundary(paths["dflash"], upstream)

    readiness_audit = text_evidence(
        paths["readiness_audit"],
        ("A3B-first candidate", "Dense MQ6", "dense perf-blocked"),
    )
    i8_decision = text_evidence(
        paths["i8_grouped_decision"],
        ("do not port an MQ6-specific i8 grouped-MMQ path now", "dense MQ6 throughput", "grouped HFQ6 WMMA"),
    )

    dense_ar_promotable = all(
        ratio_at_least(pair, "prefill_ratio", DENSE_PROMOTION_RATIO)
        and ratio_at_least(pair, "decode_ratio", DENSE_PROMOTION_RATIO)
        for pair in dense_ar_pairs.values()
    )
    dense_dflash_promotable = all(
        pair.get("all_runs_ok")
        and pair.get("token_attractor_ok")
        and ratio_at_least(pair, "prefill_ratio", DENSE_PROMOTION_RATIO)
        and ratio_at_least(pair, "decode_ratio", DENSE_PROMOTION_RATIO)
        for pair in dflash_pairs.values()
    )
    a3b_prefill_speedup = all(
        ratio_at_least(pair, "prefill_ratio", A3B_PREFILL_SPEEDUP_RATIO) for pair in a3b_ar_pairs.values()
    )
    a3b_decode_close = all(
        ratio_at_least(pair, "decode_ratio", A3B_DECODE_CLOSE_RATIO) for pair in a3b_ar_pairs.values()
    )

    refresh_required = bool(upstream["mq6_evidence_refresh_required"])

    gates = {
        "provenance_canonical_mq6_artifacts_present": artifacts["canonical_mq6_artifacts_present"],
        "live_canonical_mq6_artifacts_present": live_artifacts["canonical_mq6_artifacts_present"],
        "canonical_mq6_artifacts_present": bool(
            artifacts["canonical_mq6_artifacts_present"]
            and live_artifacts["canonical_mq6_artifacts_present"]
        ),
        "mq6_producer_mq4_lloyd_promote6_fixed": producer_path[
            "promote6_mq4_lloyd_emits_mq6"
        ],
        "mq6_batched_rope_compact_offset_fixed": batched_rope_compact_offset[
            "source_reconciled_in_worktree"
        ],
        "mq6_daemon_dflash_first_token_eos_fixed": daemon_dflash_first_token_eos[
            "source_reconciled_in_worktree"
        ],
        "mq6_daemon_ar_exact_match_delta_replay_fixed": daemon_ar_exact_match_delta_replay[
            "source_reconciled_in_worktree"
        ],
        "mq6_ddtree_eviction_target_hidden_host_fixed": ddtree_eviction_target_hidden_host[
            "source_reconciled_in_worktree"
        ],
        "mq6_moe_prefill_grouped_scratch_free_fixed": moe_prefill_grouped_scratch_free[
            "source_reconciled_in_worktree"
        ],
        "mq6_cask_scratch_teardown_fixed": cask_scratch_teardown[
            "source_reconciled_in_worktree"
        ],
        "mq6_mtp_gpu_teardown_fixed": mtp_gpu_teardown["source_reconciled_in_worktree"],
        "mq6_asym4_fwht_fallback_fixed": asym4_fwht_fallback[
            "source_reconciled_in_worktree"
        ],
        "mq6_qwen36_a3b_runtime_source_reconciled": qwen36_a3b_runtime_source_reconciliation[
            "source_reconciled_in_worktree"
        ],
        "mq6_hunt3_nan_safe_sampling_fixed": hunt3_nan_safe_sampling[
            "source_reconciled_in_worktree"
        ],
        "dense_decode_repo_kernel_inventory_present": kernels["repo_required_kernels_present"],
        "dense_decode_install_kernel_inventory_present": kernels["install_required_kernels_present"],
        "dense_decode_no_gpu_route_test_present": dense_decode_route_test["all_phrase_checks_present"],
        "moe_decode_no_gpu_route_test_present": moe_decode_route_test["all_phrase_checks_present"],
        "dense_9b_ppl_beats_mq4": dense_ppl["candidate_beats_control"],
        "qwen35_a3b_ppl_beats_mq4": qwen35_a3b_ppl["candidate_beats_control"],
        "qwen36_a3b_ppl_beats_mq4": qwen36_a3b_ppl["candidate_beats_control"],
        "dense_9b_c512_kld_beats_mq4": kld_c512["candidate_beats_control"],
        "dense_ar_perf_present": all_pairs_ok(dense_ar_pairs),
        "a3b_ar_perf_present": all_pairs_ok(a3b_ar_pairs),
        "dense_dflash_perf_present": all_pairs_ok(dflash_pairs)
        and all(pair.get("token_attractor_ok") for pair in dflash_pairs.values()),
        "dflash_target_side_verify_only": dflash_boundary["target_side_verify_only"],
        "dflash_mq6_draft_evidence_present": dflash_boundary["mq6_draft_evidence_present"],
        "dflash_a3b_evidence_present": dflash_boundary["a3b_dflash_evidence_present"],
        "dflash_spec_boundary_refresh_required": dflash_boundary["refresh_required"],
        "dflash_spec_boundary_contract_test_present": dflash_boundary_contract_test[
            "all_phrase_checks_present"
        ],
        "dense_ar_perf_promotable": dense_ar_promotable,
        "dense_dflash_perf_promotable": dense_dflash_promotable,
        "a3b_prefill_speedup_present": a3b_prefill_speedup,
        "a3b_decode_close_to_mq4": a3b_decode_close,
        "origin_refresh_required": refresh_required,
        "remote_branch_refresh_required": bool(upstream["remote_branch_refresh_required"]),
        "a3b_first_scope_supported": bool(
            qwen35_a3b_ppl["candidate_beats_control"]
            and qwen36_a3b_ppl["candidate_beats_control"]
            and all_pairs_ok(a3b_ar_pairs)
            and a3b_prefill_speedup
            and a3b_decode_close
            and not refresh_required
        ),
        "i8_grouped_port_recommended": False,
    }
    gates["dense_promotion_allowed"] = bool(
        gates["canonical_mq6_artifacts_present"]
        and gates["dense_9b_ppl_beats_mq4"]
        and gates["dense_9b_c512_kld_beats_mq4"]
        and gates["dense_ar_perf_present"]
        and gates["dense_dflash_perf_present"]
        and gates["dense_ar_perf_promotable"]
        and gates["dense_dflash_perf_promotable"]
        and not refresh_required
    )
    gates["broad_promotion_allowed"] = bool(gates["dense_promotion_allowed"] and gates["a3b_first_scope_supported"])

    surface_coverage = mq6_surface_coverage(
        gates=gates,
        kernels=kernels,
        batched_rope_compact_offset=batched_rope_compact_offset,
        daemon_dflash_first_token_eos=daemon_dflash_first_token_eos,
        daemon_ar_exact_match_delta_replay=daemon_ar_exact_match_delta_replay,
        ddtree_eviction_target_hidden_host=ddtree_eviction_target_hidden_host,
        moe_prefill_grouped_scratch_free=moe_prefill_grouped_scratch_free,
        cask_scratch_teardown=cask_scratch_teardown,
        mtp_gpu_teardown=mtp_gpu_teardown,
        asym4_fwht_fallback=asym4_fwht_fallback,
        dense_decode_route_test=dense_decode_route_test,
        moe_decode_route_test=moe_decode_route_test,
        dflash_boundary_contract_test=dflash_boundary_contract_test,
        dense_ar_pairs=dense_ar_pairs,
        a3b_ar_pairs=a3b_ar_pairs,
        dflash_boundary=dflash_boundary,
        refresh_required=refresh_required,
    )
    gates["surface_coverage_complete"] = bool(surface_coverage["all_required_surfaces_recorded"])
    gates["all_surfaces_promotion_ready"] = bool(surface_coverage["all_surfaces_promotion_ready"])
    locally_reconciled_refresh_reasons: dict[str, Any] = {}
    if producer_path["origin_fix_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["producer_context"] = {
            "source": "producer_path",
            "commit_short": producer_path["origin_fix_short"],
            "origin_fix_short": producer_path["origin_fix_short"],
            "origin_fix_commit": producer_path["origin_fix_commit"],
            "evidence_class": producer_path["evidence_class"],
            "path": producer_path["path"],
        }
    if batched_rope_compact_offset["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["spec_verify"] = {
            "source": "batched_rope_compact_offset",
            "commit_short": batched_rope_compact_offset["origin_fix_short"],
            "origin_fix_short": batched_rope_compact_offset["origin_fix_short"],
            "origin_fix_commit": batched_rope_compact_offset["origin_fix_commit"],
            "evidence_class": batched_rope_compact_offset["evidence_class"],
            "validation_command": batched_rope_compact_offset["validation_command"],
            "perf_rows_refreshed_by_this_check": batched_rope_compact_offset[
                "perf_rows_refreshed_by_this_check"
            ],
            "long_context_or_eviction_rows_refreshed_by_this_check": batched_rope_compact_offset[
                "long_context_or_eviction_rows_refreshed_by_this_check"
            ],
        }
    if daemon_ar_exact_match_delta_replay["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["daemon_ar_exact_match_delta_replay"] = {
            "source": "daemon_ar_exact_match_delta_replay",
            "commit_short": daemon_ar_exact_match_delta_replay["origin_fix_short"],
            "origin_fix_short": daemon_ar_exact_match_delta_replay["origin_fix_short"],
            "origin_fix_commit": daemon_ar_exact_match_delta_replay["origin_fix_commit"],
            "evidence_class": daemon_ar_exact_match_delta_replay["evidence_class"],
            "path": daemon_ar_exact_match_delta_replay["path"],
            "perf_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                "perf_rows_refreshed_by_this_check"
            ],
            "coherence_rows_refreshed_by_this_check": daemon_ar_exact_match_delta_replay[
                "coherence_rows_refreshed_by_this_check"
            ],
        }
    if ddtree_eviction_target_hidden_host["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["ddtree_eviction_stability"] = {
            "source": "ddtree_eviction_target_hidden_host",
            "commit_short": ddtree_eviction_target_hidden_host["origin_fix_short"],
            "origin_fix_short": ddtree_eviction_target_hidden_host["origin_fix_short"],
            "origin_fix_commit": ddtree_eviction_target_hidden_host["origin_fix_commit"],
            "remote_ref": ddtree_eviction_target_hidden_host["remote_ref"],
            "evidence_class": ddtree_eviction_target_hidden_host["evidence_class"],
            "validation_command": ddtree_eviction_target_hidden_host["validation_command"],
            "perf_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                "perf_rows_refreshed_by_this_check"
            ],
            "coherence_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                "coherence_rows_refreshed_by_this_check"
            ],
            "long_context_or_eviction_rows_refreshed_by_this_check": ddtree_eviction_target_hidden_host[
                "long_context_or_eviction_rows_refreshed_by_this_check"
            ],
        }
    if moe_prefill_grouped_scratch_free["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["moe_prefill_scratch_teardown"] = {
            "source": "moe_prefill_grouped_scratch_free",
            "commit_short": moe_prefill_grouped_scratch_free["origin_fix_short"],
            "origin_fix_short": moe_prefill_grouped_scratch_free["origin_fix_short"],
            "origin_fix_commit": moe_prefill_grouped_scratch_free["origin_fix_commit"],
            "evidence_class": moe_prefill_grouped_scratch_free["evidence_class"],
            "path": moe_prefill_grouped_scratch_free["path"],
            "perf_rows_refreshed_by_this_check": moe_prefill_grouped_scratch_free[
                "perf_rows_refreshed_by_this_check"
            ],
            "long_running_moe_stability_refreshed_by_this_check": moe_prefill_grouped_scratch_free[
                "long_running_moe_stability_refreshed_by_this_check"
            ],
        }
    if cask_scratch_teardown["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["cask_scratch_teardown"] = {
            "source": "cask_scratch_teardown",
            "commit_short": cask_scratch_teardown["origin_fix_short"],
            "origin_fix_short": cask_scratch_teardown["origin_fix_short"],
            "origin_fix_commit": cask_scratch_teardown["origin_fix_commit"],
            "evidence_class": cask_scratch_teardown["evidence_class"],
            "path": cask_scratch_teardown["path"],
            "perf_rows_refreshed_by_this_check": cask_scratch_teardown[
                "perf_rows_refreshed_by_this_check"
            ],
            "long_context_or_eviction_rows_refreshed_by_this_check": cask_scratch_teardown[
                "long_context_or_eviction_rows_refreshed_by_this_check"
            ],
        }
    if mtp_gpu_teardown["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["mtp_gpu_teardown"] = {
            "source": "mtp_gpu_teardown",
            "commit_short": mtp_gpu_teardown["origin_fix_short"],
            "origin_fix_short": mtp_gpu_teardown["origin_fix_short"],
            "origin_fix_commit": mtp_gpu_teardown["origin_fix_commit"],
            "evidence_class": mtp_gpu_teardown["evidence_class"],
            "paths": mtp_gpu_teardown["paths"],
            "perf_rows_refreshed_by_this_check": mtp_gpu_teardown[
                "perf_rows_refreshed_by_this_check"
            ],
            "long_running_mtp_rows_refreshed_by_this_check": mtp_gpu_teardown[
                "long_running_mtp_rows_refreshed_by_this_check"
            ],
        }
    if asym4_fwht_fallback["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["asym4_fwht_fallback"] = {
            "source": "asym4_fwht_fallback",
            "commit_short": asym4_fwht_fallback["origin_fix_short"],
            "origin_fix_short": asym4_fwht_fallback["origin_fix_short"],
            "origin_fix_commit": asym4_fwht_fallback["origin_fix_commit"],
            "evidence_class": asym4_fwht_fallback["evidence_class"],
            "path": asym4_fwht_fallback["path"],
            "perf_rows_refreshed_by_this_check": asym4_fwht_fallback[
                "perf_rows_refreshed_by_this_check"
            ],
            "coherence_rows_refreshed_by_this_check": asym4_fwht_fallback[
                "coherence_rows_refreshed_by_this_check"
            ],
        }
    if qwen36_a3b_runtime_source_reconciliation["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["qwen36_a3b_runtime_source_reconciliation"] = {
            "source": "qwen36_a3b_runtime_source_reconciliation",
            "commit_short": qwen36_a3b_runtime_source_reconciliation["origin_fix_short"],
            "origin_fix_short": qwen36_a3b_runtime_source_reconciliation["origin_fix_short"],
            "origin_fix_commit": qwen36_a3b_runtime_source_reconciliation["origin_fix_commit"],
            "evidence_class": qwen36_a3b_runtime_source_reconciliation["evidence_class"],
            "paths": qwen36_a3b_runtime_source_reconciliation["paths"],
            "validation_commands": qwen36_a3b_runtime_source_reconciliation[
                "validation_commands"
            ],
            "ppl_rows_refreshed_by_this_check": qwen36_a3b_runtime_source_reconciliation[
                "ppl_rows_refreshed_by_this_check"
            ],
            "perf_rows_refreshed_by_this_check": qwen36_a3b_runtime_source_reconciliation[
                "perf_rows_refreshed_by_this_check"
            ],
            "coherence_rows_refreshed_by_this_check": qwen36_a3b_runtime_source_reconciliation[
                "coherence_rows_refreshed_by_this_check"
            ],
        }
    if hunt3_nan_safe_sampling["source_reconciled_in_worktree"]:
        locally_reconciled_refresh_reasons["hunt3_nan_safe_sampling"] = {
            "source": "hunt3_nan_safe_sampling",
            "commit_short": hunt3_nan_safe_sampling["origin_fix_short"],
            "origin_fix_short": hunt3_nan_safe_sampling["origin_fix_short"],
            "origin_fix_commit": hunt3_nan_safe_sampling["origin_fix_commit"],
            "remote_ref": hunt3_nan_safe_sampling["remote_ref"],
            "evidence_class": hunt3_nan_safe_sampling["evidence_class"],
            "paths": hunt3_nan_safe_sampling["paths"],
            "validation_commands": hunt3_nan_safe_sampling["validation_commands"],
            "perf_rows_refreshed_by_this_check": hunt3_nan_safe_sampling[
                "perf_rows_refreshed_by_this_check"
            ],
            "coherence_rows_refreshed_by_this_check": hunt3_nan_safe_sampling[
                "coherence_rows_refreshed_by_this_check"
            ],
        }
    refresh_plan = evidence_refresh_plan(
        upstream,
        surface_coverage,
        locally_reconciled_refresh_reasons=locally_reconciled_refresh_reasons,
    )

    blockers = []
    if not gates["live_canonical_mq6_artifacts_present"]:
        blockers.append(
            "live /home/sadara/Models candidate-root scan is missing canonical MQ6 artifacts; "
            "refresh symlinks or provenance before promotion evidence is trusted"
        )
    if not gates["mq6_producer_mq4_lloyd_promote6_fixed"]:
        blockers.append(
            "GGUF --kmap-promote 6 for Mq4Lloyd still emits MQ4G256Lloyd in the current worktree; "
            "reconcile origin d5985c3e or repair the producer path before regenerating Promote6 artifacts"
        )
    if not gates["mq6_batched_rope_compact_offset_fixed"]:
        blockers.append(
            "batched RoPE still lacks compact_offset phase correction in the current worktree; "
            "reconcile origin 676338a4 before trusting DFlash/spec verify evidence after KV compaction"
        )
    if not gates["mq6_daemon_dflash_first_token_eos_fixed"]:
        blockers.append(
            "daemon DFlash still enters the spec loop after an already-terminal first token; "
            "reconcile origin d5bc3f6 before trusting daemon DFlash stability"
        )
    if not gates["mq6_daemon_ar_exact_match_delta_replay_fixed"]:
        blockers.append(
            "daemon AR still appears vulnerable to exact-match DeltaNet double-advance; "
            "reconcile origin d5bc3f6 before trusting daemon AR multi-turn stability"
        )
    if not gates["mq6_ddtree_eviction_target_hidden_host_fixed"]:
        blockers.append(
            "DDTree/CASK eviction still leaves target_hidden_host uncompact in the current worktree; "
            "reconcile origin/fix/issue-triage-2026-06-03 d12b3ad4 before trusting DDTree eviction stability"
        )
    if not gates["mq6_moe_prefill_grouped_scratch_free_fixed"]:
        blockers.append(
            "Qwen35 Path-2 grouped-MoE prefill scratch teardown is missing grouped scratch buffers; "
            "reconcile origin b4adca1f before trusting long-running A3B serve stability"
        )
    if not gates["mq6_cask_scratch_teardown_fixed"]:
        blockers.append(
            "CASK eviction still leaks per-call indices/weights scratch on an early error path; "
            "reconcile origin a7dcfb0d before trusting eviction stability"
        )
    if not gates["mq6_mtp_gpu_teardown_fixed"]:
        blockers.append(
            "MTP teardown still drops GPU-owned KV/snapshot handles without freeing device buffers; "
            "reconcile origin a7dcfb0d before trusting long-running MTP/spec stability"
        )
    if not gates["mq6_asym4_fwht_fallback_fixed"]:
        blockers.append(
            "per-token FA fallback still routes FWHT asym4 KV through the plain asym4 kernels; "
            "reconcile origin a7dcfb0d before trusting asym4/FWHT decode evidence"
        )
    if not gates["mq6_qwen36_a3b_runtime_source_reconciled"]:
        blockers.append(
            "Qwen3.6-A3B runtime source reconciliation for native OpenAI penalties and DeltaNet Q8 dither "
            "is missing; reconcile origin 379484ba before trusting refreshed Qwen3.6-A3B serve evidence"
        )
    if not gates["mq6_hunt3_nan_safe_sampling_fixed"]:
        blockers.append(
            "hunt-3 NaN-safe CPU/MTP sampling source reconciliation is missing; aggressive quant candidates "
            "can still panic on non-finite logits before coherence or quality gates can reject them"
        )
    if not gates["dense_decode_repo_kernel_inventory_present"]:
        blockers.append(
            "repo gfx1151 compiled-kernel inventory is missing at least one required MQ6 dense-decode blob; "
            "rebuild or restore kernels before rerunning dense perf evidence"
        )
    if not gates["dense_decode_no_gpu_route_test_present"]:
        blockers.append(
            "hipfire-runtime no-GPU MQ6 dense-decode route test is missing or incomplete; "
            "restore mq6_dense_decode_routes_through_hfq6_family before trusting route coverage"
        )
    if not gates["dflash_spec_boundary_contract_test_present"]:
        blockers.append(
            "MQ6 DFlash/spec no-GPU boundary contract test is missing or incomplete; "
            "restore target-verify-only classification tests before trusting DFlash scope"
        )
    if not gates["moe_decode_no_gpu_route_test_present"]:
        blockers.append(
            "Qwen35 no-GPU MQ6 MoE decode route test is missing or incomplete; "
            "restore moe_decode_routes_mq6_indexed_path_for_k8 before trusting MoE decode runtime coverage"
        )
    if not gates["dense_ar_perf_promotable"]:
        blockers.append("dense MQ6 AR prefill/decode throughput remains below the 0.90x MQ4 promotion threshold")
    if not gates["dense_dflash_perf_promotable"]:
        blockers.append("dense MQ6 DFlash target-side throughput remains below the 0.90x MQ4 promotion threshold")
    blockers.append("DFlash evidence covers MQ6 target-side verify with an MQ4 draft; no MQ6 draft lane is claimed")
    if dflash_boundary["target_side_verify_only"]:
        blockers.append(
            "MQ6 DFlash/spec evidence is dense 27B target-side verify only; "
            "no MQ6 draft, A3B DFlash sidecar, or non-greedy sampling evidence is present"
        )
    if refresh_required:
        blockers.append(
            "origin-only MQ6-relevant fixes touch A3B/MoE, dense AR, and DFlash/spec surfaces; "
            "rerun affected evidence after reconciling qwen35-native-mtp with origin/master"
        )
    if gates["remote_branch_refresh_required"]:
        blockers.append(
            "origin/fix/hunt3-engine-bugs has sampling+MoE/daemon/spec fixes not present on HEAD; "
            "rerun MQ6 model-backed evidence on the selected branch"
        )
    next_work = [
        "Reconcile qwen35-native-mtp with origin/master and origin/fix/hunt3-engine-bugs before trusting current MQ6 promotion evidence.",
        "Rerun MQ6 AR perf baselines after reconciliation and keep prompt md5 plus binary md5 in the artifact.",
        "Rerun MQ6 DFlash/spec baselines after reconciliation; current evidence is MQ6 target-side verify with an MQ4 draft.",
        "Rerun MQ6 coherence with ./scripts/coherence-gate.sh --full for any refreshed candidate artifact or reconciled runtime.",
        "Keep A3B/MoE as the first scope unless dense prefill/decode and DFlash throughput clear the 0.90x MQ4 threshold.",
        "Do not port MQ6 i8 grouped-MMQ unless a future A3B profile shows grouped HFQ6 WMMA is the dominant bottleneck.",
    ]

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "origin_context": upstream,
        "arch": "gfx1151",
        "format": "mq6",
        "status": "a3b-first-candidate-dense-perf-gated",
        "promotion_allowed": gates["broad_promotion_allowed"],
        "thresholds": {
            "dense_promotion_ratio": DENSE_PROMOTION_RATIO,
            "a3b_prefill_speedup_ratio": A3B_PREFILL_SPEEDUP_RATIO,
            "a3b_decode_close_ratio": A3B_DECODE_CLOSE_RATIO,
        },
        "artifact_inventory": artifacts,
        "live_candidate_inventory": live_artifacts,
        "producer_path": producer_path,
        "batched_rope_compact_offset": batched_rope_compact_offset,
        "daemon_dflash_first_token_eos": daemon_dflash_first_token_eos,
        "daemon_ar_exact_match_delta_replay": daemon_ar_exact_match_delta_replay,
        "ddtree_eviction_target_hidden_host": ddtree_eviction_target_hidden_host,
        "moe_prefill_grouped_scratch_free": moe_prefill_grouped_scratch_free,
        "cask_scratch_teardown": cask_scratch_teardown,
        "mtp_gpu_teardown": mtp_gpu_teardown,
        "asym4_fwht_fallback": asym4_fwht_fallback,
        "qwen36_a3b_runtime_source_reconciliation": qwen36_a3b_runtime_source_reconciliation,
        "hunt3_nan_safe_sampling": hunt3_nan_safe_sampling,
        "kernel_inventory": kernels,
        "dense_decode_route_test": dense_decode_route_test,
        "moe_decode_route_test": moe_decode_route_test,
        "dflash_boundary_contract_test": dflash_boundary_contract_test,
        "quality": {
            "dense_9b_ppl": dense_ppl,
            "qwen35_a3b_ppl": qwen35_a3b_ppl,
            "qwen36_a3b_ppl": qwen36_a3b_ppl,
            "dense_9b_c512_kld": kld_c512,
        },
        "performance": {
            "ar_pairs": ar_pairs,
            "dflash_pairs": dflash_pairs,
            "dflash_spec_boundary": dflash_boundary,
        },
        "surface_coverage": surface_coverage,
        "evidence_refresh_plan": refresh_plan,
        "audit_evidence": {
            "readiness_audit": readiness_audit,
            "i8_grouped_decision": i8_decision,
        },
        "gates": gates,
        "blockers": blockers,
        "next_work": next_work,
        "decision": (
            "keep MQ6 as the closest non-MQ4 candidate; review an A3B/MoE-first scope, "
            "do not claim broad promotion until dense AR and dense DFlash throughput are fixed, "
            "and do not port MQ6 i8 grouped-MMQ unless a future A3B profile makes HFQ6 grouped WMMA "
            "the dominant bottleneck"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Write JSON status to this path")
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT), help="Model fixture root to scan")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_status(models_root=Path(args.models_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty) + "\n",
        encoding="utf-8",
    )
    print(out)
    return 0 if payload["schema"] == SCHEMA else 1


if __name__ == "__main__":
    raise SystemExit(main())
