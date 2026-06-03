#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Check readiness-matrix evidence artifacts for drift.

The quant readiness matrix is intentionally text-heavy because it records
engineering judgment.  This companion check keeps the cited evidence honest:
every matrix evidence path must exist, JSON artifacts must parse, and the
structured negative-decision status artifacts must still carry their expected
schemas and blockers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hipfire.quant_readiness_integrity.gfx1151.v0"
DEFAULT_READINESS_SCRIPT = ROOT / "scripts" / "quant_readiness.py"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness" / "2026-06-03-evidence-integrity.json"


EXPECTED_STRUCTURED_ARTIFACTS: dict[str, dict[str, Any]] = {
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-readiness-matrix.json": {
        "schema": "hipfire.quant_readiness.gfx1151.v0",
        "format_id": None,
        "expected": {
            "origin_context.upstream": "origin/master",
            "origin_context.upstream_reconciliation_required": True,
            "origin_context.promotion_evidence_refresh_required": True,
            "origin_context.relevant_upstream_commits.0.short": "676338a4",
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.0.present_on_head": False,
            "origin_context.relevant_upstream_commits.1.short": "d5985c3e",
            "origin_context.relevant_upstream_commits.1.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.1.present_on_head": False,
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq8-status.json": {
        "schema": "hipfire.mq8_status.gfx1151.v0",
        "format_id": "mq8",
        "expected": {
            "promotion_allowed": False,
            "active_promotion_backlog": False,
            "gates.product_role_defined": False,
            "gates.canonical_mq8_artifact_present": False,
            "gates.example_or_benchmark_harness_present": True,
            "gates.candidate_model_benchmark_harness_present": False,
            "gates.quality_evidence_present": False,
            "gates.perf_evidence_present": False,
            "purpose_decision.classification": "permanent-runtime-research",
            "purpose_decision.promotion_backlog_closed": True,
            "purpose_decision.runtime_substrate_maintenance_allowed": True,
            "purpose_decision.artifact_generation_without_product_role_allowed": False,
            "purpose_decision.reopen_requires_all_contract_gates": True,
            "promotion_boundary.promotion_lane_state": "closed_until_reopen_contract",
            "promotion_boundary.artifact_generation_state": "blocked_missing_product_role",
            "promotion_boundary.runtime_substrate_maintenance_allowed": True,
            "promotion_boundary.candidate_model_baseline_available": False,
            "promotion_boundary.reopen_contract_satisfied": False,
            "promotion_boundary.high_bit_justification_required": True,
            "promotion_boundary.missing_reopen_requirements.0": "product_role_over_q8_or_mq6",
            "promotion_boundary.missing_reopen_requirements.2": "mq8_example_or_benchmark_harness",
            "promotion_boundary.next_unblocked_step": "define_product_role_over_q8_or_mq6",
            "research_closure.classification": "permanent-runtime-research",
            "research_closure.closed_without_product_role": True,
            "research_closure.artifact_generation_blocked": True,
            "research_closure.perf_collection_blocked": True,
            "research_closure.runtime_substrate_maintenance_allowed": True,
            "research_closure.candidate_artifact_generation_requires_product_role": True,
            "research_closure.candidate_perf_collection_requires_artifact_and_quality": True,
            "research_closure.high_bit_value_must_exceed_mq6_or_q8": True,
            "research_closure.minimum_reopen_requirements.0": "product_role_over_q8_or_mq6",
            "research_closure.missing_reopen_requirements.2": "mq8_example_or_benchmark_harness",
            "research_closure.next_unblocked_step": "define_product_role_over_q8_or_mq6",
            "reopen_contract.satisfied.mq8_example_or_benchmark_harness": False,
            "harness.present": True,
            "harness.generic_substrate_harness_present": True,
            "harness.candidate_model_harness_present": False,
            "harness.candidate_harness_requires_canonical_artifact": True,
            "origin_context.local_runtime_context_commits.0.short": "282cc9c5",
            "origin_context.local_runtime_context_commits.0.present_on_head": True,
            "origin_context.local_runtime_context_commits.0.present_on_origin_master": False,
            "origin_context.local_runtime_context_commits.3.short": "6bec9f71",
            "origin_context.local_runtime_context_commits.3.present_on_head": True,
            "origin_context.local_runtime_context_commits.3.present_on_origin_master": False,
            "origin_context.upstream_mq8_product_role_commit_found": False,
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq2-status.json": {
        "schema": "hipfire.mq2_status.gfx1151.v0",
        "format_id": "mq2",
        "expected": {
            "promotion_allowed": False,
            "gates.canonical_dense_artifacts_present": True,
            "gates.runtime_smoke_no_hard_errors": True,
            "gates.dense_quality_clean": False,
            "gates.all_dense_rows_collapsed": True,
            "gates.kld_ppl_evidence_present": False,
            "gates.perf_promotion_evidence_present": False,
            "quality_rejection_boundary.dense_text_status": "rejected_all_current_rows_collapsed",
            "quality_rejection_boundary.runtime_smoke_interpretation": "runtime_ok_not_quality_evidence",
            "quality_rejection_boundary.quality_override_required": True,
            "quality_rejection_boundary.sub_9b_failures_explicit": True,
            "quality_rejection_boundary.dense_9b_rejected": True,
            "quality_rejection_boundary.promotion_perf_collection_allowed": False,
            "quality_rejection_boundary.performance_rows_promotable": False,
            "quality_rejection_boundary.next_unblocked_step": "produce_new_mq2_calibration_that_passes_bounded_dense_quality",
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-status.json": {
        "schema": "hipfire.mq6_status.gfx1151.v0",
        "format_id": "mq6",
        "expected": {
            "promotion_allowed": False,
            "gates.canonical_mq6_artifacts_present": True,
            "gates.mq6_producer_mq4_lloyd_promote6_fixed": True,
            "gates.mq6_daemon_ar_exact_match_delta_replay_fixed": True,
            "gates.mq6_ddtree_eviction_target_hidden_host_fixed": True,
            "gates.mq6_moe_prefill_grouped_scratch_free_fixed": True,
            "gates.mq6_cask_scratch_teardown_fixed": True,
            "gates.mq6_mtp_gpu_teardown_fixed": True,
            "gates.mq6_asym4_fwht_fallback_fixed": True,
            "gates.mq6_qwen36_a3b_runtime_source_reconciled": True,
            "gates.dense_9b_ppl_beats_mq4": True,
            "gates.qwen35_a3b_ppl_beats_mq4": True,
            "gates.qwen36_a3b_ppl_beats_mq4": True,
            "gates.dense_9b_c512_kld_beats_mq4": True,
            "gates.dense_ar_perf_present": True,
            "gates.a3b_ar_perf_present": True,
            "gates.dense_dflash_perf_present": True,
            "gates.moe_decode_no_gpu_route_test_present": True,
            "gates.dflash_target_side_verify_only": True,
            "gates.dflash_mq6_draft_evidence_present": False,
            "gates.dflash_a3b_evidence_present": False,
            "gates.dflash_spec_boundary_refresh_required": True,
            "gates.dflash_spec_boundary_contract_test_present": True,
            "gates.surface_coverage_complete": True,
            "gates.all_surfaces_promotion_ready": False,
            "gates.dense_ar_perf_promotable": False,
            "gates.dense_dflash_perf_promotable": False,
            "gates.a3b_prefill_speedup_present": True,
            "gates.a3b_decode_close_to_mq4": True,
            "gates.origin_refresh_required": True,
            "gates.remote_branch_refresh_required": True,
            "gates.a3b_first_scope_supported": False,
            "gates.i8_grouped_port_recommended": False,
            "gates.dense_promotion_allowed": False,
            "gates.broad_promotion_allowed": False,
            "origin_context.mq6_evidence_refresh_required": True,
            "origin_context.remote_branch_refresh_required": True,
            "origin_context.relevant_upstream_commits.0.short": "676338a4",
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.0.present_on_head": False,
            "origin_context.relevant_upstream_commits.5.short": "379484ba",
            "origin_context.relevant_upstream_commits.5.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.5.present_on_head": False,
            "origin_context.relevant_remote_branch_commits.0.short": "fb36f539",
            "origin_context.relevant_remote_branch_commits.0.remote_ref": "origin/fix/hunt3-engine-bugs",
            "origin_context.relevant_remote_branch_commits.0.present_on_remote_ref": True,
            "origin_context.relevant_remote_branch_commits.0.present_on_origin_master": False,
            "origin_context.relevant_remote_branch_commits.0.present_on_head": False,
            "origin_context.relevant_remote_branch_commits.1.short": "d12b3ad4",
            "origin_context.relevant_remote_branch_commits.1.remote_ref": "origin/fix/issue-triage-2026-06-03",
            "origin_context.relevant_remote_branch_commits.1.present_on_remote_ref": True,
            "origin_context.relevant_remote_branch_commits.1.present_on_origin_master": False,
            "origin_context.relevant_remote_branch_commits.1.present_on_head": False,
            "producer_path.promote6_mq4_lloyd_emits_mq6": True,
            "producer_path.origin_fix_reconciled_in_worktree": True,
            "producer_path.origin_fix_short": "d5985c3e",
            "producer_path.target_quant_type": "MQ6G256",
            "producer_path.target_group_size": 256,
            "daemon_ar_exact_match_delta_replay.source_reconciled_in_worktree": True,
            "daemon_ar_exact_match_delta_replay.origin_fix_short": "d5bc3f6",
            "performance.dflash_spec_boundary.target_side_verify_only": True,
            "performance.dflash_spec_boundary.mq4_draft_used_for_all_mq6_rows": True,
            "performance.dflash_spec_boundary.mq6_draft_evidence_present": False,
            "performance.dflash_spec_boundary.dense_27b_only": True,
            "performance.dflash_spec_boundary.a3b_dflash_evidence_present": False,
            "performance.dflash_spec_boundary.sampling_or_non_greedy_evidence_present": False,
            "performance.dflash_spec_boundary.evidence_class": "target_side_verify_only_mq4_draft",
            "performance.dflash_spec_boundary.scope_expanded_beyond_target_verify": False,
            "performance.dflash_spec_boundary.contract_ready": True,
            "performance.dflash_spec_boundary.promotion_scope_supported": False,
            "performance.dflash_spec_boundary.refresh_required": True,
            "performance.dflash_spec_boundary.min_runs_per_case": 3,
            "performance.dflash_spec_boundary.token_attractor_clean": True,
            "dflash_boundary_contract_test.all_phrase_checks_present": True,
            "surface_coverage.schema": "hipfire.mq6_surface_coverage.gfx1151.v0",
            "surface_coverage.all_required_surfaces_recorded": True,
            "surface_coverage.all_surfaces_promotion_ready": False,
            "surface_coverage.refresh_required": True,
            "surface_coverage.surfaces.dense_decode.status": "perf_gated",
            "surface_coverage.surfaces.dense_decode.runtime_evidence_present": True,
            "surface_coverage.surfaces.dense_decode.quality_evidence_present": True,
            "surface_coverage.surfaces.dense_decode.perf_evidence_present": True,
            "surface_coverage.surfaces.dense_decode.promotion_ready": False,
            "surface_coverage.surfaces.dense_decode.details.daemon_ar_exact_match_delta_replay.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dense_decode.details.daemon_ar_exact_match_delta_replay.origin_fix_short": "d5bc3f6",
            "surface_coverage.surfaces.dense_prefill.status": "perf_gated",
            "surface_coverage.surfaces.dense_prefill.promotion_ready": False,
            "surface_coverage.surfaces.dense_prefill.details.daemon_ar_exact_match_delta_replay.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dense_prefill.details.daemon_ar_exact_match_delta_replay.origin_fix_short": "d5bc3f6",
            "surface_coverage.surfaces.moe_prefill.status": "a3b_first_candidate_refresh_blocked",
            "surface_coverage.surfaces.moe_prefill.runtime_evidence_present": True,
            "surface_coverage.surfaces.moe_prefill.quality_evidence_present": True,
            "surface_coverage.surfaces.moe_prefill.perf_evidence_present": True,
            "surface_coverage.surfaces.moe_prefill.promotion_ready": False,
            "surface_coverage.surfaces.moe_prefill.details.grouped_scratch_free.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.moe_prefill.details.grouped_scratch_free.origin_fix_short": "b4adca1f",
            "surface_coverage.surfaces.moe_decode.status": "a3b_first_candidate_refresh_blocked",
            "surface_coverage.surfaces.moe_decode.runtime_evidence_present": True,
            "surface_coverage.surfaces.moe_decode.promotion_ready": False,
            "surface_coverage.surfaces.moe_decode.details.no_gpu_route_test.all_phrase_checks_present": True,
            "surface_coverage.surfaces.dflash_spec.status": "target_verify_only_refresh_blocked",
            "surface_coverage.surfaces.dflash_spec.runtime_evidence_present": True,
            "surface_coverage.surfaces.dflash_spec.promotion_ready": False,
            "surface_coverage.surfaces.dflash_spec.details.evidence_class": "target_side_verify_only_mq4_draft",
            "surface_coverage.surfaces.dflash_spec.details.scope_expanded_beyond_target_verify": False,
            "surface_coverage.surfaces.dflash_spec.details.boundary_contract_test.all_phrase_checks_present": True,
            "surface_coverage.surfaces.dflash_spec.details.ddtree_eviction_target_hidden_host.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dflash_spec.details.ddtree_eviction_target_hidden_host.origin_fix_short": "d12b3ad4",
            "surface_coverage.surfaces.dflash_spec.details.cask_scratch_teardown.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dflash_spec.details.cask_scratch_teardown.origin_fix_short": "a7dcfb0d",
            "surface_coverage.surfaces.dflash_spec.details.mtp_gpu_teardown.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dflash_spec.details.mtp_gpu_teardown.origin_fix_short": "a7dcfb0d",
            "surface_coverage.surfaces.dflash_spec.details.asym4_fwht_fallback.source_reconciled_in_worktree": True,
            "surface_coverage.surfaces.dflash_spec.details.asym4_fwht_fallback.origin_fix_short": "a7dcfb0d",
            "ddtree_eviction_target_hidden_host.source_reconciled_in_worktree": True,
            "ddtree_eviction_target_hidden_host.origin_fix_short": "d12b3ad4",
            "ddtree_eviction_target_hidden_host.remote_ref": "origin/fix/issue-triage-2026-06-03",
            "moe_prefill_grouped_scratch_free.source_reconciled_in_worktree": True,
            "moe_prefill_grouped_scratch_free.origin_fix_short": "b4adca1f",
            "cask_scratch_teardown.source_reconciled_in_worktree": True,
            "cask_scratch_teardown.origin_fix_short": "a7dcfb0d",
            "mtp_gpu_teardown.source_reconciled_in_worktree": True,
            "mtp_gpu_teardown.origin_fix_short": "a7dcfb0d",
            "asym4_fwht_fallback.source_reconciled_in_worktree": True,
            "asym4_fwht_fallback.origin_fix_short": "a7dcfb0d",
            "qwen36_a3b_runtime_source_reconciliation.source_reconciled_in_worktree": True,
            "qwen36_a3b_runtime_source_reconciliation.origin_fix_short": "379484ba",
            "qwen36_a3b_runtime_source_reconciliation.ppl_rows_refreshed_by_this_check": False,
            "qwen36_a3b_runtime_source_reconciliation.perf_rows_refreshed_by_this_check": False,
            "evidence_refresh_plan.refresh_required": True,
            "evidence_refresh_plan.next_unblocked_step": "reconcile_qwen35_native_mtp_with_origin_and_hunt3_then_rerun_mq6_evidence",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.0": "asym4_fwht_fallback",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.1": "cask_scratch_teardown",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.2": "daemon_ar_exact_match_delta_replay",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.3": "ddtree_eviction_stability",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.4": "moe_prefill_scratch_teardown",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.5": "mtp_gpu_teardown",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.6": "producer_context",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.7": "qwen36_a3b_runtime_source_reconciliation",
            "evidence_refresh_plan.locally_reconciled_refresh_reasons.8": "spec_verify",
            "evidence_refresh_plan.stale_origin_commit_shorts.0": "676338a4",
            "evidence_refresh_plan.stale_origin_commit_shorts.5": "379484ba",
            "evidence_refresh_plan.stale_commits.1.short": "d5985c3e",
            "evidence_refresh_plan.stale_commits.1.locally_reconciled_refreshes.0": "producer_context",
            "evidence_refresh_plan.stale_commits.1.unreconciled_refreshes.0": "dflash_stability",
            "evidence_refresh_plan.stale_commits.2.short": "a7dcfb0d",
            "evidence_refresh_plan.stale_commits.2.locally_reconciled_refreshes.0": "asym4_fwht_fallback",
            "evidence_refresh_plan.stale_commits.2.locally_reconciled_refreshes.1": "cask_scratch_teardown",
            "evidence_refresh_plan.stale_commits.2.locally_reconciled_refreshes.2": "mtp_gpu_teardown",
            "evidence_refresh_plan.stale_commits.2.unreconciled_refreshes.0": "a3b_ar_perf",
            "evidence_refresh_plan.stale_commits.2.unreconciled_refreshes.1": "a3b_ppl_smoke",
            "evidence_refresh_plan.stale_commits.2.unreconciled_refreshes.2": "spec_verify",
            "evidence_refresh_plan.stale_commits.3.short": "b4adca1f",
            "evidence_refresh_plan.stale_commits.3.locally_reconciled_refreshes.0": "moe_prefill_scratch_teardown",
            "evidence_refresh_plan.stale_commits.3.unreconciled_refreshes.0": "a3b_ar_perf",
            "evidence_refresh_plan.stale_commits.3.unreconciled_refreshes.1": "long_running_moe_stability",
            "evidence_refresh_plan.stale_commits.4.short": "d5bc3f6",
            "evidence_refresh_plan.stale_commits.4.locally_reconciled_refreshes.0": "daemon_ar_exact_match_delta_replay",
            "evidence_refresh_plan.stale_commits.4.unreconciled_refreshes.0": "coherence",
            "evidence_refresh_plan.stale_commits.4.unreconciled_refreshes.1": "dense_ar_perf",
            "evidence_refresh_plan.stale_commits.4.unreconciled_refreshes.2": "dense_dflash_perf",
            "evidence_refresh_plan.stale_commits.5.short": "379484ba",
            "evidence_refresh_plan.stale_commits.5.locally_reconciled_refreshes.0": "qwen36_a3b_runtime_source_reconciliation",
            "evidence_refresh_plan.stale_commits.5.unreconciled_refreshes.0": "qwen36_a3b_ar_perf",
            "evidence_refresh_plan.stale_commits.5.unreconciled_refreshes.1": "qwen36_a3b_ppl",
            "evidence_refresh_plan.stale_remote_commit_shorts.0": "fb36f539",
            "evidence_refresh_plan.stale_remote_commit_shorts.1": "d12b3ad4",
            "evidence_refresh_plan.stale_commits.7.short": "d12b3ad4",
            "evidence_refresh_plan.stale_commits.7.source": "origin/fix/issue-triage-2026-06-03",
            "evidence_refresh_plan.stale_commits.7.locally_reconciled_refreshes.0": "ddtree_eviction_stability",
            "evidence_refresh_plan.refresh_reasons.0": "a3b_ar_perf",
            "evidence_refresh_plan.refresh_reasons.3": "dense_ar_perf",
            "evidence_refresh_plan.refresh_reasons.4": "dense_dflash_perf",
            "evidence_refresh_plan.refresh_reasons.6": "long_context_or_eviction_rows",
            "evidence_refresh_plan.refresh_reasons.8": "qwen36_a3b_ar_perf",
            "evidence_refresh_plan.refresh_reasons.9": "qwen36_a3b_ppl",
            "evidence_refresh_plan.refresh_reasons.10": "spec_verify",
            "evidence_refresh_plan.affected_surfaces.0": "dense_decode",
            "evidence_refresh_plan.affected_surfaces.1": "dense_prefill",
            "evidence_refresh_plan.affected_surfaces.2": "dflash_spec",
            "evidence_refresh_plan.affected_surfaces.3": "moe_decode",
            "evidence_refresh_plan.affected_surfaces.4": "moe_prefill",
            "evidence_refresh_plan.surface_next_steps.dflash_spec": "reconcile_origin_and_remote_fixes_then_rerun_dense_and_a3b_dflash_spec",
            "evidence_refresh_plan.rerun_commands.0": "./scripts/coherence-gate.sh --full",
            "evidence_refresh_plan.rerun_commands.1": "python3 scripts/mq6_ar_perf_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "evidence_refresh_plan.rerun_commands.2": "python3 scripts/mq6_dflash_baseline.py --pretty --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "evidence_refresh_plan.rerun_commands.3": "python3 scripts/mq6_ppl_compare.py --pretty",
            "evidence_refresh_plan.evidence_artifacts_to_refresh.0": "benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-ar-perf.json",
            "evidence_refresh_plan.evidence_artifacts_to_refresh.5": "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-dflash-r3.json",
            "next_work.0": "Reconcile qwen35-native-mtp with origin/master and origin/fix/hunt3-engine-bugs before trusting current MQ6 promotion evidence.",
            "next_work.4": "Keep A3B/MoE as the first scope unless dense prefill/decode and DFlash throughput clear the 0.90x MQ4 threshold.",
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-size-gate-status.json": {
        "schema": "hipfire.mq3_size_gate_status.gfx1151.v0",
        "format_id": "mq3",
        "expected": {
            "promotion_allowed": False,
            "gates.canonical_mq3_artifacts_present": True,
            "gates.live_canonical_mq3_artifacts_present": True,
            "gates.dense_4b_rejected": True,
            "gates.dense_9b_rejected": True,
            "gates.dense_27b_ppl_beats_mq4": True,
            "gates.dense_27b_kld_reference_present": False,
            "gates.dense_27b_kld_evidence_present": False,
            "gates.dense_27b_ar_perf_present": True,
            "gates.dense_27b_ar_reached_token_cap": False,
            "gates.dense_27b_dflash_perf_present": True,
            "gates.dense_27b_dflash_token_attractor_clean": True,
            "gates.a3b_qwen35_ppl_beats_mq4": True,
            "gates.a3b_qwen36_ppl_beats_mq4": False,
            "gates.a3b_broader_coherence_no_hard_errors": True,
            "gates.a3b_broader_coherence_capped_rows_present": True,
            "gates.a3b_broader_coherence_promotion_grade": False,
            "gates.a3b_kld_references_present": False,
            "gates.a3b_dflash_sidecars_present": False,
            "gates.a3b_moe_decode_boundary_recorded": True,
            "gates.a3b_moe_decode_indexed_route_supported": False,
            "gates.long_prefill_shape_audit_present": True,
            "gates.long_prefill_shape_contract_covered": True,
            "gates.long_prefill_runtime_evidence_present": False,
            "runtime.a3b_moe_decode_boundary.status": "decode_indexed_route_missing_recorded",
            "runtime.a3b_moe_decode_boundary.boundary_recorded": True,
            "runtime.a3b_moe_decode_boundary.indexed_route_supported": False,
            "runtime.a3b_moe_decode_boundary.all_phrase_checks_present": True,
            "runtime.long_prefill_shape_contract.status": "covered_no_gpu_not_runtime",
            "runtime.long_prefill_shape_contract.all_phrase_checks_present": True,
            "runtime.long_prefill_shape_contract.promotion_grade_runtime_result": False,
            "runtime.long_prefill_shape_contract.artifact_backed_long_prefill_evidence_present": False,
            "runtime.long_prefill_shape_contract.invariants.n_tokens": 256,
            "runtime.long_prefill_shape_contract.invariants.k_top": 8,
            "runtime.long_prefill_shape_contract.invariants.num_experts": 256,
            "runtime.long_prefill_shape_contract.invariants.total_slots": 2048,
            "runtime.long_prefill_shape_contract.invariants.m_total_bound": 5888,
            "runtime.long_prefill_shape_contract.invariants.m_total_bound_multiple_of_16": True,
            "runtime.long_prefill_shape_contract.invariants.gate_up_x_row_div": 8,
            "runtime.long_prefill_shape_contract.invariants.down_x_row_div": 1,
            "size_gate_boundary.status": "size_gated_incomplete",
            "size_gate_boundary.promotion_allowed": False,
            "size_gate_boundary.dense_4b.status": "rejected",
            "size_gate_boundary.dense_9b.status": "rejected",
            "size_gate_boundary.dense_27b.status": "candidate_incomplete_kld_reference",
            "size_gate_boundary.a3b.status": "research_blocked",
            "size_gate_boundary.a3b.coherence_status": "no_hard_errors_but_capped_rows_not_promotion_grade",
            "size_gate_boundary.a3b.broader_coherence_capped_rows_present": True,
            "size_gate_boundary.a3b.broader_coherence_hit_max_tokens_count": 7,
            "size_gate_boundary.a3b.promotion_grade_coherence_present": False,
            "size_gate_boundary.long_prefill_no_gpu_contract.status": "covered_no_gpu_not_runtime",
            "size_gate_boundary.long_prefill_no_gpu_contract.all_phrase_checks_present": True,
            "size_gate_boundary.long_prefill_no_gpu_contract.artifact_backed_long_prefill_evidence_present": False,
            "size_gate_boundary.long_prefill_no_gpu_contract.invariants.total_slots": 2048,
            "size_gate_boundary.long_prefill_no_gpu_contract.invariants.m_total_bound": 5888,
            "size_gate_boundary.moe_decode.status": "decode_indexed_route_missing_recorded",
            "size_gate_boundary.moe_decode.indexed_route_supported": False,
            "quality_boundary.dense_small_model_status": "4b_and_9b_rejected_current_calibration",
            "quality_boundary.dense_27b_status": "candidate_incomplete_kld_reference",
            "quality_boundary.a3b_status": "research_blocked",
            "quality_boundary.promotion_requires_new_quality_evidence": True,
            "next_work.3": "Run artifact-backed MQ3 A3B long-prefill coherence/perf; the current audit is no-GPU shape coverage only.",
            "live_artifact_inventory.artifact_count": 23,
            "live_artifact_inventory.candidate_root_artifact_count": 5,
            "live_artifact_inventory.cache_artifact_count": 18,
            "live_artifact_inventory.awq_mtp_artifact_count": 9,
            "live_artifact_inventory.canonical_required_complete": True,
            "live_artifact_inventory.canonical_required_missing": [],
            "live_artifact_inventory.canonical_required_present.0": "qwen3.5-27b-mq3.hfq",
            "live_artifact_inventory.canonical_required_present.1": "qwen3.5-35b-a3b-mq3.hfq",
            "live_artifact_inventory.canonical_required_present.2": "qwen3.5-4b-mq3.hfq",
            "live_artifact_inventory.canonical_required_present.3": "qwen3.5-9b-mq3.hfq",
            "live_artifact_inventory.canonical_required_present.4": "qwen3.6-35b-a3b-mq3.hfq",
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-productization-status.json": {
        "schema": "hipfire.paro_q4g128_productization_status.gfx1151.v0",
        "format_id": "paro-q4g128",
        "expected": {
            "promotion_allowed": False,
            "gates.source_probe_coverage_complete": True,
            "gates.source_inventory_present": True,
            "gates.source_inventory_consistent": True,
            "gates.source_inventory_native_paro_g128_found": False,
            "gates.native_paro_source_found": False,
            "gates.imported_hfq_exists": False,
            "gates.runtime_env_boundary_recorded": True,
            "gates.research_only_env_evidence_absent": True,
            "gates.promotion_main_path_clean": True,
            "gates.package_contract_present": True,
            "gates.oracle_evidence_present": False,
            "gates.coherence_evidence_present": False,
            "gates.finite_logit_nan_evidence_present": False,
            "gates.quality_evidence_present": False,
            "gates.gfx1151_perf_evidence_present": False,
            "gates.typed_evidence_files_exist": False,
            "gates.oracle_quality_perf_evidence_present": False,
            "evidence.typed_artifact_existence.all_positive_typed_evidence_files_exist": False,
            "dependency_graph.native_source_absent": True,
            "dependency_graph.import_blocked_by_source_absence": True,
            "dependency_graph.oracle_blocked_by_import_absence": True,
            "dependency_graph.nodes.typed_evidence_files.satisfied": False,
            "dependency_graph.nodes.source_inventory.satisfied": True,
            "dependency_graph.nodes.source_inventory_consistency.satisfied": True,
            "dependency_graph.all_required_suffixes_absent_across_expected_probes": True,
            "dependency_graph.source_inventory.native_paro_g128_source_found": False,
            "dependency_graph.source_inventory.g128_complete_module_count": 0,
            "dependency_graph.next_unblocked_step": "locate_or_generate_native_paro_q4g128_checkpoint",
            "source_inventory.present": True,
            "source_inventory.schema_ok": True,
            "source_inventory.native_paro_source_found": False,
            "source_inventory.native_paro_g128_source_found": False,
            "source_inventory.g128_complete_module_count": 0,
            "source_probe_coverage.all_required_suffixes_absent_across_expected_probes": True,
            "source_probe_coverage.aggregate_required_suffix_counts.qweight": 0,
            "source_probe_coverage.aggregate_required_suffix_counts.qzeros": 0,
            "source_probe_coverage.aggregate_required_suffix_counts.scales": 0,
            "source_probe_coverage.aggregate_required_suffix_counts.pairs": 0,
            "source_probe_coverage.aggregate_required_suffix_counts.theta": 0,
            "source_probe_coverage.aggregate_required_suffix_counts.channel_scales": 0,
            "productization_boundary.current_stage": "blocked_before_native_source_import",
            "productization_boundary.importer_state": "blocked_no_native_paro_source",
            "productization_boundary.package_state": "contract_only_missing_weights",
            "productization_boundary.astrea_state": "bundle_plan_contract_only_not_model_writer",
            "productization_boundary.main_path_state": "clean_but_unpromoted",
            "productization_boundary.hidden_research_env_dependency_present": False,
            "productization_boundary.artifact_generation_without_native_source_allowed": False,
            "productization_boundary.research_only_env_blocks_promotion": True,
            "productization_boundary.promoted_main_path_requires_typed_evidence": True,
            "productization_boundary.native_source_contract.all_required_suffixes_absent_across_expected_probes": True,
            "productization_boundary.native_source_contract.broader_source_inventory.native_paro_g128_source_found": False,
            "productization_boundary.native_source_contract.broader_source_inventory.g128_complete_module_count": 0,
            "productization_boundary.native_source_contract.required_tensor_families.0": "qweight",
            "productization_boundary.native_source_contract.required_tensor_families.5": "channel_scales",
            "productization_plan.status": "blocked_before_native_source_import",
            "productization_plan.next_unblocked_step": "locate_or_generate_native_paro_q4g128_checkpoint",
            "productization_plan.stage_order.0": "source_inventory",
            "productization_plan.stage_order.1": "native_source",
            "productization_plan.stage_order.3": "imported_hfq",
            "productization_plan.stage_order.5": "paro_oracle",
            "productization_plan.stage_order.10": "promotion_review",
            "productization_plan.stages.source_inventory.satisfied": True,
            "productization_plan.stages.source_inventory.command": (
                "python3 scripts/paroquant_inventory.py --pretty --out "
                "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-source-inventory.json"
            ),
            "productization_plan.stages.source_inventory.native_paro_g128_source_found": False,
            "productization_plan.stages.source_inventory.g128_complete_module_count": 0,
            "productization_plan.stages.native_source.satisfied": False,
            "productization_plan.stages.native_source.required_tensor_families.0": "qweight",
            "productization_plan.stages.native_source.required_tensor_families.5": "channel_scales",
            "productization_plan.stages.imported_hfq.satisfied": False,
            "productization_plan.stages.imported_hfq.blocked_by.0": "native_source",
            "productization_plan.stages.paro_oracle.satisfied": False,
            "productization_plan.stages.paro_oracle.blocked_by.0": "imported_hfq",
            "productization_plan.stages.coherence.satisfied": False,
            "productization_plan.stages.finite_logit_nan.satisfied": False,
            "productization_plan.stages.quality_kld_ppl.satisfied": False,
            "productization_plan.stages.gfx1151_perf.satisfied": False,
            "productization_plan.stages.gfx1151_perf.blocked_by.0": "imported_hfq",
            "productization_plan.stages.promotion_review.satisfied": False,
            "productization_plan.evidence_artifact_targets.oracle": (
                "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-oracle.json"
            ),
            "productization_plan.evidence_artifact_targets.gfx1151_perf": (
                "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-gfx1151-perf.json"
            ),
            "productization_plan.refresh_required_after_origin_reconciliation": True,
            "next_work.0": (
                "Locate or produce a native ParoQ4G128 checkpoint with "
                "qweight/qzeros/scales/pairs/theta/channel_scales."
            ),
            "next_work.1": "Run paro-probe, paro-import, and paro-oracle against the native source and imported HFQ.",
            "origin_context.relevant_upstream_commits.0.short": "676338a4",
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.0.present_on_head": False,
            "origin_context.relevant_upstream_commits.3.short": "8de45455",
            "origin_context.relevant_upstream_commits.3.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.3.present_on_head": False,
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-source-inventory.json": {
        "schema": "hipfire.astrea.paro_source_inventory.v0",
        "format_id": "paro-q4g128",
        "expected": {
            "files_seen": 861,
            "safetensor_dirs_scanned": 17,
            "safetensor_files_scanned": 151,
            "filename_hit_count": 0,
            "native_paro.complete_module_count": 0,
            "native_paro.incomplete_module_count": 0,
            "native_paro.g128_complete_module_count": 0,
            "native_paro.g256_complete_module_count": 0,
            "decision.native_paro_source_found": False,
            "decision.native_paro_g256_source_found": False,
            "decision.quality_state": "UNVERIFIABLE",
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-status.json": {
        "schema": "hipfire.paro_q4g256_status.gfx1151.v0",
        "format_id": "paro-q4g256",
        "expected": {
            "promotion_allowed": False,
            "gates.upstream_g256_research_evidence_present": True,
            "gates.native_g256_source_found": False,
            "gates.cpu_probe_runnable_now": False,
            "gates.regrouped_payload_ratio_gate_passed": True,
            "gates.current_true_g256_payload_ratio_evidence_present": False,
            "gates.current_true_g256_payload_ratio_gate_passed": False,
            "gates.payload_ratio_gate_passed": False,
            "gates.true_g256_quality_evidence_present": False,
            "gates.runtime_dtype_container_ready": False,
            "contract_state.current_stage": "blocked_before_true_group_size_256_source",
            "contract_state.producer_contract.required_tensor_families.0": "qweight",
            "contract_state.producer_contract.required_tensor_families.5": "channel_scales",
            "contract_state.producer_contract.required_group_size": 256,
            "contract_state.producer_contract.complete_native_paro_module_count": 0,
            "contract_state.producer_contract.complete_g256_module_count": 0,
            "contract_state.producer_contract.source_search_result": "no_paro_suffix_or_filename_hits",
            "contract_state.producer_contract.quality_verifiable_only_with_true_g256_source": True,
            "contract_state.upstream_g256_evidence.summary.exit_decision": "B",
            "contract_state.upstream_g256_evidence.summary.g256_opt_in_research_not_default": True,
            "contract_state.upstream_g256_evidence.probes.probe_count": 2,
            "contract_state.upstream_g256_evidence.probes.all_probe_caveats_regrouped_g128": True,
            "contract_state.upstream_g256_evidence.research_evidence_present": True,
            "contract_state.upstream_g256_evidence.true_g256_source_evidence_present": False,
            "contract_state.upstream_g256_evidence.promotion_proof": False,
            "contract_state.quality_state": "UNVERIFIABLE",
            "contract_state.quality_marked_unverifiable": True,
            "contract_state.evidence_class": "regrouped_g128_format_loss_only",
            "contract_state.runtime_work_allowed": False,
            "prototype_boundary.current_stage": "blocked_before_true_group_size_256_source",
            "prototype_boundary.true_group_size_256_source_required": True,
            "prototype_boundary.true_group_size_256_source_found": False,
            "prototype_boundary.producer_contract.required_group_size": 256,
            "prototype_boundary.producer_contract.source_search_result": "no_paro_suffix_or_filename_hits",
            "prototype_boundary.source_inventory_quality_state": "UNVERIFIABLE",
            "prototype_boundary.quality_marked_unverifiable": True,
            "prototype_boundary.format_loss_ratio_passed": True,
            "prototype_boundary.regrouped_payload_ratio_gate_passed": True,
            "prototype_boundary.current_true_g256_payload_ratio_evidence_present": False,
            "prototype_boundary.current_true_g256_payload_ratio_gate_passed": False,
            "prototype_boundary.format_loss_evidence_class": "regrouped_g128_format_loss_only",
            "prototype_boundary.payload_ratio_only_not_promotion_evidence": True,
            "prototype_boundary.true_g256_quality_evidence_present": False,
            "prototype_boundary.kld_ppl_within_5_percent_of_paro_q4g128_required": True,
            "prototype_boundary.kld_ppl_within_5_percent_of_paro_q4g128_proven": False,
            "prototype_boundary.runtime_dtype_container_work_allowed": False,
            "prototype_boundary.upstream_g256_research_evidence_present": True,
            "prototype_boundary.upstream_g256_evidence_class": "regrouped_g128_opt_in_research",
            "prototype_boundary.upstream_g256_evidence_is_promotion_proof": False,
            "prototype_boundary.artifact_generation_without_true_source_allowed": False,
            "prototype_boundary.next_unblocked_step": "locate_or_generate_true_group_size_256_paro_source",
            "prototype_plan.status": "blocked_before_true_group_size_256_source",
            "prototype_plan.next_unblocked_step": "locate_or_generate_true_group_size_256_paro_source",
            "prototype_plan.stage_order.0": "true_group_size_256_source",
            "prototype_plan.stage_order.1": "cpu_g256_probe",
            "prototype_plan.stage_order.5": "runtime_dtype_container_work",
            "prototype_plan.stages.true_group_size_256_source.satisfied": False,
            "prototype_plan.stages.true_group_size_256_source.required_group_size": 256,
            "prototype_plan.stages.true_group_size_256_source.required_tensor_families.0": "qweight",
            "prototype_plan.stages.true_group_size_256_source.required_tensor_families.5": "channel_scales",
            "prototype_plan.stages.cpu_g256_probe.satisfied": False,
            "prototype_plan.stages.cpu_g256_probe.blocked_by.0": "true_group_size_256_source",
            "prototype_plan.stages.paro_oracle_comparison.satisfied": False,
            "prototype_plan.stages.payload_ratio_lte_1_03.satisfied": False,
            "prototype_plan.stages.payload_ratio_lte_1_03.target_ratio": 1.03,
            "prototype_plan.stages.kld_ppl_within_5_percent_of_paro_q4g128.satisfied": False,
            "prototype_plan.stages.kld_ppl_within_5_percent_of_paro_q4g128.tolerance_ratio": 1.05,
            "prototype_plan.stages.runtime_dtype_container_work.satisfied": False,
            "prototype_plan.stages.runtime_dtype_container_work.allowed": False,
            "prototype_plan.stages.runtime_dtype_container_work.blocked_by.0": "true_group_size_256_source",
            "prototype_plan.evidence_artifact_targets.cpu_g256_probe": (
                "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-cpu-probe.json"
            ),
            "prototype_plan.evidence_artifact_targets.kld_ppl": (
                "benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-kld-ppl.json"
            ),
            "prototype_plan.quality_unverifiable_until_true_g256_source": True,
            "prototype_plan.runtime_dtype_container_work_allowed": False,
            "prototype_plan.upstream_research_evidence_is_promotion_proof": False,
            "next_unblocked_step": "locate_or_generate_true_group_size_256_paro_source",
            "next_work.0": (
                "Locate or produce a true group_size=256 Paro source with "
                "qweight/qzeros/scales/pairs/theta/channel_scales."
            ),
            "next_work.1": (
                "Rerun scripts/paroquant_g256_probe.py --local-only against that "
                "true G256 source; regrouped-G128 probes remain research context only."
            ),
            "origin_context.relevant_upstream_commits.0.short": "f1e2cef8",
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.1.short": "907cbb2f",
            "origin_context.relevant_upstream_commits.1.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.2.short": "79d30777",
            "origin_context.relevant_upstream_commits.2.present_on_origin_master": True,
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq2-lloyd-4w-bench.json": {
        "schema": "hipfire.mq2_lloyd_4w_bench.gfx1151.v0",
        "format_id": "mq2-lloyd",
        "expected": {
            "summary.all_correct": True,
            "summary.all_candidate_slower_than_baseline": True,
            "summary.promote_4w_default": False,
            "summary.default_kernel_change_allowed": False,
            "summary.model_level_promotion_allowed": False,
            "summary.bandwidth_bottleneck_proven": False,
            "promotion_allowed": False,
            "status": "specialty_blocked_no_model_artifact",
            "model_artifact_inventory.mq2_lloyd_artifact_present": False,
            "model_artifact_inventory.current_a3b_or_deepseek_artifact_present": False,
            "specialty_scope.routed_expert_status": "specialty-research-only",
            "specialty_scope.default_kernel_change_allowed": False,
            "specialty_scope.model_level_evidence_present": False,
            "specialty_scope.bandwidth_bottleneck_proven": False,
            "specialty_scope.next_experiments.0": "batch_selected_experts_across_tokens",
            "specialty_boundary.status": "specialty_blocked_no_model_artifact",
            "specialty_boundary.synthetic_kernel_result": "correct_but_slower_than_k2",
            "specialty_boundary.current_a3b_or_deepseek_artifact_present": False,
            "specialty_boundary.mq2_lloyd_artifact_present": False,
            "specialty_boundary.routed_expert_coherence_present": False,
            "specialty_boundary.model_level_perf_evidence_present": False,
            "specialty_boundary.perf_collection_allowed": False,
            "specialty_boundary.default_kernel_change_allowed": False,
            "specialty_boundary.requires_model_backed_win_over_k2": True,
            "specialty_boundary.origin_master_refresh_required": True,
            "specialty_boundary.remote_branch_refresh_required": True,
            "specialty_boundary.remote_moe_fix_refresh_required": True,
            "specialty_boundary.model_evidence_refresh_required": True,
            "specialty_boundary.next_unblocked_step": "generate_or_locate_a3b_or_deepseek_mq2_lloyd_artifact",
            "origin_context.origin_master_refresh_required": True,
            "origin_context.remote_branch_refresh_required": True,
            "origin_context.model_evidence_refresh_required": True,
            "origin_context.relevant_upstream_commits.0.short": "89f42e4b",
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.0.present_on_head": False,
            "origin_context.relevant_upstream_commits.2.short": "edf922db",
            "origin_context.relevant_upstream_commits.2.present_on_remote_ref": True,
            "origin_context.relevant_upstream_commits.2.present_on_origin_master": False,
            "origin_context.relevant_upstream_commits.2.present_on_head": False,
            "origin_context.relevant_upstream_commits.3.short": "fb36f539",
            "origin_context.relevant_upstream_commits.3.remote_ref": "origin/fix/hunt3-engine-bugs",
            "origin_context.relevant_upstream_commits.3.present_on_remote_ref": True,
            "origin_context.relevant_upstream_commits.3.present_on_origin_master": False,
            "origin_context.relevant_upstream_commits.3.present_on_head": False,
            "origin_context.remote_branch_missing_commit_shorts.1": "fb36f539",
        },
    },
    "benchmarks/results/gfx1151-quant-readiness/2026-06-03-lloyd-status.json": {
        "schema": "hipfire.lloyd_status.gfx1151.v0",
        "format_id": ["mq3-lloyd", "mq4-lloyd"],
        "expected": {
            "summary.promotion_allowed": False,
            "formats.mq3-lloyd.promotion_allowed": False,
            "formats.mq3-lloyd.gates.current_kld_present": True,
            "formats.mq3-lloyd.gates.current_kld_finite": True,
            "formats.mq3-lloyd.gates.current_kld_beats_mq4": False,
            "formats.mq3-lloyd.gates.current_kld_loses_to_mq4": True,
            "formats.mq3-lloyd.gates.origin_refresh_required": True,
            "formats.mq3-lloyd.size_scope.only_9b_artifact_present": True,
            "formats.mq3-lloyd.size_scope.live_artifact_count": 2,
            "formats.mq3-lloyd.size_scope.live_only_9b_artifact_present": True,
            "formats.mq3-lloyd.size_scope.dense_27b_artifact_present": False,
            "formats.mq3-lloyd.size_scope.live_dense_27b_artifact_present": False,
            "formats.mq3-lloyd.size_scope.a3b_artifact_present": False,
            "formats.mq3-lloyd.size_scope.live_a3b_artifact_present": False,
            "formats.mq3-lloyd.size_scope.perf_evidence_allowed": False,
            "formats.mq3-lloyd.live_artifact_inventory.artifact_count": 2,
            "formats.mq3-lloyd.live_artifact_inventory.only_9b_artifact_present": True,
            "formats.mq4-lloyd.promotion_allowed": False,
            "formats.mq4-lloyd.live_artifact_inventory.artifact_count": 2,
            "formats.mq4-lloyd.artifact_scope.only_9b_artifact_present": True,
            "formats.mq4-lloyd.artifact_scope.a3b_artifact_present": False,
            "formats.mq4-lloyd.gates.coherence_9b_clean": False,
            "formats.mq4-lloyd.gates.current_same_run_kld_present": True,
            "formats.mq4-lloyd.gates.current_same_run_candidate_valid": False,
            "formats.mq4-lloyd.gates.current_same_run_invalid_zero_kld_no_ppl": True,
            "formats.mq4-lloyd.gates.current_same_run_beats_mq4": False,
            "formats.mq4-lloyd.gates.current_same_run_beats_mq6": False,
            "formats.mq4-lloyd.gates.origin_refresh_required": True,
            "formats.mq4-lloyd.value_boundary.status": "value_not_justified_current_artifact",
            "formats.mq4-lloyd.value_boundary.bytes_per_group": 160,
            "formats.mq4-lloyd.value_boundary.coherence_gate_passed": False,
            "formats.mq4-lloyd.value_boundary.same_run_kld_ppl_valid": False,
            "formats.mq4-lloyd.value_boundary.invalid_zero_kld_no_ppl_blocks_quality": True,
            "formats.mq4-lloyd.value_boundary.beats_mq4_same_run": False,
            "formats.mq4-lloyd.value_boundary.beats_mq6_same_run": False,
            "formats.mq4-lloyd.value_boundary.historical_rows_justify_value": False,
            "formats.mq4-lloyd.value_boundary.origin_refresh_required": True,
            "formats.mq4-lloyd.value_boundary.branch_reconciled_for_promotion": False,
            "formats.mq4-lloyd.value_boundary.perf_collection_allowed": False,
            "formats.mq4-lloyd.value_boundary.performance_rows_promotable": False,
            "formats.mq4-lloyd.value_boundary.next_unblocked_step": "produce_new_coherent_mq4_lloyd_artifact",
            "formats.mq4-lloyd.promotion_contract.promotion_allowed": False,
            "formats.mq4-lloyd.promotion_contract.current_same_run_invalid_zero_kld_no_ppl_must_be_false": True,
            "formats.mq4-lloyd.promotion_contract.historical_rows_required_for_promotion": False,
            "formats.mq4-lloyd.promotion_contract.forbidden_clear.current_same_run_invalid_zero_kld_no_ppl": False,
            "formats.mq4-lloyd.promotion_contract.forbidden_clear.origin_refresh_required": False,
            "formats.mq4-lloyd.promotion_contract.required_satisfied.non_9b_artifact_present": False,
            "formats.mq4-lloyd.promotion_contract.required_satisfied.coherence_9b_clean": False,
            "formats.mq4-lloyd.promotion_contract.required_satisfied.current_same_run_candidate_valid": False,
            "formats.mq4-lloyd.producer_repair_plan.status": "blocked_before_perf",
            "formats.mq4-lloyd.producer_repair_plan.current_artifact_status": "coherence_rejected",
            "formats.mq4-lloyd.producer_repair_plan.requires_new_artifact_hash": True,
            "formats.mq4-lloyd.producer_repair_plan.requires_origin_reconciliation": True,
            "formats.mq4-lloyd.producer_repair_plan.origin_commits_to_reconcile.0": "d5985c3e",
            "formats.mq4-lloyd.producer_repair_plan.origin_commits_to_reconcile.4": "89f42e4b",
            "formats.mq4-lloyd.producer_repair_plan.same_run_quality_ready": False,
            "formats.mq4-lloyd.producer_repair_plan.invalid_zero_kld_no_ppl_blocks_quality": True,
            "formats.mq4-lloyd.producer_repair_plan.perf_collection_allowed": False,
            "formats.mq4-lloyd.producer_repair_plan.next_unblocked_step": "produce_new_coherent_mq4_lloyd_artifact",
            "formats.mq4-lloyd.next_work.0": "Produce a new coherent MQ4-Lloyd 9B artifact hash before collecting perf.",
            "formats.mq4-lloyd.next_work.3": "Rerun the current MQ4/MQ4-Lloyd/MQ6 KLD cohort and require finite PPL plus wins over MQ4 and MQ6.",
            "origin_context.relevant_upstream_commits.0.short": "d5985c3e",
            "origin_context.lloyd_evidence_refresh_required": True,
            "origin_context.format_refresh_required.mq3-lloyd": True,
            "origin_context.format_refresh_required.mq4-lloyd": True,
            "origin_context.relevant_upstream_commits.0.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.0.present_on_head": False,
            "origin_context.relevant_upstream_commits.1.short": "5d09c2ee",
            "origin_context.relevant_upstream_commits.1.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.1.present_on_head": False,
            "origin_context.relevant_upstream_commits.2.short": "46898ecd",
            "origin_context.relevant_upstream_commits.2.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.2.present_on_head": False,
            "origin_context.relevant_upstream_commits.4.short": "aa781d74",
            "origin_context.relevant_upstream_commits.4.present_on_origin_master": True,
            "origin_context.relevant_upstream_commits.4.present_on_head": False,
        },
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_value(args: list[str], *, source_root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_quant_readiness_module(path: Path = DEFAULT_READINESS_SCRIPT):
    spec = importlib.util.spec_from_file_location("quant_readiness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["quant_readiness"] = module
    spec.loader.exec_module(module)
    return module


def nested_value(payload: dict[str, Any], dotted_key: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in dotted_key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return False, None
            current = current[index]
            continue
        else:
            return False, None
    return True, current


def evidence_entries(readiness_items: Iterable[Any]) -> list[dict[str, Any]]:
    entries = []
    for item in readiness_items:
        for relpath in item.evidence_artifacts:
            entries.append(
                {
                    "format_id": item.id,
                    "promotion_target": bool(item.promotion_target),
                    "path": relpath,
                }
            )
    return entries


def check_evidence_paths(source_root: Path, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checked = []
    issues = []
    for entry in entries:
        relpath = entry["path"]
        path = Path(relpath)
        issue_base = {
            "severity": "error",
            "format_id": entry["format_id"],
            "path": relpath,
        }
        if path.is_absolute():
            issues.append({**issue_base, "kind": "absolute_evidence_path"})
            resolved = path
        else:
            resolved = source_root / path
        exists = resolved.exists()
        kind = "missing"
        size_bytes = None
        json_schema = None
        if exists and resolved.is_dir():
            kind = "directory"
        elif exists and resolved.is_file():
            kind = "file"
            size_bytes = resolved.stat().st_size
            if resolved.suffix == ".json":
                try:
                    payload = json.loads(resolved.read_text(encoding="utf-8"))
                    json_schema = payload.get("schema") if isinstance(payload, dict) else None
                except json.JSONDecodeError as exc:
                    issues.append(
                        {
                            **issue_base,
                            "kind": "invalid_json",
                            "message": f"{exc.msg} at line {exc.lineno}, column {exc.colno}",
                        }
                    )
        if not exists:
            issues.append({**issue_base, "kind": "missing_evidence"})
        checked.append(
            {
                **entry,
                "exists": exists,
                "kind": kind,
                "size_bytes": size_bytes,
                "json_schema": json_schema,
            }
        )
    return checked, issues


def check_promotion_targets(readiness_items: Iterable[Any]) -> list[dict[str, Any]]:
    issues = []
    for item in readiness_items:
        if bool(item.promotion_target) and not item.evidence_artifacts:
            issues.append(
                {
                    "severity": "error",
                    "kind": "promotion_target_without_evidence",
                    "format_id": item.id,
                    "path": None,
                }
            )
    return issues


def check_structured_artifacts(
    source_root: Path,
    entries: list[dict[str, Any]],
    expectations: dict[str, dict[str, Any]] = EXPECTED_STRUCTURED_ARTIFACTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cited_by: dict[str, set[str]] = {}
    for entry in entries:
        cited_by.setdefault(entry["path"], set()).add(entry["format_id"])

    checked = []
    issues = []
    for relpath, expectation in expectations.items():
        path = source_root / relpath
        format_id = expectation.get("format_id")
        if isinstance(format_id, list):
            required_format_ids = format_id
        elif format_id:
            required_format_ids = [format_id]
        else:
            required_format_ids = []
        cited_formats = sorted(cited_by.get(relpath, set()))
        record = {
            "path": relpath,
            "format_id": format_id,
            "exists": path.exists(),
            "cited_by": cited_formats,
            "expected_schema": expectation.get("schema"),
            "actual_schema": None,
            "expected_values": expectation.get("expected", {}),
        }
        issue_base = {
            "severity": "error",
            "path": relpath,
            "format_id": format_id,
        }
        if not path.exists():
            issues.append({**issue_base, "kind": "missing_structured_artifact"})
            checked.append(record)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    **issue_base,
                    "kind": "invalid_structured_json",
                    "message": f"{exc.msg} at line {exc.lineno}, column {exc.colno}",
                }
            )
            checked.append(record)
            continue
        if not isinstance(payload, dict):
            issues.append({**issue_base, "kind": "structured_artifact_not_object"})
            checked.append(record)
            continue
        actual_schema = payload.get("schema")
        record["actual_schema"] = actual_schema
        if actual_schema != expectation.get("schema"):
            issues.append(
                {
                    **issue_base,
                    "kind": "schema_mismatch",
                    "expected": expectation.get("schema"),
                    "actual": actual_schema,
                }
            )
        for required_format_id in required_format_ids:
            if required_format_id not in cited_formats:
                issues.append(
                    {
                        **issue_base,
                        "kind": "structured_artifact_not_cited_by_format",
                        "required_format_id": required_format_id,
                        "cited_by": cited_formats,
                    }
                )
        for dotted_key, expected in expectation.get("expected", {}).items():
            found, actual = nested_value(payload, dotted_key)
            if not found:
                issues.append({**issue_base, "kind": "missing_expected_key", "key": dotted_key})
                continue
            if actual != expected:
                issues.append(
                    {
                        **issue_base,
                        "kind": "expected_value_mismatch",
                        "key": dotted_key,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        checked.append(record)
    return checked, issues


def build_integrity_report(
    *,
    source_root: Path = ROOT,
    readiness_items: Iterable[Any] | None = None,
    readiness_script: Path = DEFAULT_READINESS_SCRIPT,
    expectations: dict[str, dict[str, Any]] = EXPECTED_STRUCTURED_ARTIFACTS,
) -> dict[str, Any]:
    if readiness_items is None:
        readiness_items = load_quant_readiness_module(readiness_script).READINESS
    readiness_items = tuple(readiness_items)
    entries = evidence_entries(readiness_items)
    checked_evidence, evidence_issues = check_evidence_paths(source_root, entries)
    structured, structured_issues = check_structured_artifacts(source_root, entries, expectations)
    promotion_issues = check_promotion_targets(readiness_items)
    issues = [*evidence_issues, *structured_issues, *promotion_issues]
    severity_counts: dict[str, int] = {}
    for issue in issues:
        severity = issue.get("severity", "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "commit": git_value(["rev-parse", "HEAD"], source_root=source_root),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], source_root=source_root),
        "source_root": str(source_root),
        "readiness_script": str(readiness_script),
        "summary": {
            "ok": not any(issue.get("severity") == "error" for issue in issues),
            "format_count": len(readiness_items),
            "evidence_count": len(entries),
            "unique_evidence_count": len({entry["path"] for entry in entries}),
            "structured_artifact_count": len(structured),
            "issue_count": len(issues),
            "severity_counts": severity_counts,
        },
        "evidence": checked_evidence,
        "structured_artifacts": structured,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--readiness-script", default=str(DEFAULT_READINESS_SCRIPT))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Write the JSON report to this path")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    readiness_script = Path(args.readiness_script)
    report = build_integrity_report(source_root=source_root, readiness_script=readiness_script)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2 if args.pretty else None, sort_keys=args.pretty) + "\n",
        encoding="utf-8",
    )
    print(out)
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
