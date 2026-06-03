#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mq8_status.py"
CURRENT_ARTIFACT = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness" / "2026-06-03-mq8-status.json"


spec = importlib.util.spec_from_file_location("mq8_status", MODULE_PATH)
assert spec and spec.loader
mq8_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq8_status
spec.loader.exec_module(mq8_status)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def source_tree(root: Path):
    write(
        root / "crates" / "hipfire-quantize" / "src" / "main.rs",
        'let use_mq8g256 = format == "mq8" || format == "mq8g256"; '
        "fn quantize_mq8g256() {} enum QuantType { MQ8G256 = 14 } Q8F16",
    )
    write(root / "crates" / "hipfire-runtime" / "src" / "hfq.rs", "DType::MQ8G256")
    write(
        root / "crates" / "hipfire-runtime" / "src" / "llama.rs",
        "gemv_mq8g256_with_rotate rotate_quantize_x_mq8 gemv_mq8g256_prerotated",
    )
    write(
        root / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs",
        "14 => Some(DType::MQ8G256) "
        "gemm_gate_up_hfq8g256 "
        "gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched "
        "gemv_hfq8g256_moe_gate_up_k8_indexed_batched "
        "gemv_hfq8g256_moe_down_k8_indexed_batched_expanded "
        "moe_prefill_quant_matrix_documents_mq2_mq3_mq4_mq6_mq8 "
        "moe_prefill_admits_gfx1151_scalar_bringup_families "
        "moe_prefill_rejects_mixed_routed_family_without_grouped_gemm",
    )
    write(
        root / "crates" / "rdna-compute" / "src" / "dispatch.rs",
        "rotate_quantize_x_mq8 gemv_mq8g256_prerotated",
    )
    write(root / "crates" / "rdna-compute" / "src" / "kernels.rs", "GEMV_MQ8G256_SRC")


def compiled_blobs(root: Path):
    for name in mq8_status.REQUIRED_BLOBS:
        write(root / name, "blob")


def harness(root: Path):
    write(root / "crates" / "rdna-compute" / "examples" / "bench_mq8.rs", "MQ8 benchmark")


class Mq8StatusTest(unittest.TestCase):
    def test_build_status_keeps_mq8_research_without_role_artifact_or_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            compiled = root / "compiled"
            models = root / "models"
            candidate = root / "candidates"
            source_tree(source)
            compiled_blobs(compiled)
            models.mkdir()
            candidate.mkdir()

            status = mq8_status.build_status(
                source_root=source,
                model_roots=[models],
                candidate_root=candidate,
                compiled_root=compiled,
            )

        self.assertEqual(status["schema"], mq8_status.SCHEMA)
        self.assertEqual(status["status"], "permanent-runtime-research")
        self.assertFalse(status["promotion_allowed"])
        self.assertTrue(status["gates"]["producer_surface_present"])
        self.assertTrue(status["gates"]["runtime_surface_present"])
        self.assertTrue(status["gates"]["gfx1151_compiled_blobs_present"])
        self.assertTrue(status["gates"]["no_gpu_admission_covered"])
        self.assertFalse(status["gates"]["product_role_defined"])
        self.assertFalse(status["gates"]["canonical_mq8_artifact_present"])
        self.assertFalse(status["gates"]["example_or_benchmark_harness_present"])
        self.assertFalse(status["gates"]["candidate_model_benchmark_harness_present"])
        self.assertFalse(status["active_promotion_backlog"])
        self.assertTrue(status["purpose_decision"]["promotion_backlog_closed"])
        self.assertTrue(status["purpose_decision"]["runtime_substrate_maintenance_allowed"])
        self.assertFalse(status["purpose_decision"]["artifact_generation_without_product_role_allowed"])
        self.assertTrue(status["purpose_decision"]["reopen_requires_all_contract_gates"])
        boundary = status["promotion_boundary"]
        self.assertEqual(boundary["promotion_lane_state"], "closed_until_reopen_contract")
        self.assertEqual(boundary["artifact_generation_state"], "blocked_missing_product_role")
        self.assertTrue(boundary["runtime_substrate_maintenance_allowed"])
        self.assertFalse(boundary["candidate_model_baseline_available"])
        self.assertFalse(boundary["reopen_contract_satisfied"])
        self.assertIn("product_role_over_q8_or_mq6", boundary["missing_reopen_requirements"])
        self.assertIn("mq8_example_or_benchmark_harness", boundary["missing_reopen_requirements"])
        self.assertEqual(boundary["next_unblocked_step"], "define_product_role_over_q8_or_mq6")
        closure = status["research_closure"]
        self.assertEqual(closure["classification"], "permanent-runtime-research")
        self.assertTrue(closure["closed_without_product_role"])
        self.assertTrue(closure["artifact_generation_blocked"])
        self.assertTrue(closure["perf_collection_blocked"])
        self.assertTrue(closure["runtime_substrate_maintenance_allowed"])
        self.assertTrue(closure["candidate_artifact_generation_requires_product_role"])
        self.assertTrue(closure["candidate_perf_collection_requires_artifact_and_quality"])
        self.assertTrue(closure["high_bit_value_must_exceed_mq6_or_q8"])
        self.assertEqual(closure["minimum_reopen_requirements"], list(mq8_status.REOPEN_REQUIREMENTS))
        self.assertIn("product_role_over_q8_or_mq6", closure["missing_reopen_requirements"])
        self.assertEqual(closure["next_unblocked_step"], "define_product_role_over_q8_or_mq6")
        self.assertIn("no product role", closure["rationale"])
        self.assertGreater(status["byte_tradeoff"]["mq8_to_mq6_ratio"], 1.0)
        self.assertFalse(status["reopen_contract"]["satisfied"]["mq8_example_or_benchmark_harness"])
        self.assertIn("no MQ8 product role", " ".join(status["blockers"]))

    def test_all_gates_can_become_promotion_ready_when_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            compiled = root / "compiled"
            models = root / "models"
            candidate = root / "candidates"
            source_tree(source)
            harness(source)
            compiled_blobs(compiled)
            models.mkdir()
            candidate.mkdir()
            write(candidate / "qwen3.5-9b-mq8.hfq", "hfq")

            status = mq8_status.build_status(
                source_root=source,
                model_roots=[models],
                candidate_root=candidate,
                compiled_root=compiled,
                product_role="high-precision MQ-family oracle",
                quality_artifacts=["kld.json"],
                perf_artifacts=["perf.json"],
            )

        self.assertEqual(status["status"], "promotion-ready")
        self.assertTrue(status["promotion_allowed"])
        self.assertTrue(status["active_promotion_backlog"])
        self.assertTrue(status["gates"]["example_or_benchmark_harness_present"])
        self.assertTrue(status["gates"]["candidate_model_benchmark_harness_present"])
        boundary = status["promotion_boundary"]
        self.assertEqual(boundary["promotion_lane_state"], "open_for_promotion_review")
        self.assertEqual(boundary["artifact_generation_state"], "allowed_for_reopen_candidate")
        self.assertTrue(boundary["candidate_model_baseline_available"])
        self.assertTrue(boundary["reopen_contract_satisfied"])
        self.assertEqual(boundary["missing_reopen_requirements"], [])
        self.assertEqual(boundary["next_unblocked_step"], "verify_readiness_matrix_before_promotion_claim")
        closure = status["research_closure"]
        self.assertEqual(closure["classification"], "promotion-ready")
        self.assertFalse(closure["closed_without_product_role"])
        self.assertFalse(closure["artifact_generation_blocked"])
        self.assertFalse(closure["perf_collection_blocked"])
        self.assertTrue(closure["candidate_artifact_generation_requires_product_role"])
        self.assertTrue(closure["candidate_perf_collection_requires_artifact_and_quality"])
        self.assertEqual(closure["missing_reopen_requirements"], [])
        self.assertEqual(closure["next_unblocked_step"], "verify_readiness_matrix_before_promotion_claim")
        self.assertEqual(status["blockers"], [])

    def test_role_artifact_and_evidence_still_need_a_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            compiled = root / "compiled"
            models = root / "models"
            candidate = root / "candidates"
            source_tree(source)
            compiled_blobs(compiled)
            models.mkdir()
            candidate.mkdir()
            write(candidate / "qwen3.5-9b-mq8.hfq", "hfq")

            status = mq8_status.build_status(
                source_root=source,
                model_roots=[models],
                candidate_root=candidate,
                compiled_root=compiled,
                product_role="high-precision MQ-family oracle",
                quality_artifacts=["kld.json"],
                perf_artifacts=["perf.json"],
            )

        self.assertFalse(status["gates"]["example_or_benchmark_harness_present"])
        self.assertFalse(status["gates"]["candidate_model_benchmark_harness_present"])
        self.assertFalse(status["promotion_allowed"])
        self.assertEqual(status["promotion_boundary"]["next_unblocked_step"], "add_candidate_model_mq8_harness")
        self.assertIn("benchmark harness", " ".join(status["blockers"]))

    def test_missing_compiled_blob_blocks_even_with_other_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            compiled = root / "compiled"
            models = root / "models"
            candidate = root / "candidates"
            source_tree(source)
            models.mkdir()
            candidate.mkdir()
            write(candidate / "qwen3.5-9b-mq8.hfq", "hfq")

            status = mq8_status.build_status(
                source_root=source,
                model_roots=[models],
                candidate_root=candidate,
                compiled_root=compiled,
                product_role="oracle",
                quality_artifacts=["kld.json"],
                perf_artifacts=["perf.json"],
            )

        self.assertFalse(status["gates"]["gfx1151_compiled_blobs_present"])
        self.assertFalse(status["promotion_allowed"])
        self.assertIn("compiled blobs", " ".join(status["blockers"]))

    def test_current_status_artifact_is_permanent_runtime_research(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], mq8_status.SCHEMA)
        self.assertEqual(payload["status"], "permanent-runtime-research")
        self.assertFalse(payload["promotion_allowed"])
        self.assertFalse(payload["gates"]["product_role_defined"])
        self.assertFalse(payload["gates"]["canonical_mq8_artifact_present"])
        self.assertTrue(payload["gates"]["producer_surface_present"])
        self.assertTrue(payload["gates"]["runtime_surface_present"])
        self.assertTrue(payload["gates"]["no_gpu_admission_covered"])
        self.assertTrue(payload["gates"]["example_or_benchmark_harness_present"])
        self.assertFalse(payload["gates"]["candidate_model_benchmark_harness_present"])
        self.assertFalse(payload["gates"]["quality_evidence_present"])
        self.assertFalse(payload["gates"]["perf_evidence_present"])
        self.assertFalse(payload["active_promotion_backlog"])
        self.assertEqual(payload["purpose_decision"]["classification"], "permanent-runtime-research")
        self.assertTrue(payload["purpose_decision"]["promotion_backlog_closed"])
        self.assertTrue(payload["purpose_decision"]["runtime_substrate_maintenance_allowed"])
        self.assertFalse(payload["purpose_decision"]["artifact_generation_without_product_role_allowed"])
        self.assertTrue(payload["purpose_decision"]["reopen_requires_all_contract_gates"])
        boundary = payload["promotion_boundary"]
        self.assertEqual(boundary["promotion_lane_state"], "closed_until_reopen_contract")
        self.assertEqual(boundary["artifact_generation_state"], "blocked_missing_product_role")
        self.assertTrue(boundary["runtime_substrate_maintenance_allowed"])
        self.assertFalse(boundary["candidate_model_baseline_available"])
        self.assertFalse(boundary["reopen_contract_satisfied"])
        self.assertEqual(boundary["next_unblocked_step"], "define_product_role_over_q8_or_mq6")
        self.assertIn("product_role_over_q8_or_mq6", boundary["missing_reopen_requirements"])
        self.assertIn("mq8_example_or_benchmark_harness", boundary["missing_reopen_requirements"])
        self.assertTrue(boundary["high_bit_justification_required"])
        closure = payload["research_closure"]
        self.assertEqual(closure["classification"], "permanent-runtime-research")
        self.assertTrue(closure["closed_without_product_role"])
        self.assertTrue(closure["artifact_generation_blocked"])
        self.assertTrue(closure["perf_collection_blocked"])
        self.assertTrue(closure["runtime_substrate_maintenance_allowed"])
        self.assertTrue(closure["candidate_artifact_generation_requires_product_role"])
        self.assertTrue(closure["candidate_perf_collection_requires_artifact_and_quality"])
        self.assertTrue(closure["high_bit_value_must_exceed_mq6_or_q8"])
        self.assertEqual(closure["minimum_reopen_requirements"], list(mq8_status.REOPEN_REQUIREMENTS))
        self.assertEqual(closure["next_unblocked_step"], "define_product_role_over_q8_or_mq6")
        self.assertIn("product_role_over_q8_or_mq6", closure["missing_reopen_requirements"])
        self.assertIn("1.29x MQ6", closure["rationale"])
        self.assertIn("1.90x MQ4", closure["rationale"])
        self.assertEqual(payload["reopen_contract"]["requirements"], list(mq8_status.REOPEN_REQUIREMENTS))
        self.assertFalse(payload["reopen_contract"]["satisfied"]["mq8_example_or_benchmark_harness"])
        self.assertTrue(payload["harness"]["present"])
        self.assertFalse(payload["harness"]["candidate_model_harness_present"])
        commits = payload["origin_context"]["local_runtime_context_commits"]
        self.assertEqual([commit["short"] for commit in commits], ["282cc9c5", "94794414", "e038b986", "6bec9f71"])
        self.assertTrue(all(commit["present_on_head"] for commit in commits))
        self.assertTrue(all(commit["present_on_origin_master"] is False for commit in commits))
        self.assertFalse(payload["origin_context"]["upstream_mq8_product_role_commit_found"])


if __name__ == "__main__":
    unittest.main()
