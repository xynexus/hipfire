#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mq2_lloyd_4w_bench_report.py"
CURRENT_ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "gfx1151-quant-readiness"
    / "2026-06-03-mq2-lloyd-4w-bench.json"
)


spec = importlib.util.spec_from_file_location("mq2_lloyd_4w_bench_report", MODULE_PATH)
assert spec and spec.loader
bench_report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench_report
spec.loader.exec_module(bench_report)


SAMPLE_OUTPUT = """
Arch: gfx1151

=== gate/up B=128 | M=2048 K=4096 batch=128 m_total=768 ===
  correctness: max_abs=0.000e0  max_rel=0.000e0  bad=0/1572864  nan=0  OK
  ref (1w16x16):   1636.1 us (  7700 GFLOPS)   4w (64x16):      1806.4 us (  6974 GFLOPS)   speedup: 0.91x

=== down B=256 | M=4096 K=2048 batch=256 m_total=1536 ===
  correctness: max_abs=0.000e0  max_rel=0.000e0  bad=0/6291456  nan=0  OK
  ref (1w16x16):   3530.3 us (  7301 GFLOPS)   4w (64x16):      3793.8 us (  6794 GFLOPS)   speedup: 0.93x
"""


class Mq2Lloyd4wBenchReportTest(unittest.TestCase):
    def test_parse_benchmark_output_records_correctness_and_speedup(self):
        arch, rows = bench_report.parse_benchmark_output(SAMPLE_OUTPUT)

        self.assertEqual(arch, "gfx1151")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "gate_up_b128")
        self.assertEqual(rows[0]["phase"], "gate_up")
        self.assertEqual(rows[0]["m_total"], 768)
        self.assertEqual(rows[0]["correctness"]["status"], "ok")
        self.assertEqual(rows[0]["correctness"]["bad"], 0)
        self.assertLess(rows[0]["speedup"], 1.0)
        self.assertEqual(rows[1]["id"], "down_b256")

    def test_payload_keeps_4w_opt_in_when_all_rows_are_slower(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = bench_report.build_payload(SAMPLE_OUTPUT, arch="gfx1151", model_roots=[Path(tmp)])

        self.assertEqual(payload["schema"], bench_report.SCHEMA)
        self.assertFalse(payload["summary"]["promote_4w_default"])
        self.assertTrue(payload["summary"]["all_correct"])
        self.assertTrue(payload["summary"]["all_candidate_slower_than_baseline"])
        self.assertFalse(payload["summary"]["default_kernel_change_allowed"])
        self.assertFalse(payload["summary"]["model_level_promotion_allowed"])
        self.assertTrue(payload["summary"]["specialty_research_allowed"])
        self.assertFalse(payload["summary"]["bandwidth_bottleneck_proven"])
        self.assertFalse(payload["model_artifact_inventory"]["mq2_lloyd_artifact_present"])
        self.assertFalse(
            payload["model_artifact_inventory"]["current_a3b_or_deepseek_artifact_present"]
        )
        self.assertEqual(
            payload["summary"]["recommended_next_lever"],
            "batch_selected_experts_across_tokens_before_more_packing",
        )
        self.assertEqual(payload["specialty_scope"]["routed_expert_status"], "specialty-research-only")
        self.assertFalse(payload["specialty_scope"]["model_level_evidence_present"])
        boundary = payload["specialty_boundary"]
        self.assertEqual(boundary["status"], "specialty_blocked_no_model_artifact")
        self.assertEqual(boundary["synthetic_kernel_result"], "correct_but_slower_than_k2")
        self.assertFalse(boundary["current_a3b_or_deepseek_artifact_present"])
        self.assertFalse(boundary["mq2_lloyd_artifact_present"])
        self.assertFalse(boundary["routed_expert_coherence_present"])
        self.assertFalse(boundary["model_level_perf_evidence_present"])
        self.assertTrue(boundary["origin_master_refresh_required"])
        self.assertTrue(boundary["remote_branch_refresh_required"])
        self.assertTrue(boundary["remote_moe_fix_refresh_required"])
        self.assertTrue(boundary["model_evidence_refresh_required"])
        self.assertFalse(boundary["perf_collection_allowed"])
        self.assertFalse(boundary["default_kernel_change_allowed"])
        self.assertTrue(boundary["requires_model_backed_win_over_k2"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "generate_or_locate_a3b_or_deepseek_mq2_lloyd_artifact",
        )
        commits = {
            commit["short"]: commit
            for commit in payload["origin_context"]["relevant_upstream_commits"]
        }
        self.assertIn("89f42e4b", commits)
        self.assertIn("edf922db", commits)
        self.assertIn("fb36f539", commits)
        self.assertEqual(commits["edf922db"]["remote_ref"], "origin/minimax/m2.7-impl")
        self.assertIn("long-context attractor", commits["edf922db"]["subject"])
        self.assertEqual(commits["fb36f539"]["remote_ref"], "origin/fix/hunt3-engine-bugs")
        self.assertIn("sampling+MoE", commits["fb36f539"]["subject"])
        origin = payload["origin_context"]
        self.assertTrue(origin["origin_master_refresh_required"])
        self.assertTrue(origin["remote_branch_refresh_required"])
        self.assertTrue(origin["model_evidence_refresh_required"])
        self.assertIn("fb36f539", origin["remote_branch_missing_commit_shorts"])
        self.assertIn("opt-in/research", payload["summary"]["decision"])

    def test_artifact_inventory_advances_boundary_to_routed_coherence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "qwen3.5-35b-a3b.mq2lloyd.hfq"
            artifact.write_bytes(b"hfq")

            payload = bench_report.build_payload(SAMPLE_OUTPUT, arch="gfx1151", model_roots=[root])

        inventory = payload["model_artifact_inventory"]
        self.assertTrue(inventory["mq2_lloyd_artifact_present"])
        self.assertTrue(inventory["current_a3b_or_deepseek_artifact_present"])
        self.assertEqual(inventory["current_a3b_or_deepseek_artifacts"][0]["name"], artifact.name)
        boundary = payload["specialty_boundary"]
        self.assertEqual(boundary["status"], "specialty_blocked_no_routed_coherence")
        self.assertTrue(boundary["current_a3b_or_deepseek_artifact_present"])
        self.assertTrue(boundary["mq2_lloyd_artifact_present"])
        self.assertFalse(boundary["routed_expert_coherence_present"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "run_routed_expert_coherence",
        )

    def test_markdown_writer_preserves_decision_and_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = bench_report.build_payload(SAMPLE_OUTPUT, arch="gfx1151", model_roots=[Path(tmp)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.md"
            bench_report.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("MQ2-Lloyd 4-Warp Kernel A/B", text)
        self.assertIn("opt-in/research", text)
        self.assertIn("specialty-research-only", text)
        self.assertIn("batch_selected_experts_across_tokens_before_more_packing", text)
        self.assertIn("gate/up B=128", text)
        self.assertIn("0.9057", text)

    def test_current_structured_artifact_is_non_promoting(self):
        payload = bench_report.json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], bench_report.SCHEMA)
        self.assertEqual(payload["summary"]["row_count"], 6)
        self.assertTrue(payload["summary"]["all_correct"])
        self.assertTrue(payload["summary"]["all_candidate_slower_than_baseline"])
        self.assertFalse(payload["summary"]["promote_4w_default"])
        self.assertFalse(payload["summary"]["default_kernel_change_allowed"])
        self.assertFalse(payload["summary"]["model_level_promotion_allowed"])
        self.assertFalse(payload["summary"]["bandwidth_bottleneck_proven"])
        self.assertFalse(payload["promotion_allowed"])
        self.assertEqual(payload["status"], "specialty_blocked_no_model_artifact")
        self.assertFalse(payload["model_artifact_inventory"]["mq2_lloyd_artifact_present"])
        self.assertFalse(
            payload["model_artifact_inventory"]["current_a3b_or_deepseek_artifact_present"]
        )
        self.assertEqual(payload["specialty_scope"]["dense_text_status"], "rejected_by_mq2_dense_quality_gate")
        self.assertEqual(payload["specialty_scope"]["routed_expert_status"], "specialty-research-only")
        self.assertFalse(payload["specialty_scope"]["model_level_evidence_present"])
        boundary = payload["specialty_boundary"]
        self.assertEqual(boundary["status"], "specialty_blocked_no_model_artifact")
        self.assertEqual(boundary["synthetic_kernel_result"], "correct_but_slower_than_k2")
        self.assertFalse(boundary["current_a3b_or_deepseek_artifact_present"])
        self.assertFalse(boundary["mq2_lloyd_artifact_present"])
        self.assertFalse(boundary["routed_expert_coherence_present"])
        self.assertFalse(boundary["model_level_perf_evidence_present"])
        self.assertTrue(boundary["remote_moe_fix_refresh_required"])
        self.assertTrue(boundary["model_evidence_refresh_required"])
        self.assertFalse(boundary["perf_collection_allowed"])
        self.assertFalse(boundary["default_kernel_change_allowed"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "generate_or_locate_a3b_or_deepseek_mq2_lloyd_artifact",
        )
        commits = {
            commit["short"]: commit
            for commit in payload["origin_context"]["relevant_upstream_commits"]
        }
        self.assertTrue(commits["89f42e4b"]["present_on_origin_master"])
        self.assertFalse(commits["89f42e4b"]["present_on_head"])
        self.assertTrue(commits["edf922db"]["present_on_remote_ref"])
        self.assertFalse(commits["edf922db"]["present_on_origin_master"])
        self.assertTrue(commits["fb36f539"]["present_on_remote_ref"])
        self.assertFalse(commits["fb36f539"]["present_on_origin_master"])
        self.assertFalse(commits["fb36f539"]["present_on_head"])
        self.assertIn(
            "fb36f539",
            payload["origin_context"]["remote_branch_missing_commit_shorts"],
        )
        self.assertIn(
            "batch_selected_experts_across_tokens",
            payload["specialty_scope"]["next_experiments"],
        )
        self.assertEqual({row["correctness"]["status"] for row in payload["rows"]}, {"ok"})
        self.assertLess(payload["summary"]["max_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
