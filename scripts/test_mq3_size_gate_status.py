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
MODULE_PATH = ROOT / "scripts" / "mq3_size_gate_status.py"
CURRENT_ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "gfx1151-quant-readiness"
    / "2026-06-03-mq3-size-gate-status.json"
)


spec = importlib.util.spec_from_file_location("mq3_size_gate_status", MODULE_PATH)
assert spec and spec.loader
mq3_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq3_status
spec.loader.exec_module(mq3_status)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload):
    write(path, json.dumps(payload))


class Mq3SizeGateStatusTest(unittest.TestCase):
    def test_kld_comparison_marks_candidate_worse_than_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kld.json"
            write_json(
                path,
                {
                    "reduced_rows": [
                        {"variant": "qwen3.5-9b.mq3-kvq8-c20", "mean_kld": 0.8, "ppl": 12.5},
                        {"variant": "qwen3.5-9b.mq4-kvq8-c20", "mean_kld": 0.2, "ppl": 9.0},
                    ]
                },
            )

            comparison = mq3_status.kld_comparison(path, "qwen3.5-9b.mq3", "qwen3.5-9b.mq4")

        self.assertFalse(comparison["candidate_beats_control"])
        self.assertEqual(comparison["candidate_to_control_mean_kld_ratio"], 4.0)

    def test_ppl_comparison_detects_candidate_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ppl.json"
            write_json(
                path,
                {
                    "cases": [
                        {"id": "qwen35-27b-mq4", "format_id": "mq4", "result": {"ppl": 12.7}},
                        {"id": "qwen35-27b-mq3", "format_id": "mq3", "result": {"ppl": 12.2}},
                    ]
                },
            )

            comparison = mq3_status.ppl_comparison(path, "qwen35-27b-mq3", "qwen35-27b-mq4")

        self.assertTrue(comparison["candidate_beats_control"])
        self.assertLess(comparison["candidate_to_control_ppl_ratio"], 1.0)

    def test_reference_status_reports_missing_required_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refs.json"
            write_json(
                path,
                {
                    "schema": "hipfire.quant_kld_reference_inventory.v0",
                    "expected_missing_or_required": [
                        {
                            "fixture": "qwen3.5-27b",
                            "local_manifest_ref_sha256_ok": False,
                        },
                        {
                            "fixture": "qwen3.5-35b-a3b",
                            "local_manifest_ref_sha256_ok": False,
                        },
                        {
                            "fixture": "qwen3.6-35b-a3b",
                            "local_manifest_ref_sha256_ok": True,
                        },
                    ],
                },
            )

            refs = mq3_status.reference_status(path)

        self.assertFalse(refs["qwen35_27b_present"])
        self.assertFalse(refs["qwen35_a3b_present"])
        self.assertTrue(refs["qwen36_a3b_present"])

    def test_live_artifact_inventory_normalizes_candidate_and_cache_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "qwen3.5-4b-mq3.hfq",
                "qwen3.5-9b-mq3.hfq",
                "qwen3.5-27b-mq3.hfq",
                "qwen3.5-35b-a3b.mq3",
                "qwen3.6-35b-a3b.mq3",
                "qwen3.5-9b-awq-mtp.mq3",
                "qwen3.5-9b-lloyd-mq3.hfq",
                "qwen35-9b-dflash-mq3.hfq",
            ]:
                (root / name).write_bytes(b"hfq")

            inventory = mq3_status.live_artifact_inventory([root])

        self.assertEqual(inventory["artifact_count"], 6)
        self.assertEqual(inventory["awq_mtp_artifact_count"], 1)
        self.assertTrue(inventory["canonical_required_complete"])
        self.assertEqual(inventory["canonical_required_missing"], [])
        self.assertEqual(
            inventory["canonical_required_present"],
            sorted(mq3_status.REQUIRED_MQ3_ARTIFACTS),
        )

    def test_perf_summary_tracks_token_attractor_and_token_cap_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.json"
            write_json(
                path,
                {
                    "cases": [
                        {
                            "id": "qwen35-27b-mq3-dflash-prose",
                            "max_tokens": 128,
                            "prompt_md5": "p",
                            "runs": [
                                {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}},
                                {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}},
                            ],
                            "summary": {
                                "ok_runs": 2,
                                "total_runs": 2,
                                "median_tokens": 128,
                            },
                        },
                    ]
                },
            )

            summary = mq3_status.perf_summary(path, format_token="qwen35-27b-mq3")

        self.assertTrue(summary["all_matching_cases_ok"])
        self.assertTrue(summary["all_matching_token_attractor_clean"])
        self.assertTrue(summary["all_matching_cases_reached_token_cap"])

    def test_mq3_moe_decode_boundary_records_missing_indexed_route(self):
        evidence = mq3_status.mq3_moe_decode_boundary_evidence()

        self.assertEqual(
            evidence["test_command"],
            mq3_status.MQ3_MOE_DECODE_BOUNDARY_TEST,
        )
        self.assertTrue(evidence["all_phrase_checks_present"])
        self.assertTrue(evidence["boundary_recorded"])
        self.assertFalse(evidence["indexed_route_supported"])

    def test_long_prefill_shape_contract_records_no_gpu_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.md"
            write(
                path,
                """# MQ3 Long-Prefill Path2 Shape Audit

- scope: no-GPU contract for MQ3 A3B grouped MoE long-prefill routing

This audit records coverage. It does not claim artifact-backed promotion evidence.

```text
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
```

```text
qwen35::tests::moe_prefill_mq3_long_prefill_path2_shape_is_production_shaped
```

- MQ3 routed experts remain admitted for `gfx1151` when router and scalar gate stay on Q8.
- MQ3 forces grouped path2 even when `HIPFIRE_MOE_GROUPED_GEMM=0` because no
  indexed MQ3 fallback is wired.
- For a full long-prefill chunk (`N=256`, `K_TOP=8`, `num_experts=256`), the
  grouped scatter shape is:
  - `total_slots = 2048`
  - `m_total_bound = 5888`
  - `m_total_bound % 16 == 0`
- Gate/up grouped GEMM consumes `x_rot_batch [N x dim]` with `x_row_div = 8`.
- Down grouped GEMM consumes `rot_batch [N*K_TOP x mi]` with `x_row_div = 1`.

This is not a promotion-grade runtime result. MQ3 A3B still needs
artifact-backed long-prefill coherence/perf rows.
""",
            )

            contract = mq3_status.long_prefill_shape_contract(path)

        self.assertEqual(contract["status"], "covered_no_gpu_not_runtime")
        self.assertTrue(contract["all_phrase_checks_present"])
        self.assertFalse(contract["promotion_grade_runtime_result"])
        self.assertFalse(contract["artifact_backed_long_prefill_evidence_present"])
        self.assertEqual(contract["test_command"], mq3_status.MQ3_LONG_PREFILL_PATH2_TEST)
        self.assertEqual(contract["test_name"], mq3_status.MQ3_LONG_PREFILL_PATH2_TEST_NAME)
        self.assertEqual(contract["invariants"]["n_tokens"], 256)
        self.assertEqual(contract["invariants"]["k_top"], 8)
        self.assertEqual(contract["invariants"]["num_experts"], 256)
        self.assertEqual(contract["invariants"]["total_slots"], 2048)
        self.assertEqual(contract["invariants"]["m_total_bound"], 5888)
        self.assertTrue(contract["invariants"]["m_total_bound_multiple_of_16"])
        self.assertEqual(contract["invariants"]["gate_up_x_row_div"], 8)
        self.assertEqual(contract["invariants"]["down_x_row_div"], 1)

    def test_current_mq3_size_gate_status_records_incomplete_promotion(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], mq3_status.SCHEMA)
        self.assertEqual(payload["status"], "candidate-size-gated-incomplete")
        self.assertFalse(payload["promotion_allowed"])
        self.assertTrue(payload["gates"]["canonical_mq3_artifacts_present"])
        self.assertTrue(payload["gates"]["live_canonical_mq3_artifacts_present"])
        self.assertTrue(payload["live_artifact_inventory"]["canonical_required_complete"])
        self.assertEqual(payload["live_artifact_inventory"]["canonical_required_missing"], [])
        self.assertEqual(
            payload["live_artifact_inventory"]["canonical_required_present"],
            sorted(mq3_status.REQUIRED_MQ3_ARTIFACTS),
        )
        self.assertEqual(payload["live_artifact_inventory"]["artifact_count"], 23)
        self.assertEqual(payload["live_artifact_inventory"]["candidate_root_artifact_count"], 5)
        self.assertEqual(payload["live_artifact_inventory"]["cache_artifact_count"], 18)
        self.assertEqual(payload["live_artifact_inventory"]["awq_mtp_artifact_count"], 9)
        self.assertTrue(payload["gates"]["dense_4b_rejected"])
        self.assertTrue(payload["gates"]["dense_9b_rejected"])
        self.assertTrue(payload["gates"]["dense_27b_ppl_beats_mq4"])
        self.assertFalse(payload["gates"]["dense_27b_kld_reference_present"])
        self.assertFalse(payload["gates"]["dense_27b_kld_evidence_present"])
        self.assertTrue(payload["gates"]["dense_27b_ar_perf_present"])
        self.assertFalse(payload["gates"]["dense_27b_ar_reached_token_cap"])
        self.assertTrue(payload["gates"]["dense_27b_dflash_perf_present"])
        self.assertTrue(payload["gates"]["dense_27b_dflash_token_attractor_clean"])
        self.assertTrue(payload["gates"]["a3b_qwen35_ppl_beats_mq4"])
        self.assertFalse(payload["gates"]["a3b_qwen36_ppl_beats_mq4"])
        self.assertTrue(payload["gates"]["a3b_broader_coherence_no_hard_errors"])
        self.assertTrue(payload["gates"]["a3b_broader_coherence_capped_rows_present"])
        self.assertFalse(payload["gates"]["a3b_broader_coherence_promotion_grade"])
        self.assertFalse(payload["gates"]["a3b_kld_references_present"])
        self.assertFalse(payload["gates"]["a3b_dflash_sidecars_present"])
        self.assertTrue(payload["gates"]["a3b_moe_decode_boundary_recorded"])
        self.assertFalse(payload["gates"]["a3b_moe_decode_indexed_route_supported"])
        self.assertTrue(payload["gates"]["long_prefill_shape_audit_present"])
        self.assertTrue(payload["gates"]["long_prefill_shape_contract_covered"])
        self.assertFalse(payload["gates"]["long_prefill_runtime_evidence_present"])
        self.assertEqual(
            payload["runtime"]["a3b_moe_decode_boundary"]["test_command"],
            mq3_status.MQ3_MOE_DECODE_BOUNDARY_TEST,
        )
        self.assertEqual(
            payload["runtime"]["a3b_moe_decode_boundary"]["status"],
            "decode_indexed_route_missing_recorded",
        )
        long_prefill = payload["runtime"]["long_prefill_shape_contract"]
        self.assertEqual(long_prefill["status"], "covered_no_gpu_not_runtime")
        self.assertTrue(long_prefill["all_phrase_checks_present"])
        self.assertFalse(long_prefill["artifact_backed_long_prefill_evidence_present"])
        self.assertEqual(long_prefill["invariants"]["n_tokens"], 256)
        self.assertEqual(long_prefill["invariants"]["k_top"], 8)
        self.assertEqual(long_prefill["invariants"]["num_experts"], 256)
        self.assertEqual(long_prefill["invariants"]["total_slots"], 2048)
        self.assertEqual(long_prefill["invariants"]["m_total_bound"], 5888)
        self.assertEqual(long_prefill["invariants"]["gate_up_x_row_div"], 8)
        self.assertEqual(long_prefill["invariants"]["down_x_row_div"], 1)
        boundary = payload["size_gate_boundary"]
        self.assertEqual(boundary["status"], "size_gated_incomplete")
        self.assertEqual(boundary["dense_4b"]["status"], "rejected")
        self.assertEqual(boundary["dense_9b"]["status"], "rejected")
        self.assertEqual(boundary["dense_27b"]["status"], "candidate_incomplete_kld_reference")
        self.assertEqual(boundary["a3b"]["status"], "research_blocked")
        self.assertEqual(
            boundary["a3b"]["coherence_status"],
            "no_hard_errors_but_capped_rows_not_promotion_grade",
        )
        self.assertTrue(boundary["a3b"]["broader_coherence_capped_rows_present"])
        self.assertEqual(boundary["a3b"]["broader_coherence_hit_max_tokens_count"], 7)
        self.assertFalse(boundary["a3b"]["promotion_grade_coherence_present"])
        self.assertEqual(
            boundary["long_prefill_no_gpu_contract"]["status"],
            "covered_no_gpu_not_runtime",
        )
        self.assertFalse(
            boundary["long_prefill_no_gpu_contract"][
                "artifact_backed_long_prefill_evidence_present"
            ]
        )
        self.assertEqual(boundary["moe_decode"]["status"], "decode_indexed_route_missing_recorded")
        self.assertIn("qwen3.5-27B KLD", boundary["next_unblocked_step"])
        self.assertEqual(
            payload["quality_boundary"]["dense_small_model_status"],
            "4b_and_9b_rejected_current_calibration",
        )
        self.assertIn("long-prefill coherence/perf", " ".join(payload["next_work"]))
        self.assertIn("qwen3.5-4b", payload["fixture_decisions"])
        self.assertIn("Qwen3.5 27B MQ3 lacks", " ".join(payload["blockers"]))
        self.assertIn("stopped before the max-token cap", " ".join(payload["blockers"]))
        self.assertIn("no-hard-error smoke only", " ".join(payload["blockers"]))
        self.assertIn("7/12 rows hit the max-token cap", " ".join(payload["blockers"]))
        self.assertIn("MoE decode lacks an indexed routed-expert path", " ".join(payload["blockers"]))
        self.assertIn("long-prefill has no-GPU shape coverage", " ".join(payload["blockers"]))


if __name__ == "__main__":
    unittest.main()
