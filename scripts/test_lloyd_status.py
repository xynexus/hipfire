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
MODULE_PATH = ROOT / "scripts" / "lloyd_status.py"
CURRENT_ARTIFACT = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness" / "2026-06-03-lloyd-status.json"


spec = importlib.util.spec_from_file_location("lloyd_status", MODULE_PATH)
assert spec and spec.loader
lloyd_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lloyd_status
spec.loader.exec_module(lloyd_status)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload):
    write(path, json.dumps(payload))


def provenance(format_id: str, artifact_name: str):
    return {
        "schema": "hipfire.quant_readiness.gfx1151.v0",
        "formats": [
            {
                "id": format_id,
                "status": "candidate-research-gated",
                "candidate_artifacts": {
                    "artifacts": [
                        {
                            "name": artifact_name,
                            "path": f"/models/{artifact_name}",
                            "size_bytes": 123,
                            "hashes": {"sha256": "abc", "md5": "def"},
                        }
                    ]
                },
            }
        ],
    }


def kld_payload(rows):
    return {
        "schema": "test.kld",
        "reduced_rows": rows,
    }


class LloydStatusTest(unittest.TestCase):
    def test_current_status_records_mq3_kld_reject_and_mq4_coherence_reject(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))
        mq3 = payload["formats"]["mq3-lloyd"]
        mq4 = payload["formats"]["mq4-lloyd"]

        self.assertEqual(payload["schema"], lloyd_status.SCHEMA)
        self.assertEqual(mq3["status"], "dense-quality-rejected-current-9b")
        self.assertFalse(mq3["promotion_allowed"])
        self.assertTrue(mq3["gates"]["canonical_9b_artifact_present"])
        self.assertTrue(mq3["gates"]["coherence_9b_clean"])
        self.assertTrue(mq3["gates"]["current_kld_present"])
        self.assertTrue(mq3["gates"]["current_kld_finite"])
        self.assertFalse(mq3["gates"]["current_kld_beats_mq4"])
        self.assertTrue(mq3["gates"]["current_kld_loses_to_mq4"])
        self.assertGreater(mq3["current_kld"]["candidate_to_control_mean_kld_ratio"], 2.0)
        self.assertTrue(mq3["size_scope"]["only_9b_artifact_present"])
        self.assertEqual(mq3["size_scope"]["live_artifact_count"], 2)
        self.assertTrue(mq3["size_scope"]["live_only_9b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["dense_4b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["live_dense_4b_artifact_present"])
        self.assertTrue(mq3["size_scope"]["dense_9b_artifact_present"])
        self.assertTrue(mq3["size_scope"]["live_dense_9b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["dense_27b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["live_dense_27b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["a3b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["live_a3b_artifact_present"])
        self.assertFalse(mq3["size_scope"]["perf_evidence_allowed"])
        self.assertEqual(mq3["live_artifact_inventory"]["artifact_count"], 2)
        self.assertTrue(mq3["live_artifact_inventory"]["only_9b_artifact_present"])
        self.assertTrue(mq3["size_scope"]["historical_only_beats_uniform_mq3"])
        self.assertTrue(mq3["gates"]["historical_beats_uniform_mq3"])
        self.assertFalse(mq3["gates"]["historical_beats_mq4"])
        self.assertFalse(mq3["gates"]["historical_beats_mq6"])
        self.assertTrue(mq3["gates"]["origin_refresh_required"])
        self.assertIn("2.34x MQ4 control", " ".join(mq3["blockers"]))
        self.assertIn("only the 9B MQ3-Lloyd artifact exists", " ".join(mq3["blockers"]))
        self.assertIn("live local scan also found only 9B MQ3-Lloyd artifacts", " ".join(mq3["blockers"]))
        self.assertIn("origin/master has MQ3-Lloyd-relevant commits missing from HEAD", " ".join(mq3["blockers"]))

        self.assertEqual(mq4["status"], "coherence-rejected-current-9b")
        self.assertFalse(mq4["promotion_allowed"])
        self.assertTrue(mq4["gates"]["canonical_9b_artifact_present"])
        self.assertEqual(mq4["live_artifact_inventory"]["artifact_count"], 2)
        self.assertTrue(mq4["artifact_scope"]["only_9b_artifact_present"])
        self.assertFalse(mq4["artifact_scope"]["a3b_artifact_present"])
        self.assertTrue(mq4["gates"]["container_qtype_named"])
        self.assertTrue(mq4["gates"]["container_bounded"])
        self.assertFalse(mq4["gates"]["coherence_9b_clean"])
        self.assertIn("!!!!!!!!!!!", mq4["coherence"]["forbidden_tokens_present"])
        self.assertFalse(mq4["gates"]["historical_best_beats_current_mq4"])
        self.assertFalse(mq4["gates"]["historical_best_beats_mq6"])
        self.assertTrue(mq4["gates"]["current_same_run_kld_present"])
        self.assertFalse(mq4["gates"]["current_same_run_candidate_valid"])
        self.assertTrue(mq4["gates"]["current_same_run_invalid_zero_kld_no_ppl"])
        self.assertFalse(mq4["gates"]["current_same_run_beats_mq4"])
        self.assertFalse(mq4["gates"]["current_same_run_beats_mq6"])
        boundary = mq4["value_boundary"]
        self.assertEqual(boundary["bytes_per_group"], 160)
        self.assertEqual(boundary["status"], "value_not_justified_current_artifact")
        self.assertFalse(boundary["coherence_gate_passed"])
        self.assertFalse(boundary["same_run_kld_ppl_valid"])
        self.assertTrue(boundary["invalid_zero_kld_no_ppl_blocks_quality"])
        self.assertFalse(boundary["beats_mq4_same_run"])
        self.assertFalse(boundary["beats_mq6_same_run"])
        self.assertFalse(boundary["historical_rows_justify_value"])
        self.assertTrue(boundary["origin_refresh_required"])
        self.assertFalse(boundary["branch_reconciled_for_promotion"])
        self.assertFalse(boundary["perf_collection_allowed"])
        self.assertEqual(boundary["next_unblocked_step"], "produce_new_coherent_mq4_lloyd_artifact")
        contract = mq4["promotion_contract"]
        self.assertFalse(contract["promotion_allowed"])
        self.assertIn("current_same_run_invalid_zero_kld_no_ppl", contract["forbidden_gates"])
        self.assertTrue(contract["current_same_run_invalid_zero_kld_no_ppl_must_be_false"])
        self.assertTrue(contract["historical_rows_required_for_promotion"] is False)
        self.assertFalse(contract["forbidden_clear"]["current_same_run_invalid_zero_kld_no_ppl"])
        self.assertFalse(contract["forbidden_clear"]["origin_refresh_required"])
        self.assertFalse(contract["required_satisfied"]["non_9b_artifact_present"])
        self.assertFalse(contract["required_satisfied"]["coherence_9b_clean"])
        self.assertFalse(contract["required_satisfied"]["current_same_run_candidate_valid"])
        repair = mq4["producer_repair_plan"]
        self.assertEqual(repair["status"], "blocked_before_perf")
        self.assertEqual(repair["current_artifact_status"], "coherence_rejected")
        self.assertTrue(repair["requires_new_artifact_hash"])
        self.assertTrue(repair["requires_origin_reconciliation"])
        self.assertIn("d5985c3e", repair["origin_commits_to_reconcile"])
        self.assertFalse(repair["same_run_quality_ready"])
        self.assertFalse(repair["perf_collection_allowed"])
        self.assertEqual(repair["next_unblocked_step"], "produce_new_coherent_mq4_lloyd_artifact")
        self.assertIn("python3 scripts/lloyd_status.py", " ".join(repair["rerun_commands"]))
        self.assertIn("./scripts/coherence-gate.sh --full", repair["rerun_commands"])
        self.assertIn("Produce a new coherent MQ4-Lloyd", mq4["next_work"][0])
        self.assertIn("finite PPL plus wins over MQ4 and MQ6", " ".join(mq4["next_work"]))
        self.assertIn("invalid zero-KLD/NaN-PPL", " ".join(mq4["blockers"]))
        self.assertIn("live local scan found only 9B MQ4-Lloyd artifacts", " ".join(mq4["blockers"]))
        self.assertTrue(mq4["gates"]["origin_refresh_required"])
        self.assertIn("origin/master has MQ4-Lloyd-relevant commits missing from HEAD", " ".join(mq4["blockers"]))
        self.assertTrue(payload["origin_context"]["lloyd_evidence_refresh_required"])
        self.assertTrue(payload["origin_context"]["format_refresh_required"]["mq3-lloyd"])
        self.assertTrue(payload["origin_context"]["format_refresh_required"]["mq4-lloyd"])
        commits = payload["origin_context"]["relevant_upstream_commits"]
        self.assertEqual(commits[0]["short"], "d5985c3e")
        self.assertTrue(commits[0]["present_on_origin_master"])
        self.assertFalse(commits[0]["present_on_head"])
        self.assertIn("producer-flow fix only", commits[0]["promotion_effect"])
        self.assertEqual(commits[2]["short"], "46898ecd")
        self.assertTrue(commits[2]["present_on_origin_master"])
        self.assertFalse(commits[2]["present_on_head"])
        self.assertIn("current MQ3-Lloyd KLD still loses to MQ4", commits[2]["promotion_effect"])
        self.assertFalse(mq4["promotion_allowed"])

    def test_mq4_value_boundary_blocks_perf_when_origin_refresh_required(self):
        boundary = lloyd_status.mq4_lloyd_value_boundary(
            gates={
                "container_qtype_named": True,
                "container_bounded": True,
                "coherence_9b_clean": True,
                "historical_best_beats_current_mq4": True,
                "historical_best_beats_mq6": True,
                "current_same_run_kld_present": True,
                "current_same_run_candidate_valid": True,
                "current_same_run_invalid_zero_kld_no_ppl": False,
                "current_same_run_beats_mq4": True,
                "current_same_run_beats_mq6": True,
                "perf_evidence_present": False,
                "origin_refresh_required": True,
            }
        )

        self.assertFalse(boundary["perf_collection_allowed"])
        self.assertTrue(boundary["origin_refresh_required"])
        self.assertFalse(boundary["branch_reconciled_for_promotion"])
        self.assertEqual(boundary["next_unblocked_step"], "reconcile_origin_master_and_rerun_lloyd_evidence")

    def test_mq4_promotion_contract_treats_invalid_zero_as_forbidden_not_required(self):
        gates = {
            "canonical_9b_artifact_present": True,
            "non_9b_artifact_present": True,
            "container_qtype_named": True,
            "container_bounded": True,
            "coherence_9b_clean": True,
            "historical_best_beats_current_mq4": False,
            "historical_best_beats_mq6": False,
            "current_same_run_kld_present": True,
            "current_same_run_candidate_valid": True,
            "current_same_run_invalid_zero_kld_no_ppl": False,
            "current_same_run_beats_mq4": True,
            "current_same_run_beats_mq6": True,
            "perf_evidence_present": True,
            "origin_refresh_required": False,
        }

        self.assertTrue(lloyd_status.mq4_lloyd_promotion_allowed(gates))

        invalid = {**gates, "current_same_run_invalid_zero_kld_no_ppl": True}
        self.assertFalse(lloyd_status.mq4_lloyd_promotion_allowed(invalid))

    def test_mq4_producer_repair_plan_blocks_perf_until_new_coherent_artifact(self):
        origin = {
            "format_refresh_required": {"mq4-lloyd": True},
            "relevant_upstream_commits": [
                {
                    "short": "d5985c3e",
                    "formats": ["mq4-lloyd", "mq6"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                    "promotion_effect": "producer-flow fix only",
                },
                {
                    "short": "89f42e4b",
                    "formats": ["mq3-lloyd", "mq4-lloyd"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                    "promotion_effect": "runtime routing cleanup",
                },
            ],
        }
        gates = {
            "coherence_9b_clean": False,
            "current_same_run_candidate_valid": False,
            "current_same_run_invalid_zero_kld_no_ppl": True,
            "current_same_run_beats_mq4": False,
            "current_same_run_beats_mq6": False,
            "origin_refresh_required": True,
            "perf_evidence_present": False,
        }
        value_boundary = lloyd_status.mq4_lloyd_value_boundary(gates={
            **gates,
            "container_qtype_named": True,
            "container_bounded": True,
            "historical_best_beats_current_mq4": False,
            "historical_best_beats_mq6": False,
        })

        plan = lloyd_status.mq4_lloyd_producer_repair_plan(
            gates=gates,
            value_boundary=value_boundary,
            origin=origin,
        )

        self.assertEqual(plan["status"], "blocked_before_perf")
        self.assertEqual(plan["next_unblocked_step"], "produce_new_coherent_mq4_lloyd_artifact")
        self.assertTrue(plan["requires_new_artifact_hash"])
        self.assertTrue(plan["requires_origin_reconciliation"])
        self.assertFalse(plan["perf_collection_allowed"])
        self.assertFalse(plan["same_run_quality_ready"])
        self.assertIn("d5985c3e", plan["origin_commits_to_reconcile"])
        self.assertIn("89f42e4b", plan["origin_commits_to_reconcile"])
        self.assertIn("python3 scripts/lloyd_status.py", " ".join(plan["rerun_commands"]))
        self.assertIn("./scripts/coherence-gate.sh --full", plan["rerun_commands"])
        self.assertIn("python3 scripts/test_lloyd_status.py", plan["validation_commands"])

    def test_mq3_current_kld_gate_can_pass_only_when_candidate_beats_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kld = root / "mq3.json"
            write_json(
                kld,
                kld_payload(
                    [
                        {"variant": "qwen3.5-9b.mq3-lloyd-kvq8-c20", "mean_kld": 0.1},
                        {"variant": "qwen3.5-9b.mq4-kvq8-c20", "mean_kld": 0.2},
                    ]
                ),
            )

            summary = lloyd_status.current_kld_summary(kld, "mq3-lloyd")

        self.assertTrue(summary["candidate_beats_control"])
        self.assertEqual(summary["candidate_to_control_mean_kld_ratio"], 0.5)

    def test_mq4_current_same_run_zero_kld_without_ppl_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mq4-lloyd-current.json"
            write_json(
                path,
                kld_payload(
                    [
                        {"variant": "qwen3.5-9b.mq4-kvq8-c20", "mean_kld": 0.2, "ppl": 9.0},
                        {"variant": "qwen3.5-9b.mq4-lloyd-kvq8-c20", "mean_kld": 0.0, "ppl": None},
                        {"variant": "qwen3.5-9b.mq6-kvq8-c20", "mean_kld": 0.05, "ppl": 9.2},
                    ]
                ),
            )

            summary = lloyd_status.current_mq4_lloyd_same_run_summary(path)

        self.assertTrue(summary["same_run_present"])
        self.assertFalse(summary["candidate_valid"])
        self.assertTrue(summary["candidate_invalid_zero_kld_no_ppl"])
        self.assertFalse(summary["candidate_beats_mq4"])
        self.assertFalse(summary["candidate_beats_mq6"])

    def test_mq4_token_attractor_blocks_coherence_even_with_valid_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "mq3_provenance": root / "mq3-prov.json",
                "mq4_provenance": root / "mq4-prov.json",
                "mq3_coherence": root / "mq3-coherence.md",
                "mq4_coherence": root / "mq4-coherence.md",
                "mq3_kld": root / "mq3-kld.json",
                "mq6_c512_kld": root / "mq6-kld.json",
                "mq3_historical_kld": root / "mq3-historical.json",
                "mq4_historical_kld": root / "mq4-historical.json",
                "mq4_container_audit": root / "container.md",
                "mq4_current_kld": root / "mq4-current.json",
            }
            write_json(paths["mq3_provenance"], provenance("mq3-lloyd", "qwen3.5-9b-lloyd-mq3.hfq"))
            write_json(paths["mq4_provenance"], provenance("mq4-lloyd", "qwen3.5-9b-lloyd-mq4.hfq"))
            write(paths["mq3_coherence"], "qwen3.5-9b.mq3-lloyd Final Number 9<|im_end|> O(1)")
            write(paths["mq4_coherence"], "qwen3.5-9b.mq4-lloyd\\n!!!!!!!!!!!")
            write_json(
                paths["mq3_kld"],
                kld_payload(
                    [
                        {"variant": "qwen3.5-9b.mq3-lloyd-kvq8-c20", "mean_kld": 0.1},
                        {"variant": "qwen3.5-9b.mq4-kvq8-c20", "mean_kld": 0.2},
                    ]
                ),
            )
            write_json(
                paths["mq6_c512_kld"],
                kld_payload(
                    [
                        {"variant": "qwen3.5-9b.mq4-kvq8-c512", "mean_kld": 0.2},
                        {"variant": "qwen3.5-9b.mq6-kvq8-c512", "mean_kld": 0.05},
                    ]
                ),
            )
            write_json(
                paths["mq3_historical_kld"],
                [
                    {"variant": "qwen3.5-9b.mq3-lloyd", "mean_kld": 0.4},
                    {"variant": "qwen3.5-9b.mq3", "mean_kld": 0.5},
                    {"variant": "qwen3.5-9b.mq4", "mean_kld": 0.3},
                    {"variant": "qwen3.5-9b.mq6", "mean_kld": 0.2},
                ],
            )
            write_json(
                paths["mq4_historical_kld"],
                [
                    {"variant": "qwen3.5-9b.mq4-lloyd-kvq8-c512", "mean_kld": 0.3},
                    {"variant": "qwen3.5-9b.mq6-q8conv1d-kvq8-c512", "mean_kld": 0.05},
                ],
            )
            write_json(
                paths["mq4_current_kld"],
                kld_payload(
                    [
                        {"variant": "qwen3.5-9b.mq4-kvq8-c20", "mean_kld": 0.2, "ppl": 9.0},
                        {"variant": "qwen3.5-9b.mq4-lloyd-kvq8-c20", "mean_kld": 0.0, "ppl": None},
                        {"variant": "qwen3.5-9b.mq6-kvq8-c20", "mean_kld": 0.05, "ppl": 9.2},
                    ]
                ),
            )
            write(paths["mq4_container_audit"], "MQ4G256_LLOYD MQ3G256_LLOYD data_end equals the file size !!!!!!!!!!!")

            status = lloyd_status.build_status(paths, model_roots=[root / "models"])

        mq4 = status["formats"]["mq4-lloyd"]
        self.assertFalse(mq4["gates"]["coherence_9b_clean"])
        self.assertFalse(mq4["promotion_allowed"])
        self.assertEqual(
            mq4["value_boundary"]["next_unblocked_step"],
            "produce_new_coherent_mq4_lloyd_artifact",
        )
        self.assertFalse(mq4["value_boundary"]["perf_collection_allowed"])
        repair = mq4["producer_repair_plan"]
        self.assertEqual(repair["status"], "blocked_before_perf")
        self.assertTrue(repair["requires_new_artifact_hash"])
        self.assertTrue(repair["requires_origin_reconciliation"])
        self.assertFalse(repair["same_run_quality_ready"])
        self.assertFalse(repair["perf_collection_allowed"])
        self.assertIn("d5985c3e", repair["origin_commits_to_reconcile"])
        self.assertIn("token attractor", " ".join(mq4["blockers"]))
        self.assertEqual(status["origin_context"]["relevant_upstream_commits"][0]["short"], "d5985c3e")

    def test_live_artifact_inventory_tracks_size_and_a3b_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "qwen3.5-9b.mq3-lloyd").write_bytes(b"9b")
            (root / "qwen3.5-27b.mq3-lloyd").write_bytes(b"27b")
            (root / "qwen3.5-35b-a3b.mq4-lloyd").write_bytes(b"a3b")

            inventory = lloyd_status.lloyd_artifact_inventory([root])

        mq3 = inventory["formats"]["mq3-lloyd"]
        self.assertEqual(mq3["artifact_count"], 2)
        self.assertFalse(mq3["only_9b_artifact_present"])
        self.assertTrue(mq3["dense_9b_artifact_present"])
        self.assertTrue(mq3["dense_27b_artifact_present"])
        self.assertFalse(mq3["a3b_artifact_present"])
        mq4 = inventory["formats"]["mq4-lloyd"]
        self.assertEqual(mq4["artifact_count"], 1)
        self.assertTrue(mq4["a3b_artifact_present"])
        self.assertFalse(mq4["only_9b_artifact_present"])

    def test_mq4_value_boundary_requires_direct_mq4_and_mq6_wins_before_perf(self):
        boundary = lloyd_status.mq4_lloyd_value_boundary(
            gates={
                "container_qtype_named": True,
                "container_bounded": True,
                "coherence_9b_clean": True,
                "historical_best_beats_current_mq4": False,
                "historical_best_beats_mq6": False,
                "current_same_run_kld_present": True,
                "current_same_run_candidate_valid": True,
                "current_same_run_invalid_zero_kld_no_ppl": False,
                "current_same_run_beats_mq4": True,
                "current_same_run_beats_mq6": False,
                "perf_evidence_present": False,
            }
        )

        self.assertEqual(boundary["status"], "value_not_justified_current_artifact")
        self.assertFalse(boundary["perf_collection_allowed"])
        self.assertEqual(boundary["next_unblocked_step"], "prove_same_run_kld_beats_mq6")

        boundary = lloyd_status.mq4_lloyd_value_boundary(
            gates={
                "container_qtype_named": True,
                "container_bounded": True,
                "coherence_9b_clean": True,
                "historical_best_beats_current_mq4": False,
                "historical_best_beats_mq6": False,
                "current_same_run_kld_present": True,
                "current_same_run_candidate_valid": True,
                "current_same_run_invalid_zero_kld_no_ppl": False,
                "current_same_run_beats_mq4": True,
                "current_same_run_beats_mq6": True,
                "perf_evidence_present": False,
            }
        )

        self.assertEqual(boundary["status"], "quality_value_candidate")
        self.assertTrue(boundary["perf_collection_allowed"])
        self.assertEqual(boundary["next_unblocked_step"], "run_fresh_process_gfx1151_perf_baselines")


if __name__ == "__main__":
    unittest.main()
