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
MODULE_PATH = ROOT / "scripts" / "paro_q4g128_status.py"
CURRENT_ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "gfx1151-quant-readiness"
    / "2026-06-03-paro-q4g128-productization-status.json"
)


spec = importlib.util.spec_from_file_location("paro_q4g128_status", MODULE_PATH)
assert spec and spec.loader
paro_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paro_status
spec.loader.exec_module(paro_status)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_evidence(root: Path, *names: str) -> list[str]:
    paths = []
    for name in names:
        path = root / name
        path.write_text("ok\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def probe_payload(*, model: str, complete_modules: int) -> dict:
    return {
        "schema": "hipfire.astrea.paro_probe.v0",
        "model": model,
        "resolved_path": model,
        "architecture": "qwen3_5",
        "tensor_count": 12,
        "paro": {
            "complete_module_count": complete_modules,
            "incomplete_module_count": 0,
            "modules": [],
            "incomplete": [],
            "suffix_counts": (
                {
                    "qweight": complete_modules,
                    "qzeros": complete_modules,
                    "scales": complete_modules,
                    "pairs": complete_modules,
                    "theta": complete_modules,
                    "channel_scales": complete_modules,
                }
                if complete_modules > 0
                else {"other": 12}
            ),
        },
        "runtime_contract": {
            "group_size": 128,
            "hfq_quant_type": 28,
            "hfq_quant_type_name": "PARO4G128",
        },
    }


def write_expected_probes(root: Path, *, complete_modules: int) -> list[Path]:
    probes = [
        (root / "qwen35-9b.json", "/models/Qwen3.5-9B/snapshots/x"),
        (root / "qwen35-a3b.json", "/models/Qwen3.5-35B-A3B/snapshots/x"),
        (root / "qwen36-a3b.json", "/models/Qwen3.6-35B-A3B/snapshots/x"),
    ]
    paths = []
    for path, model in probes:
        write_json(path, probe_payload(model=model, complete_modules=complete_modules))
        paths.append(path)
    return paths


def bundle_payload(*, weights_exist: bool = False) -> dict:
    return {
        "schema": "hipfire.astrea.bundle_plan.v0",
        "bundle_id": "test-paro",
        "container": {"format": "hfq-package-v0"},
        "external_sidecars_target": False,
        "sections": {
            "weights": {
                "source": {
                    "path": "/tmp/test.hfq",
                    "exists": weights_exist,
                    "is_file": weights_exist,
                }
            },
            "transform.paro": {
                "runtime_status": "deferred_until_loader_and_fused_kernel_exist",
                "runtime_env_boundary": {
                    "productization_candidate": [
                        {"name": "HIPFIRE_PARO_BATCHED"},
                        {"name": "HIPFIRE_MOE_PARO_I8"},
                        {"name": "HIPFIRE_MOE_PARO_I8_K8"},
                    ],
                    "promotion_report_requirements": [
                        "record oracle, finite-logit/NaN, coherence, KLD/PPL, and gfx1151 perf artifacts before promotion"
                    ],
                    "research_only": [
                        "HIPFIRE_PARO_PREROTATE",
                        "HIPFIRE_PARO_FUSE_RMSNORM",
                        "HIPFIRE_PARO_FUSED_PACK2",
                    ],
                },
            },
        },
    }


def source_inventory_payload(*, native: bool = False, g128_count: int = 0) -> dict:
    return {
        "schema": "hipfire.astrea.paro_source_inventory.v0",
        "files_seen": 3,
        "safetensor_dirs_scanned": 1,
        "safetensor_files_scanned": 1,
        "filename_hit_count": 0,
        "roots": [{"root": "/models", "exists": True}],
        "native_paro": {
            "complete_module_count": g128_count,
            "incomplete_module_count": 0,
            "g128_complete_module_count": g128_count,
            "g256_complete_module_count": 0,
        },
        "decision": {
            "native_paro_source_found": native,
            "native_paro_g256_source_found": False,
            "quality_state": "verifiable" if native else "UNVERIFIABLE",
        },
    }


class ParoQ4G128StatusTest(unittest.TestCase):
    def test_build_status_blocks_when_native_source_and_import_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            write_json(probe, probe_payload(model="/models/Qwen3.5-9B/snapshots/x", complete_modules=0))
            write_json(bundle, bundle_payload(weights_exist=False))

            status = paro_status.build_status([probe], bundle)

        self.assertEqual(status["schema"], paro_status.SCHEMA)
        self.assertEqual(status["status"], "candidate-productization")
        self.assertFalse(status["promotion_allowed"])
        self.assertFalse(status["gates"]["native_paro_source_found"])
        self.assertTrue(status["gates"]["source_inventory_present"])
        self.assertTrue(status["gates"]["source_inventory_consistent"])
        self.assertFalse(status["gates"]["imported_hfq_exists"])
        self.assertTrue(status["gates"]["runtime_env_boundary_recorded"])
        self.assertTrue(status["gates"]["package_contract_present"])
        self.assertFalse(status["gates"]["oracle_evidence_present"])
        self.assertFalse(status["gates"]["coherence_evidence_present"])
        self.assertFalse(status["gates"]["finite_logit_nan_evidence_present"])
        self.assertFalse(status["gates"]["quality_evidence_present"])
        self.assertFalse(status["gates"]["gfx1151_perf_evidence_present"])
        self.assertFalse(status["gates"]["typed_evidence_files_exist"])
        self.assertFalse(status["gates"]["oracle_quality_perf_evidence_present"])
        self.assertIn("qweight/qzeros/scales/pairs/theta/channel_scales", " ".join(status["blockers"]))
        coverage = status["source_probe_coverage"]
        self.assertEqual(coverage["required_native_tensor_families"], list(paro_status.REQUIRED_PARO_SUFFIXES))
        self.assertTrue(coverage["all_required_suffixes_absent_across_expected_probes"])
        deps = status["dependency_graph"]
        self.assertTrue(deps["native_source_absent"])
        self.assertTrue(deps["import_blocked_by_source_absence"])
        self.assertTrue(deps["oracle_blocked_by_import_absence"])
        self.assertFalse(deps["nodes"]["typed_evidence_files"]["satisfied"])
        self.assertTrue(deps["all_required_suffixes_absent_across_expected_probes"])
        self.assertEqual(deps["next_unblocked_step"], "locate_or_generate_native_paro_q4g128_checkpoint")
        boundary = status["productization_boundary"]
        self.assertEqual(boundary["current_stage"], "blocked_before_native_source_import")
        self.assertEqual(boundary["importer_state"], "blocked_no_native_paro_source")
        self.assertEqual(boundary["package_state"], "contract_only_missing_weights")
        self.assertEqual(boundary["astrea_state"], "bundle_plan_contract_only_not_model_writer")
        self.assertEqual(boundary["main_path_state"], "clean_but_unpromoted")
        self.assertFalse(boundary["hidden_research_env_dependency_present"])
        self.assertTrue(boundary["research_only_env_blocks_promotion"])
        self.assertTrue(boundary["promoted_main_path_requires_typed_evidence"])
        self.assertTrue(
            boundary["native_source_contract"]["all_required_suffixes_absent_across_expected_probes"]
        )
        plan = status["productization_plan"]
        self.assertEqual(plan["status"], "blocked_before_native_source_import")
        self.assertEqual(plan["next_unblocked_step"], "locate_or_generate_native_paro_q4g128_checkpoint")
        self.assertEqual(
            plan["stage_order"][:4],
            ["source_inventory", "native_source", "source_probe", "imported_hfq"],
        )
        self.assertIn("scripts/paroquant_inventory.py", plan["stages"]["source_inventory"]["command"])
        self.assertFalse(plan["stages"]["native_source"]["satisfied"])
        self.assertEqual(
            plan["stages"]["native_source"]["required_tensor_families"],
            list(paro_status.REQUIRED_PARO_SUFFIXES),
        )
        self.assertEqual(plan["stages"]["imported_hfq"]["blocked_by"], ["native_source"])
        self.assertIn("paro-import", plan["stages"]["imported_hfq"]["command_template"])
        self.assertIn("paro-oracle", plan["stages"]["paro_oracle"]["command_template"])
        self.assertIn("coherence-gate.sh --full", plan["stages"]["coherence"]["command_template"])
        self.assertEqual(status["next_work"], plan["next_work"])

    def test_source_inventory_is_joined_and_checked_for_probe_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            inventory = root / "inventory.json"
            write_json(probe, probe_payload(model="/models/Qwen3.5-9B/snapshots/x", complete_modules=0))
            write_json(bundle, bundle_payload(weights_exist=False))
            write_json(inventory, source_inventory_payload(native=False, g128_count=0))

            status = paro_status.build_status([probe], bundle, source_inventory=inventory)

        self.assertTrue(status["gates"]["source_inventory_present"])
        self.assertTrue(status["gates"]["source_inventory_consistent"])
        self.assertFalse(status["gates"]["source_inventory_native_paro_g128_found"])
        self.assertFalse(status["source_inventory"]["native_paro_g128_source_found"])
        self.assertEqual(
            status["productization_boundary"]["native_source_contract"][
                "broader_source_inventory"
            ]["g128_complete_module_count"],
            0,
        )
        self.assertIn("broader local Paro source inventory found zero", " ".join(status["blockers"]))

    def test_source_inventory_contradiction_blocks_productization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            inventory = root / "inventory.json"
            write_json(probe, probe_payload(model="/models/Qwen3.5-9B/snapshots/x", complete_modules=0))
            write_json(bundle, bundle_payload(weights_exist=False))
            write_json(inventory, source_inventory_payload(native=True, g128_count=2))

            status = paro_status.build_status([probe], bundle, source_inventory=inventory)

        self.assertTrue(status["gates"]["source_inventory_present"])
        self.assertFalse(status["gates"]["source_inventory_consistent"])
        self.assertTrue(status["gates"]["source_inventory_native_paro_g128_found"])
        self.assertFalse(status["promotion_allowed"])
        self.assertIn("contradicts the named source probes", " ".join(status["blockers"]))

    def test_env_boundary_reports_missing_required_knobs(self):
        payload = bundle_payload(weights_exist=True)
        boundary = payload["sections"]["transform.paro"]["runtime_env_boundary"]
        boundary["productization_candidate"] = [{"name": "HIPFIRE_PARO_BATCHED"}]
        boundary["research_only"] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            write_json(probe, probe_payload(model="/models/native-paro", complete_modules=1))
            write_json(bundle, payload)

            status = paro_status.build_status([probe], bundle)

        missing = status["bundle_plan"]["runtime_env_boundary"]["required_product_env_missing"]
        self.assertEqual(missing, ["HIPFIRE_MOE_PARO_I8", "HIPFIRE_MOE_PARO_I8_K8"])
        self.assertFalse(status["gates"]["runtime_env_boundary_recorded"])
        self.assertFalse(status["gates"]["promotion_main_path_clean"])
        self.assertFalse(status["promotion_allowed"])

    def test_generic_evidence_does_not_satisfy_typed_promotion_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            write_json(probe, probe_payload(model="/models/native-paro", complete_modules=1))
            write_json(bundle, bundle_payload(weights_exist=True))

            status = paro_status.build_status([probe], bundle, evidence_artifacts=["some-report.json"])

        self.assertFalse(status["promotion_allowed"])
        self.assertFalse(status["gates"]["oracle_quality_perf_evidence_present"])
        self.assertTrue(status["gates"]["research_only_env_evidence_absent"])
        self.assertEqual(status["evidence"]["generic_artifacts"], ["some-report.json"])
        self.assertIn("paro-oracle evidence is missing", status["blockers"])

    def test_typed_evidence_requires_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = write_expected_probes(root, complete_modules=1)
            bundle = root / "bundle.json"
            write_json(bundle, bundle_payload(weights_exist=True))

            status = paro_status.build_status(
                probes,
                bundle,
                oracle_artifacts=[str(root / "missing-oracle.json")],
                coherence_artifacts=write_evidence(root, "coherence.md"),
                nan_artifacts=write_evidence(root, "nan.json"),
                quality_artifacts=write_evidence(root, "kld.json"),
                perf_artifacts=write_evidence(root, "perf.json"),
            )

        self.assertFalse(status["promotion_allowed"])
        self.assertFalse(status["gates"]["oracle_evidence_present"])
        self.assertFalse(status["gates"]["typed_evidence_files_exist"])
        self.assertFalse(status["gates"]["oracle_quality_perf_evidence_present"])
        self.assertFalse(status["evidence"]["typed_artifact_existence"]["oracle"])
        self.assertIn("typed positive evidence artifact paths do not exist", " ".join(status["blockers"]))

    def test_research_only_env_artifact_blocks_otherwise_complete_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = write_expected_probes(root, complete_modules=1)
            bundle = root / "bundle.json"
            write_json(bundle, bundle_payload(weights_exist=True))
            oracle, coherence, nan, quality, perf = write_evidence(
                root,
                "oracle.json",
                "coherence.md",
                "nan.json",
                "kld.json",
                "perf.json",
            )

            status = paro_status.build_status(
                probes,
                bundle,
                oracle_artifacts=[oracle],
                coherence_artifacts=[coherence],
                nan_artifacts=[nan],
                quality_artifacts=[quality],
                perf_artifacts=[perf],
                research_env_artifacts=["pack2.json"],
            )

        self.assertFalse(status["promotion_allowed"])
        self.assertFalse(status["gates"]["research_only_env_evidence_absent"])
        self.assertFalse(status["gates"]["promotion_main_path_clean"])
        self.assertEqual(status["evidence"]["research_only_env_artifacts"], ["pack2.json"])
        self.assertIn("research-only Paro env evidence", " ".join(status["blockers"]))

    def test_all_gates_can_become_promotion_ready_when_typed_evidence_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = write_expected_probes(root, complete_modules=1)
            bundle = root / "bundle.json"
            write_json(bundle, bundle_payload(weights_exist=True))
            oracle, coherence, nan, quality, perf = write_evidence(
                root,
                "oracle.json",
                "coherence.md",
                "nan.json",
                "kld.json",
                "perf.json",
            )

            status = paro_status.build_status(
                probes,
                bundle,
                oracle_artifacts=[oracle],
                coherence_artifacts=[coherence],
                nan_artifacts=[nan],
                quality_artifacts=[quality],
                perf_artifacts=[perf],
            )

        self.assertEqual(status["status"], "promotion-ready")
        self.assertTrue(status["promotion_allowed"])
        self.assertTrue(status["gates"]["source_probe_coverage_complete"])
        self.assertTrue(status["gates"]["research_only_env_evidence_absent"])
        self.assertTrue(status["gates"]["promotion_main_path_clean"])
        self.assertTrue(status["gates"]["typed_evidence_files_exist"])
        self.assertTrue(status["gates"]["oracle_quality_perf_evidence_present"])
        self.assertEqual(status["blockers"], [])
        self.assertEqual(status["dependency_graph"]["next_unblocked_step"], "evaluate_readiness_matrix_for_promotion")

    def test_origin_context_records_paro_cleanup_commit_without_satisfying_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            bundle = root / "bundle.json"
            write_json(probe, probe_payload(model="/models/Qwen3.5-9B/snapshots/x", complete_modules=0))
            write_json(bundle, bundle_payload(weights_exist=False))

            status = paro_status.build_status([probe], bundle)

        commits = {
            item["short"]: item
            for item in status["origin_context"]["relevant_upstream_commits"]
        }
        self.assertIn("676338a4", commits)
        self.assertIn("d5985c3e", commits)
        self.assertIn("b4adca1f", commits)
        self.assertIn("8de45455", commits)
        self.assertIn("does not provide a native ParoQ4G128 checkpoint", commits["8de45455"]["impact"])
        self.assertIn("rerun A3B/MoE prefill", commits["b4adca1f"]["action"])
        self.assertFalse(status["gates"]["native_paro_source_found"])
        self.assertFalse(status["gates"]["oracle_evidence_present"])

    def test_current_productization_artifact_is_not_promoted(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], paro_status.SCHEMA)
        self.assertEqual(payload["status"], "candidate-productization")
        self.assertFalse(payload["promotion_allowed"])
        self.assertTrue(payload["gates"]["source_probe_coverage_complete"])
        self.assertTrue(payload["gates"]["source_inventory_present"])
        self.assertTrue(payload["gates"]["source_inventory_consistent"])
        self.assertFalse(payload["gates"]["source_inventory_native_paro_g128_found"])
        self.assertFalse(payload["gates"]["native_paro_source_found"])
        self.assertFalse(payload["gates"]["imported_hfq_exists"])
        self.assertTrue(payload["gates"]["runtime_env_boundary_recorded"])
        self.assertTrue(payload["gates"]["research_only_env_evidence_absent"])
        self.assertTrue(payload["gates"]["promotion_main_path_clean"])
        self.assertTrue(payload["gates"]["package_contract_present"])
        self.assertFalse(payload["gates"]["oracle_evidence_present"])
        self.assertFalse(payload["gates"]["coherence_evidence_present"])
        self.assertFalse(payload["gates"]["finite_logit_nan_evidence_present"])
        self.assertFalse(payload["gates"]["quality_evidence_present"])
        self.assertFalse(payload["gates"]["gfx1151_perf_evidence_present"])
        self.assertFalse(payload["gates"]["typed_evidence_files_exist"])
        self.assertFalse(payload["gates"]["oracle_quality_perf_evidence_present"])
        boundary = payload["productization_boundary"]
        self.assertEqual(boundary["current_stage"], "blocked_before_native_source_import")
        self.assertEqual(boundary["importer_state"], "blocked_no_native_paro_source")
        self.assertEqual(boundary["package_state"], "contract_only_missing_weights")
        self.assertEqual(boundary["astrea_state"], "bundle_plan_contract_only_not_model_writer")
        self.assertEqual(boundary["main_path_state"], "clean_but_unpromoted")
        self.assertFalse(boundary["hidden_research_env_dependency_present"])
        self.assertFalse(boundary["artifact_generation_without_native_source_allowed"])
        self.assertTrue(boundary["research_only_env_blocks_promotion"])
        self.assertTrue(boundary["promoted_main_path_requires_typed_evidence"])
        self.assertEqual(
            boundary["native_source_contract"]["required_tensor_families"],
            list(paro_status.REQUIRED_PARO_SUFFIXES),
        )
        self.assertFalse(
            boundary["native_source_contract"]["broader_source_inventory"][
                "native_paro_g128_source_found"
            ]
        )
        self.assertEqual(
            boundary["native_source_contract"]["broader_source_inventory"][
                "g128_complete_module_count"
            ],
            0,
        )
        self.assertTrue(
            boundary["native_source_contract"][
                "all_required_suffixes_absent_across_expected_probes"
            ]
        )
        self.assertTrue(payload["dependency_graph"]["native_source_absent"])
        self.assertTrue(payload["dependency_graph"]["import_blocked_by_source_absence"])
        self.assertTrue(payload["dependency_graph"]["oracle_blocked_by_import_absence"])
        self.assertFalse(payload["dependency_graph"]["nodes"]["typed_evidence_files"]["satisfied"])
        self.assertTrue(
            payload["dependency_graph"]["all_required_suffixes_absent_across_expected_probes"]
        )
        self.assertEqual(
            payload["dependency_graph"]["next_unblocked_step"],
            "locate_or_generate_native_paro_q4g128_checkpoint",
        )
        plan = payload["productization_plan"]
        self.assertEqual(plan["status"], "blocked_before_native_source_import")
        self.assertEqual(plan["next_unblocked_step"], "locate_or_generate_native_paro_q4g128_checkpoint")
        self.assertEqual(plan["stage_order"][0], "source_inventory")
        self.assertEqual(plan["stage_order"][3], "imported_hfq")
        self.assertFalse(plan["stages"]["native_source"]["satisfied"])
        self.assertEqual(plan["stages"]["imported_hfq"]["blocked_by"], ["native_source"])
        self.assertIn("paro-import", plan["stages"]["imported_hfq"]["command_template"])
        self.assertIn("paro-oracle", plan["stages"]["paro_oracle"]["command_template"])
        self.assertIn("coherence-gate.sh --full", plan["stages"]["coherence"]["command_template"])
        self.assertFalse(plan["stages"]["gfx1151_perf"]["satisfied"])
        self.assertTrue(plan["refresh_required_after_origin_reconciliation"])
        self.assertEqual(payload["next_work"], plan["next_work"])
        commits = {
            item["short"]: item
            for item in payload["origin_context"]["relevant_upstream_commits"]
        }
        self.assertIn("8de45455", commits)
        self.assertIn("676338a4", commits)
        self.assertIn("b4adca1f", commits)
        self.assertEqual(len(payload["source_probes"]), 3)
        self.assertEqual(payload["source_probe_coverage"]["expected_probe_ids_missing"], [])
        self.assertEqual(
            payload["source_probe_coverage"]["aggregate_required_suffix_counts"],
            {
                "channel_scales": 0,
                "pairs": 0,
                "qweight": 0,
                "qzeros": 0,
                "scales": 0,
                "theta": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
