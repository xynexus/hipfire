#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "quant_readiness.py"
INVENTORY_SCRIPT_PATH = ROOT / "scripts" / "paroquant_inventory.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quant_readiness", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["quant_readiness"] = module
    spec.loader.exec_module(module)
    return module


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("paroquant_inventory", INVENTORY_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["paroquant_inventory"] = module
    spec.loader.exec_module(module)
    return module


class QuantReadinessTests(unittest.TestCase):
    def setUp(self):
        self.qr = load_module()
        self.inventory = load_inventory_module()

    def write_hf_snapshot(self, root, cache_dir, snapshot="abc123", *, safetensors=1):
        snap = Path(root) / cache_dir / "snapshots" / snapshot
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "test"}))
        (snap / "tokenizer.json").write_text("{}")
        for idx in range(safetensors):
            (snap / f"model-{idx + 1:05d}-of-{safetensors:05d}.safetensors").write_bytes(b"")
        return snap

    def test_matrix_covers_required_formats(self):
        formats = {item.id for item in self.qr.READINESS}
        self.assertEqual(
            formats,
            {
                "mq4",
                "mq6",
                "mq3",
                "mq8",
                "mq2",
                "mq2-lloyd",
                "mq3-lloyd",
                "mq4-lloyd",
                "paro-q4g128",
                "paro-q4g256",
            },
        )

    def test_report_records_origin_reconciliation_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.qr.build_report(Path(tmp), format_ids=("mq6",))

        origin = report["origin_context"]
        self.assertEqual(origin["upstream"], "origin/master")
        self.assertIn("origin_master_commit", origin)
        self.assertIn("promotion_evidence_refresh_required", origin)
        self.assertGreaterEqual(len(origin["relevant_upstream_commits"]), 1)
        by_short = {commit["short"]: commit for commit in origin["relevant_upstream_commits"]}
        self.assertIn("676338a4", by_short)
        self.assertIn("d5985c3e", by_short)
        self.assertIn("f1e2cef8", by_short)
        self.assertIn("907cbb2f", by_short)
        self.assertIn("mq6", by_short["676338a4"]["formats"])
        self.assertIn("dflash", by_short["676338a4"]["surfaces"])
        self.assertIn("GGUF Promote6 Mq4Lloyd", by_short["d5985c3e"]["subject"])
        self.assertIn("paro-q4g256", by_short["f1e2cef8"]["formats"])

    def test_mq4_is_control_and_mq2_stays_rejected(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        self.assertEqual(by_id["mq4"].status, "promoted-control")
        self.assertFalse(by_id["mq4"].promotion_target)
        mq2 = by_id["mq2"]
        self.assertEqual(mq2.status, "quality-rejected")
        self.assertFalse(mq2.promotion_target)
        self.assertIn("--allow-mq2", mq2.producer_gate)
        self.assertIn("SHA-256 provenance", mq2.producer_gate)
        self.assertIn("zero hard runtime errors", mq2.runtime_gate)
        self.assertIn("confirms collapse", mq2.quality_gate)
        self.assertIn("garbled non-answers", mq2.quality_gate)
        self.assertIn("dense_quality_clean=false", mq2.quality_gate)
        self.assertIn("all_dense_rows_collapsed=true", mq2.quality_gate)
        self.assertIn("quality_rejection_boundary.dense_text_status=rejected_all_current_rows_collapsed", mq2.quality_gate)
        self.assertIn("quality_rejection_boundary.sub_9b_failures_explicit=true", mq2.quality_gate)
        self.assertIn("quality_rejection_boundary.dense_9b_rejected=true", mq2.quality_gate)
        self.assertIn(
            "quality_rejection_boundary.next_unblocked_step=produce_new_mq2_calibration_that_passes_bounded_dense_quality",
            mq2.quality_gate,
        )
        self.assertIn("promotion_allowed=false", mq2.quality_gate)
        self.assertIn("not promotion evidence", mq2.perf_gate)
        self.assertIn("quality_rejection_boundary.promotion_perf_collection_allowed=false", mq2.perf_gate)
        self.assertIn("quality_rejection_boundary.performance_rows_promotable=false", mq2.perf_gate)
        self.assertIn("explicit dense-text rejection", mq2.next_work[0])
        self.assertIn("mq2-artifact-provenance", " ".join(mq2.evidence_artifacts))
        self.assertIn("mq2-sweep", " ".join(mq2.evidence_artifacts))
        self.assertIn("mq2-status", " ".join(mq2.evidence_artifacts))
        self.assertIn("mq2-mq2-lloyd-readiness-audit", " ".join(mq2.evidence_artifacts))

    def test_mq2_lloyd_stays_routed_expert_research_only(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq2_lloyd = by_id["mq2-lloyd"]
        self.assertEqual(mq2_lloyd.status, "quality-rejected-moe-research")
        self.assertFalse(mq2_lloyd.promotion_target)
        self.assertIn("--allow-mq2-lloyd", mq2_lloyd.producer_gate)
        self.assertIn("deepseek-v4-flash.mq2lloyd", mq2_lloyd.producer_gate)
        self.assertIn("model_artifact_inventory.mq2_lloyd_artifact_present=false", mq2_lloyd.producer_gate)
        self.assertIn(
            "model_artifact_inventory.current_a3b_or_deepseek_artifact_present=false",
            mq2_lloyd.producer_gate,
        )
        self.assertIn("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W", mq2_lloyd.runtime_gate)
        self.assertIn("correctness-clean but slower than k2", mq2_lloyd.runtime_gate)
        self.assertIn("six OK rows", mq2_lloyd.runtime_gate)
        self.assertIn("promote_4w_default=false", mq2_lloyd.runtime_gate)
        self.assertIn("default_kernel_change_allowed=false", mq2_lloyd.runtime_gate)
        self.assertIn("model_level_promotion_allowed=false", mq2_lloyd.runtime_gate)
        self.assertIn("status=specialty_blocked_no_model_artifact", mq2_lloyd.runtime_gate)
        self.assertIn("current_a3b_or_deepseek_artifact_present=false", mq2_lloyd.runtime_gate)
        self.assertIn("mq2_lloyd_artifact_present=false", mq2_lloyd.runtime_gate)
        self.assertIn("synthetic_kernel_result=correct_but_slower_than_k2", mq2_lloyd.runtime_gate)
        self.assertIn("remote_moe_fix_refresh_required=true", mq2_lloyd.runtime_gate)
        self.assertIn("model_evidence_refresh_required=true", mq2_lloyd.runtime_gate)
        self.assertIn("perf_collection_allowed=false", mq2_lloyd.runtime_gate)
        self.assertIn(
            "next_unblocked_step=generate_or_locate_a3b_or_deepseek_mq2_lloyd_artifact",
            mq2_lloyd.runtime_gate,
        )
        self.assertIn("A3B/DeepSeek coherence", mq2_lloyd.quality_gate)
        self.assertIn("no current artifact-backed coherence", mq2_lloyd.quality_gate)
        self.assertIn("long-context attractor revert", mq2_lloyd.quality_gate)
        self.assertIn("origin/fix/hunt3-engine-bugs", mq2_lloyd.quality_gate)
        self.assertIn("fb36f539", mq2_lloyd.quality_gate)
        self.assertIn("sampling+MoE", mq2_lloyd.quality_gate)
        self.assertIn("0.9057x-0.9898x", mq2_lloyd.perf_gate)
        self.assertIn("structured six-row bench artifact", mq2_lloyd.perf_gate)
        self.assertIn("opt-in/research", mq2_lloyd.perf_gate)
        self.assertIn("remote_moe_fix_refresh_required=true", mq2_lloyd.perf_gate)
        self.assertIn("model_evidence_refresh_required=true", mq2_lloyd.perf_gate)
        self.assertIn("perf_collection_allowed=false", mq2_lloyd.perf_gate)
        self.assertIn("remote engine fixes are reconciled", mq2_lloyd.perf_gate)
        self.assertIn("model-level perf beats k2", mq2_lloyd.perf_gate)
        self.assertIn("effective-bandwidth shortcuts overcount inactive experts", mq2_lloyd.perf_gate)
        self.assertIn("launch overhead", mq2_lloyd.perf_gate)
        self.assertIn("codebook unpack/lookup cost", mq2_lloyd.perf_gate)
        self.assertIn("origin/fix/hunt3-engine-bugs", mq2_lloyd.next_work[0])
        self.assertIn("fb36f539", mq2_lloyd.next_work[0])
        self.assertIn("deepseek-v4-flash.mq2lloyd", mq2_lloyd.next_work[1])
        self.assertIn("routed-expert coherence", mq2_lloyd.next_work[2])
        self.assertIn("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W", mq2_lloyd.next_work[3])
        self.assertIn("Batch selected experts across tokens", mq2_lloyd.next_work[4])
        self.assertIn("mq2-mq2-lloyd-readiness-audit", " ".join(mq2_lloyd.evidence_artifacts))
        self.assertIn("mq2-lloyd-4w-bench", " ".join(mq2_lloyd.evidence_artifacts))

    def test_paro_g256_is_prototype_only(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        paro = by_id["paro-q4g256"]
        self.assertEqual(paro.status, "prototype-only")
        self.assertFalse(paro.promotion_target)
        self.assertIn("true Paro group_size=256", paro.producer_gate)
        self.assertIn("regrouped G128 evidence is not quality proof", paro.producer_gate)
        self.assertIn("zero complete native Paro modules", paro.producer_gate)
        self.assertIn("zero group_size=256 Paro source modules", paro.producer_gate)
        self.assertIn("CPU probe cannot be re-run", paro.producer_gate)
        self.assertIn("native_g256_source_found=false", paro.producer_gate)
        self.assertIn("cpu_probe_runnable_now=false", paro.producer_gate)
        self.assertIn("upstream G256 perfmax", paro.producer_gate)
        self.assertIn("Exit B", paro.producer_gate)
        self.assertIn("opt-in research", paro.producer_gate)
        self.assertIn("not promotion proof", paro.producer_gate)
        self.assertIn("upstream_g256_research_evidence_present=true", paro.producer_gate)
        self.assertIn("producer_contract.required_group_size=256", paro.producer_gate)
        self.assertIn(
            "producer_contract.required_tensor_families=qweight/qzeros/scales/pairs/theta/channel_scales",
            paro.producer_gate,
        )
        self.assertIn("producer_contract.complete_g256_module_count=0", paro.producer_gate)
        self.assertIn(
            "producer_contract.source_search_result=no_paro_suffix_or_filename_hits",
            paro.producer_gate,
        )
        self.assertIn("prototype_plan.status=blocked_before_true_group_size_256_source", paro.producer_gate)
        self.assertIn("prototype_plan.next_unblocked_step", paro.producer_gate)
        self.assertIn("prototype_plan.stages.cpu_g256_probe.blocked_by=true_group_size_256_source", paro.producer_gate)
        self.assertIn("G128-only native source cannot", paro.producer_gate)
        self.assertIn("Runtime DType work waits", paro.runtime_gate)
        self.assertIn("runtime_dtype_container_ready=false", paro.runtime_gate)
        self.assertIn("runtime_dtype_container_unblocked_by_upstream=false", paro.runtime_gate)
        self.assertIn("runtime_dtype_container_work.allowed=false", paro.runtime_gate)
        self.assertIn("true_group_size_256_source, cpu_g256_probe", paro.runtime_gate)
        self.assertIn("kld_ppl_within_5_percent_of_paro_q4g128", paro.runtime_gate)
        self.assertIn("UNVERIFIABLE", paro.quality_gate)
        self.assertIn("format-loss evidence only", paro.quality_gate)
        self.assertIn("within 5%", paro.quality_gate)
        self.assertIn("true_g256_quality_evidence_present=false", paro.quality_gate)
        self.assertIn("quality_comparable_to_paro_q4g128=false", paro.quality_gate)
        self.assertIn("quality_unverifiable_until_true_g256_source=true", paro.quality_gate)
        self.assertIn("oracle/payload/KLD stages remain unsatisfied", paro.quality_gate)
        self.assertIn("<=1.03x", paro.perf_gate)
        self.assertIn("1.0220x", paro.perf_gate)
        self.assertIn("regrouped_payload_ratio_gate_passed=true", paro.perf_gate)
        self.assertIn("current_true_g256_payload_ratio_gate_passed=false", paro.perf_gate)
        self.assertIn("promotion_allowed=false", paro.perf_gate)
        self.assertIn("true group_size=256 Paro source", paro.next_work[0])
        self.assertIn("paroquant_g256_probe.py", " ".join(paro.next_work))
        self.assertIn("--local-only", " ".join(paro.next_work))
        self.assertIn("true-G256 oracle comparison", " ".join(paro.next_work))
        self.assertIn("<=1.05x", " ".join(paro.next_work))
        self.assertIn("prototype_plan.runtime_dtype_container_work_allowed=true", " ".join(paro.next_work))
        self.assertIn("source-inventory", " ".join(paro.evidence_artifacts))
        self.assertIn("docs/investigations/paro-g256-perfmax/SUMMARY.md", " ".join(paro.evidence_artifacts))
        self.assertIn("g256-probe-9b.json", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g256-contract-audit", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g256-status", " ".join(paro.evidence_artifacts))

    def test_paro_inventory_classifies_complete_g256_modules(self):
        tensors = {
            "layers.0.mlp.down_proj.qweight": {"shape": [4096, 512]},
            "layers.0.mlp.down_proj.qzeros": {"shape": [16, 512]},
            "layers.0.mlp.down_proj.scales": {"shape": [16, 4096]},
            "layers.0.mlp.down_proj.pairs": {"shape": [8, 4096]},
            "layers.0.mlp.down_proj.theta": {"shape": [8, 2048]},
            "layers.0.mlp.down_proj.channel_scales": {"shape": [4096]},
        }

        modules, incomplete, suffix_counts = self.inventory.discover_paro_modules(tensors)

        self.assertEqual(incomplete, [])
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0]["group_size"], 256)
        self.assertEqual(suffix_counts["qweight"], 1)
        self.assertTrue(self.inventory.path_has_hint(Path("qwen3.5-9b-paro-g256.safetensors")))

    def test_paro_inventory_reports_unverifiable_when_no_native_source_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "qwen3.5-9b-paro-g256-mq4.hfq").write_bytes(b"hfq")
            payload = self.inventory.scan_roots([root])

        self.assertEqual(payload["schema"], self.inventory.SCHEMA)
        self.assertEqual(payload["native_paro"]["complete_module_count"], 0)
        self.assertEqual(payload["native_paro"]["g256_complete_module_count"], 0)
        self.assertFalse(payload["decision"]["native_paro_g256_source_found"])
        self.assertEqual(payload["decision"]["quality_state"], "UNVERIFIABLE")
        self.assertEqual(payload["filename_hit_count"], 1)

    def test_paro_g128_records_source_probe_without_artifact_claim(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        paro = by_id["paro-q4g128"]
        self.assertEqual(paro.status, "candidate-productization")
        self.assertIn("zero complete Paro modules", paro.producer_gate)
        self.assertIn("Qwen3.5 35B-A3B", paro.producer_gate)
        self.assertIn("Qwen3.6 35B-A3B", paro.producer_gate)
        self.assertIn("native Paro checkpoint", paro.producer_gate)
        self.assertIn("importer/oracle contract is explicit", paro.producer_gate)
        self.assertIn("transform.paro package section", paro.producer_gate)
        self.assertIn("runtime_env_boundary", paro.producer_gate)
        self.assertIn("structured Paro productization", paro.producer_gate)
        self.assertIn("source_probe_coverage_complete=true", paro.producer_gate)
        self.assertIn("source_inventory_present=true", paro.producer_gate)
        self.assertIn("source_inventory_consistent=true", paro.producer_gate)
        self.assertIn("source_inventory_native_paro_g128_found=false", paro.producer_gate)
        self.assertIn("all_required_suffixes_absent_across_expected_probes=true", paro.producer_gate)
        self.assertIn("aggregate_required_suffix_counts=0", paro.producer_gate)
        self.assertIn("qweight/qzeros/scales/pairs/theta/channel_scales", paro.producer_gate)
        self.assertIn("scanned 151 safetensors across 17 dirs", paro.producer_gate)
        self.assertIn("zero complete native Paro modules", paro.producer_gate)
        self.assertIn("no local imported Paro HFQ artifact", paro.producer_gate)
        self.assertIn("productization_plan.status=blocked_before_native_source_import", paro.producer_gate)
        self.assertIn("productization_plan.next_unblocked_step", paro.producer_gate)
        self.assertIn("676338a4", paro.producer_gate)
        self.assertIn("d5985c3e", paro.producer_gate)
        self.assertIn("b4adca1f", paro.producer_gate)
        self.assertIn("do not provide source/import/oracle/quality/perf", paro.producer_gate)
        self.assertIn("Paro batched-admit env policy", paro.runtime_gate)
        self.assertIn("HIPFIRE_MOE_PARO_I8/K8 opt-out policy", paro.runtime_gate)
        self.assertIn("HIPFIRE_PARO_BATCHED", paro.runtime_gate)
        self.assertIn("HIPFIRE_MOE_PARO_I8", paro.runtime_gate)
        self.assertIn("runtime_env_boundary", paro.runtime_gate)
        self.assertIn("HIPFIRE_PARO_PREROTATE", paro.runtime_gate)
        self.assertIn("HIPFIRE_PARO_FUSED_PACK2", paro.runtime_gate)
        self.assertIn("runtime_env_boundary_recorded=true", paro.runtime_gate)
        self.assertIn("research_only_env_evidence_absent=true", paro.runtime_gate)
        self.assertIn("promotion_main_path_clean=true", paro.runtime_gate)
        self.assertIn("promotion_allowed=false", paro.runtime_gate)
        self.assertIn("g128_complete_module_count=0", paro.runtime_gate)
        self.assertIn("productization_plan.stages.source_inventory.command", paro.runtime_gate)
        self.assertIn("scripts/paroquant_inventory.py --pretty --out", paro.runtime_gate)
        self.assertIn("productization_plan.refresh_required_after_origin_reconciliation=true", paro.runtime_gate)
        self.assertIn("cannot satisfy the Paro importer contract", paro.quality_gate)
        self.assertIn("missing native tensor families", paro.quality_gate)
        self.assertIn("qweight/qzeros/scales/pairs/theta/channel_scales", paro.quality_gate)
        self.assertIn("A3B snapshots", paro.quality_gate)
        self.assertIn("package metadata only", paro.quality_gate)
        self.assertIn("oracle_evidence_present=false", paro.quality_gate)
        self.assertIn("coherence_evidence_present=false", paro.quality_gate)
        self.assertIn("finite_logit_nan_evidence_present=false", paro.quality_gate)
        self.assertIn("quality_evidence_present=false", paro.quality_gate)
        self.assertIn("gfx1151_perf_evidence_present=false", paro.quality_gate)
        self.assertIn("typed_evidence_files_exist=false", paro.quality_gate)
        self.assertIn("artifact files land", paro.quality_gate)
        self.assertIn("oracle_quality_perf_evidence_present=false", paro.quality_gate)
        self.assertIn("command templates for", paro.quality_gate)
        self.assertIn("paro-import", paro.quality_gate)
        self.assertIn("bundle-plan", paro.quality_gate)
        self.assertIn("paro-oracle", paro.quality_gate)
        self.assertIn("full coherence gate", paro.quality_gate)
        self.assertIn("productization_plan.stages.imported_hfq", paro.perf_gate)
        self.assertIn("qweight/qzeros/scales/pairs/theta/channel_scales", paro.next_work[0])
        self.assertIn("paro-probe, paro-import, and paro-oracle", paro.next_work[1])
        self.assertIn("Regenerate the Astrea bundle plan", paro.next_work[2])
        self.assertIn("imported HFQ", paro.next_work[2])
        self.assertIn("coherence, finite-logit/NaN, KLD/PPL", paro.next_work[3])
        self.assertIn("Refresh the evidence after reconciling origin commits", paro.next_work[4])
        self.assertIn("paro-q4g128-qwen35-9b-source-probe", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g128-qwen35-a3b-source-probe", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g128-qwen36-a3b-source-probe", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-source-inventory", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g128-productization-audit", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g128-astrea-bundle-plan", " ".join(paro.evidence_artifacts))
        self.assertIn("paro-q4g128-productization-status", " ".join(paro.evidence_artifacts))

    def test_mq6_records_routing_evidence_without_promotion_claim(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq6 = by_id["mq6"]
        self.assertEqual(mq6.status, "candidate-runtime-ready")
        self.assertTrue(mq6.promotion_target)
        self.assertIn("four canonical MQ6 artifacts", mq6.producer_gate)
        self.assertIn("`.hfq` symlinks", mq6.producer_gate)
        self.assertIn("live_canonical_mq6_artifacts_present=true", mq6.producer_gate)
        self.assertIn("provenance_canonical_mq6_artifacts_present=true", mq6.producer_gate)
        self.assertIn("GGUF `--kmap-promote 6` Mq4Lloyd producer path", mq6.producer_gate)
        self.assertIn("promote6_mq4_lloyd_emits_mq6=true", mq6.producer_gate)
        self.assertIn("origin d5985c3e reconciled in the worktree", mq6.producer_gate)
        self.assertIn("Dense and MoE gfx1151 paths are wired", mq6.runtime_gate)
        self.assertIn("Qwen35 dense prefill", mq6.runtime_gate)
        self.assertIn("MQ6/HFQ6 `is_batchable_la` coverage", mq6.runtime_gate)
        self.assertIn("grouped MoE path-2 routing", mq6.runtime_gate)
        self.assertIn("no-GPU admission tests", mq6.runtime_gate)
        self.assertIn("dense-decode compiled-blob inventory", mq6.runtime_gate)
        self.assertIn("origin a7dcfb0d CASK scratch", mq6.runtime_gate)
        self.assertIn("MTP GPU teardown", mq6.runtime_gate)
        self.assertIn("FWHT asym4 fallback dispatch", mq6.runtime_gate)
        self.assertIn("mq6_dense_decode_routes_through_hfq6_family", mq6.runtime_gate)
        self.assertIn("moe_decode_routes_mq6_indexed_path_for_k8", mq6.runtime_gate)
        self.assertIn("not a dispatch or promotion proof", mq6.runtime_gate)
        self.assertIn("DFlash/spec boundary", mq6.runtime_gate)
        self.assertIn("test_dflash_spec_boundary_marks_mq4_draft_target_verify_only", mq6.runtime_gate)
        self.assertIn("MQ6-draft and A3B scope expansion", mq6.runtime_gate)
        self.assertIn("PPL evidence", mq6.quality_gate)
        self.assertIn("KLD evidence", mq6.quality_gate)
        self.assertIn("bounded 9B dense PPL", mq6.quality_gate)
        self.assertIn("Qwen3.5 A3B PPL", mq6.quality_gate)
        self.assertIn("Qwen3.6 A3B PPL", mq6.quality_gate)
        self.assertIn("BF16-referenced KLD evidence", mq6.quality_gate)
        self.assertIn("c64", mq6.quality_gate)
        self.assertIn("c256", mq6.quality_gate)
        self.assertIn("c512", mq6.quality_gate)
        self.assertIn("DFlash target-verify evidence", mq6.quality_gate)
        self.assertIn("3-run", mq6.quality_gate)
        self.assertIn("Structured MQ6 promotion gates", mq6.quality_gate)
        self.assertIn("origin_refresh_required=true", mq6.quality_gate)
        self.assertIn("a3b_first_scope_supported=false", mq6.quality_gate)
        self.assertIn("remote_branch_refresh_required=true", mq6.quality_gate)
        self.assertIn("dflash_target_side_verify_only=true", mq6.quality_gate)
        self.assertIn("dflash_mq6_draft_evidence_present=false", mq6.quality_gate)
        self.assertIn("dflash_a3b_evidence_present=false", mq6.quality_gate)
        self.assertIn("dflash_spec_boundary_refresh_required=true", mq6.quality_gate)
        self.assertIn("dflash_spec_boundary_contract_test_present=true", mq6.quality_gate)
        self.assertIn("surface_coverage_complete=true", mq6.quality_gate)
        self.assertIn("dense_decode_repo_kernel_inventory_present=true", mq6.quality_gate)
        self.assertIn("dense_decode_install_kernel_inventory_present=true", mq6.quality_gate)
        self.assertIn("dense_decode_no_gpu_route_test_present=true", mq6.quality_gate)
        self.assertIn("moe_decode_no_gpu_route_test_present=true", mq6.quality_gate)
        self.assertIn("all_surfaces_promotion_ready=false", mq6.quality_gate)
        self.assertIn("evidence_refresh_plan.refresh_required=true", mq6.quality_gate)
        self.assertIn("evidence_refresh_plan.affected_surfaces", mq6.quality_gate)
        self.assertIn("promotion_allowed=false", mq6.quality_gate)
        self.assertIn("gfx1151 AR", mq6.perf_gate)
        self.assertIn("DFlash/spec rows", mq6.perf_gate)
        self.assertIn("3-run", mq6.perf_gate)
        self.assertIn("2x slower", mq6.perf_gate)
        self.assertIn("A3B-first", mq6.perf_gate)
        self.assertIn("i8 grouped-path decision", mq6.perf_gate)
        self.assertIn("do not port MQ6 i8 grouped-MMQ now", mq6.perf_gate)
        self.assertIn("dense MQ6 throughput", mq6.perf_gate)
        self.assertIn("dense_ar_perf_promotable=false", mq6.perf_gate)
        self.assertIn("dense_dflash_perf_promotable=false", mq6.perf_gate)
        self.assertIn("origin_refresh_required=true", mq6.perf_gate)
        self.assertIn("remote_branch_refresh_required=true", mq6.perf_gate)
        self.assertIn("dense_decode_repo_kernel_inventory_present=true", mq6.perf_gate)
        self.assertIn("dense_decode_install_kernel_inventory_present=true", mq6.perf_gate)
        self.assertIn("dense_decode_no_gpu_route_test_present=true", mq6.perf_gate)
        self.assertIn("moe_decode_no_gpu_route_test_present=true", mq6.perf_gate)
        self.assertIn("dflash_target_side_verify_only=true", mq6.perf_gate)
        self.assertIn("dflash_mq6_draft_evidence_present=false", mq6.perf_gate)
        self.assertIn("dflash_a3b_evidence_present=false", mq6.perf_gate)
        self.assertIn("dflash_spec_boundary_contract_test_present=true", mq6.perf_gate)
        self.assertIn("surface_coverage.surfaces.dense_decode.status=perf_gated", mq6.perf_gate)
        self.assertIn(
            "surface_coverage.surfaces.moe_prefill.status=a3b_first_candidate_refresh_blocked",
            mq6.perf_gate,
        )
        self.assertIn(
            "surface_coverage.surfaces.dflash_spec.status=target_verify_only_refresh_blocked",
            mq6.perf_gate,
        )
        self.assertIn("evidence_refresh_plan.next_unblocked_step", mq6.perf_gate)
        self.assertIn("broad_promotion_allowed=false", mq6.perf_gate)
        self.assertIn("artifact-provenance", " ".join(mq6.evidence_artifacts))
        self.assertIn("coherence-md5", " ".join(mq6.evidence_artifacts))
        self.assertIn("ar-perf", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-ppl", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-a3b-ppl", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-qwen36-a3b-ppl", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-kld", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-kld-c64", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-kld-c256", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-kld-c512", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-dflash", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-dflash-r3", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-readiness-audit", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-i8-grouped-decision", " ".join(mq6.evidence_artifacts))
        self.assertIn("mq6-status", " ".join(mq6.evidence_artifacts))
        self.assertIn("A3B/MoE as the first promotion scope", " ".join(mq6.next_work))
        self.assertIn("c512 BF16-referenced KLD row", " ".join(mq6.next_work))
        self.assertIn("rerun MQ6 A3B/MoE", " ".join(mq6.next_work))
        self.assertIn("origin/fix/hunt3-engine-bugs", " ".join(mq6.next_work))
        self.assertIn("fb36f539", " ".join(mq6.next_work))
        self.assertIn("Do not port MQ6 i8 grouped-MMQ", " ".join(mq6.next_work))
        self.assertNotIn("Record provenance", " ".join(mq6.next_work))
        self.assertNotIn("Populate fresh-process gfx1151 AR", " ".join(mq6.next_work))
        self.assertNotIn("Run a Q8 or BF16-referenced KLD pass", " ".join(mq6.next_work))
        self.assertNotIn("Decide whether an i8 MMQ grouped path", " ".join(mq6.next_work))
        self.assertNotIn("c256/c512", " ".join(mq6.next_work))
        self.assertNotIn("Optionally promote", " ".join(mq6.next_work))
        self.assertNotIn("one-run smoke", " ".join(mq6.next_work))

    def test_mq8_is_runtime_research_without_artifact(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq8 = by_id["mq8"]
        self.assertEqual(mq8.status, "permanent-runtime-research")
        self.assertFalse(mq8.promotion_target)
        self.assertIn("hipfire-quantize can emit MQ8", mq8.producer_gate)
        self.assertIn("--format mq8", mq8.producer_gate)
        self.assertIn("Q8F16 fallback", mq8.producer_gate)
        self.assertIn("No local canonical MQ8 candidate artifact", mq8.producer_gate)
        self.assertIn("producer_surface_present=true", mq8.producer_gate)
        self.assertIn("canonical_mq8_artifact_present=false", mq8.producer_gate)
        self.assertIn("DType::MQ8G256", mq8.runtime_gate)
        self.assertIn("gemv_mq8g256_with_rotate", mq8.runtime_gate)
        self.assertIn("rotate_quantize_x_mq8", mq8.runtime_gate)
        self.assertIn("gfx1151 MQ8/HFQ8 compiled blobs", mq8.runtime_gate)
        self.assertIn("gfx1151-scoped", mq8.runtime_gate)
        self.assertIn("runtime_surface_present=true", mq8.runtime_gate)
        self.assertIn("gfx1151_compiled_blobs_present=true", mq8.runtime_gate)
        self.assertIn("no_gpu_admission_covered=true", mq8.runtime_gate)
        self.assertIn("candidate_model_benchmark_harness_present=false", mq8.runtime_gate)
        self.assertIn("harness.candidate_model_harness_present=false", mq8.runtime_gate)
        self.assertIn("No KLD/PPL or coherence evidence", mq8.quality_gate)
        self.assertIn("Q8/MQ6", mq8.quality_gate)
        self.assertIn("removed from the active promotion backlog", mq8.quality_gate)
        self.assertIn("do not generate a multi-GB", mq8.quality_gate)
        self.assertIn("quality_evidence_present=false", mq8.quality_gate)
        self.assertIn("product_role_defined=false", mq8.quality_gate)
        self.assertIn("promotion_boundary.promotion_lane_state=closed_until_reopen_contract", mq8.quality_gate)
        self.assertIn("promotion_boundary.artifact_generation_state=blocked_missing_product_role", mq8.quality_gate)
        self.assertIn("reopen_contract.satisfied.mq8_example_or_benchmark_harness=false", mq8.quality_gate)
        self.assertIn("promotion_boundary.next_unblocked_step=define_product_role_over_q8_or_mq6", mq8.quality_gate)
        self.assertIn("research_closure.artifact_generation_blocked=true", mq8.quality_gate)
        self.assertIn("candidate_artifact_generation_requires_product_role=true", mq8.quality_gate)
        self.assertIn("research_closure.next_unblocked_step=define_product_role_over_q8_or_mq6", mq8.quality_gate)
        self.assertIn("258 B/group", mq8.perf_gate)
        self.assertIn("Runtime work is limited to maintenance", mq8.perf_gate)
        self.assertIn("perf_evidence_present=false", mq8.perf_gate)
        self.assertIn("promotion_allowed=false", mq8.perf_gate)
        self.assertIn("high_bit_justification_required=true", mq8.perf_gate)
        self.assertIn("candidate_model_baseline_available=false", mq8.perf_gate)
        self.assertIn("keeps mq8_example_or_benchmark_harness in missing_reopen_requirements", mq8.perf_gate)
        self.assertIn("research_closure.perf_collection_blocked=true", mq8.perf_gate)
        self.assertIn("candidate_perf_collection_requires_artifact_and_quality=true", mq8.perf_gate)
        self.assertIn("high_bit_value_must_exceed_mq6_or_q8=true", mq8.perf_gate)
        self.assertIn("permanent runtime-research lane", mq8.next_work[0])
        self.assertIn("qwen3.5-9b-mq8.hfq", mq8.next_work[1])
        self.assertIn("hipfire-quantize --format mq8", mq8.next_work[1])
        self.assertIn("KLD/PPL against MQ4/Q8/MQ6", mq8.next_work[2])
        self.assertIn("mq8-runtime-audit", " ".join(mq8.evidence_artifacts))
        self.assertIn("mq8-status", " ".join(mq8.evidence_artifacts))

    def test_mq3_records_artifact_provenance_without_promotion_claim(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq3 = by_id["mq3"]
        self.assertEqual(mq3.status, "candidate-size-gated")
        self.assertTrue(mq3.promotion_target)
        self.assertIn("SHA-256 provenance", mq3.producer_gate)
        self.assertIn("live_canonical_mq3_artifacts_present=true", mq3.producer_gate)
        self.assertIn("/home/sadara/.hipfire/models", mq3.producer_gate)
        self.assertIn("all five canonical required MQ3 artifacts", mq3.producer_gate)
        self.assertIn("A3B MQ3 rows", mq3.runtime_gate)
        self.assertIn("3-run 27B DFlash/spec", mq3.runtime_gate)
        self.assertIn("No-GPU long-prefill path2 shape coverage", mq3.runtime_gate)
        self.assertIn("N=256, K_TOP=8, x_row_div=8", mq3.runtime_gate)
        self.assertIn("artifact-backed long-prefill promotion coverage", mq3.runtime_gate)
        self.assertIn("MoE decode boundary coverage", mq3.runtime_gate)
        self.assertIn("decode still lacks an indexed routed-expert path", mq3.runtime_gate)
        self.assertIn("hard-error coherence smoke", mq3.quality_gate)
        self.assertIn("4B boundary row", mq3.quality_gate)
        self.assertIn("max-token cap", mq3.quality_gate)
        self.assertIn("A3B routed-expert coherence smoke", mq3.quality_gate)
        self.assertIn("Bounded A3B PPL evidence", mq3.quality_gate)
        self.assertIn("8.2158", mq3.quality_gate)
        self.assertIn("6.5041", mq3.quality_gate)
        self.assertIn("Broader A3B coherence", mq3.quality_gate)
        self.assertIn("All 12 rows completed without hard errors", mq3.quality_gate)
        self.assertIn("no-hard-error smoke only", mq3.quality_gate)
        self.assertIn("7/12 rows hit the max-token cap", mq3.quality_gate)
        self.assertIn("a3b_broader_coherence_promotion_grade=false", mq3.quality_gate)
        self.assertIn("long-LRU hit the cap", mq3.quality_gate)
        self.assertIn("Bounded c20 4B BF16-referenced KLD", mq3.quality_gate)
        self.assertIn("0.517444", mq3.quality_gate)
        self.assertIn("0.130965", mq3.quality_gate)
        self.assertIn("Bounded 9B PPL is worse", mq3.quality_gate)
        self.assertIn("c20 BF16-referenced KLD", mq3.quality_gate)
        self.assertIn("large 9B quality regression", mq3.quality_gate)
        self.assertIn("Bounded 27B PPL is favorable", mq3.quality_gate)
        self.assertIn("no comparable local qwen3.5-27B", mq3.quality_gate)
        self.assertIn("KLD reference inventory", mq3.quality_gate)
        self.assertIn("verifies local 4B, 9B, and qwen3.6-27B refs", mq3.quality_gate)
        self.assertIn("qwen3.5-27B plus both A3B refs are absent", mq3.quality_gate)
        self.assertIn("qwen3.6-27B is not a valid qwen3.5 substitute", mq3.quality_gate)
        self.assertIn("size-gate audit", mq3.quality_gate)
        self.assertIn("current non-promotes", mq3.quality_gate)
        self.assertIn("MQ3-Lloyd", mq3.quality_gate)
        self.assertIn("A3B DFlash fixture audit", mq3.quality_gate)
        self.assertIn("no paired local Qwen3.5/Qwen3.6 35B-A3B draft sidecar", mq3.quality_gate)
        self.assertIn("blocked on fixture availability", mq3.quality_gate)
        self.assertIn("A3B KLD reference audit", mq3.quality_gate)
        self.assertIn("no local or remote Qwen3.5/Qwen3.6 35B-A3B HFKLDR reference", mq3.quality_gate)
        self.assertIn("rejects accidental 9B or 27B reference reuse", mq3.quality_gate)
        self.assertIn("structured MQ3 size-gate status", mq3.quality_gate)
        self.assertIn("dense_4b_rejected=true", mq3.quality_gate)
        self.assertIn("dense_9b_rejected=true", mq3.quality_gate)
        self.assertIn("live_canonical_mq3_artifacts_present=true", mq3.quality_gate)
        self.assertIn("dense_27b_kld_reference_present=false", mq3.quality_gate)
        self.assertIn("a3b_dflash_sidecars_present=false", mq3.quality_gate)
        self.assertIn("a3b_broader_coherence_capped_rows_present=true", mq3.quality_gate)
        self.assertIn("long_prefill_shape_contract_covered=true", mq3.quality_gate)
        self.assertIn("long_prefill_runtime_evidence_present=false", mq3.quality_gate)
        self.assertIn("a3b_moe_decode_boundary_recorded=true", mq3.quality_gate)
        self.assertIn("a3b_moe_decode_indexed_route_supported=false", mq3.quality_gate)
        self.assertIn("a3b_qwen36_ppl_beats_mq4=false", mq3.quality_gate)
        self.assertIn("promotion_allowed=false", mq3.quality_gate)
        self.assertIn("size-gate audit", mq3.next_work[0])
        self.assertIn("4B non-promotable", mq3.next_work[1])
        self.assertIn("coherence and KLD evidence", mq3.next_work[1])
        self.assertIn("9B non-promotable", mq3.next_work[2])
        self.assertIn("qwen3.5-27B BF16/Q8 KLD reference", mq3.next_work[3])
        self.assertIn("manifest-pin", mq3.next_work[3])
        self.assertIn("35B-A3B HFKLDR references", mq3.next_work[4])
        self.assertIn("A3B MQ4/MQ3 KLD cases", mq3.next_work[4])
        self.assertIn("paired A3B DFlash draft sidecars", mq3.next_work[5])
        self.assertIn("DFlash/spec rows", mq3.next_work[5])
        self.assertIn("capped A3B trains", mq3.next_work[6])
        self.assertIn("Three-run 27B", mq3.perf_gate)
        self.assertIn("12 tokens", mq3.perf_gate)
        self.assertIn("mq3-long-prefill-path2-shape-audit", " ".join(mq3.evidence_artifacts))
        self.assertIn("80-token cap", mq3.perf_gate)
        self.assertIn("dense_27b_ar_reached_token_cap=false", mq3.perf_gate)
        self.assertIn("dense_27b_dflash_token_attractor_clean=true", mq3.perf_gate)
        self.assertIn("missing 27B KLD reference", mq3.perf_gate)
        self.assertIn("broader prompt envelope", mq3.perf_gate)
        self.assertIn("Broaden 27B AR and DFlash/spec prompts", mq3.next_work[7])
        self.assertIn("A3B AR speed rows", mq3.next_work[8])
        self.assertIn("indexed MQ3 A3B MoE decode route", mq3.next_work[9])
        self.assertIn("artifact-backed MQ3 A3B long-prefill", mq3.next_work[10])
        self.assertIn("qwen3.6-35b-a3b", mq3.fixtures)
        self.assertIn("mq3-artifact-provenance", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-size-gate-audit", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-size-gate-status", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-coherence", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-boundary-coherence", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-4b-kld", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-9b-ppl", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-27b-ppl", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-kld-reference-inventory", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-27b-kld-reference-audit", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-9b-kld", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-a3b-coherence", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-a3b-ppl", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-a3b-broader-coherence", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-a3b-kld-reference-audit", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-a3b-dflash-fixture-audit", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-ar-perf", " ".join(mq3.evidence_artifacts))
        self.assertIn("mq3-dflash", " ".join(mq3.evidence_artifacts))

    def test_mq3_lloyd_records_9b_artifact_and_research_gate(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq3_lloyd = by_id["mq3-lloyd"]
        self.assertEqual(mq3_lloyd.status, "candidate-research-gated")
        self.assertIn("qwen3.5-9b-lloyd-mq3.hfq", mq3_lloyd.producer_gate)
        self.assertIn("Current 4B, 27B, and A3B", mq3_lloyd.producer_gate)
        self.assertIn("live local scan found two MQ3-Lloyd artifacts, both 9B", mq3_lloyd.producer_gate)
        self.assertIn("guarded 4B, 9B, 27B, and A3B MQ3-Lloyd cases", mq3_lloyd.producer_gate)
        self.assertIn("loads through the daemon", mq3_lloyd.runtime_gate)
        self.assertIn("short reasoning row and long-prefill LRU row", mq3_lloyd.runtime_gate)
        self.assertIn("routed expert decode and batched MoE", mq3_lloyd.runtime_gate)
        self.assertIn("final number 9", mq3_lloyd.quality_gate)
        self.assertIn("O(1) get/put", mq3_lloyd.quality_gate)
        self.assertIn("Current 2026-06-03 c20", mq3_lloyd.quality_gate)
        self.assertIn("0.553297", mq3_lloyd.quality_gate)
        self.assertIn("0.236385", mq3_lloyd.quality_gate)
        self.assertIn("dense promotion", mq3_lloyd.quality_gate)
        self.assertIn("Historical 2026-05-08", mq3_lloyd.quality_gate)
        self.assertIn("1.6913", mq3_lloyd.quality_gate)
        self.assertIn("lagged MQ4", mq3_lloyd.quality_gate)
        self.assertIn("live_only_9b_artifact_present=true", mq3_lloyd.quality_gate)
        self.assertIn("live_dense_27b_artifact_present=false", mq3_lloyd.quality_gate)
        self.assertIn("live_a3b_artifact_present=false", mq3_lloyd.quality_gate)
        self.assertIn("origin_refresh_required=true", mq3_lloyd.quality_gate)
        self.assertIn("wrong-fixture BF16 reference filenames", mq3_lloyd.quality_gate)
        self.assertIn("fresh-process gfx1151 speed baseline", mq3_lloyd.perf_gate)
        self.assertIn("branch reconciliation plus rerun evidence", mq3_lloyd.perf_gate)
        self.assertIn("dense-promotion rejected", mq3_lloyd.next_work[0])
        self.assertIn("new artifact hash", mq3_lloyd.next_work[0])
        self.assertIn("Do not spend 9B promotion perf time", mq3_lloyd.next_work[1])
        self.assertIn("BF16-referenced KLD/PPL gate", mq3_lloyd.next_work[1])
        self.assertIn("4B/27B/A3B MQ3-Lloyd artifacts", mq3_lloyd.next_work[2])
        self.assertIn("mq3-lloyd-artifact-provenance", " ".join(mq3_lloyd.evidence_artifacts))
        self.assertIn("coherence-full-after-mq3-lloyd", " ".join(mq3_lloyd.evidence_artifacts))
        self.assertIn("mq3-lloyd-9b-kld", " ".join(mq3_lloyd.evidence_artifacts))
        self.assertIn("lloyd-status", " ".join(mq3_lloyd.evidence_artifacts))
        self.assertIn("lloyd-kld-harness-audit", " ".join(mq3_lloyd.evidence_artifacts))
        self.assertIn("lloyd-historical-2026-05-08", " ".join(mq3_lloyd.evidence_artifacts))

    def test_mq4_lloyd_records_current_artifact_and_coherence_reject(self):
        by_id = {item.id: item for item in self.qr.READINESS}
        mq4_lloyd = by_id["mq4-lloyd"]
        self.assertEqual(mq4_lloyd.status, "candidate-research-gated")
        self.assertIn("qwen3.5-9b-lloyd-mq4.hfq", mq4_lloyd.producer_gate)
        self.assertIn("Astrea container inspect", mq4_lloyd.producer_gate)
        self.assertIn("qtype 30", mq4_lloyd.producer_gate)
        self.assertIn("Current 27B and A3B", mq4_lloyd.producer_gate)
        self.assertIn("live local scan found two MQ4-Lloyd artifacts, both 9B", mq4_lloyd.producer_gate)
        self.assertIn("guarded 9B, 27B, and A3B MQ4-Lloyd cases", mq4_lloyd.producer_gate)
        self.assertIn("loads through the daemon without a hard error", mq4_lloyd.runtime_gate)
        self.assertIn("container-inspect audit removes", mq4_lloyd.runtime_gate)
        self.assertIn("exclamation-token attractor", mq4_lloyd.runtime_gate)
        self.assertIn("Historical 2026-05-13", mq4_lloyd.quality_gate)
        self.assertIn("0.3114", mq4_lloyd.quality_gate)
        self.assertIn("0.2519", mq4_lloyd.quality_gate)
        self.assertIn("0.0510", mq4_lloyd.quality_gate)
        self.assertIn("160 B/group quality win", mq4_lloyd.quality_gate)
        self.assertIn("!!!!!!!!!!!", mq4_lloyd.quality_gate)
        self.assertIn("rejected for promotion", mq4_lloyd.quality_gate)
        self.assertIn("mq4-lloyd-9b-kld-c20", mq4_lloyd.quality_gate)
        self.assertIn("invalid zero-KLD/NaN-PPL", mq4_lloyd.quality_gate)
        self.assertIn("wrong-fixture BF16 reference filenames", mq4_lloyd.quality_gate)
        self.assertIn("producer or calibration quality", mq4_lloyd.quality_gate)
        self.assertIn("artifact_scope.only_9b_artifact_present=true", mq4_lloyd.quality_gate)
        self.assertIn("artifact_scope.a3b_artifact_present=false", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.status=value_not_justified_current_artifact", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.coherence_gate_passed=false", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.same_run_kld_ppl_valid=false", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.historical_rows_justify_value=false", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.origin_refresh_required=true", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.perf_collection_allowed=false", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.next_unblocked_step=produce_new_coherent_mq4_lloyd_artifact", mq4_lloyd.quality_gate)
        self.assertIn("producer_repair_plan.status=blocked_before_perf", mq4_lloyd.quality_gate)
        self.assertIn("producer_repair_plan.requires_new_artifact_hash=true", mq4_lloyd.quality_gate)
        self.assertIn("producer_repair_plan.same_run_quality_ready=false", mq4_lloyd.quality_gate)
        self.assertIn("producer_repair_plan.perf_collection_allowed=false", mq4_lloyd.quality_gate)
        self.assertIn("current_same_run_invalid_zero_kld_no_ppl as a forbidden gate", mq4_lloyd.quality_gate)
        self.assertIn("promotion_contract.historical_rows_required_for_promotion=false", mq4_lloyd.quality_gate)
        self.assertIn("promotion source of truth", mq4_lloyd.quality_gate)
        self.assertIn("value_boundary.performance_rows_promotable=false", mq4_lloyd.perf_gate)
        self.assertIn("readiness and container-inspect audits", mq4_lloyd.next_work[0])
        self.assertIn("coherence-rejected", mq4_lloyd.next_work[1])
        self.assertIn("current MQ4/MQ4-Lloyd/MQ6 KLD cohort", mq4_lloyd.next_work[3])
        self.assertIn("manifest-pinned refs", mq4_lloyd.next_work[3])
        self.assertIn("current same-run KLD/PPL", mq4_lloyd.next_work[4])
        self.assertIn("historical Lloyd rows are context only", mq4_lloyd.next_work[4])
        self.assertIn("producer_repair_plan", " ".join(mq4_lloyd.next_work))
        self.assertIn("mq4-lloyd-artifact-provenance", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("coherence-full-after-mq4-lloyd", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("lloyd-kld-harness-audit", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("mq4-lloyd-readiness-audit", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("mq4-lloyd-container-audit", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("lloyd-status", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("mq4-lloyd-9b-kld-c20", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("mq4-lloyd-astrea-inspect", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("mq6-control-astrea-inspect", " ".join(mq4_lloyd.evidence_artifacts))
        self.assertIn("lloyd-historical-2026-05-13", " ".join(mq4_lloyd.evidence_artifacts))

    def test_discover_fixture_reports_available_hf_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_hf_snapshot(tmp, "models--Qwen--Qwen3.5-9B", safetensors=4)
            report = self.qr.build_report(Path(tmp))
        fixtures = {item["id"]: item for item in report["fixtures"]}
        self.assertTrue(fixtures["qwen3.5-9b"]["present"])
        self.assertEqual(fixtures["qwen3.5-9b"]["snapshots"][0]["safetensors"], 4)
        self.assertFalse(fixtures["qwen3.5-27b"]["present"])

    def test_candidate_artifact_discovery_separates_format_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_root = root / "candidates"
            candidate_root.mkdir()
            for name in [
                "qwen3.5-9b-mq3.hfq",
                "qwen3.5-9b-lloyd-mq3.hfq",
                "qwen3.5-9b-lloyd-mq4.hfq",
                "qwen3.5-9b-mq6.hfq",
                "qwen3.5-9b-paro-mq4.hfq",
                "qwen3.5-9b-paro-g256-mq4.hfq",
            ]:
                (candidate_root / name).write_bytes(b"hfq")

            report = self.qr.build_report(root, candidate_root=candidate_root)

        by_id = {item["id"]: item for item in report["formats"]}
        self.assertEqual(
            [item["name"] for item in by_id["mq3"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-mq3.hfq"],
        )
        self.assertEqual(
            [item["name"] for item in by_id["mq3-lloyd"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-lloyd-mq3.hfq"],
        )
        self.assertEqual(
            [item["name"] for item in by_id["mq4-lloyd"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-lloyd-mq4.hfq"],
        )
        self.assertEqual(
            [item["name"] for item in by_id["mq6"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-mq6.hfq"],
        )
        self.assertEqual(
            [item["name"] for item in by_id["paro-q4g128"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-paro-mq4.hfq"],
        )
        self.assertEqual(
            [item["name"] for item in by_id["paro-q4g256"]["candidate_artifacts"]["artifacts"]],
            ["qwen3.5-9b-paro-g256-mq4.hfq"],
        )

    def test_candidate_artifact_discovery_reports_symlink_target_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backing = root / "legacy" / "qwen3.5-9b.mq6"
            backing.parent.mkdir()
            backing.write_bytes(b"legacy-hfq")
            candidate_root = root / "candidates"
            candidate_root.mkdir()
            candidate = candidate_root / "qwen3.5-9b-mq6.hfq"
            candidate.symlink_to(backing)

            report = self.qr.build_report(
                root,
                candidate_root=candidate_root,
                hash_algorithms=("sha256",),
            )

        by_id = {item["id"]: item for item in report["formats"]}
        artifact = by_id["mq6"]["candidate_artifacts"]["artifacts"][0]
        self.assertEqual(artifact["name"], "qwen3.5-9b-mq6.hfq")
        self.assertEqual(artifact["size_bytes"], len(b"legacy-hfq"))
        self.assertEqual(Path(artifact["symlink_target"]).name, "qwen3.5-9b.mq6")
        self.assertEqual(
            artifact["hashes"]["sha256"],
            hashlib.sha256(b"legacy-hfq").hexdigest(),
        )

    def test_format_filter_limits_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.qr.build_report(Path(tmp), format_ids=("mq6",))
        self.assertEqual([item["id"] for item in report["formats"]], ["mq6"])
        self.assertEqual(report["format_filter"], ["mq6"])

    def test_markdown_marks_present_and_missing_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_hf_snapshot(tmp, "models--Qwen--Qwen3.5-9B", safetensors=1)
            report = self.qr.build_report(Path(tmp))
        markdown = self.qr.markdown_table(report)
        self.assertIn("qwen3.5-9b (present)", markdown)
        self.assertIn("qwen3.5-27b (missing)", markdown)
        self.assertIn("| mq6 | candidate-runtime-ready |", markdown)
        self.assertIn("artifact-provenance", markdown)


if __name__ == "__main__":
    unittest.main()
