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
MODULE_PATH = ROOT / "scripts" / "paro_q4g256_status.py"
CURRENT_ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "gfx1151-quant-readiness"
    / "2026-06-03-paro-q4g256-status.json"
)


spec = importlib.util.spec_from_file_location("paro_q4g256_status", MODULE_PATH)
assert spec and spec.loader
g256_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g256_status
spec.loader.exec_module(g256_status)


CONTRACT_AUDIT = """
# Contract

| Variant | Avg output NRMSE vs source Paro | Worst output NRMSE | Avg payload ratio vs source |
|---|---:|---:|---:|
| `PARO4G256` AWQ regroup | 0.0859 | 0.1095 | 0.9817x |
| `PARO4G256_MQ` row-major G256 body + Paro side metadata | 0.0951 | 0.1114 | 1.0220x |
"""


PROBE_SCRIPT = """
parser.add_argument("--local-only", action="store_true")
payload = {
  "schema": "hipfire.astrea.paro_g256_probe.v0",
  "caveat": "Regroups/dequantizes existing G128 Paro weights; not a true G256 ParoQuant calibration run."
}
"""

PERFMAX_SUMMARY = """
# `feat/paro-g256-perfmax` -- final summary

## Exit decision: B

**Verdict: G256 quality-viable as opt-in research, not default.**

Per-expert PARO MoE kernels shipped in fivetide's PR #319.

## Next-steps

- **Phase 7 conditional ports** -- gfx1100 + gfx1151 rebuilds + benches.
"""

PROBE_RESULT = {
    "schema": "hipfire.astrea.paro_g256_probe.v0",
    "caveat": "Regroups/dequantizes existing G128 Paro weights; not a true G256 ParoQuant calibration run.",
    "model": "/models/qwen-paro",
    "modules_probed": 12,
    "summary": {
        "avg_output_nrmse": {
            "paro4g256_awq": 0.085,
            "paro4g256_mq_body": 0.096,
        },
        "avg_payload_ratio_vs_source": {
            "paro4g256_awq": 0.981,
            "paro4g256_mq_body_plus_side": 1.022,
        },
        "worst_output_nrmse": {
            "paro4g256_awq": 0.144,
            "paro4g256_mq_body": 0.155,
        },
    },
}


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_inventory(path: Path, *, native: bool = False, g256: bool = False):
    write(
        path,
        json.dumps(
            {
                "schema": "hipfire.astrea.paro_source_inventory.v0",
                "files_seen": 10,
                "filename_hit_count": 0 if not native else 1,
                "roots": [{"root": "/models", "exists": True}],
                "safetensor_dirs_scanned": 1,
                "safetensor_files_scanned": 1,
                "scan_errors": [],
                "native_paro": {
                    "complete_module_count": 1 if native else 0,
                    "incomplete_module_count": 0,
                    "g128_complete_module_count": 0 if g256 else (1 if native else 0),
                    "g256_complete_module_count": 1 if g256 else 0,
                },
                "decision": {
                    "native_paro_source_found": native,
                    "native_paro_g256_source_found": g256,
                    "quality_state": "verifiable" if g256 else "UNVERIFIABLE",
                },
            }
        ),
    )


def runtime_source(root: Path, *, wired: bool = False):
    token = "PARO4G256_MQ" if wired else "PARO4G128"
    write(root / "crates" / "hipfire-runtime" / "src" / "hfq.rs", token)
    write(root / "crates" / "hipfire-arch-qwen35" / "src" / "qwen35.rs", token)


def upstream_evidence(root: Path):
    summary = root / "SUMMARY.md"
    probe_a = root / "g256-probe-0.8b.json"
    probe_b = root / "g256-probe-9b.json"
    write(summary, PERFMAX_SUMMARY)
    write(probe_a, json.dumps(PROBE_RESULT))
    probe_b_payload = dict(PROBE_RESULT)
    probe_b_payload["model"] = "/models/qwen-9b-paro"
    probe_b_payload["summary"] = {
        **PROBE_RESULT["summary"],
        "avg_payload_ratio_vs_source": {
            "paro4g256_awq": 0.982,
            "paro4g256_mq_body_plus_side": 1.023,
        },
    }
    write(probe_b, json.dumps(probe_b_payload))
    return summary, (probe_a, probe_b)


class ParoQ4G256StatusTest(unittest.TestCase):
    def test_parse_format_loss_table_records_payload_ratio_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.md"
            write(audit, CONTRACT_AUDIT)

            parsed = g256_status.parse_format_loss_table(audit)

        self.assertTrue(parsed["regrouped_mq_payload_ratio_gate_passed"])
        self.assertFalse(parsed["current_true_g256_payload_ratio_evidence_present"])
        self.assertFalse(parsed["current_true_g256_payload_ratio_gate_passed"])
        self.assertFalse(parsed["mq_payload_ratio_gate_passed"])
        self.assertEqual(parsed["variants"]["PARO4G256_MQ"]["avg_payload_ratio_vs_source"], 1.022)
        self.assertFalse(parsed["true_g256_quality_evidence_present"])

    def test_build_status_blocks_runtime_when_true_source_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = root / "inventory.json"
            audit = root / "audit.md"
            probe = root / "probe.py"
            source = root / "src"
            summary, probe_results = upstream_evidence(root / "upstream")
            write_inventory(inv, native=False, g256=False)
            write(audit, CONTRACT_AUDIT)
            write(probe, PROBE_SCRIPT)
            runtime_source(source, wired=False)

            status = g256_status.build_status(
                source_inventory=inv,
                contract_audit=audit,
                probe_script=probe,
                source_root=source,
                upstream_summary=summary,
                probe_results=probe_results,
            )

        self.assertEqual(status["status"], "prototype-only")
        self.assertFalse(status["promotion_allowed"])
        self.assertTrue(status["gates"]["producer_contract_explicit"])
        self.assertTrue(status["gates"]["cpu_probe_available"])
        self.assertTrue(status["gates"]["upstream_g256_research_evidence_present"])
        self.assertFalse(status["gates"]["native_g256_source_found"])
        self.assertFalse(status["gates"]["cpu_probe_runnable_now"])
        self.assertTrue(status["gates"]["regrouped_payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["current_true_g256_payload_ratio_evidence_present"])
        self.assertFalse(status["gates"]["current_true_g256_payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["true_g256_quality_evidence_present"])
        self.assertFalse(status["gates"]["runtime_dtype_container_ready"])
        self.assertEqual(
            status["contract_state"]["current_stage"],
            "blocked_before_true_group_size_256_source",
        )
        self.assertTrue(status["contract_state"]["quality_marked_unverifiable"])
        self.assertEqual(status["contract_state"]["evidence_class"], "regrouped_g128_format_loss_only")
        producer = status["contract_state"]["producer_contract"]
        self.assertEqual(producer["required_tensor_families"], list(g256_status.REQUIRED_PARO_SUFFIXES))
        self.assertEqual(producer["required_group_size"], 256)
        self.assertEqual(producer["complete_g256_module_count"], 0)
        self.assertEqual(producer["source_search_result"], "no_paro_suffix_or_filename_hits")
        self.assertTrue(producer["quality_verifiable_only_with_true_g256_source"])
        self.assertFalse(status["contract_state"]["runtime_work_allowed"])
        self.assertIn("true_group_size_256_source", status["contract_state"]["runtime_work_blocked_by"])
        upstream = status["contract_state"]["upstream_g256_evidence"]
        self.assertEqual(upstream["summary"]["exit_decision"], "B")
        self.assertTrue(upstream["summary"]["g256_opt_in_research_not_default"])
        self.assertTrue(upstream["probes"]["all_probe_caveats_regrouped_g128"])
        self.assertFalse(upstream["true_g256_source_evidence_present"])
        self.assertFalse(upstream["promotion_proof"])
        boundary = status["prototype_boundary"]
        self.assertEqual(boundary["current_stage"], "blocked_before_true_group_size_256_source")
        self.assertTrue(boundary["true_group_size_256_source_required"])
        self.assertFalse(boundary["true_group_size_256_source_found"])
        self.assertEqual(boundary["producer_contract"]["required_group_size"], 256)
        self.assertEqual(boundary["producer_contract"]["source_search_result"], "no_paro_suffix_or_filename_hits")
        self.assertEqual(boundary["source_inventory_quality_state"], "UNVERIFIABLE")
        self.assertTrue(boundary["quality_marked_unverifiable"])
        self.assertTrue(boundary["format_loss_ratio_passed"])
        self.assertFalse(boundary["current_true_g256_payload_ratio_gate_passed"])
        self.assertEqual(boundary["format_loss_evidence_class"], "regrouped_g128_format_loss_only")
        self.assertTrue(boundary["payload_ratio_only_not_promotion_evidence"])
        self.assertFalse(boundary["true_g256_quality_evidence_present"])
        self.assertTrue(boundary["kld_ppl_within_5_percent_of_paro_q4g128_required"])
        self.assertFalse(boundary["kld_ppl_within_5_percent_of_paro_q4g128_proven"])
        self.assertFalse(boundary["runtime_dtype_container_work_allowed"])
        self.assertTrue(boundary["upstream_g256_research_evidence_present"])
        self.assertEqual(boundary["upstream_g256_evidence_class"], "regrouped_g128_opt_in_research")
        self.assertFalse(boundary["upstream_g256_evidence_is_promotion_proof"])
        self.assertFalse(boundary["artifact_generation_without_true_source_allowed"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "locate_or_generate_true_group_size_256_paro_source",
        )
        self.assertIn("true group_size=256", " ".join(status["blockers"]))

    def test_runtime_dtypes_do_not_unblock_without_true_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = root / "inventory.json"
            audit = root / "audit.md"
            probe = root / "probe.py"
            source = root / "src"
            summary, probe_results = upstream_evidence(root / "upstream")
            write_inventory(inv, native=True, g256=True)
            write(audit, CONTRACT_AUDIT)
            write(probe, PROBE_SCRIPT)
            runtime_source(source, wired=True)

            status = g256_status.build_status(
                source_inventory=inv,
                contract_audit=audit,
                probe_script=probe,
                source_root=source,
                upstream_summary=summary,
                probe_results=probe_results,
            )

        self.assertTrue(status["gates"]["native_g256_source_found"])
        self.assertTrue(status["gates"]["cpu_probe_runnable_now"])
        self.assertTrue(status["gates"]["upstream_g256_research_evidence_present"])
        self.assertTrue(status["gates"]["runtime_dtype_container_ready"])
        self.assertTrue(status["gates"]["regrouped_payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["current_true_g256_payload_ratio_evidence_present"])
        self.assertFalse(status["gates"]["current_true_g256_payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["payload_ratio_gate_passed"])
        self.assertFalse(status["gates"]["true_g256_quality_evidence_present"])
        self.assertFalse(status["gates"]["quality_comparable_to_paro_q4g128"])
        self.assertFalse(status["promotion_allowed"])
        boundary = status["prototype_boundary"]
        self.assertFalse(boundary["runtime_dtype_container_work_allowed"])
        self.assertTrue(boundary["payload_ratio_only_not_promotion_evidence"])
        self.assertFalse(boundary["current_true_g256_payload_ratio_gate_passed"])
        self.assertIn("paro_oracle_comparison", boundary["runtime_dtype_container_work_blocked_by"])
        self.assertIn(
            "kld_ppl_within_5_percent_of_paro_q4g128",
            boundary["runtime_dtype_container_work_blocked_by"],
        )
        self.assertEqual(boundary["next_unblocked_step"], "run_paro_oracle_against_true_g256_export")

    def test_g128_native_source_does_not_make_g256_probe_runnable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = root / "inventory.json"
            audit = root / "audit.md"
            probe = root / "probe.py"
            source = root / "src"
            summary, probe_results = upstream_evidence(root / "upstream")
            write_inventory(inv, native=True, g256=False)
            write(audit, CONTRACT_AUDIT)
            write(probe, PROBE_SCRIPT)
            runtime_source(source, wired=False)

            status = g256_status.build_status(
                source_inventory=inv,
                contract_audit=audit,
                probe_script=probe,
                source_root=source,
                upstream_summary=summary,
                probe_results=probe_results,
            )

        self.assertFalse(status["gates"]["native_g256_source_found"])
        self.assertFalse(status["gates"]["cpu_probe_runnable_now"])
        self.assertEqual(
            status["prototype_boundary"]["next_unblocked_step"],
            "locate_or_generate_true_group_size_256_paro_source",
        )
        self.assertIn(
            "true_group_size_256_source",
            status["contract_state"]["runtime_work_blocked_by"],
        )

    def test_missing_upstream_summary_blocks_research_evidence_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = root / "inventory.json"
            audit = root / "audit.md"
            probe = root / "probe.py"
            source = root / "src"
            missing_summary = root / "missing-summary.md"
            probe_results = (root / "missing-probe.json",)
            write_inventory(inv, native=False, g256=False)
            write(audit, CONTRACT_AUDIT)
            write(probe, PROBE_SCRIPT)
            runtime_source(source, wired=False)

            status = g256_status.build_status(
                source_inventory=inv,
                contract_audit=audit,
                probe_script=probe,
                source_root=source,
                upstream_summary=missing_summary,
                probe_results=probe_results,
            )

        self.assertFalse(status["gates"]["upstream_g256_research_evidence_present"])
        self.assertFalse(status["contract_state"]["upstream_g256_evidence"]["research_evidence_present"])
        self.assertIn("origin G256 perfmax summary/probe evidence is missing", status["blockers"])

    def test_current_status_artifact_is_prototype_only(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], g256_status.SCHEMA)
        self.assertEqual(payload["status"], "prototype-only")
        self.assertFalse(payload["promotion_allowed"])
        self.assertFalse(payload["gates"]["native_g256_source_found"])
        self.assertFalse(payload["gates"]["cpu_probe_runnable_now"])
        self.assertTrue(payload["gates"]["cpu_probe_available"])
        self.assertTrue(payload["gates"]["upstream_g256_research_evidence_present"])
        self.assertTrue(payload["gates"]["regrouped_payload_ratio_gate_passed"])
        self.assertFalse(payload["gates"]["current_true_g256_payload_ratio_evidence_present"])
        self.assertFalse(payload["gates"]["current_true_g256_payload_ratio_gate_passed"])
        self.assertFalse(payload["gates"]["payload_ratio_gate_passed"])
        self.assertFalse(payload["gates"]["true_g256_quality_evidence_present"])
        self.assertFalse(payload["gates"]["quality_comparable_to_paro_q4g128"])
        self.assertFalse(payload["gates"]["runtime_dtype_container_ready"])
        self.assertEqual(
            payload["contract_state"]["required_stage_order"],
            list(g256_status.CONTRACT_STAGES),
        )
        self.assertEqual(
            payload["contract_state"]["current_stage"],
            "blocked_before_true_group_size_256_source",
        )
        self.assertEqual(payload["contract_state"]["quality_state"], "UNVERIFIABLE")
        self.assertTrue(payload["contract_state"]["quality_marked_unverifiable"])
        self.assertEqual(payload["contract_state"]["evidence_class"], "regrouped_g128_format_loss_only")
        producer = payload["contract_state"]["producer_contract"]
        self.assertEqual(producer["required_tensor_families"], list(g256_status.REQUIRED_PARO_SUFFIXES))
        self.assertEqual(producer["required_group_size"], 256)
        self.assertEqual(producer["complete_native_paro_module_count"], 0)
        self.assertEqual(producer["complete_g256_module_count"], 0)
        self.assertEqual(producer["source_search_result"], "no_paro_suffix_or_filename_hits")
        self.assertTrue(producer["quality_verifiable_only_with_true_g256_source"])
        self.assertFalse(payload["contract_state"]["runtime_work_allowed"])
        self.assertIn("true_group_size_256_source", payload["contract_state"]["runtime_work_blocked_by"])
        upstream = payload["contract_state"]["upstream_g256_evidence"]
        self.assertEqual(upstream["summary"]["exit_decision"], "B")
        self.assertTrue(upstream["summary"]["g256_opt_in_research_not_default"])
        self.assertTrue(upstream["probes"]["all_probe_caveats_regrouped_g128"])
        self.assertEqual(upstream["probes"]["probe_count"], 2)
        self.assertFalse(upstream["true_g256_source_evidence_present"])
        self.assertFalse(upstream["promotion_proof"])
        boundary = payload["prototype_boundary"]
        self.assertEqual(boundary["current_stage"], "blocked_before_true_group_size_256_source")
        self.assertTrue(boundary["true_group_size_256_source_required"])
        self.assertFalse(boundary["true_group_size_256_source_found"])
        self.assertEqual(boundary["producer_contract"]["required_tensor_families"], list(g256_status.REQUIRED_PARO_SUFFIXES))
        self.assertEqual(boundary["producer_contract"]["source_search_result"], "no_paro_suffix_or_filename_hits")
        self.assertEqual(boundary["source_inventory_quality_state"], "UNVERIFIABLE")
        self.assertTrue(boundary["quality_marked_unverifiable"])
        self.assertTrue(boundary["format_loss_ratio_passed"])
        self.assertFalse(boundary["current_true_g256_payload_ratio_gate_passed"])
        self.assertEqual(boundary["format_loss_evidence_class"], "regrouped_g128_format_loss_only")
        self.assertTrue(boundary["payload_ratio_only_not_promotion_evidence"])
        self.assertFalse(boundary["true_g256_quality_evidence_present"])
        self.assertTrue(boundary["kld_ppl_within_5_percent_of_paro_q4g128_required"])
        self.assertFalse(boundary["kld_ppl_within_5_percent_of_paro_q4g128_proven"])
        self.assertFalse(boundary["runtime_dtype_container_work_allowed"])
        self.assertTrue(boundary["upstream_g256_research_evidence_present"])
        self.assertEqual(boundary["upstream_g256_evidence_class"], "regrouped_g128_opt_in_research")
        self.assertFalse(boundary["upstream_g256_evidence_is_promotion_proof"])
        self.assertFalse(boundary["artifact_generation_without_true_source_allowed"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "locate_or_generate_true_group_size_256_paro_source",
        )
        self.assertEqual(
            payload["next_unblocked_step"],
            "locate_or_generate_true_group_size_256_paro_source",
        )
        plan = payload["prototype_plan"]
        self.assertEqual(plan["status"], "blocked_before_true_group_size_256_source")
        self.assertEqual(plan["stage_order"], list(g256_status.CONTRACT_STAGES))
        self.assertFalse(plan["stages"]["true_group_size_256_source"]["satisfied"])
        self.assertEqual(
            plan["stages"]["true_group_size_256_source"]["required_group_size"],
            256,
        )
        self.assertFalse(plan["stages"]["cpu_g256_probe"]["satisfied"])
        self.assertEqual(
            plan["stages"]["cpu_g256_probe"]["blocked_by"],
            ["true_group_size_256_source"],
        )
        self.assertIn(
            "scripts/paroquant_g256_probe.py --model",
            plan["stages"]["cpu_g256_probe"]["command_template"],
        )
        self.assertFalse(plan["stages"]["runtime_dtype_container_work"]["satisfied"])
        self.assertIn(
            "kld_ppl_within_5_percent_of_paro_q4g128",
            plan["stages"]["runtime_dtype_container_work"]["blocked_by"],
        )
        self.assertEqual(payload["next_work"], plan["next_work"])
        by_short = {
            item["short"]: item
            for item in payload["origin_context"]["relevant_upstream_commits"]
        }
        self.assertTrue(by_short["79d30777"]["present_on_origin_master"])
        self.assertTrue(by_short["f1e2cef8"]["present_on_origin_master"])
        self.assertTrue(by_short["907cbb2f"]["present_on_origin_master"])


if __name__ == "__main__":
    unittest.main()
