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
MODULE_PATH = ROOT / "scripts" / "mq6_status.py"
CURRENT_ARTIFACT = (
    ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness" / "2026-06-03-mq6-status.json"
)


spec = importlib.util.spec_from_file_location("mq6_status", MODULE_PATH)
assert spec and spec.loader
mq6_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq6_status
spec.loader.exec_module(mq6_status)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload):
    write(path, json.dumps(payload))


class Mq6StatusTest(unittest.TestCase):
    def test_ppl_comparison_detects_candidate_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ppl.json"
            write_json(
                path,
                {
                    "cases": [
                        {"id": "qwen35-9b-mq4", "format_id": "mq4", "result": {"ppl": 12.1}},
                        {"id": "qwen35-9b-mq6", "format_id": "mq6", "result": {"ppl": 11.5}},
                    ]
                },
            )

            comparison = mq6_status.ppl_comparison(path, "qwen35-9b-mq6", "qwen35-9b-mq4")

        self.assertTrue(comparison["candidate_beats_control"])
        self.assertLess(comparison["candidate_to_control_ppl_ratio"], 1.0)

    def test_kld_comparison_detects_candidate_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kld.json"
            write_json(
                path,
                {
                    "reduced_rows": [
                        {"variant": "qwen3.5-9b.mq4-kvq8-c512", "mean_kld": 0.25, "ppl": 8.8},
                        {"variant": "qwen3.5-9b.mq6-kvq8-c512", "mean_kld": 0.05, "ppl": 9.2},
                    ]
                },
            )

            comparison = mq6_status.kld_comparison(path, "qwen3.5-9b.mq6", "qwen3.5-9b.mq4")

        self.assertTrue(comparison["candidate_beats_control"])
        self.assertEqual(comparison["candidate_to_control_mean_kld_ratio"], 0.2)

    def test_perf_pair_records_dense_candidate_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.json"
            write_json(
                path,
                {
                    "cases": [
                        {
                            "id": "control",
                            "summary": {
                                "median_prefill_tok_s": 100.0,
                                "median_decode_tok_s": 40.0,
                                "ok_runs": 3,
                                "total_runs": 3,
                            },
                        },
                        {
                            "id": "candidate",
                            "summary": {
                                "median_prefill_tok_s": 45.0,
                                "median_decode_tok_s": 30.0,
                                "ok_runs": 3,
                                "total_runs": 3,
                            },
                        },
                    ]
                },
            )

            pair = mq6_status.perf_pair(path, "candidate", "control")

        self.assertTrue(pair["all_runs_ok"])
        self.assertEqual(pair["prefill_ratio"], 0.45)
        self.assertEqual(pair["decode_ratio"], 0.75)

    def test_dflash_pair_requires_token_attractor_clean_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dflash.json"
            run = {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}}
            write_json(
                path,
                {
                    "cases": [
                        {
                            "id": "control",
                            "runs": [run, run, run],
                            "summary": {
                                "median_prefill_tok_s": 50.0,
                                "median_decode_tok_s": 25.0,
                                "median_decode_tau": 2.0,
                                "median_decode_accept_rate": 0.2,
                                "ok_runs": 3,
                                "total_runs": 3,
                            },
                        },
                        {
                            "id": "candidate",
                            "runs": [run, run, run],
                            "summary": {
                                "median_prefill_tok_s": 25.0,
                                "median_decode_tok_s": 10.0,
                                "median_decode_tau": 2.0,
                                "median_decode_accept_rate": 0.2,
                                "ok_runs": 3,
                                "total_runs": 3,
                            },
                        },
                    ]
                },
            )

            pair = mq6_status.dflash_pair(path, "candidate", "control")

        self.assertTrue(pair["token_attractor_ok"])
        self.assertEqual(pair["prefill_ratio"], 0.5)
        self.assertEqual(pair["decode_ratio"], 0.4)
        self.assertEqual(pair["tau_ratio"], 1.0)

    def test_dflash_spec_boundary_marks_mq4_draft_target_verify_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dflash.json"
            run = {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}}
            write_json(
                path,
                {
                    "schema": "hipfire.mq6_dflash.gfx1151.v0",
                    "cases": [
                        {
                            "id": "control",
                            "target_format": "mq4",
                            "target_file": "qwen3.5-27b.mq4",
                            "draft_file": "qwen35-27b-dflash-mq4.hfq",
                            "prompt_md5": "abc",
                            "runs": [run, run, run],
                            "summary": {"total_runs": 3},
                        },
                        {
                            "id": "candidate",
                            "target_format": "mq6",
                            "target_file": "qwen3.5-27b.mq6",
                            "draft_file": "qwen35-27b-dflash-mq4.hfq",
                            "prompt_md5": "abc",
                            "runs": [run, run, run],
                            "summary": {"total_runs": 3},
                        },
                    ],
                },
            )

            boundary = mq6_status.dflash_spec_boundary(
                path,
                {"mq6_evidence_refresh_required": True},
            )

        self.assertEqual(boundary["schema"], "hipfire.mq6_dflash.gfx1151.v0")
        self.assertEqual(boundary["mq6_case_count"], 1)
        self.assertTrue(boundary["target_side_verify_only"])
        self.assertEqual(boundary["evidence_class"], "target_side_verify_only_mq4_draft")
        self.assertFalse(boundary["scope_expanded_beyond_target_verify"])
        self.assertTrue(boundary["contract_ready"])
        self.assertFalse(boundary["promotion_scope_supported"])
        self.assertTrue(boundary["mq4_draft_used_for_all_mq6_rows"])
        self.assertFalse(boundary["mq6_draft_evidence_present"])
        self.assertTrue(boundary["dense_27b_only"])
        self.assertFalse(boundary["a3b_dflash_evidence_present"])
        self.assertFalse(boundary["sampling_or_non_greedy_evidence_present"])
        self.assertTrue(boundary["all_mq6_cases_three_run"])
        self.assertTrue(boundary["token_attractor_clean"])
        self.assertTrue(boundary["refresh_required"])

    def test_dflash_spec_boundary_detects_mq6_draft_scope_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dflash.json"
            run = {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}}
            write_json(
                path,
                {
                    "schema": "hipfire.mq6_dflash.gfx1151.v0",
                    "cases": [
                        {
                            "id": "candidate",
                            "target_format": "mq6",
                            "target_file": "qwen3.5-27b.mq6",
                            "draft_file": "qwen35-27b-dflash-mq6.hfq",
                            "prompt_md5": "abc",
                            "runs": [run, run, run],
                            "summary": {"total_runs": 3},
                        },
                    ],
                },
            )

            boundary = mq6_status.dflash_spec_boundary(
                path,
                {"mq6_evidence_refresh_required": False},
            )

        self.assertEqual(boundary["evidence_class"], "mq6_draft_evidence_present")
        self.assertFalse(boundary["target_side_verify_only"])
        self.assertTrue(boundary["scope_expanded_beyond_target_verify"])
        self.assertTrue(boundary["mq6_draft_evidence_present"])
        self.assertFalse(boundary["a3b_dflash_evidence_present"])
        self.assertTrue(boundary["contract_ready"])
        self.assertFalse(boundary["promotion_scope_supported"])

    def test_dflash_spec_boundary_detects_a3b_scope_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dflash.json"
            run = {"status": "ok", "exit_code": 0, "token_attractor": {"ok": True}}
            write_json(
                path,
                {
                    "schema": "hipfire.mq6_dflash.gfx1151.v0",
                    "cases": [
                        {
                            "id": "candidate",
                            "target_format": "mq6",
                            "target_file": "qwen3.5-35b-a3b.mq6",
                            "draft_file": "qwen35-a3b-dflash-mq4.hfq",
                            "prompt_md5": "abc",
                            "runs": [run, run, run],
                            "summary": {"total_runs": 3},
                        },
                    ],
                },
            )

            boundary = mq6_status.dflash_spec_boundary(
                path,
                {"mq6_evidence_refresh_required": False},
            )

        self.assertEqual(boundary["evidence_class"], "a3b_dflash_evidence_present")
        self.assertFalse(boundary["target_side_verify_only"])
        self.assertTrue(boundary["scope_expanded_beyond_target_verify"])
        self.assertFalse(boundary["mq6_draft_evidence_present"])
        self.assertTrue(boundary["a3b_dflash_evidence_present"])
        self.assertFalse(boundary["dense_27b_only"])
        self.assertTrue(boundary["contract_ready"])
        self.assertFalse(boundary["promotion_scope_supported"])

    def test_live_candidate_inventory_follows_symlinked_canonical_mq6_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "Models"
            candidate_root = models_root / mq6_status.CANDIDATE_SUBDIR
            backing_root = root / "legacy"
            candidate_root.mkdir(parents=True)
            backing_root.mkdir()
            for name in mq6_status.REQUIRED_MQ6_ARTIFACTS:
                backing = backing_root / name.replace("-mq6.hfq", ".mq6")
                backing.write_bytes(f"payload:{name}".encode("utf-8"))
                (candidate_root / name).symlink_to(backing)

            inventory = mq6_status.live_candidate_inventory(models_root)

        self.assertTrue(inventory["candidate_root_exists"])
        self.assertEqual(inventory["artifact_count"], 4)
        self.assertEqual(inventory["symlink_count"], 4)
        self.assertTrue(inventory["canonical_mq6_artifacts_present"])
        self.assertEqual(inventory["required_artifacts_missing"], [])
        by_name = {artifact["name"]: artifact for artifact in inventory["artifacts"]}
        self.assertEqual(
            Path(by_name["qwen3.5-9b-mq6.hfq"]["symlink_target"]).name,
            "qwen3.5-9b.mq6",
        )
        self.assertGreater(by_name["qwen3.5-9b-mq6.hfq"]["size_bytes"], 0)

    def test_kernel_inventory_separates_repo_and_install_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo" / "kernels" / "compiled" / "gfx1151"
            install_root = root / "install" / "kernels" / "compiled" / "gfx1151"
            repo_root.mkdir(parents=True)
            install_root.mkdir(parents=True)
            for name in mq6_status.REQUIRED_MQ6_DENSE_DECODE_KERNELS:
                (repo_root / name).write_bytes(b"kernel")
            for name in mq6_status.REQUIRED_MQ6_DENSE_DECODE_KERNELS[:-1]:
                (install_root / name).write_bytes(b"kernel")

            inventory = mq6_status.kernel_inventory(repo_root=repo_root, install_root=install_root)

        self.assertEqual(
            inventory["schema"],
            "hipfire.mq6_dense_decode_kernel_inventory.gfx1151.v0",
        )
        self.assertTrue(inventory["repo_required_kernels_present"])
        self.assertFalse(inventory["install_required_kernels_present"])
        self.assertTrue(inventory["any_required_kernel_set_present"])
        self.assertEqual(
            inventory["roots"]["install"]["missing"],
            [mq6_status.REQUIRED_MQ6_DENSE_DECODE_KERNELS[-1]],
        )
        self.assertIn("gemv_mq6g256.hsaco", inventory["required_kernels"])
        self.assertEqual(inventory["evidence_class"], "compiled_blob_presence_only_not_dispatch_proof")

    def test_producer_path_records_mq4_lloyd_promote6_emit_mq6(self):
        evidence = mq6_status.mq6_producer_path_evidence()

        self.assertTrue(evidence["source_present"])
        self.assertTrue(evidence["promote6_mq4_lloyd_emits_mq6"])
        self.assertTrue(evidence["origin_fix_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "d5985c3e")
        self.assertEqual(evidence["target_quant_type"], "MQ6G256")
        self.assertEqual(evidence["target_group_size"], 256)
        self.assertIn("GgufFormat::Mq4Lloyd", evidence["matched_source"])
        self.assertIn("quantize_mq6g256", evidence["matched_source"])

    def test_batched_rope_compact_offset_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_batched_rope_compact_offset_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_batched_rope_compact_offset.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "676338a4")
        self.assertGreaterEqual(evidence["qwen35_compact_offset_callsite_count"], 2)
        self.assertTrue(evidence["checks"]["interleaved_kernel_applies_pos_offset"])
        self.assertTrue(evidence["checks"]["halfsplit_kernel_applies_pos_offset"])
        self.assertTrue(evidence["checks"]["dispatch_pushes_pos_offset_kernarg"])
        self.assertTrue(evidence["checks"]["qwen35_batched_fa_passes_compact_offset"])
        self.assertTrue(evidence["checks"]["mtp_head_explicit_zero_offset"])
        self.assertTrue(evidence["checks"]["tree_verify_explicit_zero_offset"])
        self.assertTrue(evidence["checks"]["no_model_equivalence_example_present"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["long_context_or_eviction_rows_refreshed_by_this_check"])
        self.assertIn("rope_compact_offset_check", evidence["validation_command"])

    def test_daemon_dflash_first_token_eos_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_daemon_dflash_first_token_eos_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_daemon_dflash_first_token_eos.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "d5bc3f6")
        self.assertTrue(evidence["checks"]["first_token_guard_declared"])
        self.assertTrue(evidence["checks"]["checks_model_eos_token"])
        self.assertTrue(evidence["checks"]["checks_chat_im_end_token"])
        self.assertTrue(evidence["checks"]["checks_tokenizer_terminator"])
        self.assertTrue(evidence["checks"]["spec_loop_guarded"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["coherence_rows_refreshed_by_this_check"])

    def test_daemon_ar_exact_match_delta_replay_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_daemon_ar_exact_match_delta_replay_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_daemon_ar_exact_match_delta_replay.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "d5bc3f6")
        self.assertTrue(evidence["checks"]["generate_window_found"])
        self.assertTrue(evidence["checks"]["qwen35_single_gpu_branch_found"])
        self.assertTrue(evidence["checks"]["legacy_exact_match_lcp_rewind_absent"])
        self.assertTrue(evidence["checks"]["qwen35_prefills_direct_new_tokens"])
        self.assertTrue(evidence["checks"]["qwen35_no_conversation_lcp_scan"])
        self.assertIn("lcp_adj", evidence["legacy_terms_checked"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["coherence_rows_refreshed_by_this_check"])

    def test_ddtree_eviction_target_hidden_host_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_ddtree_eviction_target_hidden_host_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_ddtree_eviction_target_hidden_host.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "d12b3ad4")
        self.assertEqual(evidence["remote_ref"], "origin/fix/issue-triage-2026-06-03")
        self.assertTrue(evidence["checks"]["helper_declared"])
        self.assertTrue(evidence["checks"]["helper_uses_retain_mask_rows"])
        self.assertTrue(evidence["checks"]["unit_test_selected_rows_present"])
        self.assertTrue(evidence["checks"]["unit_test_empty_mask_present"])
        self.assertTrue(evidence["checks"]["daemon_post_prefill_and_cycle_calls_present"])
        self.assertTrue(evidence["checks"]["dflash_spec_demo_post_prefill_and_cycle_calls_present"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["coherence_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["long_context_or_eviction_rows_refreshed_by_this_check"])
        self.assertIn("compact_target_hidden_host", evidence["validation_command"])

    def test_moe_prefill_grouped_scratch_free_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_moe_prefill_grouped_scratch_free_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_moe_prefill_grouped_scratch_free.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "b4adca1f")
        self.assertTrue(evidence["checks"]["prefill_batch_impl_found"])
        self.assertTrue(evidence["checks"]["prefill_batch_free_gpu_found"])
        self.assertTrue(evidence["checks"]["frees_moe_expert_token_counts"])
        self.assertTrue(evidence["checks"]["frees_moe_expert_offsets"])
        self.assertTrue(evidence["checks"]["frees_moe_sorted_slot_index"])
        self.assertTrue(evidence["checks"]["frees_moe_inverse_perm"])
        self.assertTrue(evidence["checks"]["frees_moe_expert_tile_ids"])
        self.assertTrue(evidence["checks"]["frees_moe_y_gate_up_grouped"])
        self.assertTrue(evidence["checks"]["frees_moe_y_down_grouped"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["long_running_moe_stability_refreshed_by_this_check"])

    def test_cask_scratch_teardown_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_cask_scratch_teardown_evidence()

        self.assertEqual(evidence["schema"], "hipfire.mq6_cask_scratch_teardown.gfx1151.v0")
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(evidence["checks"]["fallible_work_wrapped_in_inner_result"])
        self.assertTrue(evidence["checks"]["indices_scratch_freed_after_inner_result"])
        self.assertTrue(evidence["checks"]["weights_scratch_freed_after_inner_result"])
        self.assertTrue(evidence["checks"]["inner_result_propagated_after_free"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["long_context_or_eviction_rows_refreshed_by_this_check"])

    def test_mtp_gpu_teardown_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_mtp_gpu_teardown_evidence()

        self.assertEqual(evidence["schema"], "hipfire.mq6_mtp_gpu_teardown.gfx1151.v0")
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(evidence["checks"]["mtp_head_kv_cache_frees_inner_gpu_tensors"])
        self.assertTrue(evidence["checks"]["mtp_head_no_drop_inner_in_kv_free_gpu"])
        self.assertTrue(evidence["checks"]["mtp_spec_frees_trunk_snapshot_gpu"])
        self.assertTrue(evidence["checks"]["mtp_spec_frees_mtp_kv_inner_gpu_tensors"])
        self.assertTrue(evidence["checks"]["mtp_spec_no_drop_trunk_snapshot"])
        self.assertTrue(evidence["checks"]["mtp_compose_frees_both_mtp_kv_inner_states"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["long_running_mtp_rows_refreshed_by_this_check"])

    def test_asym4_fwht_fallback_source_fix_is_reconciled(self):
        evidence = mq6_status.mq6_asym4_fwht_fallback_evidence()

        self.assertEqual(evidence["schema"], "hipfire.mq6_asym4_fwht_fallback.gfx1151.v0")
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(evidence["checks"]["run_fa_layer_body_found"])
        self.assertTrue(evidence["checks"]["fwht_subbranch_found"])
        self.assertTrue(evidence["checks"]["fwht_writer_dispatched"])
        self.assertTrue(evidence["checks"]["fwht_attention_dispatched"])
        self.assertTrue(evidence["checks"]["fwht4_uses_branch_native_q8_v_signature"])
        self.assertTrue(evidence["checks"]["asym4_fallback_writer_preserved"])
        self.assertTrue(evidence["checks"]["asym4_fallback_attention_preserved"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["coherence_rows_refreshed_by_this_check"])

    def test_qwen36_a3b_runtime_source_reconciliation_is_recorded_without_model_rerun(self):
        evidence = mq6_status.mq6_qwen36_a3b_runtime_source_reconciliation_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_qwen36_a3b_runtime_source_reconciliation.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "379484ba")
        self.assertTrue(evidence["checks"]["sampler_config_has_native_penalties"])
        self.assertTrue(evidence["checks"]["sampler_gpu_path_threads_penalties"])
        self.assertTrue(evidence["checks"]["sampler_cpu_path_subtracts_additive_penalties"])
        self.assertTrue(evidence["checks"]["dispatch_exposes_sample_top_p_pf"])
        self.assertTrue(evidence["checks"]["sample_top_p_kernel_accepts_and_applies_penalties"])
        self.assertTrue(evidence["checks"]["daemon_parses_and_threads_native_penalties"])
        self.assertTrue(evidence["checks"]["daemon_allocates_wide_repeat_buffers"])
        self.assertTrue(evidence["checks"]["cli_sends_native_penalties_without_repeat_fold"])
        self.assertTrue(evidence["checks"]["cli_marks_open_think_for_visible_stripping"])
        self.assertTrue(evidence["checks"]["deltanet_dispatch_uses_per_launch_requant_frame"])
        self.assertTrue(evidence["checks"]["deltanet_kernel_uses_frame_independent_dither"])
        self.assertFalse(evidence["ppl_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertIn("cargo test -p hipfire-runtime --lib sampler", evidence["validation_commands"])

    def test_hunt3_nan_safe_sampling_source_fix_is_recorded_without_model_rerun(self):
        evidence = mq6_status.mq6_hunt3_nan_safe_sampling_evidence()

        self.assertEqual(
            evidence["schema"],
            "hipfire.mq6_hunt3_nan_safe_sampling.gfx1151.v0",
        )
        self.assertTrue(evidence["source_reconciled_in_worktree"])
        self.assertEqual(evidence["origin_fix_short"], "fb36f539")
        self.assertEqual(evidence["remote_ref"], "origin/fix/hunt3-engine-bugs")
        self.assertTrue(evidence["checks"]["llama_argmax_uses_nan_safe_fold"])
        self.assertTrue(evidence["checks"]["llama_cpu_sampler_rng_reset_exported"])
        self.assertTrue(evidence["checks"]["mtp_sampling_drops_nan_candidates"])
        self.assertTrue(evidence["checks"]["deepseek_sampling_drops_nan_candidates"])
        self.assertFalse(evidence["perf_rows_refreshed_by_this_check"])
        self.assertFalse(evidence["coherence_rows_refreshed_by_this_check"])
        self.assertIn(
            "cargo test -p hipfire-arch-qwen35 --lib sample_from_logits",
            evidence["validation_commands"],
        )

    def test_evidence_refresh_plan_maps_stale_commits_to_surfaces(self):
        origin = {
            "mq6_evidence_refresh_required": True,
            "origin_master_missing_commit_shorts": ["676338a4", "d5985c3e", "d5bc3f6"],
            "remote_branch_missing_commit_shorts": ["fb36f539"],
            "relevant_upstream_commits": [
                {
                    "short": "676338a4",
                    "subject": "fix(qwen35): batched RoPE ignored compact_offset",
                    "refreshes": ["dense_dflash_perf", "spec_verify"],
                    "surfaces": ["batched_rope", "dflash"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                },
                {
                    "short": "d5bc3f6",
                    "subject": "fix(daemon): AR exact-match DeltaNet",
                    "refreshes": ["dense_ar_perf", "dense_dflash_perf", "coherence"],
                    "surfaces": ["ar", "dflash"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                },
                {
                    "short": "d5985c3e",
                    "subject": "fix(stragglers): GGUF Promote6 Mq4Lloyd",
                    "refreshes": ["dflash_stability", "producer_context"],
                    "surfaces": ["dflash_drafter_unload", "gguf_producer"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                },
            ],
            "relevant_remote_branch_commits": [
                {
                    "short": "fb36f539",
                    "subject": "fix(engine): hunt-3 batch",
                    "remote_ref": "origin/fix/hunt3-engine-bugs",
                    "refreshes": [
                        "a3b_ar_perf",
                        "dense_dflash_perf",
                        "spec_verify",
                        "hunt3_nan_safe_sampling",
                    ],
                    "surfaces": ["sampling", "moe", "daemon", "dflash"],
                    "present_on_remote_ref": True,
                    "present_on_origin_master": False,
                    "present_on_head": False,
                }
            ],
        }
        coverage = {
            "surfaces": {
                "dense_decode": {"next_unblocked_step": "fix_dense_decode_throughput_or_scope_dense_out"},
                "dense_prefill": {"next_unblocked_step": "fix_dense_prefill_throughput_or_scope_dense_out"},
                "moe_prefill": {"next_unblocked_step": "rerun_a3b_moe_prefill_after_branch_reconciliation"},
                "moe_decode": {"next_unblocked_step": "rerun_a3b_moe_decode_after_branch_reconciliation"},
                "dflash_spec": {
                    "next_unblocked_step": "reconcile_origin_and_remote_fixes_then_rerun_dense_and_a3b_dflash_spec"
                },
            }
        }

        plan = mq6_status.evidence_refresh_plan(
            origin,
            coverage,
            locally_reconciled_refresh_reasons={
                "producer_context": {
                    "source": "producer_path",
                    "short": "d5985c3e",
                },
                "hunt3_nan_safe_sampling": {
                    "source": "hunt3_nan_safe_sampling",
                    "commit_short": "fb36f539",
                }
            },
        )

        self.assertTrue(plan["refresh_required"])
        self.assertEqual(plan["stale_origin_commit_shorts"], ["676338a4", "d5985c3e", "d5bc3f6"])
        self.assertEqual(plan["stale_remote_commit_shorts"], ["fb36f539"])
        self.assertEqual(
            plan["locally_reconciled_refresh_reasons"],
            ["hunt3_nan_safe_sampling", "producer_context"],
        )
        self.assertIn("dense_ar_perf", plan["refresh_reasons"])
        self.assertIn("dense_dflash_perf", plan["refresh_reasons"])
        self.assertIn("spec_verify", plan["refresh_reasons"])
        self.assertIn("a3b_ar_perf", plan["refresh_reasons"])
        self.assertIn("coherence", plan["refresh_reasons"])
        self.assertNotIn("producer_context", plan["refresh_reasons"])
        self.assertNotIn("hunt3_nan_safe_sampling", plan["refresh_reasons"])
        stale_by_short = {commit["short"]: commit for commit in plan["stale_commits"]}
        self.assertEqual(
            stale_by_short["d5bc3f6"]["unreconciled_refreshes"],
            ["coherence", "dense_ar_perf", "dense_dflash_perf"],
        )
        self.assertEqual(stale_by_short["d5bc3f6"]["locally_reconciled_refreshes"], [])
        self.assertEqual(stale_by_short["d5985c3e"]["locally_reconciled_refreshes"], ["producer_context"])
        self.assertEqual(stale_by_short["d5985c3e"]["unreconciled_refreshes"], ["dflash_stability"])
        self.assertEqual(
            stale_by_short["fb36f539"]["locally_reconciled_refreshes"],
            ["hunt3_nan_safe_sampling"],
        )
        self.assertEqual(
            stale_by_short["fb36f539"]["unreconciled_refreshes"],
            ["a3b_ar_perf", "dense_dflash_perf", "spec_verify"],
        )
        self.assertIn("dense_decode", plan["affected_surfaces"])
        self.assertIn("dense_prefill", plan["affected_surfaces"])
        self.assertIn("moe_prefill", plan["affected_surfaces"])
        self.assertIn("moe_decode", plan["affected_surfaces"])
        self.assertIn("dflash_spec", plan["affected_surfaces"])
        self.assertIn("python3 scripts/mq6_ar_perf_baseline.py", " ".join(plan["rerun_commands"]))
        self.assertIn("python3 scripts/mq6_dflash_baseline.py", " ".join(plan["rerun_commands"]))
        self.assertIn("./scripts/coherence-gate.sh --full", plan["rerun_commands"])
        self.assertIn("2026-06-02-mq6-ar-perf.json", " ".join(plan["evidence_artifacts_to_refresh"]))
        self.assertIn("2026-06-03-mq6-dflash-r3.json", " ".join(plan["evidence_artifacts_to_refresh"]))

    def test_evidence_refresh_plan_scopes_source_reconciliation_to_matching_commit(self):
        origin = {
            "mq6_evidence_refresh_required": True,
            "origin_master_missing_commit_shorts": ["676338a4"],
            "remote_branch_missing_commit_shorts": ["fb36f539"],
            "relevant_upstream_commits": [
                {
                    "short": "676338a4",
                    "subject": "fix(qwen35): batched RoPE ignored compact_offset",
                    "refreshes": ["dense_dflash_perf", "spec_verify", "long_context_or_eviction_rows"],
                    "surfaces": ["batched_rope", "dflash"],
                    "present_on_origin_master": True,
                    "present_on_head": False,
                },
            ],
            "relevant_remote_branch_commits": [
                {
                    "short": "fb36f539",
                    "subject": "fix(engine): hunt-3 batch",
                    "remote_ref": "origin/fix/hunt3-engine-bugs",
                    "refreshes": ["dense_dflash_perf", "spec_verify"],
                    "surfaces": ["sampling", "moe", "daemon", "dflash"],
                    "present_on_remote_ref": True,
                    "present_on_origin_master": False,
                    "present_on_head": False,
                }
            ],
        }
        coverage = {
            "surfaces": {
                "dflash_spec": {
                    "next_unblocked_step": "reconcile_origin_and_remote_fixes_then_rerun_dense_and_a3b_dflash_spec"
                },
            }
        }

        plan = mq6_status.evidence_refresh_plan(
            origin,
            coverage,
            locally_reconciled_refresh_reasons={
                "spec_verify": {
                    "source": "batched_rope_compact_offset",
                    "commit_short": "676338a4",
                }
            },
        )

        stale_by_short = {commit["short"]: commit for commit in plan["stale_commits"]}
        self.assertEqual(stale_by_short["676338a4"]["locally_reconciled_refreshes"], ["spec_verify"])
        self.assertEqual(
            stale_by_short["676338a4"]["unreconciled_refreshes"],
            ["dense_dflash_perf", "long_context_or_eviction_rows"],
        )
        self.assertEqual(stale_by_short["fb36f539"]["locally_reconciled_refreshes"], [])
        self.assertIn("spec_verify", stale_by_short["fb36f539"]["unreconciled_refreshes"])
        self.assertIn("spec_verify", plan["refresh_reasons"])
        self.assertIn("dense_dflash_perf", plan["refresh_reasons"])
        self.assertIn("long_context_or_eviction_rows", plan["refresh_reasons"])

    def test_current_mq6_status_records_a3b_first_dense_perf_gated(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], mq6_status.SCHEMA)
        self.assertEqual(payload["status"], "a3b-first-candidate-dense-perf-gated")
        self.assertFalse(payload["promotion_allowed"])
        self.assertEqual(payload["artifact_inventory"]["artifact_count"], 4)
        self.assertEqual(payload["live_candidate_inventory"]["artifact_count"], 4)
        self.assertEqual(payload["live_candidate_inventory"]["symlink_count"], 4)
        self.assertTrue(payload["live_candidate_inventory"]["canonical_mq6_artifacts_present"])
        self.assertTrue(payload["gates"]["canonical_mq6_artifacts_present"])
        self.assertTrue(payload["gates"]["provenance_canonical_mq6_artifacts_present"])
        self.assertTrue(payload["gates"]["live_canonical_mq6_artifacts_present"])
        self.assertTrue(payload["gates"]["mq6_producer_mq4_lloyd_promote6_fixed"])
        self.assertTrue(payload["gates"]["mq6_batched_rope_compact_offset_fixed"])
        self.assertTrue(payload["gates"]["mq6_daemon_dflash_first_token_eos_fixed"])
        self.assertTrue(payload["gates"]["mq6_daemon_ar_exact_match_delta_replay_fixed"])
        self.assertTrue(payload["gates"]["mq6_ddtree_eviction_target_hidden_host_fixed"])
        self.assertTrue(payload["gates"]["mq6_moe_prefill_grouped_scratch_free_fixed"])
        self.assertTrue(payload["gates"]["mq6_cask_scratch_teardown_fixed"])
        self.assertTrue(payload["gates"]["mq6_mtp_gpu_teardown_fixed"])
        self.assertTrue(payload["gates"]["mq6_asym4_fwht_fallback_fixed"])
        self.assertTrue(payload["gates"]["mq6_qwen36_a3b_runtime_source_reconciled"])
        self.assertTrue(payload["gates"]["mq6_hunt3_nan_safe_sampling_fixed"])
        self.assertTrue(payload["gates"]["dense_decode_repo_kernel_inventory_present"])
        self.assertTrue(payload["gates"]["dense_decode_install_kernel_inventory_present"])
        self.assertTrue(payload["gates"]["dense_decode_no_gpu_route_test_present"])
        self.assertTrue(payload["gates"]["moe_decode_no_gpu_route_test_present"])
        self.assertTrue(payload["gates"]["dense_9b_ppl_beats_mq4"])
        self.assertTrue(payload["gates"]["qwen35_a3b_ppl_beats_mq4"])
        self.assertTrue(payload["gates"]["qwen36_a3b_ppl_beats_mq4"])
        self.assertTrue(payload["gates"]["dense_9b_c512_kld_beats_mq4"])
        self.assertTrue(payload["gates"]["dense_ar_perf_present"])
        self.assertTrue(payload["gates"]["a3b_ar_perf_present"])
        self.assertTrue(payload["gates"]["dense_dflash_perf_present"])
        self.assertTrue(payload["gates"]["dflash_target_side_verify_only"])
        self.assertFalse(payload["gates"]["dflash_mq6_draft_evidence_present"])
        self.assertFalse(payload["gates"]["dflash_a3b_evidence_present"])
        self.assertTrue(payload["gates"]["dflash_spec_boundary_refresh_required"])
        self.assertTrue(payload["gates"]["dflash_spec_boundary_contract_test_present"])
        self.assertTrue(payload["gates"]["surface_coverage_complete"])
        self.assertFalse(payload["gates"]["all_surfaces_promotion_ready"])
        self.assertFalse(payload["gates"]["dense_ar_perf_promotable"])
        self.assertFalse(payload["gates"]["dense_dflash_perf_promotable"])
        self.assertTrue(payload["gates"]["a3b_prefill_speedup_present"])
        self.assertTrue(payload["gates"]["a3b_decode_close_to_mq4"])
        self.assertFalse(payload["gates"]["a3b_first_scope_supported"])
        self.assertTrue(payload["gates"]["origin_refresh_required"])
        self.assertTrue(payload["gates"]["remote_branch_refresh_required"])
        self.assertFalse(payload["gates"]["i8_grouped_port_recommended"])
        self.assertFalse(payload["gates"]["dense_promotion_allowed"])
        self.assertFalse(payload["gates"]["broad_promotion_allowed"])
        self.assertTrue(payload["origin_context"]["mq6_evidence_refresh_required"])
        self.assertTrue(payload["origin_context"]["remote_branch_refresh_required"])
        producer_path = payload["producer_path"]
        self.assertTrue(producer_path["promote6_mq4_lloyd_emits_mq6"])
        self.assertTrue(producer_path["origin_fix_reconciled_in_worktree"])
        self.assertEqual(producer_path["origin_fix_short"], "d5985c3e")
        batched_rope = payload["batched_rope_compact_offset"]
        self.assertTrue(batched_rope["source_reconciled_in_worktree"])
        self.assertEqual(batched_rope["origin_fix_short"], "676338a4")
        self.assertTrue(batched_rope["checks"]["qwen35_batched_fa_passes_compact_offset"])
        self.assertFalse(batched_rope["perf_rows_refreshed_by_this_check"])
        daemon_dflash = payload["daemon_dflash_first_token_eos"]
        self.assertTrue(daemon_dflash["source_reconciled_in_worktree"])
        self.assertEqual(daemon_dflash["origin_fix_short"], "d5bc3f6")
        self.assertTrue(daemon_dflash["checks"]["spec_loop_guarded"])
        self.assertFalse(daemon_dflash["perf_rows_refreshed_by_this_check"])
        self.assertFalse(daemon_dflash["coherence_rows_refreshed_by_this_check"])
        daemon_ar = payload["daemon_ar_exact_match_delta_replay"]
        self.assertTrue(daemon_ar["source_reconciled_in_worktree"])
        self.assertEqual(daemon_ar["origin_fix_short"], "d5bc3f6")
        self.assertTrue(daemon_ar["checks"]["legacy_exact_match_lcp_rewind_absent"])
        self.assertTrue(daemon_ar["checks"]["qwen35_prefills_direct_new_tokens"])
        self.assertFalse(daemon_ar["perf_rows_refreshed_by_this_check"])
        self.assertFalse(daemon_ar["coherence_rows_refreshed_by_this_check"])
        ddtree_eviction = payload["ddtree_eviction_target_hidden_host"]
        self.assertTrue(ddtree_eviction["source_reconciled_in_worktree"])
        self.assertEqual(ddtree_eviction["origin_fix_short"], "d12b3ad4")
        self.assertEqual(ddtree_eviction["remote_ref"], "origin/fix/issue-triage-2026-06-03")
        self.assertTrue(ddtree_eviction["checks"]["daemon_post_prefill_and_cycle_calls_present"])
        self.assertTrue(ddtree_eviction["checks"]["dflash_spec_demo_post_prefill_and_cycle_calls_present"])
        self.assertFalse(ddtree_eviction["perf_rows_refreshed_by_this_check"])
        self.assertFalse(ddtree_eviction["coherence_rows_refreshed_by_this_check"])
        grouped_scratch = payload["moe_prefill_grouped_scratch_free"]
        self.assertTrue(grouped_scratch["source_reconciled_in_worktree"])
        self.assertEqual(grouped_scratch["origin_fix_short"], "b4adca1f")
        self.assertTrue(grouped_scratch["checks"]["frees_moe_y_gate_up_grouped"])
        self.assertTrue(grouped_scratch["checks"]["frees_moe_y_down_grouped"])
        self.assertFalse(grouped_scratch["perf_rows_refreshed_by_this_check"])
        self.assertFalse(grouped_scratch["long_running_moe_stability_refreshed_by_this_check"])
        cask_scratch = payload["cask_scratch_teardown"]
        self.assertTrue(cask_scratch["source_reconciled_in_worktree"])
        self.assertEqual(cask_scratch["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(cask_scratch["checks"]["inner_result_propagated_after_free"])
        mtp_teardown = payload["mtp_gpu_teardown"]
        self.assertTrue(mtp_teardown["source_reconciled_in_worktree"])
        self.assertEqual(mtp_teardown["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(mtp_teardown["checks"]["mtp_spec_frees_trunk_snapshot_gpu"])
        asym4_fwht = payload["asym4_fwht_fallback"]
        self.assertTrue(asym4_fwht["source_reconciled_in_worktree"])
        self.assertEqual(asym4_fwht["origin_fix_short"], "a7dcfb0d")
        self.assertTrue(asym4_fwht["checks"]["fwht4_uses_branch_native_q8_v_signature"])
        qwen36_runtime = payload["qwen36_a3b_runtime_source_reconciliation"]
        self.assertTrue(qwen36_runtime["source_reconciled_in_worktree"])
        self.assertEqual(qwen36_runtime["origin_fix_short"], "379484ba")
        self.assertTrue(qwen36_runtime["checks"]["cli_sends_native_penalties_without_repeat_fold"])
        hunt3_nan = payload["hunt3_nan_safe_sampling"]
        self.assertTrue(hunt3_nan["source_reconciled_in_worktree"])
        self.assertEqual(hunt3_nan["origin_fix_short"], "fb36f539")
        self.assertTrue(hunt3_nan["checks"]["llama_argmax_uses_nan_safe_fold"])
        self.assertTrue(hunt3_nan["checks"]["mtp_sampling_drops_nan_candidates"])
        self.assertTrue(hunt3_nan["checks"]["deepseek_sampling_drops_nan_candidates"])
        self.assertTrue(qwen36_runtime["checks"]["deltanet_kernel_uses_frame_independent_dither"])
        self.assertFalse(qwen36_runtime["ppl_rows_refreshed_by_this_check"])
        self.assertFalse(qwen36_runtime["perf_rows_refreshed_by_this_check"])
        commits = {commit["short"]: commit for commit in payload["origin_context"]["relevant_upstream_commits"]}
        self.assertIn("676338a4", commits)
        self.assertIn("dflash", commits["676338a4"]["surfaces"])
        self.assertIn("dense_dflash_perf", commits["676338a4"]["refreshes"])
        self.assertIn("a7dcfb0d", commits)
        self.assertIn("a3b_ar_perf", commits["a7dcfb0d"]["refreshes"])
        self.assertIn("cask_scratch_teardown", commits["a7dcfb0d"]["refreshes"])
        self.assertIn("mtp_gpu_teardown", commits["a7dcfb0d"]["refreshes"])
        self.assertIn("asym4_fwht_fallback", commits["a7dcfb0d"]["refreshes"])
        self.assertIn("b4adca1f", commits)
        self.assertIn("moe_prefill_scratch_teardown", commits["b4adca1f"]["refreshes"])
        self.assertIn("379484ba", commits)
        self.assertIn("qwen36_a3b_runtime_source_reconciliation", commits["379484ba"]["refreshes"])
        self.assertIn("qwen36_a3b_ppl", commits["379484ba"]["refreshes"])
        self.assertIn("qwen36_a3b_ar_perf", commits["379484ba"]["refreshes"])
        remote_commits = {
            commit["short"]: commit
            for commit in payload["origin_context"]["relevant_remote_branch_commits"]
        }
        self.assertIn("fb36f539", remote_commits)
        self.assertEqual(remote_commits["fb36f539"]["remote_ref"], "origin/fix/hunt3-engine-bugs")
        self.assertIn("sampling+MoE", remote_commits["fb36f539"]["subject"])
        self.assertIn("d12b3ad4", remote_commits)
        self.assertEqual(remote_commits["d12b3ad4"]["remote_ref"], "origin/fix/issue-triage-2026-06-03")
        self.assertIn("ddtree_eviction_stability", remote_commits["d12b3ad4"]["refreshes"])
        kernel_inventory = payload["kernel_inventory"]
        self.assertEqual(
            kernel_inventory["schema"],
            "hipfire.mq6_dense_decode_kernel_inventory.gfx1151.v0",
        )
        self.assertTrue(kernel_inventory["repo_required_kernels_present"])
        self.assertTrue(kernel_inventory["install_required_kernels_present"])
        self.assertEqual(
            kernel_inventory["evidence_class"],
            "compiled_blob_presence_only_not_dispatch_proof",
        )
        self.assertIn("gemv_mq6g256.hsaco", kernel_inventory["required_kernels"])
        self.assertIn("gemv_hfq6g256_residual.hsaco", kernel_inventory["required_kernels"])
        self.assertEqual(kernel_inventory["roots"]["repo"]["missing"], [])
        route_test = payload["dense_decode_route_test"]
        self.assertEqual(
            route_test["command"],
            "cargo test -p hipfire-runtime --lib mq6_dense_decode_routes_through_hfq6_family",
        )
        self.assertTrue(route_test["present"])
        self.assertTrue(route_test["all_phrase_checks_present"])
        self.assertTrue(route_test["phrase_checks"]["DenseGemvRoute::Mq6RotateThenMq6Prerotated"])
        self.assertTrue(route_test["phrase_checks"]["DenseResidualRoute::Hfq6ResidualDirect"])
        moe_decode_route_test = payload["moe_decode_route_test"]
        self.assertEqual(
            moe_decode_route_test["command"],
            "cargo test -p hipfire-arch-qwen35 --lib moe_decode_routes_mq6_indexed_path_for_k8",
        )
        self.assertTrue(moe_decode_route_test["present"])
        self.assertTrue(moe_decode_route_test["all_phrase_checks_present"])
        self.assertTrue(moe_decode_route_test["phrase_checks"]["MoeDecodeIndexedRoutedPath::Mq6"])
        self.assertTrue(
            moe_decode_route_test["phrase_checks"][
                "moe_decode_rejects_mismatched_mq6_gate_up_and_down_from_indexed_path"
            ]
        )
        boundary_test = payload["dflash_boundary_contract_test"]
        self.assertIn("test_dflash_spec_boundary_marks_mq4_draft", boundary_test["command"])
        self.assertTrue(boundary_test["present"])
        self.assertTrue(boundary_test["all_phrase_checks_present"])
        self.assertTrue(
            boundary_test["phrase_checks"][
                "test_dflash_spec_boundary_detects_mq6_draft_scope_expansion"
            ]
        )
        boundary = payload["performance"]["dflash_spec_boundary"]
        self.assertTrue(boundary["target_side_verify_only"])
        self.assertEqual(boundary["draft_formats"], ["mq4"])
        self.assertFalse(boundary["mq6_draft_evidence_present"])
        self.assertFalse(boundary["a3b_dflash_evidence_present"])
        self.assertEqual(boundary["evidence_class"], "target_side_verify_only_mq4_draft")
        self.assertFalse(boundary["scope_expanded_beyond_target_verify"])
        self.assertTrue(boundary["contract_ready"])
        self.assertFalse(boundary["promotion_scope_supported"])
        self.assertEqual(boundary["min_runs_per_case"], 3)
        self.assertTrue(boundary["token_attractor_clean"])
        self.assertTrue(boundary["refresh_required"])
        coverage = payload["surface_coverage"]
        self.assertEqual(coverage["schema"], "hipfire.mq6_surface_coverage.gfx1151.v0")
        self.assertTrue(coverage["all_required_surfaces_recorded"])
        self.assertFalse(coverage["all_surfaces_promotion_ready"])
        self.assertTrue(coverage["refresh_required"])
        self.assertEqual(
            set(coverage["surfaces"]),
            {"dense_decode", "dense_prefill", "moe_prefill", "moe_decode", "dflash_spec"},
        )
        self.assertEqual(coverage["surfaces"]["dense_decode"]["status"], "perf_gated")
        self.assertTrue(coverage["surfaces"]["dense_decode"]["runtime_evidence_present"])
        self.assertIn(
            "mq6_dense_decode_routes_through_hfq6_family",
            " ".join(coverage["surfaces"]["dense_decode"]["evidence"]),
        )
        self.assertIn(
            "kernel_inventory.required_mq6_dense_decode_kernels",
            " ".join(coverage["surfaces"]["dense_decode"]["evidence"]),
        )
        self.assertTrue(
            coverage["surfaces"]["dense_decode"]["details"]["kernel_inventory"][
                "repo_required_kernels_present"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dense_decode"]["details"]["no_gpu_route_test"][
                "all_phrase_checks_present"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dense_decode"]["details"]["daemon_ar_exact_match_delta_replay"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertEqual(coverage["surfaces"]["dense_prefill"]["status"], "perf_gated")
        self.assertIn(
            "dense_prefill_mq6_and_hfq6_are_batchable_in_qwen35",
            " ".join(coverage["surfaces"]["dense_prefill"]["evidence"]),
        )
        self.assertTrue(
            coverage["surfaces"]["dense_prefill"]["details"]["daemon_ar_exact_match_delta_replay"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertEqual(
            coverage["surfaces"]["moe_prefill"]["status"],
            "a3b_first_candidate_refresh_blocked",
        )
        self.assertEqual(
            coverage["surfaces"]["moe_decode"]["status"],
            "a3b_first_candidate_refresh_blocked",
        )
        self.assertTrue(
            coverage["surfaces"]["moe_prefill"]["details"]["grouped_scratch_free"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertEqual(
            coverage["surfaces"]["moe_prefill"]["details"]["grouped_scratch_free"][
                "origin_fix_short"
            ],
            "b4adca1f",
        )
        self.assertTrue(coverage["surfaces"]["moe_decode"]["runtime_evidence_present"])
        self.assertIn(
            "moe_decode_routes_mq6_indexed_path_for_k8",
            " ".join(coverage["surfaces"]["moe_decode"]["evidence"]),
        )
        self.assertTrue(
            coverage["surfaces"]["moe_decode"]["details"]["no_gpu_route_test"][
                "all_phrase_checks_present"
            ]
        )
        self.assertEqual(
            coverage["surfaces"]["dflash_spec"]["status"],
            "target_verify_only_refresh_blocked",
        )
        self.assertFalse(coverage["surfaces"]["dense_decode"]["promotion_ready"])
        self.assertFalse(coverage["surfaces"]["dflash_spec"]["promotion_ready"])
        self.assertTrue(coverage["surfaces"]["dflash_spec"]["runtime_evidence_present"])
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["boundary_contract_test"][
                "all_phrase_checks_present"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["batched_rope_compact_offset"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertFalse(
            coverage["surfaces"]["dflash_spec"]["details"]["batched_rope_compact_offset"][
                "perf_rows_refreshed_by_this_check"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["daemon_dflash_first_token_eos"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertFalse(
            coverage["surfaces"]["dflash_spec"]["details"]["daemon_dflash_first_token_eos"][
                "perf_rows_refreshed_by_this_check"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["ddtree_eviction_target_hidden_host"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertEqual(
            coverage["surfaces"]["dflash_spec"]["details"]["ddtree_eviction_target_hidden_host"][
                "origin_fix_short"
            ],
            "d12b3ad4",
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["cask_scratch_teardown"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["mtp_gpu_teardown"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertTrue(
            coverage["surfaces"]["dflash_spec"]["details"]["asym4_fwht_fallback"][
                "source_reconciled_in_worktree"
            ]
        )
        self.assertFalse(
            coverage["surfaces"]["dflash_spec"]["details"]["ddtree_eviction_target_hidden_host"][
                "perf_rows_refreshed_by_this_check"
            ]
        )
        self.assertTrue(coverage["surfaces"]["moe_prefill"]["quality_evidence_present"])
        self.assertTrue(coverage["surfaces"]["moe_prefill"]["perf_evidence_present"])
        self.assertIn("no_mq6_draft_evidence", coverage["surfaces"]["dflash_spec"]["blockers"])
        self.assertIn("dense MQ6 AR", " ".join(payload["blockers"]))
        self.assertIn("DFlash evidence covers MQ6 target-side verify", " ".join(payload["blockers"]))
        self.assertIn("no MQ6 draft, A3B DFlash sidecar", " ".join(payload["blockers"]))
        self.assertIn("origin-only MQ6-relevant fixes", " ".join(payload["blockers"]))
        self.assertIn("origin/fix/hunt3-engine-bugs", " ".join(payload["blockers"]))
        self.assertIn("A3B/MoE-first", payload["decision"])
        refresh_plan = payload["evidence_refresh_plan"]
        self.assertTrue(refresh_plan["refresh_required"])
        self.assertIn("producer_context", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn(
            "daemon_ar_exact_match_delta_replay",
            refresh_plan["locally_reconciled_refresh_reasons"],
        )
        self.assertIn("spec_verify", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn("ddtree_eviction_stability", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn("moe_prefill_scratch_teardown", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn("cask_scratch_teardown", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn("mtp_gpu_teardown", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn("asym4_fwht_fallback", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertIn(
            "qwen36_a3b_runtime_source_reconciliation",
            refresh_plan["locally_reconciled_refresh_reasons"],
        )
        self.assertIn("hunt3_nan_safe_sampling", refresh_plan["locally_reconciled_refresh_reasons"])
        self.assertNotIn("producer_context", refresh_plan["refresh_reasons"])
        self.assertNotIn("daemon_ar_exact_match_delta_replay", refresh_plan["refresh_reasons"])
        self.assertNotIn("ddtree_eviction_stability", refresh_plan["refresh_reasons"])
        self.assertNotIn("moe_prefill_scratch_teardown", refresh_plan["refresh_reasons"])
        self.assertNotIn("cask_scratch_teardown", refresh_plan["refresh_reasons"])
        self.assertNotIn("mtp_gpu_teardown", refresh_plan["refresh_reasons"])
        self.assertNotIn("asym4_fwht_fallback", refresh_plan["refresh_reasons"])
        self.assertNotIn("qwen36_a3b_runtime_source_reconciliation", refresh_plan["refresh_reasons"])
        self.assertNotIn("hunt3_nan_safe_sampling", refresh_plan["refresh_reasons"])
        self.assertIn("676338a4", refresh_plan["stale_origin_commit_shorts"])
        self.assertIn("fb36f539", refresh_plan["stale_remote_commit_shorts"])
        self.assertIn("d12b3ad4", refresh_plan["stale_remote_commit_shorts"])
        self.assertIn("dense_ar_perf", refresh_plan["refresh_reasons"])
        self.assertIn("dense_dflash_perf", refresh_plan["refresh_reasons"])
        self.assertIn("a3b_ar_perf", refresh_plan["refresh_reasons"])
        self.assertIn("qwen36_a3b_ppl", refresh_plan["refresh_reasons"])
        self.assertIn("qwen36_a3b_ar_perf", refresh_plan["refresh_reasons"])
        self.assertIn("spec_verify", refresh_plan["refresh_reasons"])
        stale_by_short = {commit["short"]: commit for commit in refresh_plan["stale_commits"]}
        self.assertIn("spec_verify", stale_by_short["676338a4"]["locally_reconciled_refreshes"])
        self.assertIn("dense_dflash_perf", stale_by_short["676338a4"]["unreconciled_refreshes"])
        self.assertIn(
            "long_context_or_eviction_rows",
            stale_by_short["676338a4"]["unreconciled_refreshes"],
        )
        self.assertIn("producer_context", stale_by_short["d5985c3e"]["locally_reconciled_refreshes"])
        self.assertIn("dflash_stability", stale_by_short["d5985c3e"]["unreconciled_refreshes"])
        self.assertNotIn("producer_context", stale_by_short["d5985c3e"]["unreconciled_refreshes"])
        self.assertEqual(
            stale_by_short["a7dcfb0d"]["locally_reconciled_refreshes"],
            ["asym4_fwht_fallback", "cask_scratch_teardown", "mtp_gpu_teardown"],
        )
        self.assertEqual(
            stale_by_short["a7dcfb0d"]["unreconciled_refreshes"],
            ["a3b_ar_perf", "a3b_ppl_smoke", "spec_verify"],
        )
        self.assertIn("moe_prefill_scratch_teardown", stale_by_short["b4adca1f"]["locally_reconciled_refreshes"])
        self.assertEqual(
            stale_by_short["b4adca1f"]["unreconciled_refreshes"],
            ["a3b_ar_perf", "long_running_moe_stability"],
        )
        self.assertEqual(
            stale_by_short["d5bc3f6"]["locally_reconciled_refreshes"],
            ["daemon_ar_exact_match_delta_replay"],
        )
        self.assertEqual(
            stale_by_short["d5bc3f6"]["unreconciled_refreshes"],
            ["coherence", "dense_ar_perf", "dense_dflash_perf"],
        )
        self.assertIn("ddtree_eviction_stability", stale_by_short["d12b3ad4"]["locally_reconciled_refreshes"])
        self.assertEqual(stale_by_short["d12b3ad4"]["unreconciled_refreshes"], [])
        self.assertEqual(stale_by_short["d12b3ad4"]["source"], "origin/fix/issue-triage-2026-06-03")
        self.assertEqual(
            stale_by_short["379484ba"]["locally_reconciled_refreshes"],
            ["qwen36_a3b_runtime_source_reconciliation"],
        )
        self.assertEqual(
            stale_by_short["379484ba"]["unreconciled_refreshes"],
            ["qwen36_a3b_ar_perf", "qwen36_a3b_ppl"],
        )
        self.assertIn(
            "hunt3_nan_safe_sampling",
            stale_by_short["fb36f539"]["locally_reconciled_refreshes"],
        )
        self.assertNotIn(
            "hunt3_nan_safe_sampling",
            stale_by_short["fb36f539"]["unreconciled_refreshes"],
        )
        self.assertIn("spec_verify", stale_by_short["fb36f539"]["unreconciled_refreshes"])
        self.assertIn("dense_decode", refresh_plan["affected_surfaces"])
        self.assertIn("dense_prefill", refresh_plan["affected_surfaces"])
        self.assertIn("moe_prefill", refresh_plan["affected_surfaces"])
        self.assertIn("moe_decode", refresh_plan["affected_surfaces"])
        self.assertIn("dflash_spec", refresh_plan["affected_surfaces"])
        self.assertEqual(
            refresh_plan["surface_next_steps"]["dflash_spec"],
            "reconcile_origin_and_remote_fixes_then_rerun_dense_and_a3b_dflash_spec",
        )
        self.assertIn("python3 scripts/mq6_ar_perf_baseline.py", " ".join(refresh_plan["rerun_commands"]))
        self.assertIn("python3 scripts/mq6_dflash_baseline.py", " ".join(refresh_plan["rerun_commands"]))
        self.assertIn("./scripts/coherence-gate.sh --full", refresh_plan["rerun_commands"])
        self.assertIn("2026-06-02-mq6-ar-perf.json", " ".join(refresh_plan["evidence_artifacts_to_refresh"]))
        self.assertIn("2026-06-03-mq6-dflash-r3.json", " ".join(refresh_plan["evidence_artifacts_to_refresh"]))
        self.assertIn("Reconcile qwen35-native-mtp", payload["next_work"][0])
        self.assertIn("Rerun MQ6 AR perf baselines", " ".join(payload["next_work"]))
        self.assertIn("Rerun MQ6 DFlash/spec baselines", " ".join(payload["next_work"]))
        self.assertIn("Keep A3B/MoE as the first scope", " ".join(payload["next_work"]))


if __name__ == "__main__":
    unittest.main()
