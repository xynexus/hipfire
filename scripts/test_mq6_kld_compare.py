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
MODULE_PATH = ROOT / "scripts" / "mq6_kld_compare.py"
spec = importlib.util.spec_from_file_location("mq6_kld_compare", MODULE_PATH)
assert spec and spec.loader
mq6_kld = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq6_kld
spec.loader.exec_module(mq6_kld)


class Mq6KldCompareTest(unittest.TestCase):
    def test_default_cases_cover_mq4_control_and_mq6_candidate(self):
        ids = {case.id for case in mq6_kld.DEFAULT_CASES}
        self.assertEqual(ids, {"qwen35-9b-mq4", "qwen35-9b-mq6"})
        self.assertEqual(len(ids), len(mq6_kld.DEFAULT_CASES))
        all_ids = {case.id for case in mq6_kld.ALL_CASES}
        self.assertIn("qwen35-9b-mq3", all_ids)
        self.assertIn("qwen35-4b-mq4", all_ids)
        self.assertIn("qwen35-4b-mq3", all_ids)
        self.assertIn("qwen35-a3b-mq4", all_ids)
        self.assertIn("qwen35-a3b-mq3", all_ids)
        self.assertIn("qwen36-a3b-mq4", all_ids)
        self.assertIn("qwen36-a3b-mq3", all_ids)
        self.assertIn("qwen35-4b-mq3-lloyd", all_ids)
        self.assertIn("qwen35-9b-mq3-lloyd", all_ids)
        self.assertIn("qwen35-27b-mq3-lloyd", all_ids)
        self.assertIn("qwen35-a3b-mq3-lloyd", all_ids)
        self.assertIn("qwen35-9b-mq4-lloyd", all_ids)
        self.assertIn("qwen35-27b-mq4-lloyd", all_ids)
        self.assertIn("qwen35-a3b-mq4-lloyd", all_ids)
        self.assertEqual({case.family for case in mq6_kld.MQ3_A3B_CASES}, {"a3b"})
        self.assertEqual(
            {case.family for case in mq6_kld.MQ3_LLOYD_CASES},
            {"dense", "a3b"},
        )
        self.assertEqual(
            {case.format_id for case in mq6_kld.MQ4_LLOYD_CASES},
            {"mq4-lloyd"},
        )

    def test_case_variant_records_kv_mode_and_chunk_cap(self):
        case = mq6_kld.Case("qwen35-9b-mq6", "qwen3.5-9b.mq6", "mq6", "dense")
        self.assertEqual(
            mq6_kld.case_variant(case, "q8", 20),
            "qwen3.5-9b.mq6-kvq8-c20",
        )
        self.assertEqual(
            mq6_kld.case_variant(case, "asym3", None),
            "qwen3.5-9b.mq6-kvasym3",
        )

    def test_reference_filename_guard_rejects_wrong_fixture(self):
        cases = [
            case
            for case in mq6_kld.ALL_CASES
            if case.id in {"qwen35-a3b-mq4", "qwen35-a3b-mq3"}
        ]
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-9b-bf16.kldref.bin"), cases),
            ["qwen35-a3b-mq4", "qwen35-a3b-mq3"],
        )
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-35b-a3b-bf16.kldref.bin"), cases),
            [],
        )

    def test_reference_filename_guard_accepts_legacy_qwen35_names(self):
        cases = [
            case
            for case in mq6_kld.ALL_CASES
            if case.id in {"qwen36-a3b-mq4", "qwen36-a3b-mq3"}
        ]
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen36-35b-a3b-bf16.kldref.bin"), cases),
            [],
        )

    def test_reference_filename_guard_covers_lloyd_cases(self):
        dense_cases = [
            case
            for case in mq6_kld.ALL_CASES
            if case.id in {"qwen35-9b-mq3-lloyd", "qwen35-9b-mq4-lloyd"}
        ]
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-9b-bf16.kldref.bin"), dense_cases),
            [],
        )
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-4b-bf16.kldref.bin"), dense_cases),
            ["qwen35-9b-mq3-lloyd", "qwen35-9b-mq4-lloyd"],
        )

        a3b_cases = [
            case
            for case in mq6_kld.ALL_CASES
            if case.id in {"qwen35-a3b-mq3-lloyd", "qwen35-a3b-mq4-lloyd"}
        ]
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-35b-a3b-bf16.kldref.bin"), a3b_cases),
            [],
        )
        self.assertEqual(
            mq6_kld.reference_mismatches(Path("qwen3.5-27b-bf16.kldref.bin"), a3b_cases),
            ["qwen35-a3b-mq3-lloyd", "qwen35-a3b-mq4-lloyd"],
        )

    def test_fmt_metric_keeps_missing_values_blank(self):
        self.assertEqual(mq6_kld.fmt_metric(None), "")
        self.assertEqual(mq6_kld.fmt_metric(0.123456789), "0.123457")
        self.assertEqual(mq6_kld.fmt_metric("n/a"), "n/a")

    def test_write_markdown_summary(self):
        payload = {
            "date": "2026-06-03T00:00:00+00:00",
            "commit": "abc123",
            "branch": "test",
            "arch": "gfx1151",
            "params": {"kv_mode": "q8", "scoring_mode": "prefill", "max_chunks": 20},
            "eval_hipfire": {"md5": "eval-md5"},
            "reference": {"sha256": "ref-sha"},
            "reduced_rows": [
                {
                    "variant": "qwen3.5-9b.mq6-kvq8-c20",
                    "scoring_mode": "prefill",
                    "n_chunks": 20,
                    "mean_kld": 0.123456,
                    "mean_kld_ci_lo": 0.1,
                    "mean_kld_ci_hi": 0.2,
                    "p99_kld": 0.5,
                    "ppl": 9.25,
                }
            ],
            "cases": [
                {
                    "id": "qwen35-9b-mq6",
                    "status": "ok",
                    "output_size_bytes": 480,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            mq6_kld.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("# Quant BF16-Referenced KLD", text)
        self.assertIn("qwen3.5-9b.mq6-kvq8-c20", text)
        self.assertIn("eval-md5", text)
        self.assertIn("ref-sha", text)
        self.assertIn("qwen35-9b-mq6: ok", text)


if __name__ == "__main__":
    unittest.main()
