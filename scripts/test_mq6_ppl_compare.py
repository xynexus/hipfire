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
MODULE_PATH = ROOT / "scripts" / "mq6_ppl_compare.py"
spec = importlib.util.spec_from_file_location("mq6_ppl_compare", MODULE_PATH)
assert spec and spec.loader
mq6_ppl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq6_ppl
spec.loader.exec_module(mq6_ppl)


class Mq6PplCompareTest(unittest.TestCase):
    def test_parse_result(self):
        output = """
Scored: 2039 tokens
NLL/tok: 2.4499742159
PPL: 11.5880
Elapsed: 69.625s (29.3 tok/s)
"""
        parsed = mq6_ppl.parse_result(output)
        self.assertEqual(parsed["scored"], 2039)
        self.assertAlmostEqual(parsed["nll_tok"], 2.4499742159)
        self.assertAlmostEqual(parsed["ppl"], 11.5880)
        self.assertAlmostEqual(parsed["elapsed_s"], 69.625)
        self.assertAlmostEqual(parsed["scored_tok_s"], 29.3)

    def test_a3b_cases_include_qwen35_and_qwen36_pairs(self):
        ids = {case.id for case in mq6_ppl.A3B_CASES}
        self.assertIn("qwen35-a3b-mq4", ids)
        self.assertIn("qwen35-a3b-mq6", ids)
        self.assertIn("qwen36-a3b-mq4", ids)
        self.assertIn("qwen36-a3b-mq6", ids)
        self.assertEqual(len({case.id for case in mq6_ppl.ALL_CASES}), len(mq6_ppl.ALL_CASES))

    def test_mq3_cases_cover_dense_and_a3b_fixtures(self):
        dense = {case.id: case for case in mq6_ppl.MQ3_DENSE_CASES}
        self.assertEqual(dense["qwen35-9b-mq3"].model_file, "qwen3.5-9b.mq3")
        self.assertEqual(dense["qwen35-27b-mq4"].format_id, "mq4")
        self.assertEqual(dense["qwen35-27b-mq3"].format_id, "mq3")

        a3b = {case.id: case for case in mq6_ppl.MQ3_A3B_CASES}
        self.assertEqual(a3b["qwen35-a3b-mq3"].model_file, "qwen3.5-35b-a3b.mq3")
        self.assertEqual(a3b["qwen36-a3b-mq3"].model_file, "qwen3.6-35b-a3b.mq3")

    def test_write_markdown_summary(self):
        payload = {
            "date": "2026-06-03T00:00:00+00:00",
            "commit": "abc123",
            "branch": "test",
            "arch": "gfx1151",
            "params": {"ctx": 2048, "warmup": 8, "offset": 0, "kv_mode": "q8"},
            "perplexity": {"md5": "perp-md5"},
            "corpus": {"md5": "corpus-md5"},
            "cases": [
                {
                    "id": "qwen35-a3b-mq6",
                    "format_id": "mq6",
                    "family": "moe",
                    "status": "ok",
                    "result": {
                        "scored": 2039,
                        "nll_tok": 2.4,
                        "ppl": 11.0,
                        "scored_tok_s": 20.0,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            mq6_ppl.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("# Quant PPL Compare", text)
        self.assertIn("qwen35-a3b-mq6", text)
        self.assertIn("perp-md5", text)


if __name__ == "__main__":
    unittest.main()
