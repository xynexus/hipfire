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


def load_script(name: str):
    module_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Mq3PerfBaselineTest(unittest.TestCase):
    def test_ar_cases_compare_mq3_against_mq4_controls(self):
        ar = load_script("mq3_ar_perf_baseline")
        by_id = {case.id: case for case in ar.DEFAULT_CASES}
        self.assertEqual(by_id["qwen35-27b-mq3-cap"].model_file, "qwen3.5-27b.mq3")
        self.assertEqual(by_id["qwen35-27b-mq4-cap"].model_file, "qwen3.5-27b.mq4")
        self.assertEqual(by_id["qwen35-a3b-mq3-sheep"].family, "moe")
        self.assertEqual(by_id["qwen36-a3b-mq3-sheep"].format_id, "mq3")

    def test_dflash_cases_are_27b_target_side_rows_with_same_mq4_draft(self):
        dflash = load_script("mq3_dflash_baseline")
        by_id = {case.id: case for case in dflash.DEFAULT_CASES}
        self.assertEqual(by_id["qwen35-27b-mq3-dflash-code"].target_file, "qwen3.5-27b.mq3")
        self.assertEqual(by_id["qwen35-27b-mq4-dflash-code"].target_file, "qwen3.5-27b.mq4")
        self.assertEqual(by_id["qwen35-27b-mq3-dflash-prose"].draft_file, dflash.DEFAULT_DRAFT)
        self.assertEqual(by_id["qwen35-27b-mq3-dflash-prose"].target_format, "mq3")

    def test_markdown_writers_record_hard_errors(self):
        ar = load_script("mq3_ar_perf_baseline")
        dflash = load_script("mq3_dflash_baseline")
        with tempfile.TemporaryDirectory() as tmp:
            ar_path = Path(tmp) / "ar.md"
            ar.write_markdown(
                {
                    "date": "2026-06-03T00:00:00+00:00",
                    "commit": "abc",
                    "branch": "test",
                    "arch": "gfx1151",
                    "runs_per_case": 1,
                    "daemon": {"md5": "deadbeef"},
                    "cases": [
                        {
                            "id": "case-ar",
                            "format_id": "mq3",
                            "family": "dense",
                            "prompt_md5": "p",
                            "summary": {"ok_runs": 0, "total_runs": 1},
                            "runs": [{"run": 1, "status": "hard_error", "panic": "boom"}],
                        }
                    ],
                },
                ar_path,
            )
            self.assertIn("case-ar run 1: boom", ar_path.read_text())

            dflash_path = Path(tmp) / "dflash.md"
            dflash.write_markdown(
                {
                    "date": "2026-06-03T00:00:00+00:00",
                    "commit": "abc",
                    "branch": "test",
                    "arch": "gfx1151",
                    "dflash_spec_demo": {"md5": "deadbeef"},
                    "params": {"kv_mode": "q8", "ctx": 2048},
                    "cases": [
                        {
                            "id": "case-dflash",
                            "target_format": "mq3",
                            "prompt_md5": "p",
                            "summary": {"ok_runs": 0, "total_runs": 1},
                            "runs": [
                                {
                                    "run": 2,
                                    "status": "hard_error",
                                    "panic": "",
                                    "token_attractor": {"reason": "zero_tokens"},
                                }
                            ],
                        }
                    ],
                },
                dflash_path,
            )
            self.assertIn("case-dflash run 2", dflash_path.read_text())
            self.assertIn("zero_tokens", dflash_path.read_text())


if __name__ == "__main__":
    unittest.main()
