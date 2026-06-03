#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mq6_ar_perf_baseline.py"

spec = importlib.util.spec_from_file_location("mq6_ar_perf_baseline", MODULE_PATH)
assert spec and spec.loader
mq6_ar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq6_ar
spec.loader.exec_module(mq6_ar)


class Mq6ArPerfBaselineTest(unittest.TestCase):
    def test_documented_pretty_flag_is_accepted(self):
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertIn("--pretty", proc.stdout)

    def test_markdown_writer_records_summary_and_hard_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ar.md"
            mq6_ar.write_markdown(
                {
                    "date": "2026-06-03T00:00:00+00:00",
                    "commit": "abc",
                    "branch": "test",
                    "arch": "gfx1151",
                    "runs_per_case": 1,
                    "daemon": {"md5": "deadbeef"},
                    "cases": [
                        {
                            "id": "case-ok",
                            "format_id": "mq6",
                            "family": "dense",
                            "prompt_md5": "prompt",
                            "summary": {
                                "ok_runs": 1,
                                "total_runs": 1,
                                "median_prefill_tok_s": 12.0,
                                "median_decode_tok_s": 34.0,
                                "median_tok_s": 30.0,
                                "median_tokens": 8.0,
                            },
                            "runs": [{"run": 1, "status": "ok", "panic": ""}],
                        },
                        {
                            "id": "case-hard",
                            "format_id": "mq6",
                            "family": "moe",
                            "prompt_md5": "prompt",
                            "summary": {"ok_runs": 0, "total_runs": 1},
                            "runs": [
                                {
                                    "run": 1,
                                    "status": "hard_error",
                                    "panic": "boom",
                                    "output_prefix": "ignored",
                                }
                            ],
                        },
                    ],
                },
                path,
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("# MQ6 AR Perf Baseline", text)
        self.assertIn("deadbeef", text)
        self.assertIn("case-ok", text)
        self.assertIn("case-hard run 1: boom", text)


if __name__ == "__main__":
    unittest.main()
