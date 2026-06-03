#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mq6_dflash_baseline.py"
spec = importlib.util.spec_from_file_location("mq6_dflash_baseline", MODULE_PATH)
assert spec and spec.loader
mq6_dflash = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq6_dflash
spec.loader.exec_module(mq6_dflash)


class Mq6DflashBaselineTest(unittest.TestCase):
    def test_parse_metrics_and_tokens(self):
        output = """
=== BENCH METRICS ===
prompt_tokens: 13
prefill_secs: 1.3337
prefill_tok_s: 9.75
ttft_ms: 2179.19
decode_tokens_emitted: 36
decode_secs: 4.1478
decode_tok_s: 8.68
decode_tau: 6.0000
decode_accept_rate: 0.4000
vram_used_mb: 23037
vram_total_mb: 122880
=====================
DFlash tokens: [271, 248068, 198, 90700]
"""
        metrics = mq6_dflash.parse_metrics(output)
        self.assertEqual(metrics["prompt_tokens"], 13)
        self.assertEqual(metrics["decode_tokens_emitted"], 36)
        self.assertAlmostEqual(metrics["decode_tok_s"], 8.68)
        self.assertAlmostEqual(metrics["decode_tau"], 6.0)
        self.assertEqual(mq6_dflash.parse_tokens(output), [271, 248068, 198, 90700])

    def test_token_attractor_detector_hard_fails_single_token_loop(self):
        result = mq6_dflash.token_attractor_check([42] * 64)
        self.assertFalse(result["ok"])
        self.assertEqual(result["max_tok"], 42)
        self.assertGreater(result["max_freq"], 0.50)

    def test_token_attractor_detector_allows_short_eot_trim(self):
        result = mq6_dflash.token_attractor_check([1, 2, 248044, 42, 42, 42])
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "short_window_ok")

    def test_documented_pretty_flag_is_accepted(self):
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertIn("--pretty", proc.stdout)


if __name__ == "__main__":
    unittest.main()
