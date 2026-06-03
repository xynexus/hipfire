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
MODULE_PATH = ROOT / "scripts" / "mq3_a3b_coherence.py"
spec = importlib.util.spec_from_file_location("mq3_a3b_coherence", MODULE_PATH)
assert spec and spec.loader
coh = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = coh
spec.loader.exec_module(coh)


class Mq3A3BCoherenceTest(unittest.TestCase):
    def test_default_cases_cover_controls_candidates_and_prompts(self):
        ids = {case.id for case in coh.DEFAULT_CASES}
        for prompt in ("trains", "humaneval3", "long-lru"):
            self.assertIn(f"qwen35-a3b-mq4-{prompt}", ids)
            self.assertIn(f"qwen35-a3b-mq3-{prompt}", ids)
            self.assertIn(f"qwen36-a3b-mq4-{prompt}", ids)
            self.assertIn(f"qwen36-a3b-mq3-{prompt}", ids)
        self.assertEqual(len(ids), len(coh.DEFAULT_CASES))

    def test_prompt_fixtures_are_committed_files(self):
        for prompt in coh.PROMPT_CASES:
            path, text = coh.load_prompt(prompt)
            self.assertTrue(path.exists(), prompt.prompt_file)
            self.assertGreater(len(text), 20)

    def test_write_markdown_records_outputs_and_hard_errors(self):
        payload = {
            "date": "2026-06-03T00:00:00+00:00",
            "commit": "abc123",
            "branch": "test",
            "arch": "gfx1151",
            "daemon": {"md5": "daemon-md5"},
            "cases": [
                {
                    "id": "qwen35-a3b-mq3-trains",
                    "model": {"model_file": "qwen3.5-35b-a3b.mq3", "format_id": "mq3"},
                    "prompt": {"id": "trains", "prompt_file": "trains-meet.txt"},
                    "status": "hard_error",
                    "done": None,
                    "prompt_md5": "prompt-md5",
                    "hit_max_tokens": False,
                    "panic": "boom",
                    "output": "partial",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            coh.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("# MQ3 A3B Broader Coherence", text)
        self.assertIn("daemon-md5", text)
        self.assertIn("qwen35-a3b-mq3-trains: boom", text)
        self.assertIn("partial", text)


if __name__ == "__main__":
    unittest.main()
