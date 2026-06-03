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
MODULE_PATH = ROOT / "scripts" / "mq2_status.py"
CURRENT_ARTIFACT = ROOT / "benchmarks" / "results" / "gfx1151-quant-readiness" / "2026-06-03-mq2-status.json"


spec = importlib.util.spec_from_file_location("mq2_status", MODULE_PATH)
assert spec and spec.loader
mq2_status = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mq2_status
spec.loader.exec_module(mq2_status)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload):
    write(path, json.dumps(payload))


def provenance_payload(names: list[str]):
    return {
        "schema": "hipfire.quant_readiness.gfx1151.v0",
        "formats": [
            {
                "id": "mq2",
                "status": "quality-rejected",
                "candidate_artifacts": {
                    "artifacts": [{"name": name, "path": f"/models/{name}", "size_bytes": 1} for name in names]
                },
            }
        ],
    }


def sweep_text(output: str, *, status: str = "OK"):
    stats = '{"type":"done","id":"r1","tokens":12,"prefill_tok_s":1.0,"decode_tok_s":2.0}'
    return f"""# MQ3/MQ2 sweep

- `cap.txt` md5=`bbf2d0483ecacaa2eb7296f9c8083f95` size=60B

## qwen3.5-9b.mq2

### cap

- wall: 1.0s  status: **{status}**
- stats: `{stats}`
- prompt-file: `./benchmarks/prompts/sweep/cap.txt`  md5=`bbf2d0483ecacaa2eb7296f9c8083f95`

**Output:**

```
{output}
```
"""


def sweep_text_two_rows(first_output: str, second_output: str):
    stats = '{"type":"done","id":"r1","tokens":12,"prefill_tok_s":1.0,"decode_tok_s":2.0}'
    return f"""# MQ3/MQ2 sweep

- `cap.txt` md5=`bbf2d0483ecacaa2eb7296f9c8083f95` size=60B
- `code.txt` md5=`e2f8e5b972f5cb8560d004062094d8a8` size=60B

## qwen3.5-9b.mq2

### cap

- wall: 1.0s  status: **OK**
- stats: `{stats}`
- prompt-file: `./benchmarks/prompts/sweep/cap.txt`  md5=`bbf2d0483ecacaa2eb7296f9c8083f95`

**Output:**

```
{first_output}
```

### code

- wall: 1.0s  status: **OK**
- stats: `{stats}`
- prompt-file: `./benchmarks/prompts/sweep/code.txt`  md5=`e2f8e5b972f5cb8560d004062094d8a8`

**Output:**

```
{second_output}
```
"""


class Mq2StatusTest(unittest.TestCase):
    def test_parse_sweep_marks_garbled_cap_row_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.md"
            write(path, sweep_text("gibberish"))

            parsed = mq2_status.parse_sweep(path)

        self.assertEqual(parsed["row_count"], 1)
        self.assertEqual(parsed["hard_error_count"], 0)
        self.assertEqual(parsed["collapsed_row_count"], 1)
        self.assertTrue(parsed["all_rows_runtime_ok"])
        self.assertTrue(parsed["all_rows_collapsed"])

    def test_parse_sweep_accepts_expected_cap_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.md"
            write(path, sweep_text("The capital of France is Paris."))

            parsed = mq2_status.parse_sweep(path)

        self.assertEqual(parsed["quality_pass_row_count"], 1)
        self.assertFalse(parsed["all_rows_collapsed"])

    def test_build_status_rejects_when_all_dense_rows_collapse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = root / "provenance.json"
            sweep = root / "sweep.md"
            write_json(provenance, provenance_payload(list(mq2_status.REQUIRED_DENSE_ARTIFACTS)))
            write(sweep, sweep_text("not an answer"))

            status = mq2_status.build_status(provenance, sweep)

        self.assertEqual(status["schema"], mq2_status.SCHEMA)
        self.assertEqual(status["status"], "quality-rejected")
        self.assertFalse(status["promotion_allowed"])
        self.assertTrue(status["gates"]["canonical_dense_artifacts_present"])
        self.assertTrue(status["gates"]["runtime_smoke_no_hard_errors"])
        self.assertFalse(status["gates"]["dense_quality_clean"])
        self.assertTrue(status["gates"]["all_dense_rows_collapsed"])
        boundary = status["quality_rejection_boundary"]
        self.assertEqual(boundary["dense_text_status"], "rejected_all_current_rows_collapsed")
        self.assertEqual(boundary["runtime_smoke_interpretation"], "runtime_ok_not_quality_evidence")
        self.assertTrue(boundary["quality_override_required"])
        self.assertFalse(boundary["promotion_perf_collection_allowed"])
        self.assertFalse(boundary["performance_rows_promotable"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "produce_new_mq2_calibration_that_passes_bounded_dense_quality",
        )
        self.assertIn("quality-collapsed", " ".join(status["blockers"]))

    def test_build_status_requires_every_dense_row_to_pass_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = root / "provenance.json"
            sweep = root / "sweep.md"
            write_json(provenance, provenance_payload(list(mq2_status.REQUIRED_DENSE_ARTIFACTS)))
            write(sweep, sweep_text_two_rows("The capital of France is Paris.", "not code"))

            status = mq2_status.build_status(provenance, sweep)

        self.assertFalse(status["promotion_allowed"])
        self.assertFalse(status["gates"]["dense_quality_clean"])
        self.assertFalse(status["gates"]["all_dense_rows_collapsed"])
        self.assertEqual(status["sweep"]["quality_pass_row_count"], 1)
        self.assertEqual(status["sweep"]["row_count"], 2)
        self.assertEqual(status["quality_rejection_boundary"]["dense_text_status"], "quality_incomplete")
        self.assertFalse(status["quality_rejection_boundary"]["promotion_perf_collection_allowed"])

    def test_current_mq2_status_artifact_is_dense_quality_rejected(self):
        payload = json.loads(CURRENT_ARTIFACT.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], mq2_status.SCHEMA)
        self.assertEqual(payload["status"], "quality-rejected")
        self.assertFalse(payload["promotion_allowed"])
        self.assertTrue(payload["gates"]["canonical_dense_artifacts_present"])
        self.assertTrue(payload["gates"]["runtime_smoke_no_hard_errors"])
        self.assertTrue(payload["gates"]["all_dense_rows_collapsed"])
        self.assertFalse(payload["gates"]["dense_quality_clean"])
        self.assertFalse(payload["gates"]["kld_ppl_evidence_present"])
        self.assertFalse(payload["gates"]["perf_promotion_evidence_present"])
        self.assertEqual(payload["sweep"]["row_count"], 12)
        self.assertEqual(payload["sweep"]["collapsed_row_count"], 12)
        boundary = payload["quality_rejection_boundary"]
        self.assertEqual(boundary["dense_text_status"], "rejected_all_current_rows_collapsed")
        self.assertEqual(boundary["runtime_smoke_interpretation"], "runtime_ok_not_quality_evidence")
        self.assertTrue(boundary["quality_override_required"])
        self.assertTrue(boundary["sub_9b_failures_explicit"])
        self.assertTrue(boundary["dense_9b_rejected"])
        self.assertFalse(boundary["promotion_perf_collection_allowed"])
        self.assertFalse(boundary["performance_rows_promotable"])
        self.assertEqual(
            boundary["next_unblocked_step"],
            "produce_new_mq2_calibration_that_passes_bounded_dense_quality",
        )
        by_model = {item["model"]: item for item in boundary["fixture_quality_outcomes"]}
        self.assertEqual(by_model["qwen3.5-0.8b.mq2"]["status"], "rejected")
        self.assertEqual(by_model["qwen3.5-4b.mq2"]["status"], "rejected")
        self.assertEqual(by_model["qwen3.5-9b.mq2"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
