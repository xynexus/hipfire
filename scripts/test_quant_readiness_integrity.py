#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "quant_readiness_integrity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quant_readiness_integrity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["quant_readiness_integrity"] = module
    spec.loader.exec_module(module)
    return module


def readiness_item(format_id: str, artifacts: tuple[str, ...], *, promotion_target: bool = True):
    return SimpleNamespace(
        id=format_id,
        promotion_target=promotion_target,
        evidence_artifacts=artifacts,
    )


class QuantReadinessIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.integrity = load_module()

    def test_missing_evidence_path_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "present.json").write_text(json.dumps({"schema": "test.schema"}), encoding="utf-8")
            items = (
                readiness_item("fmt", ("present.json", "missing.md")),
                readiness_item("control", (), promotion_target=False),
            )

            report = self.integrity.build_integrity_report(
                source_root=root,
                readiness_items=items,
                expectations={},
            )

        self.assertFalse(report["summary"]["ok"])
        self.assertEqual(report["summary"]["evidence_count"], 2)
        self.assertIn("missing_evidence", {issue["kind"] for issue in report["issues"]})

    def test_structured_schema_and_value_mismatches_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "status.json"
            artifact.write_text(
                json.dumps({"schema": "wrong.schema", "promotion_allowed": True}),
                encoding="utf-8",
            )
            items = (readiness_item("fmt", ("status.json",)),)
            expectations = {
                "status.json": {
                    "schema": "expected.schema",
                    "format_id": "fmt",
                    "expected": {"promotion_allowed": False},
                }
            }

            report = self.integrity.build_integrity_report(
                source_root=root,
                readiness_items=items,
                expectations=expectations,
            )

        kinds = {issue["kind"] for issue in report["issues"]}
        self.assertFalse(report["summary"]["ok"])
        self.assertIn("schema_mismatch", kinds)
        self.assertIn("expected_value_mismatch", kinds)

    def test_expected_structured_artifact_must_be_cited_by_its_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text(
                json.dumps({"schema": "expected.schema", "promotion_allowed": False}),
                encoding="utf-8",
            )
            items = (readiness_item("other", ("status.json",)),)
            expectations = {
                "status.json": {
                    "schema": "expected.schema",
                    "format_id": "fmt",
                    "expected": {"promotion_allowed": False},
                }
            }

            report = self.integrity.build_integrity_report(
                source_root=root,
                readiness_items=items,
                expectations=expectations,
            )

        self.assertFalse(report["summary"]["ok"])
        self.assertIn("structured_artifact_not_cited_by_format", {issue["kind"] for issue in report["issues"]})

    def test_current_readiness_evidence_is_integrity_clean(self):
        report = self.integrity.build_integrity_report()

        self.assertTrue(report["summary"]["ok"], report["issues"])
        self.assertEqual(report["schema"], self.integrity.SCHEMA)
        self.assertGreater(report["summary"]["evidence_count"], 0)
        self.assertEqual(report["summary"]["issue_count"], 0)
        expected_paths = set(self.integrity.EXPECTED_STRUCTURED_ARTIFACTS)
        structured_paths = {item["path"] for item in report["structured_artifacts"]}
        self.assertEqual(structured_paths, expected_paths)


if __name__ == "__main__":
    unittest.main()
