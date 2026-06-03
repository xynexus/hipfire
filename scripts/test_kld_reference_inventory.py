#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "kld_reference_inventory.py"
spec = importlib.util.spec_from_file_location("kld_reference_inventory", MODULE_PATH)
assert spec and spec.loader
kld_inventory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kld_inventory
spec.loader.exec_module(kld_inventory)


class KldReferenceInventoryTest(unittest.TestCase):
    def test_manifest_refs_verify_local_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir = root / "refs"
            ref_dir.mkdir()
            ref = ref_dir / "qwen3.5-4b-bf16.kldref.bin"
            ref.write_bytes(b"kldref")
            digest = hashlib.sha256(b"kldref").hexdigest()
            manifest = {
                "references": {
                    ref.name: {
                        "sha256": digest,
                        "hf_repo": "hipfire-models/qwen-kldref",
                        "hf_repo_type": "dataset",
                        "slice_md5": "slice-md5",
                        "top_k": 256,
                        "n_ctx": 2048,
                    },
                    "qwen3.5-27b-bf16.kldref.bin": {
                        "sha256": "missing",
                    },
                }
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            rows = kld_inventory.manifest_refs(manifest, ref_dir)

        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["qwen3.5-4b-bf16.kldref.bin"]["exists"])
        self.assertTrue(by_name["qwen3.5-4b-bf16.kldref.bin"]["sha256_ok"])
        self.assertEqual(by_name["qwen3.5-4b-bf16.kldref.bin"]["size_bytes"], len(b"kldref"))
        self.assertFalse(by_name["qwen3.5-27b-bf16.kldref.bin"]["exists"])
        self.assertFalse(by_name["qwen3.5-27b-bf16.kldref.bin"]["sha256_ok"])

    def test_inventory_reports_expected_missing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir = root / "refs"
            ref_dir.mkdir()
            ref = ref_dir / "qwen3.5-9b-bf16.kldref.bin"
            ref.write_bytes(b"present")
            digest = hashlib.sha256(b"present").hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "references": {
                            ref.name: {
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "models").mkdir()
            (root / "models" / "qwen3.5-27b-bf16-imatrix.gguf").write_bytes(b"not-a-ref")

            payload = kld_inventory.build_inventory(
                manifest_path=manifest_path,
                ref_dir=ref_dir,
                search_roots=[root / "models", ref_dir],
                expected=["qwen3.5-27b", "qwen3.5-9b"],
                hf_repo=None,
                hf_repo_type="dataset",
                max_matches=8,
            )

        by_fixture = {row["fixture"]: row for row in payload["expected_missing_or_required"]}
        self.assertEqual(payload["schema"], kld_inventory.SCHEMA)
        self.assertFalse(by_fixture["qwen3.5-27b"]["manifest_entry_present"])
        self.assertFalse(by_fixture["qwen3.5-27b"]["local_manifest_ref_present"])
        self.assertEqual(by_fixture["qwen3.5-27b"]["match_count"], 1)
        self.assertTrue(by_fixture["qwen3.5-9b"]["manifest_entry_present"])
        self.assertTrue(by_fixture["qwen3.5-9b"]["local_manifest_ref_sha256_ok"])


if __name__ == "__main__":
    unittest.main()
