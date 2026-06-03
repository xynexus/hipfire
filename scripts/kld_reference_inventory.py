#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Inventory HFKLDR references for quant-readiness gates.

This script keeps KLD-reference blocker audits reproducible. It verifies the
manifest-pinned local refs by SHA-256 and scans local roots for missing
same-fixture references such as `qwen3.5-27b-bf16.kldref.bin` or A3B refs. The
optional Hugging Face dataset listing is recorded as observed evidence only;
promotion still requires a manifest-pinned local ref.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hipfire.quant_kld_reference_inventory.v0"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "quality-baselines" / "harness" / "manifest.json"
DEFAULT_REF_DIR = ROOT / "benchmarks" / "quality-baselines" / "refs"
DEFAULT_SEARCH_ROOTS = (
    DEFAULT_REF_DIR,
    Path("/home/sadara/Models"),
    Path("/home/sadara/.cache/huggingface/hub"),
    Path("/home/sadara/.hipfire/models"),
)
DEFAULT_EXPECTED = (
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
)
REFERENCE_HINTS = ("kldref", "bf16", "q8", "gguf")
SKIP_DIR_NAMES = {".hipfire_kernels", ".locks", "__pycache__"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return text.lower().replace("_", "-")


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        current = Path(dirpath)
        for filename in filenames:
            yield current / filename


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_refs(manifest: dict, ref_dir: Path) -> list[dict]:
    rows = []
    for name, meta in sorted(manifest.get("references", {}).items()):
        path = ref_dir / name
        expected = meta.get("sha256")
        row = {
            "name": name,
            "path": str(path),
            "expected_sha256": expected,
            "exists": path.exists(),
            "manifest_hf_repo": meta.get("hf_repo"),
            "manifest_hf_repo_type": meta.get("hf_repo_type"),
            "slice_md5": meta.get("slice_md5"),
            "top_k": meta.get("top_k"),
            "n_ctx": meta.get("n_ctx"),
        }
        if path.exists():
            actual = sha256_file(path)
            row.update(
                {
                    "size_bytes": path.stat().st_size,
                    "actual_sha256": actual,
                    "sha256_ok": actual == expected,
                }
            )
        else:
            row["sha256_ok"] = False
        rows.append(row)
    return rows


def path_matches_fixture(path: Path, fixture: str) -> bool:
    text = normalize(str(path))
    fixture_norm = normalize(fixture)
    compact = fixture_norm.replace(".", "")
    return (fixture_norm in text or compact in text) and any(hint in text for hint in REFERENCE_HINTS)


def scan_expected_refs(search_roots: Iterable[Path], expected: Iterable[str], *, max_matches: int) -> dict[str, dict]:
    reports = {
        fixture: {
            "fixture": fixture,
            "matches": [],
            "match_count": 0,
        }
        for fixture in expected
    }
    root_reports = []
    for root in search_roots:
        root = root.expanduser()
        root_report = {"root": str(root), "exists": root.exists(), "files_seen": 0}
        if root.exists():
            for path in iter_files(root):
                root_report["files_seen"] += 1
                for fixture, report in reports.items():
                    if path_matches_fixture(path, fixture):
                        report["match_count"] += 1
                        if len(report["matches"]) < max_matches:
                            report["matches"].append(str(path))
        root_reports.append(root_report)
    return {"roots": root_reports, "fixtures": reports}


def list_hf_repo(repo: str, repo_type: str) -> dict:
    try:
        from huggingface_hub import list_repo_files
    except Exception as exc:
        return {"repo": repo, "repo_type": repo_type, "status": "unavailable", "error": str(exc)}
    try:
        files = sorted(list_repo_files(repo, repo_type=repo_type))
    except Exception as exc:
        return {"repo": repo, "repo_type": repo_type, "status": "error", "error": str(exc)}
    return {"repo": repo, "repo_type": repo_type, "status": "ok", "files": files}


def build_inventory(
    *,
    manifest_path: Path,
    ref_dir: Path,
    search_roots: Iterable[Path],
    expected: Iterable[str],
    hf_repo: str | None,
    hf_repo_type: str,
    max_matches: int,
) -> dict:
    manifest = load_manifest(manifest_path)
    refs = manifest_refs(manifest, ref_dir)
    scan = scan_expected_refs(search_roots, expected, max_matches=max_matches)
    expected_reports = []
    refs_by_name = {row["name"]: row for row in refs}
    for fixture, report in scan["fixtures"].items():
        expected_name = f"{fixture}-bf16.kldref.bin"
        manifest_row = refs_by_name.get(expected_name)
        expected_reports.append(
            {
                **report,
                "expected_ref_name": expected_name,
                "manifest_entry_present": manifest_row is not None,
                "local_manifest_ref_present": bool(manifest_row and manifest_row["exists"]),
                "local_manifest_ref_sha256_ok": bool(manifest_row and manifest_row["sha256_ok"]),
            }
        )
    payload = {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "ref_dir": str(ref_dir),
        "manifest_refs": refs,
        "search_roots": scan["roots"],
        "expected_missing_or_required": expected_reports,
    }
    if hf_repo:
        payload["remote_listing"] = list_hf_repo(hf_repo, hf_repo_type)
    return payload


def write_json(payload: dict, *, pretty: bool = False, out: str | None = None):
    text = json.dumps(payload, indent=2, sort_keys=True) if pretty else json.dumps(payload, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory manifest-pinned HFKLDR references.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    parser.add_argument("--search-root", action="append")
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--hf-repo", default="hipfire-models/qwen-kldref")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--no-remote", action="store_true")
    parser.add_argument("--max-matches", type=int, default=32)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    search_roots = tuple(Path(root) for root in args.search_root) if args.search_root else DEFAULT_SEARCH_ROOTS
    expected = tuple(args.expected) if args.expected else DEFAULT_EXPECTED
    payload = build_inventory(
        manifest_path=Path(args.manifest),
        ref_dir=Path(args.ref_dir),
        search_roots=search_roots,
        expected=expected,
        hf_repo=None if args.no_remote else args.hf_repo,
        hf_repo_type=args.hf_repo_type,
        max_matches=args.max_matches,
    )
    write_json(payload, pretty=args.pretty, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
