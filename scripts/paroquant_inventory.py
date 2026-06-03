#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Inventory local roots for native ParoQuant source checkpoints.

This is a metadata scanner. It does not load tensor payloads and it does not
write model artifacts. The goal is to make ParoQ4G128/G256 producer evidence
reproducible: native Paro source checkpoints must expose the full companion
tensor set (`qweight`, `qzeros`, `scales`, `pairs`, `theta`,
`channel_scales`), and G256 requires those companions to imply
`group_size=256`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCHEMA = "hipfire.astrea.paro_source_inventory.v0"
PARO_SUFFIXES = (
    ".qweight",
    ".qzeros",
    ".scales",
    ".pairs",
    ".theta",
    ".channel_scales",
)
DEFAULT_ROOTS = (
    Path("/home/sadara/Models"),
    Path("/home/sadara/.cache/huggingface/hub"),
    Path("/home/sadara/.hipfire/models"),
)
PATH_HINTS = (
    "paro",
    "g128",
    "g256",
    "qweight",
    "qzeros",
    "pairs",
    "theta",
    "channel_scales",
)
SKIP_DIR_NAMES = {".hipfire_kernels", ".locks", "__pycache__"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def paro_base_from_key(name: str) -> str | None:
    return name.removesuffix(".qweight") if name.endswith(".qweight") else None


def group_size_from_shapes(qweight_shape: list[int] | None, qzeros_shape: list[int] | None) -> int | None:
    if not qweight_shape or not qzeros_shape:
        return None
    if len(qweight_shape) < 1 or len(qzeros_shape) < 1:
        return None
    in_features = int(qweight_shape[0])
    groups = int(qzeros_shape[0])
    if in_features <= 0 or groups <= 0:
        return None
    if in_features % groups != 0:
        return None
    return in_features // groups


def path_has_hint(path: Path) -> bool:
    text = str(path).lower()
    return any(hint in text for hint in PATH_HINTS)


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
            current = Path(dirpath)
            for filename in filenames:
                yield current / filename
    except OSError:
        return


def tensor_meta_for_file(path: Path) -> tuple[dict[str, dict], str | None]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - dependency availability varies by host.
        return {}, f"safetensors unavailable: {exc}"

    tensors: dict[str, dict] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = None
                dtype = None
                try:
                    tensor_slice = handle.get_slice(key)
                    shape = [int(x) for x in tensor_slice.get_shape()]
                    dtype = str(tensor_slice.get_dtype())
                except Exception as exc:
                    dtype = f"metadata-error:{exc}"
                tensors[key] = {"file": str(path), "shape": shape, "dtype": dtype}
    except Exception as exc:
        return {}, str(exc)
    return tensors, None


def discover_paro_modules(tensors: dict[str, dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    suffix_counts: dict[str, int] = {}
    for name in tensors:
        suffix = next((s for s in PARO_SUFFIXES if name.endswith(s)), ".other")
        suffix_counts[suffix.removeprefix(".")] = suffix_counts.get(suffix.removeprefix("."), 0) + 1

    modules = []
    incomplete = []
    for name in sorted(tensors):
        base = paro_base_from_key(name)
        if base is None:
            continue
        required = [f"{base}{suffix}" for suffix in PARO_SUFFIXES]
        missing = [key for key in required if key not in tensors]
        if missing:
            incomplete.append({"base": base, "missing": missing})
            continue
        qweight_shape = tensors[f"{base}.qweight"].get("shape")
        qzeros_shape = tensors[f"{base}.qzeros"].get("shape")
        modules.append(
            {
                "base": base,
                "group_size": group_size_from_shapes(qweight_shape, qzeros_shape),
                "qweight_shape": qweight_shape,
                "qzeros_shape": qzeros_shape,
                "scales_shape": tensors[f"{base}.scales"].get("shape"),
                "pairs_shape": tensors[f"{base}.pairs"].get("shape"),
                "theta_shape": tensors[f"{base}.theta"].get("shape"),
                "channel_scales_shape": tensors[f"{base}.channel_scales"].get("shape"),
            }
        )
    return modules, incomplete, dict(sorted(suffix_counts.items()))


def scan_roots(roots: Iterable[Path], *, max_filename_hits: int = 64, max_modules: int = 32) -> dict:
    root_reports = []
    all_modules = []
    all_incomplete = []
    filename_hits = []
    scan_errors = []
    safetensor_dirs: dict[Path, list[Path]] = defaultdict(list)
    total_files_seen = 0

    for root in roots:
        root = root.expanduser()
        root_report = {
            "root": str(root),
            "exists": root.exists(),
            "files_seen": 0,
            "safetensor_file_count": 0,
            "filename_hit_count": 0,
        }
        if root.exists():
            for path in iter_files(root):
                total_files_seen += 1
                root_report["files_seen"] += 1
                if path_has_hint(path):
                    root_report["filename_hit_count"] += 1
                    if len(filename_hits) < max_filename_hits:
                        filename_hits.append(str(path))
                if path.suffix == ".safetensors":
                    root_report["safetensor_file_count"] += 1
                    safetensor_dirs[path.parent].append(path)
        root_reports.append(root_report)

    for directory, files in sorted(safetensor_dirs.items(), key=lambda item: str(item[0])):
        tensors = {}
        for path in sorted(files):
            file_tensors, error = tensor_meta_for_file(path)
            if error:
                scan_errors.append({"file": str(path), "error": error})
                continue
            tensors.update(file_tensors)
        modules, incomplete, suffix_counts = discover_paro_modules(tensors)
        if modules or incomplete:
            all_modules.extend({"source_dir": str(directory), **module} for module in modules)
            all_incomplete.extend({"source_dir": str(directory), **module} for module in incomplete)
        elif any(name in suffix_counts for name in ("qweight", "qzeros", "scales", "pairs", "theta", "channel_scales")):
            scan_errors.append(
                {
                    "source_dir": str(directory),
                    "error": "Paro-like suffixes found but no complete qweight-rooted module was discovered",
                    "suffix_counts": suffix_counts,
                }
            )

    group_counts: dict[str, int] = {}
    for module in all_modules:
        key = str(module.get("group_size"))
        group_counts[key] = group_counts.get(key, 0) + 1

    return {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "roots": root_reports,
        "files_seen": total_files_seen,
        "safetensor_dirs_scanned": len(safetensor_dirs),
        "safetensor_files_scanned": sum(len(files) for files in safetensor_dirs.values()),
        "filename_hit_count": sum(root["filename_hit_count"] for root in root_reports),
        "filename_hits": filename_hits,
        "native_paro": {
            "complete_module_count": len(all_modules),
            "incomplete_module_count": len(all_incomplete),
            "complete_modules_by_group_size": dict(sorted(group_counts.items())),
            "g128_complete_module_count": group_counts.get("128", 0),
            "g256_complete_module_count": group_counts.get("256", 0),
            "modules": all_modules[:max_modules],
            "modules_truncated": len(all_modules) > max_modules,
            "incomplete": all_incomplete[:max_modules],
            "incomplete_truncated": len(all_incomplete) > max_modules,
        },
        "scan_errors": scan_errors,
        "decision": {
            "native_paro_source_found": bool(all_modules),
            "native_paro_g256_source_found": group_counts.get("256", 0) > 0,
            "quality_state": "verifiable" if group_counts.get("256", 0) > 0 else "UNVERIFIABLE",
        },
    }


def write_json(payload: dict, *, pretty: bool = False, out: str | None = None):
    text = json.dumps(payload, indent=2, sort_keys=True) if pretty else json.dumps(payload, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory local roots for native ParoQuant source checkpoints.")
    parser.add_argument("--root", action="append", help="Root to scan. May be repeated.")
    parser.add_argument("--max-filename-hits", type=int, default=64)
    parser.add_argument("--max-modules", type=int, default=32)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(Path(root) for root in args.root) if args.root else DEFAULT_ROOTS
    write_json(
        scan_roots(roots, max_filename_hits=args.max_filename_hits, max_modules=args.max_modules),
        pretty=args.pretty,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
