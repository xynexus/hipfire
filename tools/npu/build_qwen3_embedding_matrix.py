#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build and package the compiled Qwen3 AIE2P embedding image matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BUCKETS = (128, 256, 512, 1024, 2048)
MAX_PADDED_ROWS = 4096
BUILD_FINGERPRINT = ".hipfire-build.json"


def image_ready(path: Path) -> bool:
    return (path / "final.xclbin").is_file() and (path / "insts.bin").is_file()


def swiglu_ready(path: Path) -> bool:
    return len(list(path.glob("*.xclbin"))) == 1 and len(list(path.glob("*instr*.bin"))) == 1


def build_fingerprint(command: list[str], sources: tuple[Path, ...]) -> dict[str, object]:
    return {
        "schema": "hipfire.npu_component_build.v1",
        "command": command,
        "sources": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sorted(sources)
        },
    }


def write_build_fingerprint(output: Path, command: list[str], sources: tuple[Path, ...]) -> None:
    fingerprint = build_fingerprint(command, sources)
    temporary = output / f"{BUILD_FINGERPRINT}.tmp"
    temporary.write_text(json.dumps(fingerprint, indent=2) + "\n")
    temporary.replace(output / BUILD_FINGERPRINT)


def cached_image_ready(output: Path, ready, command: list[str], sources: tuple[Path, ...]) -> bool:
    if not ready(output):
        return False
    try:
        recorded = json.loads((output / BUILD_FINGERPRINT).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return recorded == build_fingerprint(command, sources)


def run(
    command: list[str],
    output: Path,
    ready,
    force: bool,
    sources: tuple[Path, ...],
) -> Path:
    if not force and cached_image_ready(output, ready, command, sources):
        return output
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    if not ready(output):
        raise RuntimeError(f"builder did not produce a complete image in {output}")
    write_build_fingerprint(output, command, sources)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--quant-format", default="oq8+")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    home = Path.home()
    parser.add_argument(
        "--component-root",
        type=Path,
        default=home / ".cache/hipfire/npu/components/qwen3-embedding",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=home / ".cache/hipfire/npu/embedding",
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.query_heads not in (16, 32) or args.kv_heads != 8 or args.head_dim != 128:
        parser.error("Qwen3 embedding matrix requires QH16/32, KVH8, and D128")

    tools = Path(__file__).resolve().parent
    venv = Path(os.environ.get("HIPFIRE_NPU_VENV", home / ".venv"))
    python = venv / "bin/python"
    geometries = [
        (bucket, batch) for bucket in BUCKETS for batch in (1, 2, 4, 8, 16, 32) if bucket * batch <= MAX_PADDED_ROWS
    ]
    row_counts = sorted({max(256, bucket * batch) for bucket, batch in geometries})
    builds: dict[str, tuple[list[str], Path, object, tuple[Path, ...]]] = {}

    def sources(builder: str, kernel: str) -> tuple[Path, ...]:
        return (tools / builder, tools / kernel)

    projection_shapes = {
        "projection-q": (args.hidden_size, args.query_heads * args.head_dim),
        "projection-kv": (args.hidden_size, args.kv_heads * args.head_dim),
        "projection-o": (args.query_heads * args.head_dim, args.hidden_size),
        "projection-gate-up": (args.hidden_size, args.intermediate_size),
        "projection-down": (args.intermediate_size, args.hidden_size),
    }
    for rows in row_counts:
        row_root = args.component_root / f"rows-{rows}"
        residual = row_root / "residual-rmsnorm"
        builds[f"rows-{rows}/residual-rmsnorm"] = (
            [
                str(python),
                str(tools / "build_qwen3_residual_rmsnorm.py"),
                "--rows",
                str(rows),
                "--hidden-size",
                str(args.hidden_size),
                "--output",
                str(residual),
            ],
            residual,
            image_ready,
            sources("build_qwen3_residual_rmsnorm.py", "qwen3_residual_rmsnorm_bf16.cc"),
        )
        for name, (inner, outer) in projection_shapes.items():
            output = row_root / name
            builds[f"rows-{rows}/{name}"] = (
                [
                    str(python),
                    str(tools / "build_qwen3_oq8_projection.py"),
                    "--rows",
                    str(rows),
                    "--input-columns",
                    str(inner),
                    "--output-columns",
                    str(outer),
                    "--output",
                    str(output),
                ],
                output,
                image_ready,
                sources("build_qwen3_oq8_projection.py", "qwen3_oq8_projection_bf16.cc"),
            )
        swiglu = row_root / "swiglu"
        builds[f"rows-{rows}/swiglu"] = (
            [
                str(python),
                str(tools / "build_qwen35_swiglu.py"),
                "--hidden-size",
                str(rows * args.intermediate_size),
                "--npu",
                "npu2",
                "--out-dir",
                str(swiglu),
            ],
            swiglu,
            swiglu_ready,
            sources("build_qwen35_swiglu.py", "silu_mul_bf16.cc"),
        )

    geometry_builders = {
        "headnorm-rope": (
            "build_qwen3_headnorm_rope.py",
            [
                "--query-heads",
                str(args.query_heads),
                "--kv-heads",
                str(args.kv_heads),
                "--head-dim",
                str(args.head_dim),
            ],
        ),
        "query-pack": (
            "build_qwen3_query_pack.py",
            ["--query-heads", str(args.query_heads)],
        ),
        "kv-pack": (
            "build_qwen3_kv_pack.py",
            [],
        ),
        "attention": (
            "build_qwen3_segmented_attention.py",
            [
                "--query-heads",
                str(args.query_heads),
                "--kv-heads",
                str(args.kv_heads),
                "--head-dim",
                str(args.head_dim),
            ],
        ),
        "attention-unpack": (
            "build_qwen3_attention_unpack.py",
            ["--query-heads", str(args.query_heads)],
        ),
        "final-pool-l2": (
            "build_qwen3_final_pool_l2.py",
            ["--hidden-size", str(args.hidden_size)],
        ),
    }
    for bucket, batch in geometries:
        geometry_root = args.component_root / f"s{bucket}-b{batch}"
        for name, (script, extra) in geometry_builders.items():
            output = geometry_root / name
            builds[f"s{bucket}-b{batch}/{name}"] = (
                [
                    str(python),
                    str(tools / script),
                    "--bucket",
                    str(bucket),
                    "--batch",
                    str(batch),
                    *extra,
                    "--output",
                    str(output),
                ],
                output,
                image_ready,
                sources(
                    script,
                    {
                        "build_qwen3_headnorm_rope.py": "qwen3_headnorm_rope_bf16.cc",
                        "build_qwen3_query_pack.py": "qwen3_query_pack_bf16.cc",
                        "build_qwen3_kv_pack.py": "qwen3_kv_pack_bf16.cc",
                        "build_qwen3_segmented_attention.py": "segmented_attention_bf16.cc",
                        "build_qwen3_attention_unpack.py": "qwen3_attention_unpack_bf16.cc",
                        "build_qwen3_final_pool_l2.py": "qwen3_final_pool_l2_bf16.cc",
                    }[script],
                ),
            )

    print(f"building {len(builds)} component images with {args.jobs} workers")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run, command, output, ready, args.force, build_sources): name
            for name, (command, output, ready, build_sources) in builds.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                print(f"error: {name}: {error}", file=sys.stderr)
                return 1
            print(f"ready {name}")

    for bucket, batch in geometries:
        rows = max(256, bucket * batch)
        row_root = args.component_root / f"rows-{rows}"
        geometry_root = args.component_root / f"s{bucket}-b{batch}"
        key = (
            f"aie2p-qwen3-h{args.hidden_size}-l{args.layers}-qh{args.query_heads}-"
            f"kvh{args.kv_heads}-d{args.head_dim}-i{args.intermediate_size}-"
            f"{args.quant_format}-s{bucket}-b{batch}"
        )
        output = args.output_root / key
        command = [
            str(python),
            str(tools / "package_qwen3_embedding_bundle.py"),
            "--output",
            str(output),
            "--bucket",
            str(bucket),
            "--batch",
            str(batch),
            "--hidden-size",
            str(args.hidden_size),
            "--layers",
            str(args.layers),
            "--query-heads",
            str(args.query_heads),
            "--kv-heads",
            str(args.kv_heads),
            "--head-dim",
            str(args.head_dim),
            "--intermediate-size",
            str(args.intermediate_size),
            "--quant-format",
            args.quant_format,
        ]
        for name in (
            "residual-rmsnorm",
            "projection-q",
            "projection-kv",
            "projection-o",
            "projection-gate-up",
            "projection-down",
            "swiglu",
        ):
            command += ["--component", f"{name}={row_root / name}"]
        for name in geometry_builders:
            command += ["--component", f"{name}={geometry_root / name}"]
        subprocess.run(command, check=True)
    print(f"packaged {len(geometries)} Qwen3 embedding bundles in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
