#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build segmented-attention to token-major Qwen3 unpacking for AIE2P."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


COLS = 8
ROWS = 4
QUERIES = 4
MAX_DMA_CHUNKS = 64
BUCKETS = (128, 256, 512, 1024, 2048)
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"<size = {size}, stride = {stride}>" for size, stride in entries) + "]"


def blocks(count: int, block: int) -> str:
    return dims([(count, block), (block // 512, 512), (512, 1)])


def generate_mlir(bucket: int, batch: int, query_heads: int, head_dim: int) -> str:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}")
    if batch <= 0 or bucket * batch > 4096:
        raise ValueError("batch must be positive and bucket*batch <= 4096")
    if query_heads not in (16, 32):
        raise ValueError("query_heads must be 16 or 32")
    if head_dim != 128:
        raise ValueError("head_dim must be 128")

    q_per_kv = query_heads // COLS
    token_chunks = bucket // (ROWS * QUERIES)
    groups = q_per_kv * token_chunks
    q_width = query_heads * head_dim
    tile = QUERIES * head_dim * 2
    input_join = ROWS * tile
    input_document = COLS * groups * input_join
    input_bytes = batch * input_document
    output_join = ROWS * tile
    output_document = bucket * q_width * 2
    output_bytes = batch * output_document

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(COLS):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
        for row in range(ROWS):
            out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
    for col in range(COLS):
        consumers = ", ".join(f"@i{col}_{row}" for row in range(ROWS))
        offsets = ", ".join(str(row * tile) for row in range(ROWS))
        out.append(
            f"    aie.objectfifo @ish{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{input_join}xi8>>"
        )
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @i{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{tile}xi8>>"
            )
        out.append(f"    aie.objectfifo.link [@ish{col}] -> [{consumers}] ([] [{offsets}])")
    for col in range(COLS):
        producers = ", ".join(f"@o{col}_{row}" for row in range(ROWS))
        offsets = ", ".join(str(row * tile) for row in range(ROWS))
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @o{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{tile}xi8>>"
            )
        out += [
            f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{output_join}xi8>>",
            f"    aie.objectfifo.link [{producers}] -> [@osh{col}] ([{offsets}] [])",
        ]
    out.append(
        f'    func.func private @hipfire_qwen3_copy_attention_tile(memref<{tile}xi8>, memref<{tile}xi8>) attributes {{link_with = "qwen3_attention_unpack.o"}}'
    )
    for col in range(COLS):
        for row in range(ROWS):
            out += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %groups = arith.constant {groups} : index",
                f"      %batch = arith.constant {batch} : index",
                "      scf.for %outer = %z to %inf step %one {",
                "        scf.for %document = %z to %batch step %one {",
            ]
            out += [
                "          scf.for %group = %z to %groups step %one {",
                f"            %i = aie.objectfifo.acquire @i{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{tile}xi8>>",
                f"            %iv = aie.objectfifo.subview.access %i[0] : !aie.objectfifosubview<memref<{tile}xi8>> -> memref<{tile}xi8>",
                f"            %o = aie.objectfifo.acquire @o{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{tile}xi8>>",
                f"            %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{tile}xi8>> -> memref<{tile}xi8>",
                f"            func.call @hipfire_qwen3_copy_attention_tile(%iv, %ov) : (memref<{tile}xi8>, memref<{tile}xi8>) -> ()",
                f"            aie.objectfifo.release @i{col}_{row}(Consume, 1)",
                f"            aie.objectfifo.release @o{col}_{row}(Produce, 1)",
                "          }",
                "        }",
                "      }",
                "      aie.end",
                "    } {stack_size = 1024 : i32}",
            ]

    out.append(f"    aie.runtime_sequence(%I: memref<{input_bytes}xi8>, %O: memref<{output_bytes}xi8>) {{")
    for document in range(batch):
        for local in range(q_per_kv):
            for chunk_start in range(0, token_chunks, MAX_DMA_CHUNKS):
                chunk_count = min(MAX_DMA_CHUNKS, token_chunks - chunk_start)
                phase: list[str] = []
                for col in range(COLS):
                    input_name = f"ti{col}_{document}_{local}_{chunk_start}"
                    phase.append(input_name)
                    input_offset = (
                        document * input_document
                        + col * groups * input_join
                        + (local * token_chunks + chunk_start) * input_join
                    )
                    out += [
                        f"      %{input_name} = aiex.dma_configure_task_for @ish{col} {{",
                        f"        aie.dma_bd(%I : memref<{input_bytes}xi8>, {input_offset}, {chunk_count * input_join}, {blocks(chunk_count, input_join)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{input_name})",
                    ]
                    output_name = f"to{col}_{document}_{local}_{chunk_start}"
                    phase.append(output_name)
                    output_offset = (
                        document * output_document
                        + (col * q_per_kv + local) * head_dim * 2
                        + chunk_start * ROWS * QUERIES * q_width * 2
                    )
                    layout = dims(
                        [
                            (chunk_count, ROWS * QUERIES * q_width * 2),
                            (ROWS, QUERIES * q_width * 2),
                            (QUERIES, q_width * 2),
                            (head_dim * 2, 1),
                        ]
                    )
                    out += [
                        f"      %{output_name} = aiex.dma_configure_task_for @osh{col} {{",
                        f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {output_offset}, {output_join}, {layout}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        f"      }} {{issue_token = true, repeat_count = {chunk_count - 1} : i32}}",
                        f"      aiex.dma_start_task(%{output_name})",
                    ]
                for task in phase:
                    out += [
                        f"      aiex.dma_await_task(%{task})",
                        f"      aiex.dma_free_task(%{task})",
                    ]
    out += ["    }", "  }", "}"]
    return "\n".join(out) + "\n"


def toolchain() -> tuple[Path, Path]:
    venv = Path(os.environ.get("HIPFIRE_NPU_VENV", Path.home() / ".venv"))
    python = venv / "bin/python"
    location = subprocess.run(
        [str(python), "-c", "import mlir_aie; print(list(mlir_aie.__path__)[0])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mlir_aie = Path(location)
    return mlir_aie, mlir_aie.parent / "llvm-aie"


def build(output: Path, bucket: int, batch: int, query_heads: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(bucket, batch, query_heads, 128), encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")]
        )
        source = Path(__file__).with_name("qwen3_attention_unpack_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "qwen3_attention_unpack.o"),
                f"-I{mlir_aie / 'include'}",
                "-std=c++20",
                "-O2",
                "-DNDEBUG",
                "-DHIPFIRE_HEAD_DIM=128",
                "-Wno-parentheses",
                "-Wno-attributes",
                "-Wno-macro-redefined",
                "-Wno-empty-body",
                "-Wno-deprecated-declarations",
                "--target=aie2p-none-unknown-elf",
            ],
            check=True,
            env=env,
        )
        aiecc = shutil.which("aiecc", path=env["PATH"])
        if aiecc is None:
            raise RuntimeError("aiecc not found")
        subprocess.run(
            [
                aiecc,
                str(mlir),
                "--no-compile-host",
                "--no-xchesscc",
                "--no-xbridge",
                f"--peano={peano}",
                "--aie-generate-npu-insts",
                f"--npu-insts-name={output / 'insts.bin'}",
                "--aie-generate-xclbin",
                f"--xclbin-name={output / 'final.xclbin'}",
                f"--tmpdir={output}",
            ],
            check=True,
            env=env,
        )
    manifest = {
        "schema": "hipfire.npu_qwen3_attention_unpack.v1",
        "sequence_bucket": bucket,
        "dispatch_batch": batch,
        "query_heads": query_heads,
        "head_dim": 128,
        "input_layout": "segmented_attention_output",
        "output_layout": "token_major_b_s_qh_d_bf16",
        "xclbin": "final.xclbin",
        "instructions": "insts.bin",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True, choices=BUCKETS)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--query-heads", type=int, choices=(16, 32), default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(args.output, args.bucket, args.batch, args.query_heads, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
