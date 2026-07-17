#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build token-major Qwen3 query packing for segmented AIE2P attention."""

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


def generate_mlir(bucket: int, batch: int, query_heads: int, head_dim: int) -> str:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}")
    if batch <= 0 or bucket * batch > 4096:
        raise ValueError("batch must be positive and bucket*batch <= 4096")
    if query_heads not in (16, 32) or query_heads % COLS != 0:
        raise ValueError("query_heads must be 16 or 32")
    if head_dim != 128:
        raise ValueError("head_dim must be 128")

    q_per_kv = query_heads // COLS
    token_chunks = bucket // (ROWS * QUERIES)
    q_groups = q_per_kv * token_chunks
    q_width = query_heads * head_dim
    input_document = bucket * q_width * 2
    input_bytes = batch * input_document
    q_tile = QUERIES * head_dim * 2
    raw_pair = 2 * q_tile
    raw_join = COLS // 2 * raw_pair
    q_pair = 2 * q_tile + 512
    q_join = COLS // 2 * q_pair
    output_document = ROWS * q_groups * q_join
    output_bytes = batch * output_document

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(COLS):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        for pair in range(COLS // 2):
            col = 2 * pair
            out += [
                f"    %c{pair}_{row} = aie.tile({col}, {row + 2})",
            ]
    for row in range(ROWS):
        consumers = ", ".join(f"@qr{pair}_{row}" for pair in range(COLS // 2))
        input_offsets = ", ".join(str(pair * raw_pair) for pair in range(COLS // 2))
        producers = ", ".join(f"@qo{pair}_{row}" for pair in range(COLS // 2))
        output_offsets = ", ".join(str(pair * q_pair) for pair in range(COLS // 2))
        cores = ", ".join(f"%c{pair}_{row}" for pair in range(COLS // 2))
        out.append(
            f"    aie.objectfifo @qsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{raw_join}xi8>>"
        )
        for pair in range(COLS // 2):
            out += [
                f"    aie.objectfifo @qr{pair}_{row}(%mt{row}, {{%c{pair}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{raw_pair}xi8>>",
                f"    aie.objectfifo @qo{pair}_{row}(%c{pair}_{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{q_pair}xi8>>",
            ]
        out += [
            f"    aie.objectfifo.link [@qsh{row}] -> [{consumers}] ([] [{input_offsets}])",
            f"    aie.objectfifo @lsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{batch}xi32>>",
            f"    aie.objectfifo @lbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{batch}xi32>>",
            f"    aie.objectfifo.link [@lsh{row}] -> [@lbc{row}] ([] [0])",
            f"    aie.objectfifo @qosh{row}(%mt{row}, {{%shim{row}}}, 1 : i32) : !aie.objectfifo<memref<{q_join}xi8>>",
            f"    aie.objectfifo.link [{producers}] -> [@qosh{row}] ([{output_offsets}] [])",
        ]
    out.append(
        f'    func.func private @hipfire_qwen3_pack_query_pair(memref<{raw_pair}xi8>, memref<{q_pair}xi8>, i32) attributes {{link_with = "qwen3_query_pack.o"}}'
    )
    for row in range(ROWS):
        for pair in range(COLS // 2):
            out += [
                f"    %core{pair}_{row} = aie.core(%c{pair}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %chunks = arith.constant {token_chunks} : index",
                f"      %batch = arith.constant {batch} : index",
                "      scf.for %outer = %z to %inf step %one {",
                f"        %lengths = aie.objectfifo.acquire @lbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{batch}xi32>>",
                f"        %lengthsv = aie.objectfifo.subview.access %lengths[0] : !aie.objectfifosubview<memref<{batch}xi32>> -> memref<{batch}xi32>",
                "        scf.for %document = %z to %batch step %one {",
                f"          %length = memref.load %lengthsv[%document] : memref<{batch}xi32>",
            ]
            for local in range(q_per_kv):
                out += [
                    "          scf.for %chunk = %z to %chunks step %one {",
                    f"            %qr{local} = aie.objectfifo.acquire @qr{pair}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{raw_pair}xi8>>",
                    f"            %qrv{local} = aie.objectfifo.subview.access %qr{local}[0] : !aie.objectfifosubview<memref<{raw_pair}xi8>> -> memref<{raw_pair}xi8>",
                    f"            %qo{local} = aie.objectfifo.acquire @qo{pair}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{q_pair}xi8>>",
                    f"            %qov{local} = aie.objectfifo.subview.access %qo{local}[0] : !aie.objectfifosubview<memref<{q_pair}xi8>> -> memref<{q_pair}xi8>",
                    f"            func.call @hipfire_qwen3_pack_query_pair(%qrv{local}, %qov{local}, %length) : (memref<{raw_pair}xi8>, memref<{q_pair}xi8>, i32) -> ()",
                    f"            aie.objectfifo.release @qr{pair}_{row}(Consume, 1)",
                    f"            aie.objectfifo.release @qo{pair}_{row}(Produce, 1)",
                    "          }",
                ]
            out += [
                "        }",
                f"        aie.objectfifo.release @lbc{row}(Consume, 1)",
                "      }",
                "      aie.end",
                "    } {stack_size = 2048 : i32}",
            ]

    out.append(
        f"    aie.runtime_sequence(%Q: memref<{input_bytes}xi8>, %L: memref<{batch}xi32>, %O: memref<{output_bytes}xi8>) {{"
    )
    length_tasks: list[str] = []
    for row in range(ROWS):
        length_name = f"tl{row}"
        length_tasks.append(length_name)
        out += [
            f"      %{length_name} = aiex.dma_configure_task_for @lsh{row} {{",
            f"        aie.dma_bd(%L : memref<{batch}xi32>, 0, {batch}, {dims([(batch, 1)])}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{length_name})",
        ]
    # Keep the four length broadcasts live while processing one document/head
    # group across all rows at a time. This caps live DMA descriptors at twelve
    # for every supported batch instead of growing by eight per document.
    for document in range(batch):
        for local in range(q_per_kv):
            for chunk_start in range(0, token_chunks, MAX_DMA_CHUNKS):
                chunk_count = min(MAX_DMA_CHUNKS, token_chunks - chunk_start)
                phase: list[str] = []
                for row in range(ROWS):
                    input_name = f"qi{row}_{document}_{local}_{chunk_start}"
                    output_name = f"qo{row}_{document}_{local}_{chunk_start}"
                    phase += [input_name, output_name]
                    input_stride = ROWS * QUERIES * q_width * 2
                    input_offset = (
                        document * input_document
                        + row * QUERIES * q_width * 2
                        + local * head_dim * 2
                        + chunk_start * input_stride
                    )
                    input_layout = dims(
                        [
                            (chunk_count, input_stride),
                            (COLS, q_per_kv * head_dim * 2),
                            (QUERIES, q_width * 2),
                            (head_dim * 2, 1),
                        ]
                    )
                    out += [
                        f"      %{input_name} = aiex.dma_configure_task_for @qsh{row} {{",
                        f"        aie.dma_bd(%Q : memref<{input_bytes}xi8>, {input_offset}, {raw_join}, {input_layout}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        f"      }} {{issue_token = true, repeat_count = {chunk_count - 1} : i32}}",
                        f"      aiex.dma_start_task(%{input_name})",
                    ]
                    output_offset = (
                        document * output_document + (row * q_groups + local * token_chunks + chunk_start) * q_join
                    )
                    output_layout = dims([(chunk_count, q_join), (q_join, 1)])
                    out += [
                        f"      %{output_name} = aiex.dma_configure_task_for @qosh{row} {{",
                        f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {output_offset}, {chunk_count * q_join}, {output_layout}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{output_name})",
                    ]
                for task in phase:
                    out += [
                        f"      aiex.dma_await_task(%{task})",
                        f"      aiex.dma_free_task(%{task})",
                    ]
    for task in length_tasks:
        out += [f"      aiex.dma_await_task(%{task})", f"      aiex.dma_free_task(%{task})"]
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
        source = Path(__file__).with_name("qwen3_query_pack_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "qwen3_query_pack.o"),
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
        "schema": "hipfire.npu_qwen3_query_pack.v1",
        "sequence_bucket": bucket,
        "dispatch_batch": batch,
        "query_heads": query_heads,
        "kv_heads": 8,
        "head_dim": 128,
        "input_layout": "token_major_b_s_qh_d_bf16",
        "output_layout": "segmented_attention_q_with_length_trailers",
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
