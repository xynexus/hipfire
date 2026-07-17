#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build a real-length-masked Qwen3 causal-attention image for AIE2P."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CORE_COLS = 8
CORE_ROWS = 4
QUERIES_PER_CORE = 4
BLOCK_KEYS = 16
BUCKETS = (128, 256, 512, 1024, 2048)
INF = 9_223_372_036_854_775_807


def _blocks(count: int, block: int) -> str:
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"


def generate_mlir(
    bucket: int,
    batch: int,
    query_heads: int = 16,
    kv_heads: int = 8,
    head_dim: int = 128,
) -> str:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}")
    if batch <= 0 or bucket * batch > 4096:
        raise ValueError("batch must be positive and bucket*batch must not exceed 4096")
    if kv_heads != CORE_COLS or query_heads // kv_heads not in (2, 4):
        raise ValueError("Qwen3 attention requires 8 KV heads and 16 or 32 query heads")
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim != 128:
        raise ValueError("the initial Qwen3 embedding image requires head_dim=128")

    q_heads_per_kv = query_heads // kv_heads
    token_chunks = bucket // (CORE_ROWS * QUERIES_PER_CORE)
    q_groups = token_chunks * q_heads_per_kv
    key_blocks = bucket // BLOCK_KEYS
    q_tile = QUERIES_PER_CORE * head_dim * 2
    q_pair_data = 2 * q_tile
    q_pair = q_pair_data + 512
    q_join = (CORE_COLS // 2) * q_pair
    q_document = CORE_ROWS * q_groups * q_join
    q_bytes = batch * q_document
    kv_tile = 2 * BLOCK_KEYS * head_dim * 2
    kv_head = key_blocks * kv_tile
    kv_document = kv_heads * kv_head
    kv_bytes = batch * kv_document
    out_join = CORE_ROWS * q_tile
    output_document = bucket * query_heads * head_dim * 2
    output_bytes = batch * output_document
    accum = QUERIES_PER_CORE * head_dim

    lines = ["module {", "  aie.device(npu2) {"]
    for col in range(CORE_COLS):
        lines += [
            f"    %shim{col} = aie.tile({col}, 0)",
            f"    %mt{col} = aie.tile({col}, 1)",
        ]
    for row in range(CORE_ROWS):
        for col in range(CORE_COLS):
            lines += [
                f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
                f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{accum}xf32>',
                f'    %stats{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "stats{col}_{row}"}} : memref<8xf32>',
            ]

    for row in range(CORE_ROWS):
        q_consumers = ", ".join(f"@qpair{pair}_{row}" for pair in range(CORE_COLS // 2))
        q_offsets = ", ".join(str(pair * q_pair) for pair in range(CORE_COLS // 2))
        lines.append(
            f"    aie.objectfifo @qsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{q_join}xi8>>"
        )
        for pair in range(CORE_COLS // 2):
            lines.append(
                f"    aie.objectfifo @qpair{pair}_{row}(%mt{row}, {{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{q_pair}xi8>>"
            )
        lines.append(f"    aie.objectfifo.link [@qsh{row}] -> [{q_consumers}] ([] [{q_offsets}])")

    for col in range(CORE_COLS):
        cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
        lines += [
            f"    aie.objectfifo @kvsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{kv_tile}xi8>>",
            f"    aie.objectfifo @kv{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{kv_tile}xi8>>",
            f"    aie.objectfifo.link [@kvsh{col}] -> [@kv{col}] ([] [0])",
        ]

    for col in range(CORE_COLS):
        producers = ", ".join(f"@o{col}_{row}" for row in range(CORE_ROWS))
        offsets = ", ".join(str(row * q_tile) for row in range(CORE_ROWS))
        for row in range(CORE_ROWS):
            lines.append(
                f"    aie.objectfifo @o{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{q_tile}xi8>>"
            )
        lines += [
            f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{out_join}xi8>>",
            f"    aie.objectfifo.link [{producers}] -> [@osh{col}] ([{offsets}] [])",
        ]

    lines += [
        f'    func.func private @hipfire_segmented_attention_init(memref<{accum}xf32>, memref<8xf32>) attributes {{link_with = "segmented_attention.o"}}',
        f'    func.func private @hipfire_segmented_attention_block(memref<{q_pair}xi8>, memref<{kv_tile}xi8>, memref<{accum}xf32>, memref<8xf32>, i32, i32, i32, i32, i32) attributes {{link_with = "segmented_attention.o"}}',
        f'    func.func private @hipfire_segmented_attention_finish(memref<{accum}xf32>, memref<8xf32>, memref<{q_tile}xi8>) attributes {{link_with = "segmented_attention.o"}}',
    ]

    for row in range(CORE_ROWS):
        for col in range(CORE_COLS):
            lines += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %groups = arith.constant {q_groups} : index",
                f"      %blocks = arith.constant {key_blocks} : index",
                f"      %batch = arith.constant {batch} : index",
                f"      %token_chunks = arith.constant {token_chunks} : i32",
                "      %query_stride = arith.constant 16 : i32",
                f"      %query_row = arith.constant {row * QUERIES_PER_CORE} : i32",
                "      %key_stride = arith.constant 16 : i32",
                f"      %pair_lane = arith.constant {col % 2} : i32",
                "      %causal = arith.constant 1 : i32",
                "      %no_window = arith.constant 0 : i32",
                "      scf.for %outer = %z to %inf step %one {",
                "        scf.for %document = %z to %batch step %one {",
            ]
            lines += [
                "          scf.for %group = %z to %groups step %one {",
                "          %group_i32 = arith.index_cast %group : index to i32",
                "          %token_chunk = arith.remui %group_i32, %token_chunks : i32",
                "          %query_chunk = arith.muli %token_chunk, %query_stride : i32",
                "          %query_base = arith.addi %query_chunk, %query_row : i32",
                f"          %q = aie.objectfifo.acquire @qpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{q_pair}xi8>>",
                f"          %qv = aie.objectfifo.subview.access %q[0] : !aie.objectfifosubview<memref<{q_pair}xi8>> -> memref<{q_pair}xi8>",
                f"          %o = aie.objectfifo.acquire @o{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{q_tile}xi8>>",
                f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{q_tile}xi8>> -> memref<{q_tile}xi8>",
                f"          func.call @hipfire_segmented_attention_init(%acc{col}_{row}, %stats{col}_{row}) : (memref<{accum}xf32>, memref<8xf32>) -> ()",
                "          scf.for %block = %z to %blocks step %one {",
                "            %block_i32 = arith.index_cast %block : index to i32",
                "            %key_base = arith.muli %block_i32, %key_stride : i32",
                f"            %kv = aie.objectfifo.acquire @kv{col}(Consume, 1) : !aie.objectfifosubview<memref<{kv_tile}xi8>>",
                f"            %kvv = aie.objectfifo.subview.access %kv[0] : !aie.objectfifosubview<memref<{kv_tile}xi8>> -> memref<{kv_tile}xi8>",
                f"            func.call @hipfire_segmented_attention_block(%qv, %kvv, %acc{col}_{row}, %stats{col}_{row}, %pair_lane, %query_base, %key_base, %causal, %no_window) : (memref<{q_pair}xi8>, memref<{kv_tile}xi8>, memref<{accum}xf32>, memref<8xf32>, i32, i32, i32, i32, i32) -> ()",
                f"            aie.objectfifo.release @kv{col}(Consume, 1)",
                "          }",
                f"          func.call @hipfire_segmented_attention_finish(%acc{col}_{row}, %stats{col}_{row}, %ov) : (memref<{accum}xf32>, memref<8xf32>, memref<{q_tile}xi8>) -> ()",
                f"          aie.objectfifo.release @qpair{col // 2}_{row}(Consume, 1)",
                f"          aie.objectfifo.release @o{col}_{row}(Produce, 1)",
                "          }",
                "        }",
            ]
            lines += ["      }", "      aie.end", "    } {stack_size = 4096 : i32}"]

    lines.append(
        f"    aie.runtime_sequence(%Q: memref<{q_bytes}xi8>, %KV: memref<{kv_bytes}xi8>, %O: memref<{output_bytes}xi8>) {{"
    )
    for document in range(batch):
        phase: list[str] = []
        for row in range(CORE_ROWS):
            name = f"tq{row}_{document}"
            phase.append(name)
            lines += [
                f"      %{name} = aiex.dma_configure_task_for @qsh{row} {{",
                f"        aie.dma_bd(%Q : memref<{q_bytes}xi8>, {document * q_document + row * q_groups * q_join}, {q_groups * q_join}, {_blocks(q_groups, q_join)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for col in range(CORE_COLS):
            name = f"tkv{col}_{document}"
            phase.append(name)
            lines += [
                f"      %{name} = aiex.dma_configure_task_for @kvsh{col} {{",
                f"        aie.dma_bd(%KV : memref<{kv_bytes}xi8>, {document * kv_document + col * kv_head}, {kv_head}, {_blocks(key_blocks, kv_tile)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {q_groups - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for col in range(CORE_COLS):
            name = f"to{col}_{document}"
            phase.append(name)
            lines += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {document * output_document + col * q_groups * out_join}, {q_groups * out_join}, {_blocks(q_groups, out_join)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for task in phase:
            lines += [
                f"      aiex.dma_await_task(%{task})",
                f"      aiex.dma_free_task(%{task})",
            ]
    lines += ["    }", "  }", "}"]
    return "\n".join(lines) + "\n"


def _toolchain() -> tuple[Path, Path]:
    venv = Path(os.environ.get("HIPFIRE_NPU_VENV", Path.home() / ".venv"))
    python = venv / "bin/python"
    if not python.is_file():
        raise RuntimeError(f"NPU Python environment not found at {python}")
    location = subprocess.run(
        [str(python), "-c", "import mlir_aie; print(list(mlir_aie.__path__)[0])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mlir_aie = Path(location)
    peano = mlir_aie.parent / "llvm-aie"
    if not (peano / "bin/clang++").is_file():
        raise RuntimeError(f"Peano compiler not found under {peano}")
    return mlir_aie, peano


def build(
    output: Path,
    bucket: int,
    batch: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    emit_mlir_only: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir_path = output / "aie.mlir"
    mlir_path.write_text(
        generate_mlir(bucket, batch, query_heads, kv_heads, head_dim),
        encoding="utf-8",
    )
    if not emit_mlir_only:
        mlir_aie, peano = _toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [
                "/opt/xilinx/xrt/bin",
                str(peano / "bin"),
                str(mlir_aie / "bin"),
                env.get("PATH", ""),
            ]
        )
        source = Path(__file__).with_name("segmented_attention_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "segmented_attention.o"),
                f"-I{mlir_aie / 'include'}",
                "-std=c++20",
                "-Wno-parentheses",
                "-Wno-attributes",
                "-Wno-macro-redefined",
                "-Wno-empty-body",
                "-Wno-deprecated-declarations",
                "-O2",
                "-DNDEBUG",
                f"-DHIPFIRE_HEAD_DIM={head_dim}",
                "--target=aie2p-none-unknown-elf",
            ],
            check=True,
            env=env,
        )
        aiecc = shutil.which("aiecc", path=env["PATH"])
        if aiecc is None:
            raise RuntimeError("aiecc is not available in the configured NPU toolchain")
        subprocess.run(
            [
                aiecc,
                str(mlir_path),
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
        "schema": "hipfire.npu_segmented_attention_image.v1",
        "npu_architecture": "aie2p",
        "architecture": "qwen3",
        "attention": "causal",
        "sequence_bucket": bucket,
        "dispatch_batch": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_padded_rows": 4096,
        "kernel_arguments": ["packed_q_bf16_with_length_trailers", "packed_kv_bf16", "packed_output_bf16"],
        "length_staging": "one 512-byte trailer per Q core-pair object; first u32 is the real length",
        "xclbin": "final.xclbin",
        "instructions": "insts.bin",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True, choices=BUCKETS)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--query-heads", type=int, choices=(16, 32), default=16)
    parser.add_argument("--kv-heads", type=int, choices=(8,), default=8)
    parser.add_argument("--head-dim", type=int, choices=(128,), default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--emit-mlir-only",
        action="store_true",
        help="write aie.mlir and manifest.json without invoking Peano/aiecc",
    )
    args = parser.parse_args()
    try:
        build(
            args.output,
            args.bucket,
            args.batch,
            args.query_heads,
            args.kv_heads,
            args.head_dim,
            args.emit_mlir_only,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
