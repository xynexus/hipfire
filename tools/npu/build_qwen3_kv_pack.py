#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build token-major Qwen3 K/V packing for segmented AIE2P attention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


COLS = 8
BLOCK_KEYS = 16
BUCKETS = (128, 256, 512, 1024, 2048)
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"<size = {size}, stride = {stride}>" for size, stride in entries) + "]"


def blocks(count: int, block: int) -> str:
    return dims([(count, block), (block // 512, 512), (512, 1)])


def generate_mlir(bucket: int, batch: int, head_dim: int) -> str:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}")
    if batch <= 0 or bucket * batch > 4096:
        raise ValueError("batch must be positive and bucket*batch <= 4096")
    if head_dim != 128:
        raise ValueError("head_dim must be 128")

    key_blocks = bucket // BLOCK_KEYS
    width = COLS * head_dim
    input_document = bucket * width * 2
    input_bytes = batch * input_document
    input_tile = BLOCK_KEYS * head_dim * 2
    output_tile = 2 * input_tile
    output_head = key_blocks * output_tile
    output_document = COLS * output_head
    output_bytes = batch * output_document

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(COLS):
        out += [
            f"    %shim{col} = aie.tile({col}, 0)",
            f"    %c{col} = aie.tile({col}, 2)",
            f"    aie.objectfifo @ki{col}(%shim{col}, {{%c{col}}}, 1 : i32) : !aie.objectfifo<memref<{input_tile}xi8>>",
            f"    aie.objectfifo @vi{col}(%shim{col}, {{%c{col}}}, 1 : i32) : !aie.objectfifo<memref<{input_tile}xi8>>",
            f"    aie.objectfifo @o{col}(%c{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{output_tile}xi8>>",
        ]
    out.append(
        f'    func.func private @hipfire_qwen3_pack_kv_block(memref<{input_tile}xi8>, memref<{input_tile}xi8>, memref<{output_tile}xi8>) attributes {{link_with = "qwen3_kv_pack.o"}}'
    )
    for col in range(COLS):
        out += [
            f"    %core{col} = aie.core(%c{col}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %blocks = arith.constant {key_blocks} : index",
            f"      %batch = arith.constant {batch} : index",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %document = %z to %batch step %one {",
        ]
        out += [
            "          scf.for %block = %z to %blocks step %one {",
            f"            %k = aie.objectfifo.acquire @ki{col}(Consume, 1) : !aie.objectfifosubview<memref<{input_tile}xi8>>",
            f"            %kv = aie.objectfifo.subview.access %k[0] : !aie.objectfifosubview<memref<{input_tile}xi8>> -> memref<{input_tile}xi8>",
            f"            %v = aie.objectfifo.acquire @vi{col}(Consume, 1) : !aie.objectfifosubview<memref<{input_tile}xi8>>",
            f"            %vv = aie.objectfifo.subview.access %v[0] : !aie.objectfifosubview<memref<{input_tile}xi8>> -> memref<{input_tile}xi8>",
            f"            %o = aie.objectfifo.acquire @o{col}(Produce, 1) : !aie.objectfifosubview<memref<{output_tile}xi8>>",
            f"            %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{output_tile}xi8>> -> memref<{output_tile}xi8>",
            f"            func.call @hipfire_qwen3_pack_kv_block(%kv, %vv, %ov) : (memref<{input_tile}xi8>, memref<{input_tile}xi8>, memref<{output_tile}xi8>) -> ()",
            f"            aie.objectfifo.release @ki{col}(Consume, 1)",
            f"            aie.objectfifo.release @vi{col}(Consume, 1)",
            f"            aie.objectfifo.release @o{col}(Produce, 1)",
            "          }",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]

    out.append(
        f"    aie.runtime_sequence(%K: memref<{input_bytes}xi8>, %V: memref<{input_bytes}xi8>, %O: memref<{output_bytes}xi8>) {{"
    )
    for document in range(batch):
        phase: list[str] = []
        for col in range(COLS):
            input_offset = document * input_document + col * head_dim * 2
            input_layout = dims(
                [
                    (key_blocks, BLOCK_KEYS * width * 2),
                    (BLOCK_KEYS, width * 2),
                    (head_dim * 2, 1),
                ]
            )
            for role, fifo in (("k", "ki"), ("v", "vi")):
                name = f"t{role}{col}_{document}"
                phase.append(name)
                argument = role.upper()
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @{fifo}{col} {{",
                    f"        aie.dma_bd(%{argument} : memref<{input_bytes}xi8>, {input_offset}, {key_blocks * input_tile}, {input_layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            output_name = f"to{col}_{document}"
            phase.append(output_name)
            output_offset = document * output_document + col * output_head
            out += [
                f"      %{output_name} = aiex.dma_configure_task_for @o{col} {{",
                f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {output_offset}, {output_head}, {blocks(key_blocks, output_tile)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
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


def build(output: Path, bucket: int, batch: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(bucket, batch, 128), encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")]
        )
        source = Path(__file__).with_name("qwen3_kv_pack_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "qwen3_kv_pack.o"),
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
        "schema": "hipfire.npu_qwen3_kv_pack.v1",
        "sequence_bucket": bucket,
        "dispatch_batch": batch,
        "kv_heads": 8,
        "head_dim": 128,
        "input_layout": "token_major_b_s_hkv_d_bf16",
        "output_layout": "segmented_attention_kv",
        "xclbin": "final.xclbin",
        "instructions": "insts.bin",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True, choices=BUCKETS)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(args.output, args.bucket, args.batch, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
