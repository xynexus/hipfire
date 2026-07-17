#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build segmented Qwen3 Q/K RMSNorm plus full RoPE for AIE2P."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


COLS = 8
BUCKETS = (128, 256, 512, 1024, 2048)
CHUNK = 64
PARAM_PAD = 64
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(
        f"<size = {size}, stride = {stride}>" for size, stride in entries
    ) + "]"


def generate_mlir(bucket: int, batch: int, q_heads: int, kv_heads: int, head_dim: int) -> str:
    if bucket not in BUCKETS or batch <= 0 or bucket * batch > 4096:
        raise ValueError("invalid bucket/batch geometry")
    if q_heads not in (16, 32) or kv_heads != 8 or head_dim != 128:
        raise ValueError("Qwen3 headnorm/RoPE requires QH=16/32 KVH=8 D=128")
    q_rows = q_heads // COLS
    active_rows = q_rows + 1
    if active_rows > 4:
        raise ValueError("Qwen3 headnorm/RoPE exceeds four core rows")
    rows = bucket * batch
    head_bytes = head_dim * 2
    pair_bytes = 2 * head_bytes
    joined = (COLS // 2) * pair_bytes
    parameter_bytes = ((2 * head_dim * 2 + 4 + PARAM_PAD - 1) // PARAM_PAD) * PARAM_PAD
    parameter_total = 2 * rows * parameter_bytes
    q_bytes = rows * q_heads * head_bytes
    kv_bytes = rows * kv_heads * head_bytes

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(COLS):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(active_rows):
        for pair in range(COLS // 2):
            out.append(f"    %c{pair}_{row} = aie.tile({2 * pair}, {row + 2})")
    for row in range(active_rows):
        input_consumers = ", ".join(f"@x{pair}_{row}" for pair in range(COLS // 2))
        input_offsets = ", ".join(str(pair * pair_bytes) for pair in range(COLS // 2))
        output_producers = ", ".join(f"@o{pair}_{row}" for pair in range(COLS // 2))
        output_offsets = ", ".join(str(pair * pair_bytes) for pair in range(COLS // 2))
        cores = ", ".join(f"%c{pair}_{row}" for pair in range(COLS // 2))
        out.append(
            f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{joined}xi8>>"
        )
        for pair in range(COLS // 2):
            out += [
                f"    aie.objectfifo @x{pair}_{row}(%mt{row}, {{%c{pair}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{pair_bytes}xi8>>",
                f"    aie.objectfifo @o{pair}_{row}(%c{pair}_{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{pair_bytes}xi8>>",
            ]
        out += [
            f"    aie.objectfifo.link [@xsh{row}] -> [{input_consumers}] ([] [{input_offsets}])",
            f"    aie.objectfifo @psh{row}(%shim{row + 4}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{parameter_bytes}xi8>>",
            f"    aie.objectfifo @pbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{parameter_bytes}xi8>>",
            f"    aie.objectfifo.link [@psh{row}] -> [@pbc{row}] ([] [0])",
            f"    aie.objectfifo @osh{row}(%mt{row}, {{%shim{row}}}, 1 : i32) : !aie.objectfifo<memref<{joined}xi8>>",
            f"    aie.objectfifo.link [{output_producers}] -> [@osh{row}] ([{output_offsets}] [])",
        ]
    out.append(
        f'    func.func private @hipfire_qwen3_headnorm_rope(memref<{pair_bytes}xi8>, memref<{parameter_bytes}xi8>, memref<{pair_bytes}xi8>, i32) attributes {{link_with = "qwen3_headnorm_rope.o"}}'
    )
    for row in range(active_rows):
        for pair in range(COLS // 2):
            out += [
                f"    %core{pair}_{row} = aie.core(%c{pair}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %tokens = arith.constant {rows} : index",
                "      %lane0 = arith.constant 0 : i32",
                "      %lane1 = arith.constant 1 : i32",
                "      scf.for %outer = %z to %inf step %one {",
                "        scf.for %token = %z to %tokens step %one {",
                f"          %x = aie.objectfifo.acquire @x{pair}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{pair_bytes}xi8>>",
                f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{pair_bytes}xi8>> -> memref<{pair_bytes}xi8>",
                f"          %p = aie.objectfifo.acquire @pbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{parameter_bytes}xi8>>",
                f"          %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{parameter_bytes}xi8>> -> memref<{parameter_bytes}xi8>",
                f"          %o = aie.objectfifo.acquire @o{pair}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{pair_bytes}xi8>>",
                f"          %ov_pair = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{pair_bytes}xi8>> -> memref<{pair_bytes}xi8>",
                f"          func.call @hipfire_qwen3_headnorm_rope(%xv, %pv, %ov_pair, %lane0) : (memref<{pair_bytes}xi8>, memref<{parameter_bytes}xi8>, memref<{pair_bytes}xi8>, i32) -> ()",
                f"          func.call @hipfire_qwen3_headnorm_rope(%xv, %pv, %ov_pair, %lane1) : (memref<{pair_bytes}xi8>, memref<{parameter_bytes}xi8>, memref<{pair_bytes}xi8>, i32) -> ()",
                f"          aie.objectfifo.release @x{pair}_{row}(Consume, 1)",
                f"          aie.objectfifo.release @pbc{row}(Consume, 1)",
                f"          aie.objectfifo.release @o{pair}_{row}(Produce, 1)",
                "        }",
                "      }",
                "      aie.end",
                "    } {stack_size = 2048 : i32}",
            ]
    out.append(
        f"    aie.runtime_sequence(%Q: memref<{q_bytes}xi8>, %K: memref<{kv_bytes}xi8>, %P: memref<{parameter_total}xi8>, %OQ: memref<{q_bytes}xi8>, %OK: memref<{kv_bytes}xi8>) {{"
    )
    for document in range(batch):
        for start in range(0, bucket, CHUNK):
            count = min(CHUNK, bucket - start)
            chunk_tasks: list[str] = []
            for row in range(active_rows):
                is_k = row == q_rows
                heads = kv_heads if is_k else q_heads
                buffer_name = "%K" if is_k else "%Q"
                output_name = "%OK" if is_k else "%OQ"
                buffer_bytes = kv_bytes if is_k else q_bytes
                head_base = 0 if is_k else row * COLS
                token_base = document * bucket + start
                input_offset = (token_base * heads + head_base) * head_bytes
                layout = dims(
                    [
                        (count, heads * head_bytes),
                        (COLS, head_bytes),
                        (head_bytes // 32, 32),
                        (32, 1),
                    ]
                )
                name = f"tx{row}_{document}_{start}"
                chunk_tasks.append(name)
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                    f"        aie.dma_bd({buffer_name} : memref<{buffer_bytes}xi8>, {input_offset}, {joined}, {layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }} {{issue_token = true, repeat_count = {count - 1} : i32}}",
                    f"      aiex.dma_start_task(%{name})",
                ]
                kind = 1 if is_k else 0
                parameter_offset = (kind * rows + token_base) * parameter_bytes
                pname = f"tp{row}_{document}_{start}"
                chunk_tasks.append(pname)
                param_layout = dims(
                    [
                        (count, parameter_bytes),
                        (1, parameter_bytes),
                        (parameter_bytes // 32, 32),
                        (32, 1),
                    ]
                )
                out += [
                    f"      %{pname} = aiex.dma_configure_task_for @psh{row} {{",
                    f"        aie.dma_bd(%P : memref<{parameter_total}xi8>, {parameter_offset}, {parameter_bytes}, {param_layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }} {{issue_token = true, repeat_count = {count - 1} : i32}}",
                    f"      aiex.dma_start_task(%{pname})",
                ]
                oname = f"to{row}_{document}_{start}"
                chunk_tasks.append(oname)
                out += [
                    f"      %{oname} = aiex.dma_configure_task_for @osh{row} {{",
                    f"        aie.dma_bd({output_name} : memref<{buffer_bytes}xi8>, {input_offset}, {joined}, {layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }} {{issue_token = true, repeat_count = {count - 1} : i32}}",
                    f"      aiex.dma_start_task(%{oname})",
                ]
            for task in chunk_tasks:
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
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    mlir_aie = Path(location)
    return mlir_aie, mlir_aie.parent / "llvm-aie"


def build(output: Path, bucket: int, batch: int, q_heads: int, kv_heads: int, head_dim: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(bucket, batch, q_heads, kv_heads, head_dim), encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")])
        source = Path(__file__).with_name("qwen3_headnorm_rope_bf16.cc")
        subprocess.run([
            str(peano / "bin/clang++"), str(source), "-c", "-o", str(output / "qwen3_headnorm_rope.o"),
            f"-I{mlir_aie / 'include'}", "-std=c++20", "-O2", "-DNDEBUG", f"-DHIPFIRE_HEAD_DIM={head_dim}",
            "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-Wno-deprecated-declarations",
            "--target=aie2p-none-unknown-elf",
        ], check=True, env=env)
        aiecc = shutil.which("aiecc", path=env["PATH"])
        if aiecc is None:
            raise RuntimeError("aiecc not found")
        subprocess.run([
            aiecc, str(mlir), "--no-compile-host", "--no-xchesscc", "--no-xbridge", f"--peano={peano}",
            "--aie-generate-npu-insts", f"--npu-insts-name={output / 'insts.bin'}", "--aie-generate-xclbin",
            f"--xclbin-name={output / 'final.xclbin'}", f"--tmpdir={output}",
        ], check=True, env=env)
    (output / "manifest.json").write_text(json.dumps({
        "schema": "hipfire.npu_qwen3_headnorm_rope.v1", "npu_architecture": "aie2p",
        "sequence_bucket": bucket, "dispatch_batch": batch, "query_heads": q_heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "rope_layout": "full_halfsplit",
        "xclbin": "final.xclbin", "instructions": "insts.bin",
    }, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(args.output, args.bucket, args.batch, args.query_heads, args.kv_heads, args.head_dim, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
