#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build a resident OQ8+ BF16 projection image for Qwen3 on AIE2P."""

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
CORE_ROWS = 8
GROUP = 256
OUTPUT_TILE = 16
A_TILE = CORE_ROWS * GROUP * 2
A_PAIR = 2 * A_TILE
A_JOIN = COLS * A_TILE
W_TILE = GROUP * OUTPUT_TILE * 2
O_TILE = CORE_ROWS * OUTPUT_TILE * 2
O_JOIN = ROWS * O_TILE
APACK = CORE_ROWS * GROUP * 2
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(
        f"<size = {size}, stride = {stride}>" for size, stride in entries
    ) + "]"


def blocks(count: int, block: int) -> str:
    unit = 512 if block % 512 == 0 else 64
    if block % unit:
        raise ValueError(f"block {block} is not divisible by DMA unit {unit}")
    return dims([(count, block), (block // unit, unit), (unit, 1)])


def generate_mlir(m: int, k: int, n: int) -> str:
    if m <= 0 or m > 4096 or m % (COLS * ROWS * CORE_ROWS):
        raise ValueError("m must be a multiple of 256 in 256..=4096")
    if k <= 0 or k % GROUP:
        raise ValueError("k must be a positive multiple of 256")
    if n <= 0 or n % OUTPUT_TILE:
        raise ValueError("n must be a positive multiple of 16")
    waves = m // (COLS * ROWS * CORE_ROWS)
    groups = k // GROUP
    n_tiles = n // OUTPUT_TILE
    acc_elements = waves * CORE_ROWS * OUTPUT_TILE
    input_bytes = m * k * 2
    weight_bytes = n_tiles * groups * W_TILE
    output_bytes = m * n * 2

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(COLS):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
        for row in range(ROWS):
            out += [
                f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
                f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{acc_elements}xf32>',
                f'    %apack{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "apack{col}_{row}"}} : memref<{APACK}xi8>',
            ]
    for row in range(ROWS):
        consumers = ", ".join(f"@ap{pair}_{row}" for pair in range(COLS // 2))
        offsets = ", ".join(str(pair * A_PAIR) for pair in range(COLS // 2))
        out.append(
            f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_JOIN}xi8>>"
        )
        for pair in range(COLS // 2):
            out.append(
                f"    aie.objectfifo @ap{pair}_{row}(%mt{row}, {{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_PAIR}xi8>>"
            )
        out.append(f"    aie.objectfifo.link [@ash{row}] -> [{consumers}] ([] [{offsets}])")
    for row in range(ROWS):
        cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
        out += [
            f"    aie.objectfifo @wsh{row}(%shim{row + ROWS}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{W_TILE}xi8>>",
            f"    aie.objectfifo @wbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{W_TILE}xi8>>",
            f"    aie.objectfifo.link [@wsh{row}] -> [@wbc{row}] ([] [0])",
        ]
    for col in range(COLS):
        producers = ", ".join(f"@o{col}_{row}" for row in range(ROWS))
        offsets = ", ".join(str(row * O_TILE) for row in range(ROWS))
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @o{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{O_TILE}xi8>>"
            )
        out += [
            f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{O_JOIN}xi8>>",
            f"    aie.objectfifo.link [{producers}] -> [@osh{col}] ([{offsets}] [])",
        ]
    out += [
        f'    func.func private @hipfire_qwen3_oq8_projection_group(memref<{A_PAIR}xi8>, memref<{W_TILE}xi8>, memref<{acc_elements}xf32>, memref<{APACK}xi8>, i32, i32, i32) attributes {{link_with = "qwen3_oq8_projection.o"}}',
        f'    func.func private @hipfire_qwen3_oq8_projection_finish(memref<{acc_elements}xf32>, memref<{O_TILE}xi8>, i32) attributes {{link_with = "qwen3_oq8_projection.o"}}',
    ]
    for col in range(COLS):
        for row in range(ROWS):
            out += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %ntiles = arith.constant {n_tiles} : index",
                f"      %groups = arith.constant {groups} : index",
                f"      %waves = arith.constant {waves} : index",
                f"      %pair_lane = arith.constant {col % 2} : i32",
                "      scf.for %outer = %z to %inf step %one {",
                "        scf.for %ntile = %z to %ntiles step %one {",
                "          scf.for %group = %z to %groups step %one {",
                f"            %w = aie.objectfifo.acquire @wbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{W_TILE}xi8>>",
                f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_TILE}xi8>> -> memref<{W_TILE}xi8>",
                "            %first_group = arith.cmpi eq, %group, %z : index",
                "            %initialize = arith.extui %first_group : i1 to i32",
                "            scf.for %wave = %z to %waves step %one {",
                f"              %a = aie.objectfifo.acquire @ap{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_PAIR}xi8>>",
                f"              %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{A_PAIR}xi8>> -> memref<{A_PAIR}xi8>",
                "              %wave_i32 = arith.index_cast %wave : index to i32",
                f"              func.call @hipfire_qwen3_oq8_projection_group(%av, %wv, %acc{col}_{row}, %apack{col}_{row}, %initialize, %wave_i32, %pair_lane) : (memref<{A_PAIR}xi8>, memref<{W_TILE}xi8>, memref<{acc_elements}xf32>, memref<{APACK}xi8>, i32, i32, i32) -> ()",
                f"              aie.objectfifo.release @ap{col // 2}_{row}(Consume, 1)",
                "            }",
                f"            aie.objectfifo.release @wbc{row}(Consume, 1)",
                "          }",
                "          scf.for %wave = %z to %waves step %one {",
                f"            %o = aie.objectfifo.acquire @o{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{O_TILE}xi8>>",
                f"            %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{O_TILE}xi8>> -> memref<{O_TILE}xi8>",
                "            %wave_i32 = arith.index_cast %wave : index to i32",
                f"            func.call @hipfire_qwen3_oq8_projection_finish(%acc{col}_{row}, %ov, %wave_i32) : (memref<{acc_elements}xf32>, memref<{O_TILE}xi8>, i32) -> ()",
                f"            aie.objectfifo.release @o{col}_{row}(Produce, 1)",
                "          }",
                "        }",
                "      }",
            ]
            out += ["      aie.end", "    } {stack_size = 8192 : i32}"]

    out.append(
        f"    aie.runtime_sequence(%X: memref<{input_bytes}xi8>, %W: memref<{weight_bytes}xi8>, %O: memref<{output_bytes}xi8>) {{"
    )
    for ntile in range(n_tiles):
        tasks: list[str] = []
        input_layout = dims(
            [
                (groups, GROUP * 2),
                (waves, COLS * ROWS * CORE_ROWS * k * 2),
                (COLS * CORE_ROWS, k * 2),
                (GROUP * 2, 1),
            ]
        )
        for row in range(ROWS):
            name = f"ta{ntile}_{row}"
            tasks.append(name)
            offset = row * COLS * CORE_ROWS * k * 2
            out += [
                f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                f"        aie.dma_bd(%X : memref<{input_bytes}xi8>, {offset}, {waves * A_JOIN}, {input_layout}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {groups - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for row in range(ROWS):
            weight_name = f"tw{ntile}_{row}"
            tasks.append(weight_name)
            out += [
                f"      %{weight_name} = aiex.dma_configure_task_for @wsh{row} {{",
                f"        aie.dma_bd(%W : memref<{weight_bytes}xi8>, {ntile * groups * W_TILE}, {groups * W_TILE}, {blocks(groups, W_TILE)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{weight_name})",
            ]
        output_layout = dims(
            [
                (waves, COLS * ROWS * CORE_ROWS * n * 2),
                (ROWS, COLS * CORE_ROWS * n * 2),
                (CORE_ROWS, n * 2),
                (OUTPUT_TILE * 2, 1),
            ]
        )
        for col in range(COLS):
            name = f"to{ntile}_{col}"
            tasks.append(name)
            offset = (col * CORE_ROWS) * n * 2 + ntile * OUTPUT_TILE * 2
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {offset}, {O_JOIN}, {output_layout}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {waves - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for task in tasks:
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


def build(output: Path, m: int, k: int, n: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(m, k, n), encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")]
        )
        source = Path(__file__).with_name("qwen3_oq8_projection_bf16.cc")
        compile_command = [
                str(peano / "bin/clang++"), str(source), "-c", "-o",
                str(output / "qwen3_oq8_projection.o"), f"-I{mlir_aie / 'include'}",
                "-std=c++20", "-O2", "-DNDEBUG",
                "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
                "-Wno-empty-body", "-Wno-deprecated-declarations",
                "--target=aie2p-none-unknown-elf",
            ]
        subprocess.run(
            compile_command,
            check=True,
            env=env,
        )
        aiecc = shutil.which("aiecc", path=env["PATH"])
        if aiecc is None:
            raise RuntimeError("aiecc not found")
        subprocess.run(
            [
                aiecc, str(mlir), "--no-compile-host", "--no-xchesscc", "--no-xbridge",
                f"--peano={peano}", "--aie-generate-npu-insts",
                f"--npu-insts-name={output / 'insts.bin'}", "--aie-generate-xclbin",
                f"--xclbin-name={output / 'final.xclbin'}", f"--tmpdir={output}",
            ],
            check=True,
            env=env,
        )
    manifest = {
        "schema": "hipfire.npu_qwen3_oq8_projection.v1",
        "npu_architecture": "aie2p",
        "rows": m,
        "input_columns": k,
        "output_columns": n,
        "group_size": GROUP,
        "input_layout": "token_major_bf16",
        "weight_layout": "oq8_dequant_bf16_mmul_prepacked",
        "parameter_layout": "awq_folded_at_model_load",
        "output_layout": "token_major_bf16",
        "xclbin": "final.xclbin",
        "instructions": "insts.bin",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--input-columns", type=int, required=True)
    parser.add_argument("--output-columns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(
            args.output,
            args.rows,
            args.input_columns,
            args.output_columns,
            args.emit_mlir_only,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
