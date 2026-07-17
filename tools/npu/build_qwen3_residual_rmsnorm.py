#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build Qwen3 residual-add plus weighted RMSNorm for AIE2P."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROWS = 4
SLOTS = 4
LANES = 16
TOKENS_PER_WAVE = ROWS * SLOTS * LANES
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"<size = {size}, stride = {stride}>" for size, stride in entries) + "]"


def generate_mlir(m: int, k: int) -> str:
    if m <= 0 or m > 4096 or m % TOKENS_PER_WAVE:
        raise ValueError("rows must be a multiple of 256 in 256..=4096")
    if k <= 0 or k > 4096 or k % 256:
        raise ValueError("hidden size must be a multiple of 256 in 256..=4096")
    waves = m // TOKENS_PER_WAVE
    record = 2 * k * 2
    joined = SLOTS * record
    weight = (k + 16) * 4
    output_join = ROWS * record
    input_bytes = m * record

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(8):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        for slot in range(SLOTS):
            out.append(f"    %c{slot}_{row} = aie.tile({2 * slot}, {row + 2})")
    for row in range(ROWS):
        consumers = ", ".join(f"@x{slot}_{row}" for slot in range(SLOTS))
        offsets = ", ".join(str(slot * record) for slot in range(SLOTS))
        cores = ", ".join(f"%c{slot}_{row}" for slot in range(SLOTS))
        out.append(
            f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{joined}xi8>>"
        )
        for slot in range(SLOTS):
            out.append(
                f"    aie.objectfifo @x{slot}_{row}(%mt{row}, {{%c{slot}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{record}xi8>>"
            )
        out += [
            f"    aie.objectfifo.link [@xsh{row}] -> [{consumers}] ([] [{offsets}])",
            f"    aie.objectfifo @wsh{row}(%shim{row + ROWS}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{weight}xi8>>",
            f"    aie.objectfifo @wbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{weight}xi8>>",
            f"    aie.objectfifo.link [@wsh{row}] -> [@wbc{row}] ([] [0])",
        ]
    for slot in range(SLOTS):
        producers = ", ".join(f"@o{slot}_{row}" for row in range(ROWS))
        offsets = ", ".join(str(row * record) for row in range(ROWS))
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @o{slot}_{row}(%c{slot}_{row}, {{%mt{2 * slot}}}, 1 : i32) : !aie.objectfifo<memref<{record}xi8>>"
            )
        out += [
            f"    aie.objectfifo @osh{slot}(%mt{2 * slot}, {{%shim{2 * slot}}}, 1 : i32) : !aie.objectfifo<memref<{output_join}xi8>>",
            f"    aie.objectfifo.link [{producers}] -> [@osh{slot}] ([{offsets}] [])",
        ]
    out.append(
        f'    func.func private @hipfire_qwen3_residual_rmsnorm(memref<{record}xi8>, memref<{weight}xi8>, memref<{record}xi8>, i32) attributes {{link_with = "qwen3_residual_rmsnorm.o"}}'
    )
    iterations = waves * LANES
    for slot in range(SLOTS):
        for row in range(ROWS):
            out += [
                f"    %core{slot}_{row} = aie.core(%c{slot}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %inf = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %iterations = arith.constant {iterations} : index",
                "      %pair_lane = arith.constant 0 : i32",
                "      scf.for %outer = %z to %inf step %one {",
                f"        %w = aie.objectfifo.acquire @wbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{weight}xi8>>",
                f"        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{weight}xi8>> -> memref<{weight}xi8>",
                "        scf.for %iteration = %z to %iterations step %one {",
                f"          %x = aie.objectfifo.acquire @x{slot}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{record}xi8>>",
                f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{record}xi8>> -> memref<{record}xi8>",
                f"          %o = aie.objectfifo.acquire @o{slot}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{record}xi8>>",
                f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{record}xi8>> -> memref<{record}xi8>",
                f"          func.call @hipfire_qwen3_residual_rmsnorm(%xv, %wv, %ov, %pair_lane) : (memref<{record}xi8>, memref<{weight}xi8>, memref<{record}xi8>, i32) -> ()",
                f"          aie.objectfifo.release @x{slot}_{row}(Consume, 1)",
                f"          aie.objectfifo.release @o{slot}_{row}(Produce, 1)",
                "        }",
                f"        aie.objectfifo.release @wbc{row}(Consume, 1)",
                "      }",
                "      aie.end",
                "    } {stack_size = 2048 : i32}",
            ]
    out.append(
        f"    aie.runtime_sequence(%X: memref<{input_bytes}xi8>, %W: memref<{weight}xi8>, %O: memref<{input_bytes}xi8>) {{"
    )
    weight_tasks: list[str] = []
    input_layout = dims(
        [
            (LANES, record),
            (SLOTS, LANES * record),
            (record // 512, 512),
            (512, 1),
        ]
    )
    # Weight broadcasts remain live for the whole invocation. Data transfers are
    # configured below one wave at a time so the compiler can reuse DMA buffer
    # descriptors instead of assigning one to every wave in the 4096-row case.
    for row in range(ROWS):
        wname = f"tw{row}"
        weight_tasks.append(wname)
        out += [
            f"      %{wname} = aiex.dma_configure_task_for @wsh{row} {{",
            f"        aie.dma_bd(%W : memref<{weight}xi8>, 0, {weight}, {dims([(weight // 64, 64), (64, 1)])}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{wname})",
        ]
    output_layout = dims(
        [
            (LANES, record),
            (ROWS, SLOTS * LANES * record),
            (record // 512, 512),
            (512, 1),
        ]
    )
    for wave in range(waves):
        wave_tasks: list[str] = []
        for row in range(ROWS):
            name = f"tx{row}_{wave}"
            wave_tasks.append(name)
            offset = (wave * TOKENS_PER_WAVE + row * SLOTS * LANES) * record
            out += [
                f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                f"        aie.dma_bd(%X : memref<{input_bytes}xi8>, {offset}, {joined}, {input_layout}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {LANES - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for slot in range(SLOTS):
            name = f"to{slot}_{wave}"
            wave_tasks.append(name)
            offset = (wave * TOKENS_PER_WAVE + slot * LANES) * record
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{slot} {{",
                f"        aie.dma_bd(%O : memref<{input_bytes}xi8>, {offset}, {output_join}, {output_layout}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {LANES - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for task in wave_tasks:
            out += [
                f"      aiex.dma_await_task(%{task})",
                f"      aiex.dma_free_task(%{task})",
            ]
    for task in weight_tasks:
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


def build(output: Path, m: int, k: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(m, k), encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")]
        )
        source = Path(__file__).with_name("qwen3_residual_rmsnorm_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "qwen3_residual_rmsnorm.o"),
                f"-I{mlir_aie / 'include'}",
                "-std=c++20",
                "-O2",
                "-DNDEBUG",
                f"-DHIPFIRE_HIDDEN_SIZE={k}",
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
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "hipfire.npu_qwen3_residual_rmsnorm.v1",
                "npu_architecture": "aie2p",
                "rows": m,
                "hidden_size": k,
                "input_layout": "token_major_residual_then_delta_bf16",
                "parameter_layout": "weight_f32_then_epsilon_f32",
                "output_layout": "token_major_completed_then_normalized_bf16",
                "xclbin": "final.xclbin",
                "instructions": "insts.bin",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(args.output, args.rows, args.hidden_size, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
