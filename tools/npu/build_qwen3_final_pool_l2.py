#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Build Qwen3 final RMSNorm, last-token pooling, and L2 for AIE2P."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


BUCKETS = (128, 256, 512, 1024, 2048)
PAIR_LANES = 4
CHUNK = 64
INF = 9_223_372_036_854_775_807


def dims(entries: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"<size = {s}, stride = {d}>" for s, d in entries) + "]"


def generate_mlir(bucket: int, batch: int, hidden: int) -> str:
    if bucket not in BUCKETS or batch <= 0 or batch > 32:
        raise ValueError("invalid final-pool bucket/batch")
    if hidden <= 0 or hidden > 4096 or hidden % 256:
        raise ValueError("hidden size must be a multiple of 256 in 256..=4096")
    pairs = (batch + 1) // 2
    lanes = min(pairs, PAIR_LANES)
    pair_counts = [(pairs - lane + lanes - 1) // lanes for lane in range(lanes)]
    physical_batch = 2 * pairs
    hidden_row = hidden * 2
    hidden_pair = 2 * hidden_row
    input_record = hidden_pair + 8
    weight_bytes = (hidden + 16) * 4
    output_pair = 2 * hidden * 4
    input_bytes = physical_batch * bucket * hidden_row
    length_bytes = pairs * bucket * 8
    output_bytes = physical_batch * hidden * 4

    out = ["module {", "  aie.device(npu2) {"]
    for col in range(8):
        out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for lane in range(lanes):
        out += [
            f"    %c{lane} = aie.tile({2 * lane}, 2)",
            f'    %selected{lane} = aie.buffer(%c{lane}) {{sym_name = "selected{lane}"}} : memref<{2 * hidden}xbf16>',
            f'    %scratch{lane} = aie.buffer(%c{lane}) {{sym_name = "scratch{lane}"}} : memref<{2 * hidden}xf32>',
        ]
    for lane in range(lanes):
        out += [
            f"    aie.objectfifo @hsh{lane}(%shim{lane}, {{%mt{lane}}}, 1 : i32) : !aie.objectfifo<memref<{hidden_pair}xi8>>",
            f"    aie.objectfifo @lsh{lane}(%shim{lane + 4}, {{%mt{lane}}}, 1 : i32) : !aie.objectfifo<memref<8xi8>>",
            f"    aie.objectfifo @x{lane}(%mt{lane}, {{%c{lane}}}, 1 : i32) : !aie.objectfifo<memref<{input_record}xi8>>",
            f"    aie.objectfifo.link [@hsh{lane}, @lsh{lane}] -> [@x{lane}] ([0, {hidden_pair}] [])",
            f"    aie.objectfifo @o{lane}(%c{lane}, {{%mt{lane}}}, 1 : i32) : !aie.objectfifo<memref<{output_pair}xi8>>",
            f"    aie.objectfifo @osh{lane}(%mt{lane}, {{%shim{lane}}}, 1 : i32) : !aie.objectfifo<memref<{output_pair}xi8>>",
            f"    aie.objectfifo.link [@o{lane}] -> [@osh{lane}] ([0] [])",
        ]
    cores = ", ".join(f"%c{lane}" for lane in range(lanes))
    out += [
        f"    aie.objectfifo @wsh(%shim7, {{%mt7}}, 1 : i32) : !aie.objectfifo<memref<{weight_bytes}xi8>>",
        f"    aie.objectfifo @wbc(%mt7, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{weight_bytes}xi8>>",
        "    aie.objectfifo.link [@wsh] -> [@wbc] ([] [0])",
    ]
    out += [
        f'    func.func private @hipfire_qwen3_select_last(memref<{input_record}xi8>, memref<{2 * hidden}xbf16>, i32) attributes {{link_with = "qwen3_final_pool_l2.o"}}',
        f'    func.func private @hipfire_qwen3_final_norm_l2(memref<{2 * hidden}xbf16>, memref<{weight_bytes}xi8>, memref<{2 * hidden}xf32>, memref<{output_pair}xi8>) attributes {{link_with = "qwen3_final_pool_l2.o"}}',
    ]
    for lane, pair_count in enumerate(pair_counts):
        out += [
            f"    %core{lane} = aie.core(%c{lane}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %tokens = arith.constant {bucket} : index",
            f"      %pairs = arith.constant {pair_count} : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %w = aie.objectfifo.acquire @wbc(Consume, 1) : !aie.objectfifosubview<memref<{weight_bytes}xi8>>",
            f"        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{weight_bytes}xi8>> -> memref<{weight_bytes}xi8>",
            "        scf.for %pair = %z to %pairs step %one {",
            "          scf.for %token = %z to %tokens step %one {",
            f"            %x = aie.objectfifo.acquire @x{lane}(Consume, 1) : !aie.objectfifosubview<memref<{input_record}xi8>>",
            f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{input_record}xi8>> -> memref<{input_record}xi8>",
            "            %token_i32 = arith.index_cast %token : index to i32",
            f"            func.call @hipfire_qwen3_select_last(%xv, %selected{lane}, %token_i32) : (memref<{input_record}xi8>, memref<{2 * hidden}xbf16>, i32) -> ()",
            f"            aie.objectfifo.release @x{lane}(Consume, 1)",
            "          }",
            f"          %o = aie.objectfifo.acquire @o{lane}(Produce, 1) : !aie.objectfifosubview<memref<{output_pair}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{output_pair}xi8>> -> memref<{output_pair}xi8>",
            f"          func.call @hipfire_qwen3_final_norm_l2(%selected{lane}, %wv, %scratch{lane}, %ov) : (memref<{2 * hidden}xbf16>, memref<{weight_bytes}xi8>, memref<{2 * hidden}xf32>, memref<{output_pair}xi8>) -> ()",
            f"          aie.objectfifo.release @o{lane}(Produce, 1)",
            "        }",
            "        aie.objectfifo.release @wbc(Consume, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]
    out.append(
        f"    aie.runtime_sequence(%H: memref<{input_bytes}xi8>, %L: memref<{length_bytes}xi8>, %W: memref<{weight_bytes}xi8>, %O: memref<{output_bytes}xi8>) {{"
    )
    out += [
        "      %tw = aiex.dma_configure_task_for @wsh {",
        f"        aie.dma_bd(%W : memref<{weight_bytes}xi8>, 0, {weight_bytes}, {dims([(1, weight_bytes), (weight_bytes // 64, 64), (64, 1)])}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        "      aiex.dma_start_task(%tw)",
    ]
    groups = max(pair_counts)
    for group in range(groups):
        active = [lane for lane in range(lanes) if group < pair_counts[lane]]
        output_tasks: list[str] = []
        for lane in active:
            pair = group * lanes + lane
            oname = f"to{lane}_{group}"
            output_tasks.append(oname)
            out += [
                f"      %{oname} = aiex.dma_configure_task_for @osh{lane} {{",
                f"        aie.dma_bd(%O : memref<{output_bytes}xi8>, {pair * output_pair}, {output_pair}, {dims([(output_pair // 64, 64), (64, 1)])}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{oname})",
            ]
        for start in range(0, bucket, CHUNK):
            count = min(CHUNK, bucket - start)
            phase: list[str] = []
            for lane in active:
                pair = group * lanes + lane
                hname = f"th{lane}_{group}_{start}"
                lname = f"tl{lane}_{group}_{start}"
                phase += [hname, lname]
                hidden_offset = (2 * pair * bucket + start) * hidden_row
                hidden_layout = dims(
                    [
                        (count, hidden_row),
                        (2, bucket * hidden_row),
                        (hidden_row // 64, 64),
                        (64, 1),
                    ]
                )
                out += [
                    f"      %{hname} = aiex.dma_configure_task_for @hsh{lane} {{",
                    f"        aie.dma_bd(%H : memref<{input_bytes}xi8>, {hidden_offset}, {hidden_pair}, {hidden_layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }} {{issue_token = true, repeat_count = {count - 1} : i32}}",
                    f"      aiex.dma_start_task(%{hname})",
                ]
                length_offset = (pair * bucket + start) * 8
                length_layout = dims([(count, 8), (1, 8), (2, 4), (4, 1)])
                out += [
                    f"      %{lname} = aiex.dma_configure_task_for @lsh{lane} {{",
                    f"        aie.dma_bd(%L : memref<{length_bytes}xi8>, {length_offset}, 8, {length_layout}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }} {{issue_token = true, repeat_count = {count - 1} : i32}}",
                    f"      aiex.dma_start_task(%{lname})",
                ]
            for task in phase:
                out += [
                    f"      aiex.dma_await_task(%{task})",
                    f"      aiex.dma_free_task(%{task})",
                ]
        for task in output_tasks:
            out += [
                f"      aiex.dma_await_task(%{task})",
                f"      aiex.dma_free_task(%{task})",
            ]
    out += ["      aiex.dma_await_task(%tw)", "      aiex.dma_free_task(%tw)"]
    out += ["    }", "  }", "}"]
    return "\n".join(out) + "\n"


def toolchain() -> tuple[Path, Path]:
    venv = Path(os.environ.get("HIPFIRE_NPU_VENV", Path.home() / ".venv"))
    location = subprocess.run(
        [str(venv / "bin/python"), "-c", "import mlir_aie; print(list(mlir_aie.__path__)[0])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package = Path(location)
    return package, package.parent / "llvm-aie"


def build(output: Path, bucket: int, batch: int, hidden: int, emit_mlir_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "aie.mlir"
    mlir.write_text(generate_mlir(bucket, batch, hidden), encoding="utf-8")
    if not emit_mlir_only:
        package, peano = toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(package / "bin"), env.get("PATH", "")]
        )
        source = Path(__file__).with_name("qwen3_final_pool_l2_bf16.cc")
        subprocess.run(
            [
                str(peano / "bin/clang++"),
                str(source),
                "-c",
                "-o",
                str(output / "qwen3_final_pool_l2.o"),
                f"-I{package / 'include'}",
                "-std=c++20",
                "-O2",
                "-DNDEBUG",
                f"-DHIPFIRE_HIDDEN_SIZE={hidden}",
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
                "schema": "hipfire.npu_qwen3_final_pool_l2.v1",
                "sequence_bucket": bucket,
                "dispatch_batch": batch,
                "hidden_size": hidden,
                "pooling": "last_real_token",
                "normalize": "weighted_rms_then_l2",
                "xclbin": "final.xclbin",
                "instructions": "insts.bin",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mlir-only", action="store_true")
    args = parser.parse_args()
    try:
        build(args.output, args.bucket, args.batch, args.hidden_size, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
