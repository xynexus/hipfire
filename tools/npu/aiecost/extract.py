#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase 7: MLIR -> ScheduleSpec extractor.

Turns a compiled `aie.mlir` into the ScheduleSpec the model consumes, so a
prediction can be *audited* against what actually built rather than against what
someone believed they wrote. A hand-written spec records intent; the IR records
reality, and R61/R63 are full of cases where those diverged.

What it reads, and from where:
    aie.device(npu1)                          -> device
    aie.tile(col, row)                        -> topology; row >= 2 is a core tile
    aie.objectfifo @n(prod, {cons}, depth)    -> depth, element bytes, direction
    aie.dma_bd(%arg : memref<NxT>, off, len)  -> per-task transfer length
    aiex.dma_configure_task_for @fifo         -> which fifo a task feeds
    aiex.dma_start_task                       -> live task count

What it CANNOT read, and says so rather than guessing:
    mmul_calls_per_core — lives inside the external kernel object, not the IR
    dtype_a / dtype_b   — mmul operand types are a kernel-body property
    aligned_loads       — a property of the kernel body
    host pack/deblock — host-side, outside the module entirely

Those stay at their ScheduleSpec defaults and are listed in `unknown`, so a
prediction built from extraction alone is explicitly partial.

Usage:
    python -m aiecost.extract path/to/aie.mlir
    python -m aiecost.extract path/to/aie.mlir --predict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiecost.spec import ScheduleSpec  # noqa: E402

_DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "bf16": 2, "f32": 4, "f16": 2}

RE_DEVICE = re.compile(r"aie\.device\((\w+)\)")
RE_TILE = re.compile(r"aie\.tile\((\d+),\s*(\d+)\)")
RE_FIFO = re.compile(r"aie\.objectfifo\s+@(\w+)\(%(\w+),\s*\{([^}]*)\},\s*(\d+)\s*:\s*i32\)\s*:\s*!aie\.objectfifo<memref<(\d+)x(\w+)>>")
RE_CORE = re.compile(r"aie\.core\(%(\w+)\)")
RE_TASK_FOR = re.compile(r"aiex\.dma_configure_task_for\s+@(\w+)")
RE_BD = re.compile(r"aie\.dma_bd\(%\w+\s*:\s*memref<(\d+)x(\w+)>,\s*(\d+),\s*(\d+)")
RE_START = re.compile(r"aiex\.dma_start_task")


@dataclass
class Fifo:
    name: str
    producer: str
    consumers: list[str]
    depth: int
    elems: int
    dtype: str

    @property
    def tile_bytes(self) -> int:
        return self.elems * _DTYPE_BYTES.get(self.dtype, 4)

    @property
    def is_feed(self) -> bool:
        """Shim-produced => data entering the array."""
        return "shim" in self.producer


@dataclass
class Extraction:
    device: str = "unknown"
    core_tiles: list[tuple[int, int]] = field(default_factory=list)
    shim_tiles: list[tuple[int, int]] = field(default_factory=list)
    fifos: list[Fifo] = field(default_factory=list)
    n_cores: int = 0
    tasks: list[tuple[str, int]] = field(default_factory=list)  # (fifo, bytes)
    n_start_tasks: int = 0
    unknown: list[str] = field(default_factory=list)


def parse(text: str) -> Extraction:
    e = Extraction()
    if m := RE_DEVICE.search(text):
        e.device = m.group(1)

    tile_names: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if m := re.search(r"%(\w+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", line):
            name, col, row = m.group(1), int(m.group(2)), int(m.group(3))
            tile_names[name] = (col, row)
            # Row 0 is shim, row 1 is the memory tile, rows >= 2 are compute.
            (e.shim_tiles if row == 0 else e.core_tiles).append((col, row))
    e.core_tiles = [t for t in e.core_tiles if t[1] >= 2]

    for m in RE_FIFO.finditer(text):
        name, prod, cons, depth, elems, dtype = m.groups()
        e.fifos.append(
            Fifo(name=name, producer=prod, consumers=[c.strip().lstrip("%") for c in cons.split(",") if c.strip()],
                 depth=int(depth), elems=int(elems), dtype=dtype)
        )

    e.n_cores = len(RE_CORE.findall(text))
    e.n_start_tasks = len(RE_START.findall(text))

    # Pair each configure_task_for with the dma_bd that follows it.
    pos = 0
    for m in RE_TASK_FOR.finditer(text):
        fifo = m.group(1)
        bd = RE_BD.search(text, m.end())
        if bd:
            elems, dtype, _off, length = bd.groups()
            e.tasks.append((fifo, int(length) * _DTYPE_BYTES.get(dtype, 4)))
        pos = m.end()
    _ = pos

    e.unknown = [
        "mmul_calls_per_core (inside the external kernel, not the IR)",
        "dtype_a/dtype_b (mmul operand types are a kernel-body property)",
        "aligned_loads (kernel body property)",
        "host_pack_bytes / host_deblock_bytes (host-side, outside the module)",
    ]
    return e


def to_spec(e: Extraction, name: str) -> ScheduleSpec:
    columns = len({c for c, _ in e.core_tiles}) or 1
    feed_fifos = [f for f in e.fifos if f.is_feed]
    drain_fifos = [f for f in e.fifos if not f.is_feed]
    feed_names = {f.name for f in feed_fifos}

    wire_in = sum(b for fifo, b in e.tasks if fifo in feed_names)
    out_bytes = sum(b for fifo, b in e.tasks if fifo not in feed_names)

    # Per-core local footprint: every fifo endpoint the core holds, depth-deep.
    per_core = 0
    for f in e.fifos:
        per_core += f.depth * f.tile_bytes
    n_bds = sum(f.depth for f in e.fifos)

    return ScheduleSpec(
        name=name,
        columns=columns,
        cores=max(1, e.n_cores),
        wire_bytes_in=wire_in,
        output_bytes=out_bytes,
        dma_tasks_live=max(1, e.n_start_tasks),
        bds_per_core=max(1, n_bds),
        locks_per_core=max(1, len(e.fifos) * 2),
        fifo_depth=max((f.depth for f in feed_fifos), default=1),
        local_stage_bytes=per_core,
        n_bos=len(e.tasks),
        note=f"extracted from MLIR; unknown: {'; '.join(e.unknown)}",
    )


def render(e: Extraction, spec: ScheduleSpec) -> str:
    out = [f"extracted: device={e.device}  cores={e.n_cores}  core_tiles={e.core_tiles}  shim={e.shim_tiles}"]
    out.append("  objectfifos:")
    for f in e.fifos:
        kind = "feed " if f.is_feed else "drain"
        out.append(f"    {kind} @{f.name:<10} depth={f.depth}  tile={f.tile_bytes:>7} B  ({f.elems}x{f.dtype})  -> {f.consumers}")
    out.append("  dma tasks:")
    for fifo, b in e.tasks:
        out.append(f"    @{fifo:<10} {b:>9} B")
    out.append(f"  start_task count: {e.n_start_tasks}")
    out.append("")
    out.append("  derived ScheduleSpec:")
    for k, v in spec.to_dict().items():
        if k == "note":
            continue
        out.append(f"    {k:<20} {v}")
    out.append("")
    out.append("  NOT extractable (left at defaults; prediction is partial):")
    for u in e.unknown:
        out.append(f"    - {u}")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mlir", help="path to a compiled aie.mlir")
    p.add_argument("--predict", action="store_true", help="also run the cost model on the extracted spec")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    text = Path(args.mlir).read_text()
    e = parse(text)
    spec = to_spec(e, Path(args.mlir).stem)
    print(render(e, spec))

    if args.predict:
        from aiecost import model

        print()
        print(model.predict(spec, device=e.device if e.device in ("npu1", "npu2") else "auto").render())

    if args.json:
        Path(args.json).write_text(json.dumps(spec.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
