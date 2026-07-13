#!/usr/bin/env python3
"""Inject declarative trace ops into canonical R63 MLIR without rebuilding it."""

import os
import sys
from pathlib import Path

from aie import ir
from aie.dialects import aie, aiex  # noqa: F401 - registers both dialects
from aie.dialects.aie import WireBundle
from aie.extras.context import mlir_mod_ctx
from aie.utils.trace.events import PortEvent, get_events_for_device
from aie.utils.trace.setup import configure_trace, start_trace


if len(sys.argv) != 2:
    raise SystemExit("usage: r64_inject_trace.py BASE.mlir")

source = Path(sys.argv[1]).read_text(encoding="utf-8")
trace_start = int(os.environ.get("R64_TRACE_START", "0"))
trace_columns = int(os.environ.get("R64_TRACE_COLS", "4"))
trace_tile = os.environ.get("R64_TRACE_TILE", "shim")
if trace_start < 0 or trace_columns <= 0 or trace_start + trace_columns > 8:
    raise SystemExit("R64 trace column range must fit within columns 0..7")
if trace_tile not in ("core", "shim"):
    raise SystemExit("R64_TRACE_TILE must be core or shim")
events = get_events_for_device("npu2").CoreEvent
core_events = [
    PortEvent(events.PORT_RUNNING_0, WireBundle.DMA, 0, True),
    PortEvent(events.PORT_STALLED_0, WireBundle.DMA, 0, True),
    PortEvent(events.PORT_RUNNING_1, WireBundle.DMA, 1, True),
    PortEvent(events.PORT_STALLED_1, WireBundle.DMA, 1, True),
    PortEvent(events.PORT_RUNNING_2, WireBundle.DMA, 0, False),
    PortEvent(events.PORT_STALLED_2, WireBundle.DMA, 0, False),
    events.INSTR_VECTOR,
    events.STREAM_STALL,
]
shim_events = get_events_for_device("npu2").ShimTileEvent
shimtile_events = [
    shim_events.DMA_MM2S_0_START_TASK,
    shim_events.DMA_MM2S_0_FINISHED_TASK,
    shim_events.DMA_MM2S_1_START_TASK,
    shim_events.DMA_MM2S_1_FINISHED_TASK,
    shim_events.DMA_S2MM_0_START_TASK,
    shim_events.DMA_S2MM_0_FINISHED_TASK,
    shim_events.DMA_S2MM_0_STREAM_STARVATION,
    shim_events.DMA_S2MM_1_STREAM_STARVATION,
]

with mlir_mod_ctx(src=source) as context:
    top_level = list(context.module.body.operations)
    if len(top_level) != 1 or top_level[0].operation.name != "aie.device":
        raise SystemExit("R64 expected one top-level aie.device")
    device = top_level[0]
    body = device.body_region.blocks[0]
    all_row_zero_cores = [
        operation
        for operation in body.operations
        if operation.operation.name == "aie.tile"
        and int(operation.attributes["row"]) == 2
    ]
    row_zero_cores = [
        operation
        for operation in all_row_zero_cores
        if trace_start <= int(operation.attributes["col"]) < trace_start + trace_columns
    ]
    selected_shims = [
        operation
        for operation in body.operations
        if operation.operation.name == "aie.tile"
        and int(operation.attributes["row"]) == 0
        and trace_start <= int(operation.attributes["col"]) < trace_start + trace_columns
    ]
    sequences = [
        operation
        for operation in body.operations
        if operation.operation.name == "aie.runtime_sequence"
    ]
    selected_tiles = row_zero_cores if trace_tile == "core" else selected_shims
    if (
        len(all_row_zero_cores) != 8
        or len(selected_tiles) != trace_columns
        or len(sequences) != 1
    ):
        raise SystemExit(
            f"R64 expected eight row-0 cores, {trace_columns} selected cores, and one "
            f"runtime sequence; got {len(all_row_zero_cores)}, {len(row_zero_cores)}, "
            f"and {len(sequences)}"
        )
    with ir.InsertionPoint.at_block_terminator(body):
        if trace_tile == "core":
            configure_trace(selected_tiles, coretile_events=core_events)
        else:
            configure_trace(selected_tiles, shimtile_events=shimtile_events)
    with ir.InsertionPoint.at_block_begin(sequences[0].body.blocks[0]):
        start_trace(trace_size=16 * 1024 * 1024, ddr_id=4, routing="single")
    if context.module.operation.verify() is not True:
        raise SystemExit("R64 traced MLIR failed verification")
    print(context.module)
