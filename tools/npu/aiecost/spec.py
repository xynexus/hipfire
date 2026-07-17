#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""ScheduleSpec / Prediction — the model's input and output.

Per docs/npu/aie2-cost-model-plan.md §4. A ScheduleSpec describes a candidate
dataflow *before* the kernel is written: what moves, how many DMA tasks carry
it, how much arithmetic each core does, and what the host must do around it.

The fields are deliberately the ones the aie2p corpus proved matter — task
counts, output-tile area, bytes per role, host pack/deblock — not FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ScheduleSpec:
    """A candidate schedule. All byte counts are per dispatch."""

    name: str

    # ── parallelism ──
    columns: int = 4  # compute columns used
    cores: int = 16  # total cores participating

    # ── external traffic (per dispatch, whole device) ──
    wire_bytes_in: int = 0  # bytes fed from host/DDR into the array
    output_bytes: int = 0  # bytes drained out of the array
    feed_streams: int = 0  # concurrent receive streams (0 => derive from columns)

    # ── DMA scheduling ──
    dma_tasks_live: int = 1  # live tasks (repeat_count collapses these; see R119)
    bds_per_core: int = 1  # buffer descriptors used per core
    locks_per_core: int = 1  # locks used per core
    fifo_depth: int = 2

    # ── per-core compute ──
    vmacs_per_core: int = 0  # VMAC (mmul) issues per core per dispatch
    mmul_shape: tuple[int, int, int] = (4, 8, 8)  # M,K,N of aie::mmul
    local_stage_bytes: int = 0  # tile-local staging buffer per core
    aligned_loads: bool = True  # 64 B-aligned local loads (see R118 / K3)

    # ── host wrapper ──
    host_pack_bytes: int = 0  # bytes the host packs before submit
    host_deblock_bytes: int = 0  # bytes the host deblocks after
    n_bos: int = 3  # buffer objects bound to the dispatch

    note: str = ""

    @property
    def macs_per_vmac(self) -> int:
        m, k, n = self.mmul_shape
        return m * k * n

    @property
    def useful_macs(self) -> int:
        return self.vmacs_per_core * self.macs_per_vmac * self.cores

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """Per-term breakdown + limiter. Never a bare number."""

    spec_name: str
    terms: dict[str, float] = field(default_factory=dict)  # seconds
    device_s: float = 0.0
    wrapper_s: float = 0.0
    limiter: str = "unknown"
    stall_fraction: float = 0.0

    buildable: bool = True
    build_errors: list[str] = field(default_factory=list)

    admissible: bool = True  # False => outside the calibrated envelope
    missing: list[str] = field(default_factory=list)  # constants that were absent
    assumptions: list[str] = field(default_factory=list)

    @property
    def useful_tops(self) -> float:
        return 0.0

    def render(self) -> str:
        out = [f"prediction: {self.spec_name}"]
        if not self.buildable:
            out.append("  BUILD: REJECTED — this schedule cannot be built:")
            out += [f"    - {e}" for e in self.build_errors]
            return "\n".join(out)
        out.append("  BUILD: ok")
        if not self.admissible:
            out.append(f"  ADMISSIBLE: NO — uncalibrated: {', '.join(self.missing)}")
            out.append("  (the model refuses to predict rather than extrapolate; run the named benches)")
        width = max((len(k) for k in self.terms), default=8)
        for k, v in sorted(self.terms.items(), key=lambda kv: -kv[1]):
            mark = "  <== limiter" if k == self.limiter else ""
            out.append(f"    {k:<{width}}  {v * 1e6:10.3f} us{mark}")
        if self.admissible:
            out.append(f"  device  : {self.device_s * 1e6:10.3f} us")
            out.append(f"  wrapper : {self.wrapper_s * 1e6:10.3f} us")
            out.append(f"  limiter : {self.limiter}")
            out.append(f"  predicted receive-stall fraction: {self.stall_fraction * 100:.1f}%")
        for a in self.assumptions:
            out.append(f"  assumption: {a}")
        return "\n".join(out)
