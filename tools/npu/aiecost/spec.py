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
    # Count SOURCE-level aie::mmul mac() calls, not native VMACs. One call costs
    # 1 or 2 native VMACs depending on shape and operand types (C4): mmul<8,8,8>
    # is virtual and issues 2. The model derives the native count; giving it VMACs
    # directly would make the caller responsible for knowing that, and getting it
    # wrong is a silent 2x.
    mmul_calls_per_core: int = 0
    mmul_shape: tuple[int, int, int] = (4, 8, 8)  # M,K,N of aie::mmul
    # Operand types decide MACs-per-native-VMAC: int8xint8 = 256, int8xint4 = 512
    # (C4). hipfire's OQ4/MQ4 weights are the int4 case and get 2x the int8 rate.
    dtype_a: str = "int8"
    dtype_b: str = "int8"
    local_stage_bytes: int = 0  # tile-local staging buffer per core
    aligned_loads: bool = True  # 64 B-aligned local loads (see R118 / K3)

    # ── host wrapper ──
    host_pack_bytes: int = 0  # bytes the host packs before submit
    host_deblock_bytes: int = 0  # bytes the host deblocks after
    n_bos: int = 3  # buffer objects bound to the dispatch

    note: str = ""

    @property
    def macs_per_call(self) -> int:
        """MACs one aie::mmul mac() computes — independent of how many VMACs it costs."""
        m, k, n = self.mmul_shape
        return m * k * n

    @property
    def dtype_pair(self) -> str:
        return f"{self.dtype_a}_{self.dtype_b}"

    @property
    def useful_macs(self) -> int:
        return self.mmul_calls_per_core * self.macs_per_call * self.cores

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
    # Actionable design feedback: a virtual mmul shape or an under-filled VMAC is
    # throughput the schedule is leaving on the table, and the model knows it.
    advice: list[str] = field(default_factory=list)
    useful_tops: float = 0.0
    # Energy is the second axis (E1). It does NOT co-optimise with time: below
    # ~37 MACs per byte fed, energy is set by data movement while time may be set
    # by something else entirely.
    energy_j: float = 0.0
    energy_terms: dict[str, float] = field(default_factory=dict)
    arithmetic_intensity: float = 0.0

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
            if self.useful_tops:
                out.append(f"  useful  : {self.useful_tops:.2f} TOPS")
            if self.energy_j:
                out.append(f"  energy  : {self.energy_j * 1e3:10.4f} mJ   (AI={self.arithmetic_intensity:.1f} MACs/byte)")
                for k, v in sorted(self.energy_terms.items(), key=lambda kv: -kv[1]):
                    share = v / self.energy_j * 100 if self.energy_j else 0
                    out.append(f"    E:{k:<8} {v * 1e3:9.4f} mJ  {share:5.1f}%")
        for a in self.advice:
            out.append(f"  ADVICE: {a}")
        for a in self.assumptions:
            out.append(f"  assumption: {a}")
        return "\n".join(out)
