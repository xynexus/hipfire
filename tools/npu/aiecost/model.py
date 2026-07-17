#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""The cost model: ScheduleSpec -> Prediction.

Per docs/npu/aie2-cost-model-plan.md §4.

    T_device  ~= fill + max(t_feed, t_core + t_stage, t_drain) + tail
    T_wrapper ~= t_host + t_submit + T_device

The composition is an overlap model with an explicit stall term, NOT a sum.
That is what lets it reproduce R117-style results (more work, same fixed cost,
less time); a sum cannot.

Two rules the aie2p corpus earned the hard way:
  - it refuses to predict on missing constants instead of extrapolating;
  - it checks buildability first, because the limits that killed R59 (5 arg
    slots), R61 (tile locks), and R118/R119 (DMA task budget) are knowable
    ahead of time from the target model.
"""

from __future__ import annotations

from . import calib
from .spec import Prediction, ScheduleSpec


def _hw_limits(device: str = "npu1") -> dict:
    """Buildability limits, read live from the toolchain target model."""
    from .device import probe_toolchain

    tc = probe_toolchain(device)
    return {
        "l1_bytes": tc["local_memory_bytes"],
        "bds_per_core": tc.get("bds_per_core", 16),
        "locks_per_core": tc.get("locks_per_core", 16),
        "columns": tc["columns"],
        "cores": tc["n_core_tiles"],
        "arg_slots": 5,  # H4: findings R59 (aie2p); likely transfers, unverified on npu1
    }


def check_buildable(spec: ScheduleSpec, limits: dict) -> list[str]:
    """Reject schedules the hardware/toolchain cannot express, before predicting."""
    errs = []
    if spec.local_stage_bytes > limits["l1_bytes"]:
        errs.append(f"local_stage_bytes={spec.local_stage_bytes} exceeds L1={limits['l1_bytes']} (M1)")
    if spec.bds_per_core > limits["bds_per_core"]:
        errs.append(f"bds_per_core={spec.bds_per_core} exceeds {limits['bds_per_core']} BDs/core (M6; cf. R118/R119)")
    if spec.locks_per_core > limits["locks_per_core"]:
        errs.append(f"locks_per_core={spec.locks_per_core} exceeds {limits['locks_per_core']} locks/core (M7; cf. R61)")
    if spec.columns > limits["columns"]:
        errs.append(f"columns={spec.columns} exceeds {limits['columns']} (H1/H2)")
    if spec.cores > limits["cores"]:
        errs.append(f"cores={spec.cores} exceeds {limits['cores']} (H2)")
    if spec.n_bos > limits["arg_slots"]:
        errs.append(f"n_bos={spec.n_bos} exceeds {limits['arg_slots']} DPU arg slots (H4; cf. R59)")
    return errs


def predict(spec: ScheduleSpec, key: str | None = None, device: str = "npu1") -> Prediction:
    """Predict a schedule. Refuses (admissible=False) if constants are missing."""
    key = key or calib.current_key()
    consts = calib.load(key)
    limits = _hw_limits(device)

    p = Prediction(spec_name=spec.name)
    p.build_errors = check_buildable(spec, limits)
    p.buildable = not p.build_errors
    if not p.buildable:
        p.admissible = False
        return p

    missing: list[str] = []

    def need(name: str, bench: str) -> float | None:
        c = consts.get(name)
        if c is None or not c.admissible:
            missing.append(f"{name} ({bench})")
            return None
        return c.value

    # ── t_core: the only fully calibrated term today (K1) ──
    f_h = need("f_h_hz", "K1")
    cyc_per_vmac = need("cyc_per_vmac", "C4")
    if cyc_per_vmac is None:
        # K1's disassembly showed exactly 1 VMAC per bundle at 1 bundle/cycle.
        # Recorded as an assumption, not silently treated as measured.
        cyc_per_vmac = 1.0
        missing.pop()  # not fatal: we have direct ISA evidence for this one
        p.assumptions.append("cyc_per_vmac=1.0 from K1 disassembly (1 VMAC/bundle); C4 should confirm per dtype/shape")
    if f_h:
        p.terms["t_core"] = spec.vmacs_per_core * cyc_per_vmac / f_h

    # ── t_stage / alignment ──
    if not spec.aligned_loads:
        pen = need("align_penalty_frac", "C8")
        if pen and "t_core" in p.terms:
            p.terms["t_core"] *= 1.0 + pen

    # ── t_feed / t_drain ──
    # C2 and C5 both scale near-linearly with columns, so bandwidth is stored
    # per-column and multiplied. Using a fixed 4-column figure would mispredict
    # a 1-column schedule by ~4x.
    bw_feed_col = need("bw_feed_per_col_bytes_s", "C2")
    if bw_feed_col and spec.wire_bytes_in:
        p.terms["t_feed"] = spec.wire_bytes_in / (bw_feed_col * spec.columns)

    bw_drain_col = need("bw_drain_per_col_bytes_s", "C5")
    if bw_drain_col and spec.output_bytes:
        p.terms["t_drain"] = spec.output_bytes / (bw_drain_col * spec.columns)

    # ── t_task ──
    c_issue = need("c_task_issue_s", "C3")
    if c_issue:
        p.terms["t_task"] = c_issue * spec.dma_tasks_live

    # ── fixed: submit + host ──
    c_cmd = need("c_cmd_s", "C1")
    c_bo = need("c_bo_s", "C1")
    if c_cmd is not None and c_bo is not None:
        p.terms["t_submit"] = c_cmd + c_bo * spec.n_bos

    c_call = need("c_call_s", "C7")
    c_pack = need("c_pack_s_per_byte", "C7")
    c_deblock = need("c_deblock_s_per_byte", "C7")
    if c_call is not None and c_pack is not None and c_deblock is not None:
        p.terms["t_host"] = c_call + c_pack * spec.host_pack_bytes + c_deblock * spec.host_deblock_bytes

    # ── fill / drain latency ──
    fill = need("fill_drain_s", "C6")
    if fill is not None:
        p.terms["fill_tail"] = fill * spec.fifo_depth

    p.missing = missing
    p.admissible = not missing
    if not p.admissible:
        return p

    # ── compose: overlap, not sum ──
    core = p.terms.get("t_core", 0.0)
    feed = p.terms.get("t_feed", 0.0)
    drain = p.terms.get("t_drain", 0.0)
    task = p.terms.get("t_task", 0.0)
    fill_tail = p.terms.get("fill_tail", 0.0)

    # DMA task issue is serialised against the feed it schedules.
    feed_side = feed + task
    steady = max(feed_side, core, drain)

    # The dispatch floor is DEVICE time, not host time: C1 measures it with
    # npu_time on a near-null kernel, so it lands inside the device span and
    # cannot overlap with the work. Filing it under the wrapper made the model
    # under-predict every phase-6 candidate by a near-constant ~160-260 us —
    # the residuals were uniformly negative, which is the signature of a missing
    # fixed term rather than a wrong rate.
    p.device_s = p.terms.get("t_submit", 0.0) + fill_tail + steady
    p.wrapper_s = p.terms.get("t_host", 0.0) + p.device_s

    p.limiter = max((("t_feed", feed_side), ("t_core", core), ("t_drain", drain)), key=lambda kv: kv[1])[0]
    # The receiver stalls whenever the consumer cannot keep up with the feed.
    p.stall_fraction = 0.0 if steady <= 0 else max(0.0, (steady - feed_side) / steady)
    return p


def rank(specs: list[ScheduleSpec], key: str | None = None) -> list[tuple[ScheduleSpec, Prediction]]:
    """Rank candidates fastest-first. Ordinal accuracy is the product (§3)."""
    out = [(s, predict(s, key)) for s in specs]
    ok = [(s, p) for s, p in out if p.buildable and p.admissible]
    bad = [(s, p) for s, p in out if not (p.buildable and p.admissible)]
    ok.sort(key=lambda sp: sp[1].device_s)
    return ok + bad
