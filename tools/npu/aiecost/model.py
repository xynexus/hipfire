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

import math

from . import calib
from .spec import Prediction, ScheduleSpec


def _hw_limits(device: str = "auto") -> dict:
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


def predict(spec: ScheduleSpec, key: str | None = None, device: str = "auto") -> Prediction:
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

    # ── t_core ──
    # A source-level mac() costs 1 or 2 NATIVE VMACs depending on shape and
    # operand types, and only native VMACs consume issue slots. C4 measured the
    # MACs a native VMAC can carry, per operand-type pair:
    #   int8 x int8 -> 256   int8 x int4 -> 512
    # so native_vmacs_per_call = ceil(macs_per_call / ceiling). That reproduces
    # every C4 row: <4,8,8,i8,i8> 256/256=1; <8,8,8,i8,i8> 512/256=2 (virtual);
    # <4,16,8,i8,i4> 512/512=1; <4,32,8,i8,i4> 1024/512=2; <2,8,8> ceil(0.5)=1
    # (one VMAC, half wasted). Assuming 256 for everything would mispredict any
    # OQ4/MQ4 kernel by 2x.
    f_h = need("f_h_hz", "K1")
    cyc_per_vmac = need("cyc_per_vmac", "C4")
    ceiling = need(f"macs_per_native_vmac_{spec.dtype_pair}", "C4")

    if f_h and cyc_per_vmac and ceiling and spec.mmul_calls_per_core:
        macs = spec.macs_per_call
        native_per_call = math.ceil(macs / ceiling)
        vmacs = spec.mmul_calls_per_core * native_per_call
        p.terms["t_core"] = vmacs * cyc_per_vmac / f_h

        m, k, n = spec.mmul_shape
        if native_per_call > 1:
            # Throughput-NEUTRAL, not harmful: C4 measured <4,16,8>=520.09,
            # <4,32,8>=524.98, <8,16,8>=522.82 G MACs/s — a virtual shape costs
            # N x the issue slots but delivers N x the MACs. The real cost is
            # accumulator registers (C_block holds N accums), which limits how
            # many independent chains fit; R0 found 2x2 mmul optimal for exactly
            # that reason. Do not report this as lost throughput.
            p.advice.append(
                f"mmul<{m},{k},{n},{spec.dtype_a},{spec.dtype_b}> is VIRTUAL: {macs} MACs = "
                f"{native_per_call} native VMACs at {ceiling:.0f} each. Throughput is unaffected "
                f"(same MACs/VMAC), but it uses {native_per_call} accumulators per call, leaving "
                f"fewer registers for independent chains — which is what hides VMAC latency (K1)."
            )
        if macs < ceiling:
            waste = 1.0 - macs / ceiling
            p.advice.append(
                f"mmul<{m},{k},{n}> UNDER-FILLS the VMAC: {macs} of {ceiling:.0f} MACs — "
                f"{waste * 100:.0f}% of every issue is wasted, and this IS lost throughput. "
                f"Use a shape whose M*K*N reaches {ceiling:.0f} in this dtype family."
            )
        if spec.dtype_a == "int8" and spec.dtype_b == "int8":
            p.advice.append(
                "int8 x int8 caps at 256 MACs/VMAC (8.31 TOPS peak). If the weights are 4-bit "
                "(OQ4/MQ4), mmul<4,16,8,int8,int4> is native at 512 MACs/VMAC — 2x the compute "
                "rate, 16.63 TOPS. Only worth chasing if t_core is actually the limiter."
            )

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

    if spec.useful_macs and p.device_s > 0:
        p.useful_tops = spec.useful_macs * 2 / p.device_s / 1e12

    p.limiter = max((("t_feed", feed_side), ("t_core", core), ("t_drain", drain)), key=lambda kv: kv[1])[0]
    # The receiver stalls whenever the consumer cannot keep up with the feed.
    p.stall_fraction = 0.0 if steady <= 0 else max(0.0, (steady - feed_side) / steady)
    return p


def rank(
    specs: list[ScheduleSpec], key: str | None = None, device: str = "auto"
) -> list[tuple[ScheduleSpec, Prediction]]:
    """Rank candidates fastest-first. Ordinal accuracy is the product (§3)."""
    out = [(s, predict(s, key, device)) for s in specs]
    ok = [(s, p) for s, p in out if p.buildable and p.admissible]
    bad = [(s, p) for s, p in out if not (p.buildable and p.admissible)]
    ok.sort(key=lambda sp: sp[1].device_s)
    return ok + bad
