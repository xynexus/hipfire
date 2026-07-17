#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""H-series: device facts and the claims register.

Implements docs/npu/aie2-cost-model-plan.md §5. Every static fact the cost model
relies on carries a source and a trust tier. Nothing enters the model at status
ASSUMED — an unknown narrows the declared envelope instead of getting a guess.

The register exists because the repo's own sources disagreed: UG1079 (in-repo)
documents AIE1/Versal, the npu-kernel-build skill inherited its 32 KB tile
memory from there, and findings.md asserted 64 KB. The toolchain settles it —
see TIER_TOOLCHAIN below and the `aie1_control` fact in probe_toolchain().
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from enum import IntEnum
from pathlib import Path
from typing import Any

from . import env
from .target import resolve_target

env.bootstrap()


class Tier(IntEnum):
    """Trust ranking of a source. Lower is stronger.

    TOOLCHAIN outranks vendor docs on capacity limits: what mlir-aie believes
    bounds what we can build, regardless of what the silicon can do.
    """

    PROBE = 1  # measured on this silicon
    TOOLCHAIN = 2  # mlir-aie target model — bounds buildability
    VENDOR_DOC = 3  # vendor doc for the correct generation (AM020 for AIE2)
    WRONG_GEN_DOC = 4  # vendor doc for another generation (UG1079/AIE1) — not evidence
    UNRELIABLE = 5  # self-inconsistent metadata (rocminfo) — corroboration only


class Status:
    CONFIRMED = "confirmed"  # measured or read from an authoritative source
    CORROBORATED = "corroborated"  # two independent sources agree
    LIKELY = "likely-transfers"  # aie2p evidence expected to hold on aie2
    WRONG_GEN = "wrong-generation"  # only a wrong-generation source claims it
    NO_TRANSFER = "does-not-transfer"  # halo-specific, must be re-measured
    UNKNOWN = "unknown"  # must probe; narrows the envelope until then


@dataclass
class Claim:
    """One static fact, with provenance."""

    id: str
    statement: str
    value: Any
    source: str
    tier: Tier
    status: str
    probe: str = ""
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = int(self.tier)
        return d


# ── live probes ─────────────────────────────────────────────────────────────


def probe_toolchain(device: str = "auto") -> dict[str, Any]:
    """Read the mlir-aie target model. Tier 2: what the compiler believes.

    Also reads the Versal (AIE1) model as a control. That control is what proves
    the 32 KB tile-memory figure belongs to AIE1 and not to any XDNA NPU.
    """
    import aie.dialects.aie as A

    def facts(dev_name: str) -> dict[str, Any]:
        tm = A.get_target_model(getattr(A.AIEDevice, dev_name))
        cols, rows = tm.columns(), tm.rows()
        cores = [(c, r) for c in range(cols) for r in range(rows) if tm.is_core_tile(c, r)]
        memtiles = [(c, r) for c in range(cols) for r in range(rows) if tm.is_mem_tile(c, r)]
        shims = [(c, r) for c in range(cols) for r in range(rows) if tm.is_shim_noc_or_pl_tile(c, r)]
        f: dict[str, Any] = {
            "columns": cols,
            "rows": rows,
            "local_memory_bytes": tm.get_local_memory_size(),
            "mem_tile_bytes": tm.get_mem_tile_size(),
            "mem_tile_rows": tm.get_num_mem_tile_rows(),
            "n_core_tiles": len(cores),
            "n_mem_tiles": len(memtiles),
            "n_shim_tiles": len(shims),
        }
        if cores:
            f["bds_per_core"] = tm.get_num_bds(*cores[0])
            f["locks_per_core"] = tm.get_num_locks(*cores[0])
        return f

    target = resolve_target(device)
    out = facts(target.key)
    out["target"] = {
        "key": target.key,
        "tile_isa": target.tile_isa,
        "target_arch": target.target_arch,
    }
    # AIE1 control: proves the 32 KB / no-memtile figures are Versal's, not ours.
    out["aie1_control"] = facts("xcvc1902")
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout
    except Exception:
        return ""


def parse_xrt_reports(platform: str, system: str) -> dict[str, Any]:
    """Parse xrt-smi without confusing BIOS and XRT version fields."""
    out: dict[str, Any] = {}
    if m := re.search(r"^\s*Name\s*:\s*(.*?)\s*$", platform, re.MULTILINE):
        out["device_name"] = m.group(1)
    table = re.search(r"\|\[[^]]+\]\s*\|([^|]+)\|([^|]+)\|([^|]+)\|", system)
    if table:
        out.setdefault("device_name", table.group(1).strip())
        out["architecture"] = table.group(2).strip()
        out["topology"] = table.group(3).strip()
    if m := re.search(r"Total Columns\s*:\s*(\d+)", platform):
        out["total_columns"] = int(m.group(1))
    if m := re.search(r"Power Mode\s*:\s*(\S+)", platform):
        out["power_mode"] = m.group(1)
    out["npu_clk_max_mhz"] = _search_int(platform, r"npu_clk_max\D+(\d+)")
    out["npu_tops_max"] = _search_int(platform, r"npu_tops_max\D+(\d+)")

    xrt = re.search(r"^XRT\s*$\n(?P<body>.*?)(?=^Device\(s\) Present|\Z)", system, re.MULTILINE | re.DOTALL)
    body = xrt.group("body") if xrt else ""
    if m := re.search(r"^\s*Version\s*:\s*([\d.]+)\s*$", body, re.MULTILINE):
        out["xrt_version"] = m.group(1)
    if m := re.search(r"^\s*amdxdna Version\s*:\s*(.*?)\s*$", body, re.MULTILINE):
        out["amdxdna_version"] = m.group(1)
    if m := re.search(r"^\s*NPU Firmware Version\s*:\s*(\S+)\s*$", body, re.MULTILINE):
        out["firmware_version"] = m.group(1)
    return out


def probe_xrt() -> dict[str, Any]:
    """Read xrt-smi. Tier 1 for identity, topology, and installed versions."""
    smi = env.xrt_bin("xrt-smi")
    if not smi.exists():
        return {}
    return parse_xrt_reports(
        _run([str(smi), "examine", "-r", "platform"]),
        _run([str(smi), "examine"]),
    )


def _search_int(txt: str, pat: str) -> int | None:
    m = re.search(pat, txt)
    return int(m.group(1)) if m else None


def probe_rocminfo() -> dict[str, Any]:
    """Read rocminfo's NPU agent. Tier 5 — corroboration only.

    This source reports L2 alongside Cacheline Size 0, Max Clock 0, and Compute
    Unit 0, i.e. it does not know the device. Its L2 figure is only usable
    because the toolchain independently agrees (see M5).
    """
    if not shutil.which("rocminfo"):
        return {}
    txt = _run(["rocminfo"])
    # Isolate the DSP/aie agent block before matching, so we do not read the GPU's caches.
    blocks = re.split(r"\nAgent \d+\s*\n", txt)
    aie = next((b for b in blocks if re.search(r"Device Type:\s*DSP", b)), None)
    if not aie:
        return {}
    out: dict[str, Any] = {"agent_found": True}
    if m := re.search(r"L2:\s*(\d+)\(", aie):
        out["l2_kb"] = int(m.group(1))
    for key, pat in (
        ("cacheline_size", r"Cacheline Size:\s*(\d+)"),
        ("max_clock_mhz", r"Max Clock Freq\. \(MHz\):\s*(\d+)"),
        ("compute_units", r"Compute Unit:\s*(\d+)"),
    ):
        out[key] = _search_int(aie, pat)
    # The tell: these are all zero, which is why this source cannot stand alone.
    out["self_consistent"] = bool(out.get("cacheline_size") or out.get("max_clock_mhz"))
    return out


def git_commit() -> str:
    txt = _run(["git", "rev-parse", "--short", "HEAD"]).strip()
    dirty = _run(["git", "status", "--porcelain"]).strip()
    return f"{txt}{'-dirty' if dirty else ''}" if txt else "unknown"


# ── register ────────────────────────────────────────────────────────────────


def build_register(device: str = "auto") -> list[Claim]:
    """Assemble the claims register from live sources.

    Values come from probes, not from constants in this file, so the register
    re-derives itself on a toolchain or firmware change rather than going stale.
    """
    target = resolve_target(device)
    tc = probe_toolchain(target.key)
    xrt = probe_xrt()
    roc = probe_rocminfo()
    aie1 = tc["aie1_control"]

    claims: list[Claim] = []
    add = claims.append

    # ── H: topology ──
    add(
        Claim(
            "H1",
            "Total columns (1 shim + N compute)",
            xrt.get("total_columns"),
            f"xrt-smi examine ({xrt.get('device_name', '?')})",
            Tier.PROBE,
            Status.CONFIRMED if xrt.get("total_columns") else Status.UNKNOWN,
        )
    )
    add(
        Claim(
            "H2",
            "Compute cores; rows = shim + memtile + core rows",
            {"cores": tc["n_core_tiles"], "cols": tc["columns"], "rows": tc["rows"]},
            "mlir-aie target model",
            Tier.TOOLCHAIN,
            Status.CONFIRMED,
            note=f"{tc['columns']} cols x {tc['rows']} rows; {tc['n_mem_tiles']} memtiles, {tc['n_shim_tiles']} shim",
        )
    )
    add(
        Claim(
            "H3",
            "Program memory per tile",
            None,
            "UG1079 (AIE1) claims 16 KB — wrong generation",
            Tier.WRONG_GEN_DOC,
            Status.WRONG_GEN,
            probe="AM020, then grow core text to link failure",
            note="aie2p observed max_core_text <= 11,200 B (consistent with 16 KB, not proof)",
        )
    )
    add(
        Claim(
            "H4",
            "DPU data-argument slots",
            5,
            "findings R59 (aie2p)",
            Tier.PROBE,
            Status.LIKELY,
            probe="sweep BO count to EINVAL",
            note=f"command-packet ABI; verified indirectly by existing {target.tile_isa} images, explicit sweep still pending",
        )
    )

    # ── M: memory ──
    l1 = tc["local_memory_bytes"]
    add(
        Claim(
            "M1",
            "L1 data memory per core tile",
            l1,
            "mlir-aie AIE2TargetModel::getLocalMemorySize",
            Tier.TOOLCHAIN,
            Status.CONFIRMED,
            note=(
                f"{l1 // 1024} KB. AIE1 control (xcvc1902) reports "
                f"{aie1['local_memory_bytes'] // 1024} KB — that is where the skill's 32 KB came from. "
                "Sets the output-tile area cap, the top lever in the aie2p corpus."
            ),
        )
    )
    for cid, stmt in (
        ("M2", "8 banks x (256 w x 128 b)"),
        ("M3", "128 KB via 3 neighbours + own"),
        ("M4", "3 concurrent ports if different banks"),
    ):
        add(Claim(cid, stmt, None, "UG1079 (AIE1)", Tier.WRONG_GEN_DOC, Status.WRONG_GEN, probe="AM020, then behavioural"))
    mt_total = tc["mem_tile_bytes"] * tc["n_mem_tiles"]
    roc_l2 = roc.get("l2_kb")
    agrees = roc_l2 is not None and roc_l2 * 1024 == mt_total
    add(
        Claim(
            "M5",
            "Memory-tile capacity (aggregate)",
            mt_total,
            "mlir-aie getMemTileSize x n_mem_tiles" + (f"; rocminfo L2={roc_l2} KB agrees" if agrees else ""),
            Tier.TOOLCHAIN,
            Status.CORROBORATED if agrees else Status.CONFIRMED,
            note=(
                f"{tc['mem_tile_bytes'] // 1024} KB x {tc['n_mem_tiles']} = {mt_total // 1024} KB. "
                f"AIE1 control reports {aie1['mem_tile_bytes']} — AIE1 has no memory tiles at all."
            ),
        )
    )
    add(
        Claim(
            "M6",
            "BDs per core tile (DMA descriptor budget)",
            tc.get("bds_per_core"),
            "mlir-aie target model",
            Tier.TOOLCHAIN,
            Status.CONFIRMED,
            note="buildability limit; R118/R119 hit the DMA task budget here",
        )
    )
    add(
        Claim(
            "M7",
            "Locks per core tile",
            tc.get("locks_per_core"),
            "mlir-aie target model",
            Tier.TOOLCHAIN,
            Status.CONFIRMED,
            note="buildability limit; R61 'exhausted tile locks even at FIFO depth 1'",
        )
    )

    # ── B: bandwidth (halo numbers do not transfer) ──
    same_halo = target.key == "npu2" and "halo" in xrt.get("device_name", "").lower()
    bandwidth_status = Status.LIKELY if same_halo else Status.NO_TRANSFER
    add(Claim("B1", "Per-stream feed roof", None, "halo aie2p: 14.4 GB/s", Tier.PROBE, bandwidth_status, probe="C2"))
    add(Claim("B2", "Aggregate feed roof", None, "halo aie2p: 56.5 GB/s @8col", Tier.PROBE, bandwidth_status, probe="C2"))
    add(Claim("B3", "Drain roof / shim channel capacity", None, "R61 (aie2p, qualitative)", Tier.PROBE, Status.UNKNOWN, probe="C5"))

    # ── K: clock and power ──
    add(
        Claim(
            "K1",
            "AIE compute clock",
            xrt.get("npu_clk_max_mhz"),
            f"xrt-smi clock report on {xrt.get('device_name', target.key)}",
            Tier.PROBE,
            Status.CONFIRMED if xrt.get("npu_clk_max_mhz") else Status.UNKNOWN,
            probe="K1: time a loop of known instruction count",
            note="TOP UNKNOWN: the time base for all of t_core; f_H and cyc_mmul are entangled without it",
        )
    )
    add(
        Claim(
            "K2",
            "Peak TOPS",
            xrt.get("npu_tops_max"),
            f"xrt-smi/marketing for {xrt.get('device_name', target.key)}",
            Tier.UNRELIABLE,
            Status.CONFIRMED if xrt.get("npu_tops_max") else Status.UNKNOWN,
            probe="C4, derives from K1",
        )
    )
    add(
        Claim(
            "K3",
            "Local-load alignment penalty (64 B)",
            None,
            "findings R118 (aie2p)",
            Tier.PROBE,
            Status.LIKELY,
            probe="C8",
        )
    )
    add(
        Claim(
            "K4",
            "Power mode (uncontrolled variable)",
            xrt.get("power_mode"),
            "xrt-smi; setting it needs CAP_SYS_ADMIN (sudo password-gated per findings)",
            Tier.PROBE,
            Status.CONFIRMED if xrt.get("power_mode") else Status.UNKNOWN,
            note="record with every row; do not claim clock-invariance across modes",
        )
    )

    # ── X: coherency ──
    add(
        Claim(
            "X1",
            "NPU is not cache coherent",
            True,
            "amdxdna driver documentation",
            Tier.VENDOR_DOC,
            Status.LIKELY,
            probe="X1: explicit-sync omission test",
        )
    )
    add(
        Claim(
            "X2",
            "No usable MALL path",
            None,
            "R56 (aie2p / Strix Halo)",
            Tier.PROBE,
            Status.LIKELY if same_halo else Status.NO_TRANSFER,
            note=("R56 was measured on the same Strix Halo family" if same_halo else "N/A on Phoenix: no MALL exists on this SoC"),
        )
    )
    return claims


def provenance(device: str = "auto") -> dict[str, Any]:
    """Full report: register + raw source dumps + versions, for durable rows."""
    target = resolve_target(device)
    xrt = probe_xrt()
    roc = probe_rocminfo()
    claims = build_register(target.key)
    by_status: dict[str, list[str]] = {}
    for c in claims:
        by_status.setdefault(c.status, []).append(c.id)
    return {
        "device": target.key,
        "tile_isa": target.tile_isa,
        "git_commit": git_commit(),
        "xrt": xrt,
        "rocminfo": roc,
        "toolchain": probe_toolchain(target.key),
        "claims": [c.to_row() for c in claims],
        "summary": by_status,
        "envelope_gaps": [c.id for c in claims if c.status in (Status.UNKNOWN, Status.WRONG_GEN, Status.NO_TRANSFER)],
    }


def render(device: str = "auto") -> str:
    rep = provenance(device)
    lines = [
        f"aiecost H-series provenance — device={rep['device']} git={rep['git_commit']}",
        f"  xrt={rep['xrt'].get('xrt_version', '?')} amdxdna={rep['xrt'].get('amdxdna_version', '?')} "
        f"name={rep['xrt'].get('device_name', '?')} pmode={rep['xrt'].get('power_mode', '?')}",
        "",
        f"{'id':4} {'tier':5} {'status':17} {'value':>10}  statement",
        "-" * 100,
    ]
    for c in rep["claims"]:
        val = c["value"]
        vs = json.dumps(val) if isinstance(val, dict) else ("-" if val is None else str(val))
        lines.append(f"{c['id']:4} T{c['tier']:<4} {c['status']:17} {vs:>10}  {c['statement']}")
        if c["note"]:
            lines.append(f"{'':29} note: {c['note']}")
    lines += ["", "summary:"]
    for status, ids in sorted(rep["summary"].items()):
        lines.append(f"  {status:17} {', '.join(ids)}")
    lines.append("")
    lines.append(f"envelope gaps (model must refuse or narrow): {', '.join(rep['envelope_gaps'])}")
    return "\n".join(lines)
