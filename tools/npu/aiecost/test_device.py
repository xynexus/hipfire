from __future__ import annotations

from aiecost.device import parse_xrt_reports


def test_parse_xrt_reports_uses_device_and_xrt_sections() -> None:
    platform = """
[0000:c5:00.1] : NPU Strix Halo
Platform
  Name                   : NPU Strix Halo
  Power Mode             : default
  Total Columns          : 8
"""
    system = """
System Configuration
  BIOS Version          : 1.04
XRT
  Version              : 2.25.0
  amdxdna Version      : 2.25.0_20260601, 627cee46
  NPU Firmware Version : 1.1.2.65
Device(s) Present
|[0000:c5:00.1]  |NPU Strix Halo  |aie2p         |6x8       |
"""

    assert parse_xrt_reports(platform, system) == {
        "device_name": "NPU Strix Halo",
        "architecture": "aie2p",
        "topology": "6x8",
        "total_columns": 8,
        "power_mode": "default",
        "npu_clk_max_mhz": None,
        "npu_tops_max": None,
        "xrt_version": "2.25.0",
        "amdxdna_version": "2.25.0_20260601, 627cee46",
        "firmware_version": "1.1.2.65",
    }
