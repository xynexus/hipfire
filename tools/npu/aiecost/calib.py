#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Calibration constants, version-keyed and provenance-carrying.

Per docs/npu/aie2-cost-model-plan.md §4: constants are keyed on
device + XRT + firmware so drift is detectable, and the model refuses to
predict on a key it has no calibration for rather than extrapolating.

Every constant records how it was measured and how strongly it is believed.
A constant with `admissible=False` is visible but must not be used to predict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALIB_DIR = Path(__file__).resolve().parent / "calib"


@dataclass
class Constant:
    """One calibrated value with its evidence."""

    name: str
    value: float
    unit: str
    bench: str  # which bench produced it (K1, C1, ...)
    method: str  # how, in one line
    admissible: bool  # may the model use it?
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    measured_utc: str = ""

    def __post_init__(self):
        if not self.measured_utc:
            self.measured_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")


def key_for(device: str, xrt: str, firmware: str) -> str:
    """Version key. A change in any component invalidates the calibration."""
    return f"{device}_xrt{xrt}_fw{firmware}".replace("/", "-").replace(" ", "")


def path_for(key: str) -> Path:
    return CALIB_DIR / f"{key}.json"


def load(key: str) -> dict[str, Constant]:
    p = path_for(key)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: Constant(**v) for k, v in raw.get("constants", {}).items()}


def save(key: str, constants: dict[str, Constant], meta: dict[str, Any] | None = None) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(key)
    existing = {}
    if p.exists():
        existing = json.loads(p.read_text())
    merged = {**existing.get("constants", {}), **{k: asdict(v) for k, v in constants.items()}}
    p.write_text(
        json.dumps(
            {
                "key": key,
                "meta": {**existing.get("meta", {}), **(meta or {})},
                "constants": merged,
            },
            indent=2,
        )
    )
    return p


def current_key() -> str:
    """Version key for the device in front of us."""
    from . import device as dev

    xrt = dev.probe_xrt()
    fw = _firmware()
    return key_for(xrt.get("device_name", "unknown"), xrt.get("xrt_version", "unknown"), fw)


def _firmware() -> str:
    p = Path("/sys/class/accel/accel0/device/fw_version")
    try:
        return p.read_text().strip()
    except Exception:
        return "unknown"
