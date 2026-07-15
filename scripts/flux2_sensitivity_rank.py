#!/usr/bin/env python3
"""Rank per-role quantization sensitivity from ablation traces.

Reads the step-1 model velocity (`velocity_001.bin`, HFDT format) from a bf16
baseline run and from each role-ablated run (that role's tensors forced to
low-bit fold via HIPFIRE_DIFFUSION_ABLATE), and reports the relative L2 change
in the velocity — a first-forward, chaos-free measure of how much quantizing
that role perturbs the model. Low = safe to quantize; high = keep high-precision.

    python scripts/flux2_sensitivity_rank.py <baseline_dir> <label>:<dir> [<label>:<dir> ...]
"""

import struct
import sys
from pathlib import Path

import numpy as np


def load_velocity(d: Path) -> np.ndarray:
    p = Path(d) / "velocity_001.bin"
    raw = p.read_bytes()
    if raw[:4] != b"HFDT":
        raise ValueError(f"{p}: bad magic")
    (rank,) = struct.unpack_from("<I", raw, 4)
    off = 8 + rank * 4
    return np.frombuffer(raw, dtype="<f4", offset=off).astype(np.float64)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    base = load_velocity(sys.argv[1])
    base_norm = float(np.linalg.norm(base)) or 1.0
    rows = []
    for arg in sys.argv[2:]:
        label, _, d = arg.partition(":")
        v = load_velocity(d)
        if v.shape != base.shape:
            print(f"warning: {label} shape {v.shape} != baseline {base.shape}", file=sys.stderr)
            continue
        diff = v - base
        rel_l2 = float(np.linalg.norm(diff)) / base_norm
        cos = float(np.dot(v, base) / (np.linalg.norm(v) * base_norm + 1e-30))
        rows.append((label, rel_l2, cos))

    rows.sort(key=lambda r: r[1])  # least sensitive first
    print(f"{'role':>22}  {'vel rel-L2':>10}  {'vel cos':>9}   sensitivity (low = safe to quantize)")
    print("-" * 72)
    for label, rel, cos in rows:
        bar = "#" * min(40, int(rel * 100))
        print(f"{label:>22}  {rel:>10.5f}  {cos:>9.5f}   {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
