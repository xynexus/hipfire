#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""P1: CPU throughput and energy — the third leg of the NPU/GPU/CPU split.

Same package, same rails, same RAPL counter as E1 (NPU) and G1 (GPU), so all
three are directly comparable.

The CPU is the *easy* one to measure, for a reason worth stating: E1's whole
confound was that package power tracked the HOST submit loop rather than the
device. Here the host IS the workload, so there is nothing to separate and no
rate-matching to do. Idle is subtracted and that is the whole story.

    NPU (E1): 30.8 GB/s · 5.13 GB/J   |  7.4 TOPS · 938 G MACs/J
    GPU (G1): 79.5 GB/s · 3.73 GB/J   | 15.1 TOPS · 432 G MACs/J

Usage:
    python -m aiecost.benches.p1_cpu --all
    python -m aiecost.benches.p1_cpu --kernel feed --threads 16
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost.benches.e1_energy import Window, read_energy_uj  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "p1_cpu.c"
BIN = Path.home() / ".cache" / "hipfire-aiecost" / "p1" / "p1_cpu"


def build() -> Path:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    if BIN.exists() and BIN.stat().st_mtime > SRC.stat().st_mtime:
        return BIN
    cmd = ["gcc", "-O3", "-march=native", "-mavx512vnni", "-fopenmp", str(SRC), "-o", str(BIN)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not BIN.exists():
        print("gcc failed:\n" + r.stderr[-1500:])
        raise SystemExit(1)
    return BIN


def run_one(kind: str, seconds: float, threads: int) -> dict:
    with Window() as idle:
        time.sleep(seconds)
    with Window() as load:
        r = subprocess.run([str(BIN), kind, str(seconds), str(threads)],
                           capture_output=True, text=True, check=False)
    out = r.stdout.strip()
    kv = dict(re.findall(r"(\w+)=([\d.eE+-]+)", out))
    delta = load.watts - idle.watts
    res = {"kernel": kind, "threads": threads, "idle_w": idle.watts, "load_w": load.watts,
           "delta_w": delta, "elapsed": float(kv.get("elapsed", 0))}
    if kind == "feed":
        by = float(kv.get("bytes", 0))
        res["bytes"] = by
        res["gb_s"] = by / res["elapsed"] / 1e9 if res["elapsed"] else 0
        res["gb_per_j"] = (by / 1e9) / (delta * res["elapsed"]) if delta > 0 and res["elapsed"] else 0
    else:
        macs = float(kv.get("macs", 0))
        res["macs"] = macs
        res["macs_s"] = macs / res["elapsed"] if res["elapsed"] else 0
        res["gmacs_per_j"] = (macs / 1e9) / (delta * res["elapsed"]) if delta > 0 and res["elapsed"] else 0
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", choices=["feed", "compute"], default="feed")
    p.add_argument("--all", action="store_true")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    if read_energy_uj() is None:
        print("RAPL unreadable (needs passwordless sudo).")
        return 1
    build()

    kinds = ["feed", "compute"] if args.all else [args.kernel]
    out = {}
    for kind in kinds:
        r = run_one(kind, args.seconds, args.threads)
        out[kind] = r
        if kind == "feed":
            print(f"  feed     idle {r['idle_w']:5.2f} W  load {r['load_w']:6.2f} W  delta {r['delta_w']:6.2f} W  "
                  f"-> {r['gb_s']:6.2f} GB/s   {r['gb_per_j']:5.2f} GB/J   (NPU 30.8 / 5.13, GPU 79.5 / 3.73)")
        else:
            print(f"  compute  idle {r['idle_w']:5.2f} W  load {r['load_w']:6.2f} W  delta {r['delta_w']:6.2f} W  "
                  f"-> {r['macs_s'] / 1e12:6.2f} T MACs/s  {r['gmacs_per_j']:6.1f} G MACs/J   "
                  f"(NPU 3.72 / 938, GPU 7.55 / 432)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
