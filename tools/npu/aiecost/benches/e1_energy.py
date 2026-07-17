#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""E1: energy per dispatch — the tok/joule instrument.

The cost model predicts time. tok/joule needs power, and the two do not
optimise together: a schedule that wins on tok/s can lose on tok/J if it burns
the array to hide a stall. This measures the second axis.

## What can be measured on this box, and what cannot

There is **no NPU power sensor**. `xrt-smi examine -r platform` reports
`Estimated Power : N/A` on npu1, and the AIE array exposes no rail of its own.
So NPU energy is only observable as a **package-level delta**: run a load, and
attribute the rise over idle to it. Two instruments:

  RAPL  /sys/class/powercap/intel-rapl:0/energy_uj  — package-0, an ACCUMULATING
        microjoule counter. Needs root (passwordless sudo on this box). This is
        the right instrument: integrate the counter across a window, never sample
        a wattage and multiply.

  PPT   /sys/class/drm/card*/device/hwmon/hwmon*/power1_average (label "PPT") —
        whole-SoC Package Power Tracking, readable without sudo. Cross-check
        only: it is a rolling average with unclear window, and idle sampling
        showed 3.2-4.2 W of swing on an unloaded box.

Both are **package-wide** on an APU: CPU cores, GPU, and NPU share the rails.
The delta therefore includes the host thread spinning on dispatch submission,
which is not separable here. Report it as package delta attributable to the
workload, never as "NPU power".

## Method

Idle window, then load window, both integrated from the RAPL counter. The load
must be SUSTAINED: a first attempt ran ~2.6 s of dispatches and then sampled a
6 s window that had already gone idle, measuring 0.02 W of nothing. So the load
loop here runs until the window closes, not for a fixed rep count.

Kernel choice is the experiment: `--kernel compute` (K1's resident VMAC chain,
t_core-bound, zero DMA) vs `--kernel feed` (C2's tile stream, t_feed-bound,
trivial compute) vs `--kernel null` (C1's no-op: dispatch machinery only).

## THE CONFOUND — read this before believing any number here

Package power tracks DISPATCH RATE, not NPU work. Measured 8 s windows:

    kernel    dispatch/s   package delta   per-dispatch
    null           4326        10.343 W      2.39 mJ
    feed            352         4.251 W     12.09 mJ
    compute         133         1.082 W      8.15 mJ

A kernel doing NOTHING burns 10.3 W — 10x the compute kernel — because the host
thread submits in a tight loop and RAPL is package-wide. At 133 dispatch/s the
host is mostly blocked waiting on the NPU, so the CPU idles and the delta is
small. **A raw compute-vs-feed power comparison measures the submit loop, not
the kernel.** An earlier draft of this file claimed feed costs 3.9x the power of
compute on exactly that basis; it was measuring the CPU.

First-order correction, subtracting the null baseline at each kernel's own rate
(2.39 mJ/dispatch of host+dispatch cost):

    feed     4.251 - 352*2.39mJ = ~3.41 W of DMA/array
    compute  1.082 - 133*2.39mJ = ~0.76 W of array

which still says moving bytes costs ~4x doing math — but that rests on the host
cost per dispatch being rate-independent, which is unverified. The clean
experiment is a MATCHED-RATE comparison (pad both kernels to the same
dispatch/s), not a subtraction. Until then, treat the ~4x as a hypothesis.

The one number here that is close to safe: the compute kernel at 133
dispatch/s leaves the host mostly idle, so its ~1.08 W delta is the best
available estimate of a near-saturated AIE array, giving ~201 G MACs/J.

Usage:
    python -m aiecost.benches.e1_energy --kernel compute
    python -m aiecost.benches.e1_energy --kernel feed --seconds 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def _find_ppt() -> Path | None:
    for h in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*"):
        lbl = h / "power1_label"
        if lbl.exists() and lbl.read_text().strip() == "PPT":
            return h / "power1_average"
    return None


PPT = _find_ppt()


def read_energy_uj() -> int | None:
    """Accumulating package energy. Needs root; returns None if unavailable."""
    r = subprocess.run(["sudo", "-n", "cat", str(RAPL)], capture_output=True, text=True, check=False)
    try:
        return int(r.stdout.strip())
    except (ValueError, AttributeError):
        return None


def read_ppt_w() -> float | None:
    if not PPT or not PPT.exists():
        return None
    try:
        return int(PPT.read_text().strip()) / 1e6
    except ValueError:
        return None


class Window:
    """Integrate package energy across a wall-clock window."""

    def __enter__(self):
        self.e0 = read_energy_uj()
        self.p0 = read_ppt_w()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t1 = time.perf_counter()
        self.e1 = read_energy_uj()
        self.p1 = read_ppt_w()

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0

    @property
    def watts(self) -> float | None:
        if self.e0 is None or self.e1 is None:
            return None
        # The counter wraps; a negative delta means wrap, not negative energy.
        d = self.e1 - self.e0
        if d < 0:
            return None
        return d / 1e6 / self.seconds

    @property
    def ppt_w(self) -> float | None:
        if self.p0 is None or self.p1 is None:
            return None
        return (self.p0 + self.p1) / 2


def build_kernel(kind: str, cache: Path):
    if kind == "null":
        # C1's near-null kernel: same dispatch/submit machinery, no useful work.
        # Isolates the host+dispatch power that the other kernels also pay, so a
        # compute-vs-feed comparison is not confounded by dispatch RATE (the feed
        # kernel dispatches ~2.6x more often, and RAPL is package-wide).
        from aiecost.benches import c1_dispatch

        built = c1_dispatch.build(1, cache)
        if not built:
            return None
        return {"kind": kind, "built": built, "macs_per_dispatch": 0, "bytes_per_dispatch": 0}
    if kind == "compute":
        from aiecost.benches import c4_mmul

        shape = (4, 8, 8, "int8", "int8")
        iters = 1_600_000
        built = c4_mmul.build(shape, iters, cache)
        if not built:
            return None
        macs = iters * c4_mmul.CHAINS * 4 * 8 * 8
        return {"kind": kind, "built": built, "shape": shape, "macs_per_dispatch": macs, "bytes_per_dispatch": 0}
    from aiecost.benches import c2_feed

    n_tiles, cols = 2048, 4
    built = c2_feed.build(n_tiles, cols, cache)
    if not built:
        return None
    return {
        "kind": kind,
        "built": built,
        "n_tiles": n_tiles,
        "cols": cols,
        "macs_per_dispatch": 0,
        "bytes_per_dispatch": c2_feed.TILE_ELEM * 4 * n_tiles * cols,
    }


def sustained_run(k: dict, seconds: float) -> tuple[int, float]:
    """Dispatch in a tight loop until the window closes. Returns (n, elapsed)."""
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    xclbin, insts = k["built"]
    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    h = hrt.load(kernel)

    if k["kind"] == "null":
        from aiecost.benches.c1_dispatch import N_ELEM

        args = [XRTTensor(np.ones((N_ELEM,), dtype=np.int32), dtype=np.int32, device="cpu") for _ in range(2)]
    elif k["kind"] == "compute":
        from aiecost.benches.c4_mmul import sizes

        s = sizes(k["shape"])
        rng = np.random.default_rng(0)
        args = [
            XRTTensor(rng.integers(-8, 8, size=(2 * s["bytes_A"],), dtype=np.int8), dtype=np.int8, device="cpu"),
            XRTTensor(rng.integers(-8, 8, size=(2 * s["bytes_B"],), dtype=np.int8), dtype=np.int8, device="cpu"),
            XRTTensor((8 + s["size_C"] + 8,), dtype=np.int32, device="cpu"),
        ]
    else:
        from aiecost.benches.c2_feed import ACC_ELEM, TILE_ELEM

        src = XRTTensor(np.ones((TILE_ELEM * k["n_tiles"],), dtype=np.int32), dtype=np.int32, device="cpu")
        args = [src] + [XRTTensor((ACC_ELEM,), dtype=np.int32, device="cpu") for _ in range(k["cols"])]

    hrt.run(h, args)  # warm
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        hrt.run(h, args)
        n += 1
    return n, time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", choices=["compute", "feed", "null"], default="compute")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "e1"))
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    if read_energy_uj() is None:
        print("RAPL package energy unreadable (needs passwordless sudo). Cannot measure energy.")
        return 1

    print(f"E1 energy — kernel={args.kernel} window={args.seconds}s")
    print(f"  RAPL: {RAPL}\n  PPT : {PPT}\n")

    k = build_kernel(args.kernel, Path(args.cache))
    if not k:
        print("build failed")
        return 1

    with Window() as idle:
        time.sleep(args.seconds)
    print(f"  idle : {idle.watts:6.3f} W (RAPL)   {idle.ppt_w or float('nan'):6.3f} W (PPT)")

    n = elapsed = 0
    with Window() as load:
        n, elapsed = sustained_run(k, args.seconds)
    print(f"  load : {load.watts:6.3f} W (RAPL)   {load.ppt_w or float('nan'):6.3f} W (PPT)   {n} dispatches in {elapsed:.2f}s")

    delta = load.watts - idle.watts
    per_disp_s = elapsed / n if n else float("nan")
    j_per_disp = delta * per_disp_s
    print(f"\n  package delta      : {delta:6.3f} W  (attributable to the workload; NOT 'NPU power')")
    print(f"  time per dispatch  : {per_disp_s * 1e6:9.1f} us")
    print(f"  energy per dispatch: {j_per_disp * 1e3:9.4f} mJ (delta basis)")
    print(f"  total-power basis  : {load.watts * per_disp_s * 1e3:9.4f} mJ/dispatch")

    if k["macs_per_dispatch"]:
        gmacs = k["macs_per_dispatch"] / 1e9
        print(f"  {gmacs:.3f} G MACs/dispatch -> {gmacs / j_per_disp:8.1f} G MACs/J (delta basis)")
    if k["bytes_per_dispatch"]:
        gb = k["bytes_per_dispatch"] / 1e9
        print(f"  {gb:.4f} GB/dispatch      -> {gb / j_per_disp:8.2f} GB/J (delta basis)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "kernel": args.kernel, "idle_w": idle.watts, "load_w": load.watts, "delta_w": delta,
            "ppt_idle_w": idle.ppt_w, "ppt_load_w": load.ppt_w,
            "dispatches": n, "elapsed_s": elapsed, "s_per_dispatch": per_disp_s, "j_per_dispatch": j_per_disp,
            "macs_per_dispatch": k["macs_per_dispatch"], "bytes_per_dispatch": k["bytes_per_dispatch"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
