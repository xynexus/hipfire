#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""G1: GPU energy/throughput, measured exactly like E1 measures the NPU.

The NPU-vs-GPU split decision needs both sides on the same instrument. On this
APU the NPU and the gfx1103 iGPU are on the SAME die and share the same package
rails, so RAPL package-0 sees both. Same counter, same matched-rate padding,
same null subtraction. Anything else (amdgpu's own hwmon vs RAPL) would compare
two different instruments and prove nothing.

Mirrors E1's three kernels: null (launch only), feed (grid-stride buffer read),
compute (resident sdot4 chain). Yields GB/J and G MACs/J directly comparable to
E1's NPU numbers:

    NPU (E1, matched rate): 5.13 GB/J   187.9 G MACs/J

Usage:
    python -m aiecost.benches.g1_gpu --kernel feed --rate 50
    python -m aiecost.benches.g1_gpu --all
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost.benches.e1_energy import Window, read_energy_uj  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "g1_gpu_energy.hip"
LIB = Path.home() / ".cache" / "hipfire-aiecost" / "g1" / "libg1.so"

FEED_BYTES = 256 * 1024 * 1024  # 256 MiB: far beyond any cache, so it is real DDR traffic
COMPUTE_ITERS = 6000  # sized so the kernel fits inside a 50/s slot (20 ms)
THREADS = 256
BLOCKS = 12 * 8  # 12 CUs, 8 waves each
# Each wave32 v_wmma_i32_16x16x16_iu8 does 16*16*16 = 4096 MACs; 4 chains per
# iteration; waves = BLOCKS * THREADS/32.
WMMA_MACS = 16 * 16 * 16
CHAINS = 4
WAVES = BLOCKS * THREADS // 32


def build() -> Path:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    if LIB.exists() and LIB.stat().st_mtime > SRC.stat().st_mtime:
        return LIB
    cmd = ["/opt/rocm/bin/hipcc", "--offload-arch=gfx1103", "-O3", "-fPIC", "-shared", str(SRC), "-o", str(LIB)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not LIB.exists():
        print("hipcc failed:\n" + r.stderr[-2000:])
        raise SystemExit(1)
    return LIB


class Hip:
    """Minimal HIP runtime binding — enough to launch the three kernels."""

    def __init__(self):
        self.rt = ctypes.CDLL("libamdhip64.so")
        self.rt.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.rt.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.rt.hipDeviceSynchronize.argtypes = []
        self.rt.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.rt.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.rt.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        p = ctypes.c_void_p()
        if self.rt.hipMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)) != 0:
            raise RuntimeError(f"hipMalloc({nbytes}) failed")
        self.rt.hipMemset(p, 1, ctypes.c_size_t(nbytes))
        return p

    def sync(self):
        self.rt.hipDeviceSynchronize()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", choices=["null", "feed", "compute"], default="feed")
    p.add_argument("--all", action="store_true", help="run all three and report GB/J + G MACs/J")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=50.0, help="matched dispatch/s (0 = free-run; see E1's confound)")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    if read_energy_uj() is None:
        print("RAPL unreadable (needs passwordless sudo).")
        return 1

    build()
    # ctypes + hipModuleLaunchKernel is fiddly for a shared lib; use hipcc's
    # host-side symbols directly via a tiny C ABI instead.
    lib = ctypes.CDLL(str(LIB))
    kinds = ["null", "feed", "compute"] if args.all else [args.kernel]
    out = {}

    for kind in kinds:
        fn = getattr(lib, f"g1_launch_{kind}", None)
        if fn is None:
            print(f"launcher g1_launch_{kind} not exported — see g1_gpu_energy.hip")
            return 1
        fn.restype = ctypes.c_double

        with Window() as idle:
            time.sleep(args.seconds)
        with Window() as load:
            n = fn(ctypes.c_double(args.seconds), ctypes.c_double(args.rate))
        delta = load.watts - idle.watts
        # Use the ACTUAL rate, never the target: a kernel that overruns its slot
        # silently runs slower, and dividing by the target inflates per-launch
        # energy while breaking the null subtraction (null was measured at the
        # target rate). A first run had compute at 311/500 launches and would
        # have reported a number ~1.6x wrong.
        actual = n / load.seconds if load.seconds else 0.0
        met = (not args.rate) or actual >= args.rate * 0.95
        per = delta / actual if actual else 0.0
        out[kind] = {"idle_w": idle.watts, "load_w": load.watts, "delta_w": delta, "launches": n,
                     "target_rate": args.rate, "actual_rate": actual, "rate_met": met, "j_per_launch": per}
        flag = "" if met else f"  !! RATE MISSED ({actual:.1f}/s vs {args.rate:.0f}/s target)"
        print(f"  {kind:8s} idle {idle.watts:5.3f} W  load {load.watts:5.3f} W  delta {delta:6.3f} W  "
              f"{n:.0f} launches @ {actual:.1f}/s -> {per * 1e3:7.3f} mJ/launch{flag}")

    if args.all and "null" in out:
        nullw = out["null"]["delta_w"]
        bad = [k for k, v in out.items() if not v["rate_met"]]
        if bad:
            print(f"\n  !! {bad} missed the rate target — null subtraction is INVALID for them")
            print("     (host cost only cancels when every kernel submits at the same rate). Lower --rate.")
        print()
        # NPU references come from the calibration set, never hardcoded: the
        # compute figure moved 187.9 -> ~929 G MACs/J once the NPU side ran on
        # all 16 cores instead of 1, which REVERSED the verdict. A literal here
        # would have kept printing "GPU 2.31x NPU" long after it became false.
        from aiecost import calib as _calib

        _c = _calib.load(_calib.current_key())
        npu_feed = _c["gpu_ref_npu_gb_per_j"].value if "gpu_ref_npu_gb_per_j" in _c else 5.13
        npu_comp = _c["j_per_mac_int8_16core_gmacs_j"].value if "j_per_mac_int8_16core_gmacs_j" in _c else None
        if npu_comp is None:
            print("  (no 16-core NPU compute reference in calib — compute verdict suppressed)")
        for kind, unit, work, npu_ref in (("feed", "GB/J", FEED_BYTES / 1e9, npu_feed),
                                          ("compute", "G MACs/J", COMPUTE_ITERS * CHAINS * WMMA_MACS * WAVES / 1e9, npu_comp)):
            if npu_ref is None:
                continue
            if kind not in out or not out[kind]["rate_met"] or not out["null"]["rate_met"]:
                continue
            mj = (out[kind]["delta_w"] - nullw) / out[kind]["actual_rate"]
            val = work / mj
            ratio = val / npu_ref
            verdict = f"GPU {ratio:.2f}x NPU" if ratio >= 1 else f"NPU {1 / ratio:.2f}x GPU"
            print(f"  {kind:8s} marginal {out[kind]['delta_w'] - nullw:6.3f} W -> {val:7.2f} {unit:9s} "
                  f"(NPU E1: {npu_ref}) => {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
