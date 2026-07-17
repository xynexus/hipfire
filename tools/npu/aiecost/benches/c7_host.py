#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C7: host-side pack/deblock cost — c_pack, c_deblock, c_call.

No kernel runs here. This measures only what the host must do around a
dispatch: get bytes into a device-visible buffer and get results back out.

    t_host = c_call + c_pack * B_in + c_deblock * B_out

Why it matters: R64 found a warm aie2p production wrapper was 76.6%
preparation/submit/sync/deblock and only 23.4% device work. If npu1 behaves
similarly, host cost is not a rounding error — it is most of the wrapper.

C1 measured c_call as the wall-minus-device gap of a null dispatch and got an
unstable 48-75 us, so it was recorded NOT admissible. C7 measures the host side
directly instead of by subtraction, which is why it supersedes that estimate.

Stages, timed separately so the model can attribute cost:
  alloc  — XRTTensor construction (BO allocation)  ** SETUP, NOT PER-DISPATCH **
  sync   — .to("npu")   host -> device
  back   — .to("cpu")   device -> host
  numpy  — .numpy()     materialise for the caller

CRITICAL — alloc is not part of t_host. Measured at ~18 ms and *independent of
size* (17.8 ms at 16 KB, 19.4 ms at 4 MB), BO allocation is a one-time setup
cost: real code allocates once and reuses across dispatches, as C1 does. Folding
it into the per-byte pack cost is what made a first cut of this bench report a
negative c_pack (-9 GB/s) — allocation noise swamped the signal. It is reported
and recorded separately because an 18 ms allocation is worth knowing about, but
it must never enter a per-dispatch prediction.

    t_host = c_call + c_pack * B_in + c_deblock * B_out      # sync/back/numpy
    (BO allocation is amortised setup and is excluded)

Usage:
    python -m aiecost.benches.c7_host
    python -m aiecost.benches.c7_host --save
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402


def measure(nbytes: int, reps: int, warmup: int) -> dict:
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    n_elem = nbytes // 4
    src = np.ones((n_elem,), dtype=np.int32)

    def once() -> dict:
        t0 = time.perf_counter()
        t = XRTTensor(src, dtype=np.int32, device="cpu")
        t1 = time.perf_counter()
        t.to("npu")
        t2 = time.perf_counter()
        t.to("cpu")
        t3 = time.perf_counter()
        _ = t.numpy()
        t4 = time.perf_counter()
        return {"alloc": t1 - t0, "sync": t2 - t1, "back": t3 - t2, "numpy": t4 - t3}

    for _ in range(warmup):
        once()

    samples = [once() for _ in range(reps)]
    out = {"nbytes": nbytes}
    for stage in ("alloc", "sync", "back", "numpy"):
        out[stage] = statistics.median(s[stage] for s in samples)
    # alloc is deliberately EXCLUDED: it is amortised setup, not per-dispatch cost.
    out["pack"] = out["sync"]  # host -> device, per dispatch
    out["deblock"] = out["back"] + out["numpy"]  # device -> host, per dispatch
    return out


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    return (sy - b * sx) / n, b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bytes", type=int, nargs="+", default=[4096, 16384, 65536, 262144, 1048576, 4194304])
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    print(f"C7 host pack/deblock: bytes={args.bytes} reps={args.reps}")
    print(f"  {'bytes':>9}  {'alloc':>9}  {'sync':>8}  {'back':>8}  {'numpy':>8}  {'pack':>9}  {'deblock':>9}   (us)")
    rows = []
    for nb in args.bytes:
        r = measure(nb, args.reps, args.warmup)
        rows.append(r)
        print(
            f"  {nb:>9}  {r['alloc'] * 1e6:9.2f}  {r['sync'] * 1e6:8.2f}  {r['back'] * 1e6:8.2f}  "
            f"{r['numpy'] * 1e6:8.2f}  {r['pack'] * 1e6:9.2f}  {r['deblock'] * 1e6:9.2f}"
        )

    pack_fixed, c_pack = fit([(r["nbytes"], r["pack"]) for r in rows])
    deb_fixed, c_deblock = fit([(r["nbytes"], r["deblock"]) for r in rows])

    alloc_med = statistics.median(r["alloc"] for r in rows)
    print("=" * 90)
    print(f"  c_pack    = {c_pack * 1e9:8.4f} ns/byte  ({1 / c_pack / 1e9:6.3f} GB/s)   fixed {pack_fixed * 1e6:8.2f} us")
    print(f"  c_deblock = {c_deblock * 1e9:8.4f} ns/byte  ({1 / c_deblock / 1e9:6.3f} GB/s)   fixed {deb_fixed * 1e6:8.2f} us")
    c_call = pack_fixed + deb_fixed
    print(f"  c_call    = {c_call * 1e6:8.2f} us  (byte-independent host cost per dispatch)")
    print(f"\n  BO alloc  = {alloc_med * 1e3:8.2f} ms  <== SETUP ONLY, size-independent; excluded from t_host.")
    print("              Allocate buffers once and reuse them; per-dispatch allocation would")
    print(f"              cost ~{alloc_med / 155e-6:.0f}x the entire {155:.0f} us device dispatch floor (C1).")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "c_pack": c_pack, "c_deblock": c_deblock, "c_call": c_call}, indent=2))
        print(f"  wrote {args.json}")

    if args.save:
        from aiecost import calib

        key = calib.current_key()
        ev = [
            f"{r['nbytes']:>8} B: pack={r['pack'] * 1e6:8.2f} us deblock={r['deblock'] * 1e6:8.2f} us "
            f"(alloc={r['alloc'] * 1e6:.2f} sync={r['sync'] * 1e6:.2f} back={r['back'] * 1e6:.2f} numpy={r['numpy'] * 1e6:.2f})"
            for r in rows
        ]
        cs = {
            "c_bo_alloc_s": calib.Constant(
                name="c_bo_alloc_s", value=alloc_med, unit="s", bench="C7",
                method="XRTTensor construction (BO allocation), median across the byte sweep",
                admissible=True,
                evidence=ev + [f"size-independent: {rows[0]['alloc'] * 1e3:.1f} ms at {rows[0]['nbytes']} B "
                               f"vs {rows[-1]['alloc'] * 1e3:.1f} ms at {rows[-1]['nbytes']} B"],
                caveats=[
                    "SETUP ONLY — never add this to a per-dispatch prediction. Real code allocates once and reuses.",
                    "A first cut of this bench folded alloc into pack and got a negative c_pack (-9 GB/s); "
                    "size-independence is what proves it is not a per-byte cost.",
                ],
            ),
            "c_pack_s_per_byte": calib.Constant(
                name="c_pack_s_per_byte", value=c_pack, unit="s/byte", bench="C7",
                method=".to('npu') host->device sync, byte sweep slope, medians (BO alloc excluded)",
                admissible=True, evidence=ev,
                caveats=["host->device path only; no kernel runs", "excludes any workload-specific block-layout transform"],
            ),
            "c_deblock_s_per_byte": calib.Constant(
                name="c_deblock_s_per_byte", value=c_deblock, unit="s/byte", bench="C7",
                method=".to('cpu') + .numpy(), byte sweep slope, medians",
                admissible=True, evidence=ev, caveats=["device->host path only"],
            ),
            "c_call_s": calib.Constant(
                name="c_call_s", value=c_call, unit="s", bench="C7",
                method="byte-independent intercept of pack + deblock (direct, not by subtraction)",
                admissible=True, evidence=ev,
                caveats=["supersedes C1's unstable 48-75 us wall-minus-device estimate, which was measured by subtraction"],
            ),
        }
        print(f"  saved -> {calib.save(key, cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
