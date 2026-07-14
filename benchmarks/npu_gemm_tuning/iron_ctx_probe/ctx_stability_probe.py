#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IRON multi-context stability probe — Phase-1 go/no-go for the top-down NPU strategy.

THE QUESTION
------------
The resident hipfire path (R119/R120/R121) hits a stochastic failure: some
*fresh hardware contexts* return the whole output as zeros ("six other fresh
contexts return the known whole-output zero symptom", R120). That instability is
the single thing blocking the fast N128/N1280 schedule from becoming the resident
default. hipfire drives the NPU through direct amdxdna DRM ioctls with hand-rolled
BD scheduling.

This probe asks: does the SAME failure appear when the context is created and
programmed by the IRON/MLIR-AIE + XRT stack instead of by hand?

  * If XRT-managed fresh contexts are STABLE  -> the zeros were self-inflicted by
    our direct-ioctl BD plumbing. Going top-down (let the compiler own context
    setup / BD allocation / program-fit) is very likely to dissolve the symptom.
    => GO on the two-arm plan.
  * If XRT-managed fresh contexts ALSO ZERO out -> the instability is a property
    of fresh amdxdna contexts (driver/firmware), not our plumbing. Top-down won't
    rescue it; pivot to a single-context whole-model graph strategy instead.
    => NO-GO on "just switch to compiler-managed multi-context".

METHOD
------
Build ONE trivial multi-column add-one kernel at the real projection footprint
(default 256 x 768 i32, fanned across --cols columns so several shim/mem DMA BDs
get programmed per context, and streamed as many small tiles so many output-DMA
tasks are issued — the R67/R120 regime). Then run it in N FRESH hardware
contexts and tally {correct, all-zeros(whole), partial-zeros, mismatch, error}.

Each --mode process trial runs in a fresh OS process => fresh pyxrt.device +
fresh pyxrt.hw_context, matching how the resident path sees "fresh contexts".
--mode inproc creates N fresh hw_contexts inside one process (cheaper; isolates
context-fresh from process-fresh).

The kernel body writes out[i,j] = in[i,j] + (col+1): a distinct nonzero constant
per column, so a single stuck column is both detectable and identifiable, and a
zeroed output is unambiguous.

USAGE
-----
  # build once, then run 64 fresh-process contexts:
  python3 ctx_stability_probe.py run --trials 64

  # cheaper in-process sweep across fresh contexts:
  python3 ctx_stability_probe.py run --trials 200 --mode inproc

  # knobs: --cols 8 --rows 256 --width 768 --tile-h 8 --tile-w 256
  python3 ctx_stability_probe.py build          # just compile the xclbin
  python3 ctx_stability_probe.py trial          # one fresh-context run -> JSON

Exit code: 0 if zero contexts observed (stable / GO), 2 if any zero/partial
context observed (unstable / investigate), 1 on harness error.

NOTE ON HARDWARE: uses the NPU exclusively (not the GPU), so it does not take the
hipfire GPU lock. Run it while the GPU is otherwise idle if you want clean energy
numbers, but it does not contend for the GPU lock domain.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
XRT_PY = "/opt/xilinx/xrt/python"  # pyxrt lives here on halo (see AGENTS.local / memory)


# --------------------------------------------------------------------------- #
# config / artifact naming                                                     #
# --------------------------------------------------------------------------- #
def build_dir(cfg) -> Path:
    tag = f"probe_{cfg.cols}c_{cfg.rows}x{cfg.width}_t{cfg.tile_h}x{cfg.tile_w}"
    return HERE / "_build" / tag


def artifacts(cfg):
    d = build_dir(cfg)
    return d, d / "probe.xclbin", d / "probe.insts.bin"


def add_shape_args(p):
    # Two hard aie2p limits shape the defaults:
    #  * a core's 64 KiB local memory must hold the in AND out fifo objects
    #    double-buffered, so the tile object must stay small (8x256 i32 = 8 KiB
    #    fits: 8KiB x 2 buffers x 2 fifos = 32 KiB; 16 KiB objects overflow).
    #  * every extra tile enqueues one fill+drain DMA, and the shim's static
    #    BD-ID pool is small — a few dozen tiles trip "aie.dma_bd Free called on
    #    BD chain with unassigned IDs", the same BD-ID wall the resident R75/R120
    #    path fights. That ceiling is itself a finding.
    # Default: one 8 KiB object per column across 8 columns (~16 shim BDs total).
    p.add_argument("--cols", type=int, default=8, help="parallel columns / workers (<=8)")
    p.add_argument("--rows", type=int, default=8, help="rows PER COLUMN (mult of tile-h)")
    p.add_argument("--width", type=int, default=256, help="matrix width (mult of tile-w)")
    p.add_argument("--tile-h", type=int, default=8, help="tile rows; keep tile <= 8 KiB")
    p.add_argument("--tile-w", type=int, default=256, help="tile cols; keep tile <= 8 KiB")
    p.add_argument("--fill", type=int, default=1000, help="input constant")


# --------------------------------------------------------------------------- #
# BUILD: emit the IRON module and compile it to xclbin + insts                 #
# --------------------------------------------------------------------------- #
def build_module(cfg):
    import numpy as np
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU2
    from aie.iron.controlflow import range_
    from aie.helpers.taplib import TensorTiler2D

    # The DPU regmap exposes only ~5 data-argument slots per context (R59), so
    # every column CANNOT get its own pair of BOs. Instead there is ONE shared
    # input BO and ONE shared output BO (the production ResidentContextBundle
    # shape): all columns' bands are stacked into a [cols*rows, width] tensor and
    # a per-column tap scatters column c to rows [c*rows : (c+1)*rows]. Two data
    # args total, which fits the regmap.
    band_shape = (cfg.rows, cfg.width)                       # one column's band/object
    full_shape = (cfg.cols * cfg.rows, cfg.width)            # shared, all columns stacked
    band_ty = np.ndarray[band_shape, np.dtype[np.int32]]
    full_ty = np.ndarray[full_shape, np.dtype[np.int32]]

    in_fifos, out_fifos, workers = [], [], []
    for c in range(cfg.cols):
        of_in = ObjectFifo(band_ty, name=f"in{c}")
        of_out = ObjectFifo(band_ty, name=f"out{c}")

        def make_body(bias, rows=cfg.rows, width=cfg.width):
            def core_fn(of_in1, of_out1):
                ei = of_in1.acquire(1)
                eo = of_out1.acquire(1)
                for i in range_(rows):
                    for j in range_(width):
                        eo[i, j] = ei[i, j] + bias
                of_in1.release(1)
                of_out1.release(1)
            return core_fn

        workers.append(Worker(make_body(c + 1), fn_args=[of_in.cons(), of_out.prod()]))
        in_fifos.append(of_in)
        out_fifos.append(of_out)

    # simple_tiler over the stacked tensor yields exactly `cols` taps, one per
    # column band (tile == band), so taps[c] selects column c's rows.
    taps = TensorTiler2D.simple_tiler(full_shape, band_shape)
    assert len(taps) == cfg.cols, f"expected {cfg.cols} band taps, got {len(taps)}"

    rt = Runtime()
    with rt.sequence(full_ty, full_ty) as (in_t, out_t):
        for w in workers:
            rt.start(w)
        for c in range(cfg.cols):
            rt.fill(in_fifos[c].prod(), in_t, taps[c])
            rt.drain(out_fifos[c].cons(), out_t, taps[c], wait=(c == cfg.cols - 1))

    return Program(NPU2(), rt).resolve_program()


def do_build(cfg, force=False):
    d, xclbin, insts = artifacts(cfg)
    if xclbin.exists() and insts.exists() and not force:
        print(f"[build] cached: {xclbin}", file=sys.stderr)
        return xclbin, insts
    d.mkdir(parents=True, exist_ok=True)
    from aie.utils.compile.utils import compile_mlir_module

    module = build_module(cfg)
    assert module.operation.verify() is True, "IRON module failed verification"
    print(f"[build] compiling -> {xclbin} (this runs aiecc/peano, ~tens of seconds)",
          file=sys.stderr)
    compile_mlir_module(str(module), insts_path=str(insts), xclbin_path=str(xclbin),
                        work_dir=str(d))
    if not (xclbin.exists() and insts.exists()):
        raise RuntimeError("aiecc reported success but artifacts are missing")
    print("[build] done", file=sys.stderr)
    return xclbin, insts


# --------------------------------------------------------------------------- #
# RUN: one dispatch into a FRESH hw_context via pyxrt                          #
# --------------------------------------------------------------------------- #
def _import_pyxrt():
    try:
        import pyxrt  # noqa
    except ImportError:
        if XRT_PY not in sys.path:
            sys.path.insert(0, XRT_PY)
        import pyxrt  # noqa
    return pyxrt


def run_once(pyxrt, device, xclbin, cfg, insts_u32):
    """Create ONE fresh hw_context, dispatch once, return per-column int32 arrays."""
    import numpy as np

    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    # --- the object under test: a fresh hardware context -------------------- #
    context = pyxrt.hw_context(device, xclbin.get_uuid())
    kname = xclbin.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(context, kname)

    # instruction stream BO (kernel arg slot 1)
    insts_bo = pyxrt.bo(device, insts_u32.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    insts_bo.write(insts_u32.tobytes(), 0)
    insts_bo.sync(TO)

    # ONE shared input + ONE shared output BO (arg slots 3, 4) — the two-BO
    # regmap-legal ABI. All columns stacked into [cols*rows, width].
    full_rows = cfg.cols * cfg.rows
    in_arr = np.full((full_rows, cfg.width), cfg.fill, dtype=np.int32)
    zero = np.zeros((full_rows, cfg.width), dtype=np.int32)

    in_bo = pyxrt.bo(device, in_arr.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
    in_bo.write(in_arr.tobytes(), 0); in_bo.sync(TO)
    out_bo = pyxrt.bo(device, zero.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
    out_bo.write(zero.tobytes(), 0); out_bo.sync(TO)

    run = kernel(3, insts_bo, insts_u32.nbytes, in_bo, out_bo)  # opcode 3 = npu txn
    state = run.wait()

    out_bo.sync(FROM)
    full_out = np.frombuffer(out_bo.read(zero.nbytes, 0), dtype=np.int32) \
        .reshape(full_rows, cfg.width)
    outs = [full_out[c * cfg.rows:(c + 1) * cfg.rows].copy() for c in range(cfg.cols)]
    return str(state), outs


def classify(cfg, state, outs):
    import numpy as np

    ok_state = state.endswith("COMPLETED")
    per_col = []
    zero_cols = 0
    for c, arr in enumerate(outs):
        expect = np.full((cfg.rows, cfg.width), cfg.fill + c + 1, dtype=np.int32)
        if not np.any(arr):
            per_col.append("zero"); zero_cols += 1
        elif np.array_equal(arr, expect):
            per_col.append("correct")
        else:
            per_col.append("mismatch")

    if not ok_state:
        status = "error"
    elif zero_cols == cfg.cols:
        status = "zeros_whole"       # the exact R120 symptom
    elif zero_cols > 0:
        status = "zeros_partial"
    elif all(x == "correct" for x in per_col):
        status = "correct"
    else:
        status = "mismatch"
    return {"status": status, "state": state, "zero_cols": zero_cols, "per_col": per_col}


def do_trial(cfg):
    """One fresh-process trial: build-if-needed, one context, emit one JSON line."""
    import numpy as np

    _, xclbin_path, insts_path = artifacts(cfg)
    if not (xclbin_path.exists() and insts_path.exists()):
        do_build(cfg)
    pyxrt = _import_pyxrt()
    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)
    insts_u32 = np.fromfile(str(insts_path), dtype=np.uint32)
    state, outs = run_once(pyxrt, device, xclbin, cfg, insts_u32)
    print(json.dumps(classify(cfg, state, outs)))
    return 0


# --------------------------------------------------------------------------- #
# ORCHESTRATE: N fresh contexts, tally, verdict                                #
# --------------------------------------------------------------------------- #
def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = XRT_PY + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XILINX_XRT", "/opt/xilinx/xrt")
    return env


def do_run(cfg, trials, mode):
    do_build(cfg)  # compile once up front so per-trial startup is pure dispatch
    tally = {}
    records = []

    def record(rec):
        records.append(rec)
        tally[rec["status"]] = tally.get(rec["status"], 0) + 1
        mark = {"correct": ".", "zeros_whole": "Z", "zeros_partial": "z",
                "mismatch": "x", "error": "E"}.get(rec["status"], "?")
        sys.stdout.write(mark); sys.stdout.flush()

    if mode == "process":
        base = [sys.executable, str(HERE / "ctx_stability_probe.py"), "trial",
                "--cols", str(cfg.cols), "--rows", str(cfg.rows), "--width", str(cfg.width),
                "--tile-h", str(cfg.tile_h), "--tile-w", str(cfg.tile_w),
                "--fill", str(cfg.fill)]
        env = _child_env()
        for _ in range(trials):
            r = subprocess.run(base, capture_output=True, text=True, env=env)
            line = (r.stdout.strip().splitlines() or [""])[-1]
            try:
                record(json.loads(line))
            except json.JSONDecodeError:
                record({"status": "error", "state": f"subproc_rc={r.returncode}",
                        "zero_cols": -1, "per_col": [], "stderr": r.stderr[-400:]})
    else:  # inproc — many fresh hw_contexts, one process
        import numpy as np
        _, xclbin_path, insts_path = artifacts(cfg)
        pyxrt = _import_pyxrt()
        device = pyxrt.device(0)
        xclbin = pyxrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        insts_u32 = np.fromfile(str(insts_path), dtype=np.uint32)
        for _ in range(trials):
            try:
                state, outs = run_once(pyxrt, device, xclbin, cfg, insts_u32)
                record(classify(cfg, state, outs))
            except Exception as e:  # noqa: BLE001 — a crash is itself a data point
                record({"status": "error", "state": repr(e), "zero_cols": -1, "per_col": []})

    sys.stdout.write("\n\n")
    unstable = sum(tally.get(k, 0) for k in ("zeros_whole", "zeros_partial"))
    total = len(records)
    print(f"=== IRON fresh-context stability: {mode} mode, {total} contexts ===")
    print(f"    shape: {cfg.cols} cols x {cfg.rows}x{cfg.width} i32, "
          f"tile {cfg.tile_h}x{cfg.tile_w}")
    for k in ("correct", "zeros_whole", "zeros_partial", "mismatch", "error"):
        if tally.get(k):
            print(f"    {k:14s}: {tally[k]:4d}  ({100*tally[k]/total:.1f}%)")
    zrate = 100 * unstable / total if total else 0.0
    print(f"\n    zero/partial-zero rate: {zrate:.1f}%  ({unstable}/{total})")

    if unstable == 0 and tally.get("error", 0) == 0:
        print("\n    VERDICT: STABLE. XRT-managed fresh contexts do not reproduce the")
        print("             R120 zero symptom => the zeros are self-inflicted by direct")
        print("             amdxdna BD plumbing. GO on the top-down / compiler-managed arm.")
        return 0
    print("\n    VERDICT: UNSTABLE. Fresh XRT contexts reproduce zero/partial-zero output.")
    print("             The instability follows the amdxdna context, not our hand plumbing.")
    print("             Top-down multi-context won't rescue it => prefer a SINGLE-context")
    print("             whole-model graph (cf. llama fused-decode ELF in Phase 2).")
    # dump the first few bad records for triage
    bad = [r for r in records if r["status"] in ("zeros_whole", "zeros_partial", "error")]
    for r in bad[:5]:
        print("      bad:", json.dumps(r))
    return 2


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compile the probe xclbin+insts")
    add_shape_args(b); b.add_argument("--force", action="store_true")

    t = sub.add_parser("trial", help="one fresh-context dispatch -> JSON")
    add_shape_args(t)

    r = sub.add_parser("run", help="N fresh contexts + verdict")
    add_shape_args(r)
    r.add_argument("--trials", type=int, default=64)
    r.add_argument("--mode", choices=["process", "inproc"], default="process")

    cfg = ap.parse_args()
    if cfg.cmd == "build":
        do_build(cfg, force=cfg.force); return 0
    if cfg.cmd == "trial":
        return do_trial(cfg)
    if cfg.cmd == "run":
        return do_run(cfg, cfg.trials, cfg.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
