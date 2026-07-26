#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-operator fresh-context stability probe (Phase-1 signal at a REAL heavy shape).

Dispatches a compiled aie2p llama operator xclbin (e.g. GEMM_M2048_K2048_N2048,
built by phase2_llama/run_llama.sh into the app's build/ dir) in N fresh hardware
contexts and checks *self-consistency*: the output must be non-zero and
BIT-IDENTICAL across every fresh context. Divergence or all-zeros == the R120
symptom, now tested on a real 2048x2048 bf16 projection instead of the synthetic
add-one kernel (which couldn't reach this BD/output-task weight — it hit the
static BD-ID compile wall). No golden reference is needed: identical-across-
fresh-contexts + non-zero IS the stability signal.

ABI: aiecc kernels take (opcode=3, insts_bo@group_id(1), insts_nbytes,
then the runtime_sequence buffer args @group_id(3..)). Buffer sizes/count come
from the operator's aie.runtime_sequence signature (pass via --bufsizes).

Usage (GEMM M2048xK2048xN2048, 3x [4194304 bf16] = 3x 8388608 bytes, output=arg2):
  PYTHONPATH=/opt/xilinx/xrt/python python3 real_op_probe.py \
    --xclbin <app>/build/GEMM_M2048_K2048_N2048_..._npu2.xclbin \
    --insts  <app>/build/GEMM_M2048_K2048_N2048_..._npu2.bin \
    --bufsizes 8388608,8388608,8388608 --out-index 2 --trials 100
"""
import argparse, hashlib, sys, os

XRT_PY = "/opt/xilinx/xrt/python"


def _pyxrt():
    try:
        import pyxrt
    except ImportError:
        sys.path.insert(0, XRT_PY)
        import pyxrt
    return pyxrt


def run_once(pyxrt, device, xclbin, insts_u32, bufsizes, out_index, fill_u16):
    import numpy as np
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    context = pyxrt.hw_context(device, xclbin.get_uuid())          # <-- fresh context
    kernel = pyxrt.kernel(context, xclbin.get_kernels()[0].get_name())

    insts_bo = pyxrt.bo(device, insts_u32.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    insts_bo.write(insts_u32.tobytes(), 0); insts_bo.sync(TO)

    bos = []
    for i, nbytes in enumerate(bufsizes):
        bo = pyxrt.bo(device, nbytes, pyxrt.bo.host_only, kernel.group_id(3 + i))
        if i == out_index:
            payload = np.zeros(nbytes // 2, dtype=np.uint16)      # zero the output
        else:
            payload = np.full(nbytes // 2, fill_u16, dtype=np.uint16)  # bf16 constant input
        bo.write(payload.tobytes(), 0); bo.sync(TO)
        bos.append(bo)

    state = kernel(3, insts_bo, insts_u32.nbytes, *bos).wait()

    out = bos[out_index]; out.sync(FROM)
    raw = out.read(bufsizes[out_index], 0)
    arr = np.frombuffer(raw, dtype=np.uint16)
    return str(state), (arr != 0).mean(), hashlib.sha256(raw).hexdigest()[:16]


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--bufsizes", required=True, help="comma-separated byte sizes, in arg order")
    ap.add_argument("--out-index", type=int, required=True)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--fill-bf16", default="0x3c00", help="bf16 bits to fill inputs (default 1.0)")
    cfg = ap.parse_args()

    bufsizes = [int(x) for x in cfg.bufsizes.split(",")]
    fill = int(cfg.fill_bf16, 0)
    pyxrt = _pyxrt()
    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(cfg.xclbin)
    device.register_xclbin(xclbin)
    insts_u32 = np.fromfile(cfg.insts, dtype=np.uint32)

    hashes, tally = {}, {"correct": 0, "zeros": 0, "partial": 0, "error": 0}
    first_hash = None
    for t in range(cfg.trials):
        try:
            state, nz, h = run_once(pyxrt, device, xclbin, insts_u32, bufsizes,
                                    cfg.out_index, fill)
            if not state.endswith("COMPLETED"):
                tally["error"] += 1; mark = "E"
            elif nz == 0.0:
                tally["zeros"] += 1; mark = "Z"
            elif nz < 1.0:
                tally["partial"] += 1; mark = "z"
            else:
                tally["correct"] += 1; mark = "."
                first_hash = first_hash or h
            hashes[h] = hashes.get(h, 0) + 1
        except Exception as e:  # a crash is a data point
            tally["error"] += 1; mark = "E"; hashes[f"ERR:{type(e).__name__}"] = 1
        sys.stdout.write(mark); sys.stdout.flush()

    print("\n")
    print(f"=== real-op fresh-context stability: {os.path.basename(cfg.xclbin)} ===")
    print(f"    {cfg.trials} fresh contexts, {len(bufsizes)} BOs {bufsizes}, out=arg{cfg.out_index}")
    for k, v in tally.items():
        if v:
            print(f"    {k:9s}: {v:4d} ({100*v/cfg.trials:.1f}%)")
    distinct = {h: c for h, c in hashes.items() if not h.startswith("ERR:")}
    print(f"    distinct non-error output hashes: {len(distinct)} -> {distinct}")
    stable = tally["zeros"] == 0 and tally["partial"] == 0 and tally["error"] == 0 \
        and len(distinct) == 1
    print(f"\n    VERDICT: {'STABLE (bit-identical + non-zero across all contexts)' if stable else 'UNSTABLE — see tally/hashes above'}")
    return 0 if stable else 2


if __name__ == "__main__":
    sys.exit(main())
