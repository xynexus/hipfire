#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M-sweep: is one NPU GEMM dispatch fixed-cost-dominated at small M (batch=1)
and compute-bound at large M (batched)? Fix the weight shape (K=768, N=768, a
projection-like GEMM) and sweep M (= batch x seq rows) over {256..4096}. Time each
with a completion barrier (true wall time). If per-dispatch time is flat at small M
then grows linearly, the intercept is the fixed per-dispatch floor and the slope is
the real per-row compute — which is exactly what batching amortizes.
"""
import os, time, numpy as np, ml_dtypes
from iron.common.context import AIEContext
from iron.operators.gemm.op import GEMM
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

BF16 = ml_dtypes.bfloat16
K, N = 768, 768
MS = [int(x) for x in os.environ.get("MS", "256,512,1024,2048,4096").split(",")]
ITERS, WARMUP = int(os.environ.get("ITERS", "30")), 5
rng = np.random.default_rng(0)

def fill(t, a):
    t.torch_view().copy_(torch.from_numpy(
        np.ascontiguousarray(a).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))
def T(n, val=None):
    x = XRTTensor((int(n),), dtype=BF16)
    fill(x, rng.standard_normal(int(n)) if val is None else np.zeros(int(n)))
    return x

print(f"M-sweep  K={K} N={N}  (blocking wall time, mean of {ITERS})")
print(f"{'M':>6} {'ms/call':>9} {'ms/256rows':>11} {'GMAC/s':>9} {'TOPS':>6}")
rows_ms = {}
for M in MS:
    op = GEMM(M=M, K=K, N=N, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=4,
              b_col_maj=True, context=AIEContext()).compile().get_callable()
    A, B, C = T(M*K), T(N*K), T(M*N, val=0)
    def call():
        op(A, B, C); _ = C.to_torch().float().cpu().numpy()   # barrier
    for _ in range(WARMUP): call()
    t0 = time.perf_counter()
    for _ in range(ITERS): call()
    ms = (time.perf_counter() - t0) / ITERS * 1e3
    rows_ms[M] = ms
    gmac = M*K*N / (ms*1e-3) / 1e9
    print(f"{M:>6} {ms:>9.3f} {ms*256/M:>11.3f} {gmac:>9.1f} {2*gmac/1000:>6.2f}")

# linear fit ms = floor + slope*M  (two-point from ends), report amortization
lo, hi = MS[0], MS[-1]
slope = (rows_ms[hi] - rows_ms[lo]) / (hi - lo)
floor = rows_ms[lo] - slope*lo
print(f"\nfit: ms(M) ~= {floor:.3f} (fixed per-dispatch floor) + {slope*1000:.4f} us/row")
print(f"  at M=256 (batch~1):  floor is {100*floor/rows_ms[lo]:.0f}% of the {rows_ms[lo]:.2f} ms")
print(f"  at M={hi} (batched):  floor is {100*floor/rows_ms[hi]:.0f}% of the {rows_ms[hi]:.2f} ms")
print(f"  per-row cost drops {rows_ms[lo]/256 / (rows_ms[hi]/hi):.1f}x from M=256 to M={hi}")
