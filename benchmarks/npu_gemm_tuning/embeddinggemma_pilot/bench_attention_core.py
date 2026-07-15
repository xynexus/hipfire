#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the EmbeddingGemma attention CORE at our shape, bf16 (IRON operators).

STEEL is a fused bf16 attention for XDNA, but it is tuned for LONG causal
sequences (>=2048). This measures the same primitives STEEL fuses — Q@K^T,
softmax, P@V — at OUR operating point (M256, head_dim=256, GQA 3 q-heads : 1
kv-head, bidirectional/dense) to get the raw NPU compute cost per attention and
compare it to the 9.14 ms/layer the amdxdna-direct resident attention stage
spends. This is the bf16 baseline; the int8 (A8xW8) and KVarN-K (A8xW4) upgrades
are measured separately via the r11 kernel.

Dispatch is per-op via XRT (not fused), so the assembled number is an UPPER bound
on a fused STEEL-style design — but the per-op ms isolates the compute floor and
is the apples-to-apples base for the int8 ratio.
"""
import os, time, numpy as np, ml_dtypes
from iron.common.context import AIEContext          # aie/iron before torch
from iron.operators.gemm.op import GEMM
from iron.operators.softmax.op import Softmax
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import torch

BF16 = ml_dtypes.bfloat16
M, NH, NKV, HD = 256, 3, 1, 256
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = 5

def fill(t, a):
    t.torch_view().copy_(torch.from_numpy(
        np.ascontiguousarray(a).reshape(-1).astype(BF16).view(np.uint16)).view(torch.bfloat16))
def T(n):
    x = XRTTensor((int(n),), dtype=BF16); fill(x, np.zeros(int(n), np.float32)); return x

rng = np.random.default_rng(0)
print(f"compiling GEMM(256^3) + Softmax(256x256) at M={M} HD={HD} GQA {NH}:{NKV} ...")
g = dict(M=M, K=HD, N=HD, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=4)
qkt = GEMM(**g, b_col_maj=True, context=AIEContext()).compile().get_callable()   # Q@K^T
pv  = GEMM(**g, b_col_maj=False, context=AIEContext()).compile().get_callable()  # P@V
sm  = Softmax(rows=M, cols=HD, num_aie_columns=8, context=AIEContext()).compile().get_callable()

Q = T(M*HD); fill(Q, rng.standard_normal((M,HD)))
K = T(M*HD); fill(K, rng.standard_normal((M,HD)))
V = T(M*HD); fill(V, rng.standard_normal((M,HD)))
S = T(M*HD); P = T(M*HD); C = T(M*HD)

def time_op(fn, iters):
    for _ in range(WARMUP): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e3   # ms/call

def sync():  # force NPU completion by reading the last output back to host
    _ = C.to_torch().float().cpu().numpy()

qkt_ms = time_op(lambda: qkt(Q, K, S), ITERS)
sm_ms  = time_op(lambda: sm(S, P), ITERS)
pv_ms  = time_op(lambda: pv(P, V, C), ITERS)

# blocking per-op: each call followed by a readback barrier (rules out async enqueue)
def blocking(fn):
    fn(); sync()
qkt_b = time_op(lambda: blocking(lambda: qkt(Q, K, S)), ITERS)
pv_b  = time_op(lambda: blocking(lambda: pv(P, V, C)), ITERS)

# full assembled GQA core, ONE barrier per attention (the honest wall time)
def full_core():
    for _ in range(NH):
        qkt(Q, K, S); sm(S, P); pv(P, V, C)
    sync()
core_barrier_ms = time_op(full_core, ITERS)

# assembled attention core: per q-head (Q@K^T + softmax + P@V), 3 heads (GQA rep 3)
core_ms = NH * (qkt_ms + sm_ms + pv_ms)
print(f"\n--- per-op (bf16, XRT dispatch), mean of {ITERS} ---")
print(f"  Q@K^T  256x256x256   {qkt_ms:7.3f} ms")
print(f"  softmax 256x256      {sm_ms:7.3f} ms")
print(f"  P@V    256x256x256   {pv_ms:7.3f} ms")
print(f"\n--- blocking (readback barrier per call) ---")
print(f"  Q@K^T + barrier      {qkt_b:7.3f} ms   (async skew if >> {qkt_ms:.3f})")
print(f"  P@V   + barrier      {pv_b:7.3f} ms")
print(f"\n  full GQA core, 1 barrier/attention = {core_barrier_ms:7.3f} ms  (honest wall time, unfused)")
print(f"  sum-of-per-op estimate             = {core_ms:7.3f} ms")
print(f"  resident attention stage (amdxdna-direct) = 9.140 ms  (NOTE: includes QKV+O projections + 2 norms + GPU pack/sync)")
print(f"  6x GEMM(256^3) MACs = {6*256**3/1e6:.1f} MMAC -> at bf16 this is the compute the fused core must beat")
