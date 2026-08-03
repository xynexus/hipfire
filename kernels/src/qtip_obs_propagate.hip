// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// OBS/LDLQ error propagation for the block-sequential QTIP encode.
//
// After block [c0, c1) is quantized, its per-column error must be fed forward to
// every LATER column:
//
//     residual[r, f] -= sum_{c < 256} err[r, c] * L[f, c0 + c]      for f >= c1
//
// This is the dominant CPU cost of `greedy` conditioning — O(m·k²/2) f64 ops per
// tensor, ~6.9e10 at m=2048, k=8192 — and it is a plain rank-256 update, i.e. a
// GEMM. It lives in its own kernel rather than calling gemm_f32_train_accum
// because the destination is a STRIDED sub-matrix (columns c1.. of a [m, k]
// row-major buffer) and that GEMM takes no ldc, so it cannot address one.
//
// Keeping `residual` device-resident across blocks is the point: the host copy
// never has to exist, so the per-block upload/download disappears with it.
//
// Grid: x over future columns (c1..k), y over rows. One thread per (r, f).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void qtip_obs_propagate(
    float* __restrict__ residual,   // [m, k] row-major, updated in place
    const float* __restrict__ err,  // [m, 256] row-major
    const float* __restrict__ l,    // [k, k] row-major; l[f*k + c] = L[f, c]
    int m, int k, int c0, int c1
) {
    const int f = c1 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (f >= k || r >= m) return;

    const float* erow = err + (long long)r * 256;
    const float* lrow = l + (long long)f * k + c0;   // L[f, c0 + c] for c in [0,256)
    float acc = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < 256; ++c) acc += erow[c] * lrow[c];

    residual[(long long)r * k + f] -= acc;
}

// Gather one 256-column block out of the device-resident residual into a
// contiguous [m, 256] buffer, which is what the trellis encoder consumes.
// Without this the host would have to pull the whole [m, k] residual back every
// block (64 MB × k/256 blocks — 2 GB per tensor at k=8192) just to read 2 MB.
extern "C" __global__ __launch_bounds__(256) void qtip_gather_block(
    const float* __restrict__ residual,  // [m, k]
    float* __restrict__ out,             // [m, 256]
    int m, int k, int c0
) {
    const int c = (int)threadIdx.x;      // 0..255
    const int r = (int)blockIdx.x;
    if (r >= m) return;
    out[(long long)r * 256 + c] = residual[(long long)r * k + c0 + c];
}
