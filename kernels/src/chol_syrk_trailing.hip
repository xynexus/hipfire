// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Right-looking blocked Cholesky: the trailing-submatrix update.
//
//     A[i, j] -= sum_{t < jb} L[i, j0+t] * L[j, j0+t]     for i >= j >= j0+jb
//
// This is where essentially all of the O(k^3) lives. The panel factorization and
// TRSM are O(k * jb^2) and stay on the host in f64, where they are cheap and
// where precision actually matters — an ill-conditioned Hessian breaks down in
// the panel, not in the trailing update.
//
// MEASURED motivation: faer's CPU Cholesky achieves ~20 GFLOP/s here with all 32
// threads (it is NOT misconfigured — probed, and forcing Par::rayon changes
// nothing). Cholesky scales badly at these sizes because the panel serialises,
// so ~400s of a ~600s quantize sits in this routine.
//
// Only the LOWER triangle is touched: j > i returns immediately. Writing the
// upper half would be wasted work and would break the symmetry the next panel
// download assumes.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void chol_syrk_trailing(
    float* __restrict__ a,   // [k, k] row-major, updated in place
    int k, int j0, int jb
) {
    const int start = j0 + jb;
    const int i = start + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= k || j >= k || j > i) return;

    const float* li = a + (long long)i * k + j0;
    const float* lj = a + (long long)j * k + j0;
    float acc = 0.0f;
    for (int t = 0; t < jb; ++t) acc += li[t] * lj[t];
    a[(long long)i * k + j] -= acc;
}

// Gather the panel (rows j0..k, columns j0..j0+jb) into a contiguous
// [rows, jb] buffer. Without this the host must pull the whole [k, k] matrix
// back every block just to read a k x jb slice — at k=8192 that is 256 MB per
// block against the ~4 MB actually needed, and it is why the first draft
// plateaued at 27 GF/s.
extern "C" __global__ __launch_bounds__(256) void chol_panel_gather(
    const float* __restrict__ a, float* __restrict__ out,
    int k, int j0, int jb, int rows
) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (c >= jb || r >= rows) return;
    out[(long long)r * jb + c] = a[(long long)(j0 + r) * k + j0 + c];
}

// Scatter the factored panel back, LOWER triangle only (c <= r within the
// diagonal block; every column for rows below it). Writing the upper half would
// destroy the symmetry the next gather relies on.
extern "C" __global__ __launch_bounds__(256) void chol_panel_scatter(
    float* __restrict__ a, const float* __restrict__ src,
    int k, int j0, int jb, int rows
) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (c >= jb || r >= rows || c > r) return;
    a[(long long)(j0 + r) * k + j0 + c] = src[(long long)r * jb + c];
}
