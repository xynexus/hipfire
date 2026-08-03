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
