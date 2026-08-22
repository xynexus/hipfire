// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Diagnostic for the "skip the hi pass" idea. The exact W4A8 GEMM splits every
// int8 activation as x = 16*x_hi + x_lo and spends TWO iu4 WMMA passes on it.
// x_hi is zero exactly when |x| < 16, so an H-pass WMMA is a no-op iff EVERY
// value in its 16-col x 16-K fragment satisfies that. This records max|x| per
// fragment so the skippable fraction can be counted on the host, at both
// fragment granularity and the BN=128 column-block granularity a wave actually
// needs (a wave issues WNt=4 column tiles together, so a per-fragment decision
// would diverge).
//
//   X    : [N, K] int8 activation, row-major
//   OUT  : [N/16, K/16] uint8, max|x| of each fragment (saturated at 255)
#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(256) extern "C" __global__ void act_hipass_tilemax(
    const int8_t* __restrict__ X, uint8_t* __restrict__ OUT, int N, int K) {
    const int tiles_k = K / 16;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (N / 16) * tiles_k) return;
    const int tn = id / tiles_k, tk = id % tiles_k;
    int m = 0;
    for (int r = 0; r < 16; r++) {
        const int8_t* p = X + (long long)(tn * 16 + r) * K + tk * 16;
        for (int c = 0; c < 16; c++) {
            int v = p[c];
            v = v < 0 ? -v : v;
            if (v > m) m = v;
        }
    }
    OUT[id] = (uint8_t)(m > 255 ? 255 : m);
}
