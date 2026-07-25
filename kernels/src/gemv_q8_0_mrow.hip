// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q8_0 GEMV (batch=1), MULTI-ROW register-blocked for memory-level parallelism.
//
// Root cause (docs/perf/zaya-decode-optimization.md EXP-17→22): the single-row
// gemv_q8_0 is LATENCY-BOUND on the huge lm_head (262272×2048) — GPU pinned at
// max sclk/mclk and 100% busy yet only ~50 GB/s (24% of the 211 GB/s the HW
// sustains) because one wave processes one row at a time with too few concurrent
// loads in flight. This kernel makes each wave32 own R rows and interleaves their
// per-block loads at the same block index, giving R× independent load streams →
// R× outstanding memory requests → saturates DRAM latency. Bit-identical math to
// gemv_q8_0 (same per-row Q8_0 dequant + dot), just reordered for MLP.
//
//   A_q8 : [M, K/32 blocks × 34 bytes]   x : [K] f32   y : [M] f32
// Each Q8_0 block = 2-byte f16 scale + 32 int8. Grid-strided over row-groups.

#ifndef ZMROW
#define ZMROW 8
#endif

__launch_bounds__(32, 16) extern "C" __global__ void gemv_q8_0_mrow(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x; // wave32
    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    for (int row0 = blockIdx.x * ZMROW; row0 < M; row0 += gridDim.x * ZMROW) {
        float facc[ZMROW];
        #pragma unroll
        for (int r = 0; r < ZMROW; ++r) facc[r] = 0.0f;

        for (int bi = lane; bi < blocks_per_row; bi += 32) {
            const float* xb = x + bi * 32;     // shared across the R rows
            const size_t boff = (size_t)bi * 34;
            #pragma unroll
            for (int r = 0; r < ZMROW; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const unsigned char* block = A_q8 + (size_t)row * row_bytes + boff;
                const float d = (float)*((const _Float16*)block);
                const signed char* q = (const signed char*)(block + 2);
                float s = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) s += (float)q[j] * xb[j];
                facc[r] += d * s;
            }
        }
        #pragma unroll
        for (int r = 0; r < ZMROW; ++r) {
            float v = facc[r];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
            const int row = row0 + r;
            if (lane == 0 && row < M) y[row] = v;
        }
    }
}
