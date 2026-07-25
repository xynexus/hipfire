// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q8_0 GEMV v4 (batch=1): one wave32 per output row. Each lane owns WHOLE Q8_0
// blocks (block = 2-byte f16 scale + 32 int8), striding by 32 across the row, and
// loads each block's 32 contiguous int8 + its scale ONCE. The previous version had
// every lane read the same f16 scale per block (redundant) and did strided
// byte-per-thread loads; owning whole contiguous blocks lets the compiler widen the
// loads and drops the redundant scale traffic → ~1.9× bandwidth on gfx1151
// (117 → 222 GB/s microbench), bit-exact.
//
//   A_q8 : [M, K/32 blocks × 34 bytes]   x : [K] f32   y : [M] f32
// Grid-strided: each wave32 handles MULTIPLE rows (row += gridDim.x). At huge M
// (lm_head, vocab=262272) launching one workgroup per row means the command
// processor dispatches 262k workgroups — dominant even when the per-row compute is
// cheap. A capped grid (~a few thousand blocks) that strides over rows cuts that
// dispatch cost while keeping the memory access pattern identical.
__launch_bounds__(32, 20) extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x;                 // wave32
    const int blocks_per_row = K / 32;
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;
        float facc = 0.0f;
        for (int bi = lane; bi < blocks_per_row; bi += 32) {
            const unsigned char* block = row_data + (size_t)bi * 34;
            const float d = (float)*((const _Float16*)block); // scale, loaded once
            const signed char* q = (const signed char*)(block + 2);
            const float* xb = x + bi * 32;
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) s += (float)q[j] * xb[j];
            facc += d * s;
        }
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
        if (lane == 0) y[row] = facc;
    }
}
