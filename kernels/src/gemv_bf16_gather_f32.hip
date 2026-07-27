// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fine pass of the two-stage lm_head: exact bf16 dot for a shortlist of rows.
// For each selected row k, out[idx[k]] = sum_h bf16(w[idx[k],h]) * bf16(xb[h]).
// Writing to `out[idx[k]]` (not out[k]) fuses the scatter — pre-fill `out` with
// -inf so unselected vocab entries drop out of the downstream softmax/argmax, and
// this kernel overwrites exactly the K shortlisted logits with their exact value.
// `w` = [V, H] bf16 (the tied embed), `xb` = [H] bf16 (activation, pre-cast to
// match the full bf16 lm_head's arithmetic bit-for-bit). gfx1103 wave32, F32 acc.
__device__ __forceinline__ float zbf(unsigned short h) {
    return __uint_as_float((unsigned int)h << 16);
}

__launch_bounds__(32, 16) extern "C" __global__ void gemv_bf16_gather_f32(
    const unsigned short* __restrict__ w,   // [V, H] bf16
    const unsigned int* __restrict__ idx,   // [K] selected row indices
    const unsigned short* __restrict__ xb,  // [H] bf16 activation
    float* __restrict__ out,                // [V] pre-filled with -inf; scatter target
    int K, int H
) {
    const int lane = threadIdx.x; // wave32
    for (int k = blockIdx.x; k < K; k += gridDim.x) {
        const unsigned int row = idx[k];
        if (row == 0xFFFFFFFFu) continue; // sentinel slot (GPU top-K over-allocates then fills)
        const unsigned short* wr = w + (size_t)row * H;
        float sum = 0.0f;
        for (int h = lane; h < H; h += 32) sum += zbf(wr[h]) * zbf(xb[h]);
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) out[row] = sum;
    }
}
