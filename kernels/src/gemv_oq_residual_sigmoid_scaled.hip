// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Opus-Quant shared-expert DOWN projection: N-batched decode GEMV with a fused
// sigmoid-scaled residual add — the OQ sibling of
// `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`. Used by the Qwen3.5 MoE
// shared-expert down accumulator:
//
//     y_batch[token, row] += sigmoid(c_batch[token]) * (W_down[row] · x_batch[token])
//
// Grid = (M, N, 1); one wave32 owns (row, token), so the final `+=` into the
// unique y_batch element is race-free (no atomics). `c_batch[token]` is the
// shared-expert gate logit; sigmoid is applied inline (one fewer launch, matching
// the MQ4 kernel). ZERO LDS.
//
// DECODE numerics (B=1, no WMMA): the int4/int8 weight is dequantized inline and
// multiplied by the FULL-PRECISION f32 activation (W4A16 / W8A16) — exactly like
// the mq4 decode GEMV and `gemv_oq{4,8}_grouped`. This is numerically MORE
// accurate than the W4A4/W8A8 routed path and skips the per-projection activation
// quantize. The OQ weight is SYMMETRIC (no zero point): w = qw * scale.
//
// Weights use the DENSE OQ layout the shared expert loads via load_weight_tensor
// (NOT the indexed-MoE block layout):
//   OQ4: W = [M, K/2] packed signed int4 ; Ws = [M, K/256] f32 group scales
//   OQ8: W = [M, K]   signed int8         ; Ws = [M, K/256] f32 group scales
// The caller passes Ws as a sub-offset view into the same combined buffer.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq4g256_residual_sigmoid_scaled_gpu_batched(
    const uint8_t* __restrict__ W,        // [M, K/2] packed signed int4
    const float*   __restrict__ Ws,       // [M, K/256] f32 group scales
    const float*   __restrict__ X_batch,  // [N, K] f32 rotated activations
    float*         __restrict__ Y_batch,  // [N, M] f32 residual stream (in place)
    const float*   __restrict__ c_batch,  // [N] shared-gate logit (pre-sigmoid)
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int gbytes = group / 2;
    const uint8_t* w_row = W + (long long)row * (K / 2);
    const float* ws_row = Ws + (long long)row * ng;
    const float* __restrict__ X = X_batch + (long long)bid * K;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * gbytes + lane * 4;
        const int xoff = g * group + lane * 8;
        uint32_t wv = *(const uint32_t*)(w_row + boff);
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int qw = (int)((wv >> (4 * n)) & 0xf);
            qw = (qw << 28) >> 28; // sign-extend 4-bit
            gsum += (float)qw * X[xoff + n];
        }
        facc += gsum * ws_row[g];
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        Y_batch[(long long)bid * M + row] += gate * facc;
    }
}

__launch_bounds__(32)
extern "C" __global__ void gemv_oq8g256_residual_sigmoid_scaled_gpu_batched(
    const int8_t*  __restrict__ W,        // [M, K] signed int8
    const float*   __restrict__ Ws,       // [M, K/256] f32 group scales
    const float*   __restrict__ X_batch,  // [N, K] f32 rotated activations
    float*         __restrict__ Y_batch,  // [N, M] f32 residual stream (in place)
    const float*   __restrict__ c_batch,  // [N] shared-gate logit (pre-sigmoid)
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;
    const int lane = threadIdx.x;
    const int ng = K / group;
    const int8_t* w_row = W + (long long)row * K;
    const float* ws_row = Ws + (long long)row * ng;
    const float* __restrict__ X = X_batch + (long long)bid * K;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int boff = g * group + lane * 8; // 8 int8 = two int32 per lane
        const int32_t w0 = *(const int32_t*)(w_row + boff);
        const int32_t w1 = *(const int32_t*)(w_row + boff + 4);
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int qw0 = (int)((w0 >> (8 * n)) & 0xff); qw0 = (qw0 << 24) >> 24; // sext int8
            int qw1 = (int)((w1 >> (8 * n)) & 0xff); qw1 = (qw1 << 24) >> 24;
            gsum += (float)qw0 * X[boff + n];
            gsum += (float)qw1 * X[boff + 4 + n];
        }
        facc += gsum * ws_row[g];
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        Y_batch[(long long)bid * M + row] += gate * facc;
    }
}
