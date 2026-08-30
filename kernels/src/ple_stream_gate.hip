// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// PLE per-stream value gate.
//
// Each hyper-connection stream scores the token's n-gram KEY against its own
// normalised activations, and that score gates a VALUE shared by all streams.
//
// The score is a SIGNED SQUARE ROOT of the dot product, clamped away from zero
// before the root:
//
//     g = sign(d) * sqrt(max(|d|, 1e-6)),   d = (key . query) / sqrt(hidden)
//
// Not a plain dot product, and not a softmax over streams. The clamp is inside the
// root and applies to the MAGNITUDE, so the sign survives it; clamping `d` itself
// would turn every small negative score positive.
//
// Grid: (hc_count). Block = THREADS, one reduction per stream.

#define THREADS 256

__launch_bounds__(THREADS)
extern "C" __global__ void ple_stream_gate(
    const float* __restrict__ key,     // [hc_count, hidden], normalised
    const float* __restrict__ query,   // [hc_count, hidden], normalised
    const float* __restrict__ value,   // [hidden], shared across streams
    float*       __restrict__ out,     // [hc_count, hidden]
    int hidden,
    float inv_sqrt_hidden
) {
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const long long base = (long long)s * hidden;

    __shared__ float red[THREADS];
    float acc = 0.0f;
    for (int d = tid; d < hidden; d += THREADS) acc += key[base + d] * query[base + d];
    red[tid] = acc;
    __syncthreads();
    for (int w = THREADS / 2; w > 0; w >>= 1) {
        if (tid < w) red[tid] += red[tid + w];
        __syncthreads();
    }

    __shared__ float s_gate;
    if (tid == 0) {
        const float d = red[0] * inv_sqrt_hidden;
        const float g = copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d);
        s_gate = 1.0f / (1.0f + __expf(-g));
    }
    __syncthreads();

    const float gate = s_gate;
    for (int d = tid; d < hidden; d += THREADS) out[base + d] = gate * value[d];
}
