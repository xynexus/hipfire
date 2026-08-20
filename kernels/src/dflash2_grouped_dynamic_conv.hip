// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// DFlash2 grouped dynamic causal convolution — one of the two DFlash2-only
// components (the other is the candidate selector). Semantics are pinned by
// `hipfire_runtime::dflash2::grouped_dynamic_convolve`, which is itself parity-
// checked against z-lab/dflash `_grouped_dynamic_convolve` on real checkpoint
// weights.
//
//   out[t, c] = Σ_tap (base[tap, c] + dynamic[t, tap, group_of(c)]) · h[t-tap, c]
//
// with h[<0] = 0 (causal, zero-padded at the front) and
// group_of(c) = c / group_size.
//
// Shapes (row-major):
//   H   : [len, hidden]              f32 activations
//   D   : [len, kernel_size, groups] f32 dynamic coefficients (per GROUP)
//   B   : [kernel_size, hidden]      f32 static per-channel kernel
//   Y   : [len, hidden]              f32 out
//
// One block per position, threads striding channels. The drafter is 5 layers
// with block_size 8, so `len` is tiny and per-position parallelism over `hidden`
// (5120) is the only dimension worth spending.

#include <hip/hip_runtime.h>

extern "C" __global__ void dflash2_grouped_dynamic_conv(
    const float* __restrict__ H,
    const float* __restrict__ D,
    const float* __restrict__ B,
    float*       __restrict__ Y,
    int len, int hidden, int kernel_size, int group_size
) {
    const int t = blockIdx.x;
    if (t >= len) return;
    const int groups = hidden / group_size;

    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        const int g = c / group_size;
        float acc = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            // Causal: tap 0 reads the current position, tap k reads t-k, and
            // anything before the start contributes nothing.
            const int src = t - tap;
            if (src < 0) continue;
            const float v = H[(long long)src * hidden + c];
            const float k = B[(long long)tap * hidden + c]
                          + D[((long long)t * kernel_size + tap) * groups + g];
            acc = fmaf(k, v, acc);
        }
        Y[(long long)t * hidden + c] = acc;
    }
}
