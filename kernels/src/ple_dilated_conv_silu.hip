// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// PLE short convolution for qwen4_exp: depthwise, causal, DILATED, then SiLU.
//
// Deliberately not a variant of `conv1d_silu_split`, which serves the Gated
// DeltaNet mixer in this same model: that one is UNDILATED with a 3-slot ring, so
// sharing a path would silently apply taps at t-1/t-2/t-3 instead of
// t-9/t-6/t-3. The reference makes the same split, noting it cannot use the
// standard conv path because of the dilation.
//
//   y[t] = sum_k w[c, k] * x[c, t - (kernel - 1 - k) * dilation]
//
// `w[kernel - 1]` multiplies the CURRENT input; reversing the kernel is the
// obvious mistake and produces a plausible wrong answer.
//
// State is [channels, state_len] with `state_len = (kernel - 1) * dilation`,
// OLDEST FIRST, advanced in place by one position per step.
//
// One decode step. Grid: (ceil(channels / BLOCK)). Block = BLOCK.

#define BLOCK 256
#define MAX_KERNEL 8

__launch_bounds__(BLOCK)
extern "C" __global__ void ple_dilated_conv_silu(
    float*       __restrict__ state,   // [channels, state_len], updated in place
    const float* __restrict__ x,       // [channels]
    const float* __restrict__ weight,  // [channels, kernel]
    float*       __restrict__ out,     // [channels]
    int channels,
    int state_len,
    int kernel,
    int dilation
) {
    const int c = blockIdx.x * BLOCK + threadIdx.x;
    if (c >= channels || kernel > MAX_KERNEL) return;

    float* __restrict__ st = state + (long long)c * state_len;
    const float* __restrict__ w = weight + (long long)c * kernel;

    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        const int back = (kernel - 1 - k) * dilation;
        const float v = (back == 0) ? x[c] : st[state_len - back];
        acc = fmaf(w[k], v, acc);
    }
    out[c] = acc / (1.0f + __expf(-acc));

    // Advance the ring: shift left one, current input into the newest slot. Read
    // all taps above BEFORE mutating, which the loop order already guarantees.
    for (int i = 0; i < state_len - 1; ++i) {
        st[i] = st[i + 1];
    }
    st[state_len - 1] = x[c];
}
