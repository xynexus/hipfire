// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define BLOCK 256

// `out[n, o] = sum_i w[o, i] * x[n, i] + b[o]` — a linear WITH bias over N rows.
//
// The vision tower biases every projection, unlike the text trunk which biases
// none, so the plain GEMV path does not cover it.
//
// Grid: (n_out, n_rows). One block per output element, reducing over `n_in`.
__launch_bounds__(BLOCK)
extern "C" __global__ void vision_linear_bias(
    const float* __restrict__ x,   // [n_rows, n_in]
    const float* __restrict__ w,   // [n_out, n_in]
    const float* __restrict__ b,   // [n_out] or null
    float*       __restrict__ out, // [n_rows, n_out]
    int n_rows, int n_in, int n_out
) {
    const int o = blockIdx.x;
    const int r = blockIdx.y;
    if (o >= n_out || r >= n_rows) return;

    const float* __restrict__ xr = x + (long long)r * n_in;
    const float* __restrict__ wo = w + (long long)o * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += BLOCK) acc = fmaf(xr[i], wo[i], acc);

    __shared__ float red[BLOCK];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(long long)r * n_out + o] = red[0] + (b ? b[o] : 0.0f);
    }
}

// Rotate the Q and K spans of a fused `[N, 3*hidden]` qkv, in place.
//
// V is deliberately untouched — rotary applies to queries and keys only.
//
// ⚠️ Each thread owns a PAIR `(d, d + half)` and reads BOTH before writing either.
// The obvious one-thread-per-element form races: `rotate_half` makes element `d`
// depend on `d + half`, which another thread is concurrently overwriting, and no
// `__syncthreads()` can fix that because the two may be in different blocks.
//
// Layout is `[token][q|k|v][head][dim]`; `cos`/`sin` are `[N, head_dim]`, shared
// across heads.
extern "C" __global__ void vision_rope_qkv(
    float*       __restrict__ qkv,  // [N, 3*hidden], updated in place
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    int N, int hidden, int num_heads, int head_dim
) {
    const int half = head_dim / 2;
    const int pairs_per_span = num_heads * half;
    const int total = N * 2 * pairs_per_span;   // q and k only
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= total) return;

    const int t = i / (2 * pairs_per_span);
    int rem = i - t * (2 * pairs_per_span);
    const int span = rem / pairs_per_span;      // 0 = q, 1 = k
    rem -= span * pairs_per_span;
    const int head = rem / half;
    const int d = rem - head * half;

    const long long base =
        (long long)t * 3 * hidden + (long long)span * hidden + (long long)head * head_dim;
    const float lo = qkv[base + d];
    const float hi = qkv[base + d + half];
    const long long ct = (long long)t * head_dim;
    // rotate_half: the low half pairs with -high, the high half with +low.
    qkv[base + d] = lo * cos_t[ct + d] - hi * sin_t[ct + d];
    qkv[base + d + half] = hi * cos_t[ct + d + half] + lo * sin_t[ct + d + half];
}

// Exact erf GELU — `nn.GELU()`, which the patch MERGER uses.
//
// NOT interchangeable with `gelu_tanh`, which the block MLPs use. They agree to
// ~1e-3: close enough that swapping them reads as numerical noise.
extern "C" __global__ void vision_gelu_erf(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int n
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.707106781f));
}
