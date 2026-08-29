// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// The two elementwise stages of qwen4_exp's low-rank gated-residual mix, and the
// residual write-back. The matmuls either side are stock `gemv_f32`; only these
// pieces are family-specific.
//
//   down -> [hc_scaled_silu]    -> up -> [hc_sigmoid] -> per-channel mix
//   block output + inject gates -> [hc_residual_inject] -> streams
//
// `hc_scaled_silu` divides BEFORE the activation (`silu(v / hc_count)`), which is
// where the reference puts it — applying the scale after silu is a different
// function and a plausible thing to write by mistake.

#define BLOCK 256

// out[i] = silu(x[i] * inv_scale), with silu(v) = v * sigmoid(v).
__launch_bounds__(BLOCK)
extern "C" __global__ void hc_scaled_silu(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int   n,
    float inv_scale
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * inv_scale;
    out[i] = v / (1.0f + __expf(-v));
}

// out[i] = sigmoid(x[i]).
__launch_bounds__(BLOCK)
extern "C" __global__ void hc_sigmoid(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int n
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// streams[s, d] += inject[s] * block_out[d].
//
// Accumulates into the RAW streams, not the normalised ones — the normalisation
// exists to compute the gates, not to carry the residual. Grid is (hidden-tiles,
// hc_mult) so each stream's scalar is read once per block rather than per element.
__launch_bounds__(BLOCK)
extern "C" __global__ void hc_residual_inject(
    float*       __restrict__ streams,    // [hc_mult, hidden], updated in place
    const float* __restrict__ block_out,  // [hidden]
    const float* __restrict__ inject,     // [hc_mult]
    int hidden
) {
    const int d = blockIdx.x * BLOCK + threadIdx.x;
    if (d >= hidden) return;
    const int s = blockIdx.y;
    const float g = inject[s];
    const long long i = (long long)s * hidden + d;
    streams[i] = fmaf(g, block_out[d], streams[i]);
}

// The write-side gate: `2 * sigmoid(x / hc_count)`, one value per stream.
//
// Its range is (0, 2), not (0, 1) — a stream can AMPLIFY the block output, not
// only attenuate it. Dropping the factor of two still produces plausible output
// and silently halves every residual write.
extern "C" __global__ void hc_inject_gate(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int   n,
    float inv_scale
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    out[i] = 2.0f / (1.0f + __expf(-x[i] * inv_scale));
}
