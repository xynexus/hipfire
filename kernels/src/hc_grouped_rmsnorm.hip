// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Grouped RMSNorm over a widened residual stream, for Qwen3.8-Flash-Next.
//
// The residual is `n_groups` streams of `group` floats laid out contiguously, and
// each stream normalises INDEPENDENTLY — not jointly over the full width. A flat
// norm over `n_groups * group` is the obvious wrong implementation and produces
// plausible output, so this is a separate kernel rather than a reuse of the model
// -wide RMSNorm.
//
// Two details from the reference (`Qwen4ExpTextRMSNorm`, which inherits
// `Qwen3_5RMSNorm.forward` and overrides only `_norm` to add the grouping):
//
//   * the scale is `(1.0f + weight[i])`, NOT `weight[i]` — the parameter is
//     zero-initialised, so multiplying by it alone collapses the stream to zero;
//   * `weight` spans the WHOLE vector, not one group, matching the checkpoint's
//     `hc_norm [hc_count * hidden]`.
//
// Differenced against `hipfire_arch_qwen4exp::hc::grouped_rmsnorm`, which shares
// no code with this.
//
// Grid: (n_groups). Block = BLOCK, striding over `group`.

#define BLOCK 256

__launch_bounds__(BLOCK)
extern "C" __global__ void hc_grouped_rmsnorm(
    const float* __restrict__ x,       // [n_groups, group]
    const float* __restrict__ weight,  // [n_groups, group]
    float*       __restrict__ out,     // [n_groups, group]
    int   group,
    int   n_groups,
    float eps
) {
    const int g = blockIdx.x;
    if (g >= n_groups) return;

    const long long base = (long long)g * group;
    const float* __restrict__ xg = x + base;
    const float* __restrict__ wg = weight + base;
    float*       __restrict__ og = out + base;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < group; i += BLOCK) {
        const float v = xg[i];
        acc = fmaf(v, v, acc);
    }

    __shared__ float red[BLOCK];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Mean of squares over THIS group only.
    const float scale = rsqrtf(red[0] / (float)group + eps);
    for (int i = threadIdx.x; i < group; i += BLOCK) {
        og[i] = xg[i] * scale * (1.0f + wg[i]);
    }
}
