// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-channel gated-residual read side, for Qwen3.8-Flash-Next (qwen4_exp).
//
// Sibling of `hc_input_map_4stream`, which serves DeepSeek V4. The two agree on
// the residual LAYOUT — `hc_mult` streams of `hidden` floats — and on nothing
// else, so this is a separate kernel rather than a flag on that one:
//
//   deepseek4:  x_out[d] = SUM_s  a_vec[s]            * X[s, d]   (4 scalars)
//   qwen4_exp:  x_out[d] = MEAN_s w[s * hidden + d]   * X[s, d]   (per channel)
//
// Two differences, both of which change the result rather than the performance:
// the gate is per-CHANNEL (a `[hc_mult, hidden]` plane, not four scalars), and
// the collapse is a MEAN, not a sum — a sum would come out `hc_mult` times too
// large. See `Qwen4ExpTextGatedResidual.forward` in the vendored reference, and
// `crates/hipfire-arch-qwen4exp/src/hc.rs` for the CPU oracle this is differenced
// against.
//
// `mix` is expected already sigmoid-activated and `streams` already normalised;
// both are produced by the low-rank gate and the grouped RMSNorm respectively, so
// this kernel is a pure elementwise collapse and stays trivially correct.
//
// Grid: (ceil(hidden / BLOCK_HIDDEN)). Block = BLOCK_HIDDEN.

#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_input_map_perchannel(
    const float* __restrict__ mix,      // [hc_mult, hidden] fp32, post-sigmoid
    const float* __restrict__ streams,  // [hc_mult, hidden] fp32, post-norm
    float*       __restrict__ x_out,    // [hidden] fp32
    int hidden,
    int hc_mult
) {
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= hidden) return;

    float acc = 0.0f;
    for (int s = 0; s < hc_mult; ++s) {
        const long long i = (long long)s * hidden + d;
        acc = fmaf(mix[i], streams[i], acc);
    }
    // MEAN, not sum — see the header.
    x_out[d] = acc / (float)hc_mult;
}
