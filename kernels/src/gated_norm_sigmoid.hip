// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Gated RMSNorm with a SIGMOID output gate — the Gated DeltaNet delta for
// Qwen3.8-Flash-Next (qwen4_exp).
//
// Byte-for-byte `gated_norm_f32` except for the gate activation. Qwen3.5/3.8 set
// `output_gate_type` to silu (or leave it absent, falling back to `hidden_act`);
// qwen4_exp sets it to "sigmoid", and the reference feeds that straight into
// `ACT2FN[self.activation](gate)` in `Qwen4ExpTextRMSNormGated.forward`. Confirmed
// in the GENERATED modeling file, not just the modular source — the modular class
// only overrides `__init__`, so it would be easy to conclude the field is unused.
//
// A separate kernel rather than a branch: `gated_norm_f32` is on the hot path of
// every shipping Qwen3.5/3.8 model, and adding a parameter would change its
// signature at every call site for a family that does not need it.
//
// Note the norm weight is applied PLAIN here (`weight[i]`, ones-initialised), NOT
// as `1 + weight[i]`. That differs from `Qwen4ExpTextRMSNorm`, which does carry the
// `+1` — the two norms in this model use different conventions, so they are not
// interchangeable.
//
// Grid: [n_heads, batch, 1], matching the sibling.

extern "C" __global__ void gated_norm_sigmoid_f32(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n_heads, int head_dim, float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    const float* xh = x + batch_off + h * head_dim;
    const float* zh = z + batch_off + h * head_dim;
    float* oh = out + batch_off + h * head_dim;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += 32) {
        float normed = xh[i] * inv_rms * weight[i];
        float z_val = zh[i];
        // The only line that differs from `gated_norm_f32`.
        float gate = 1.0f / (1.0f + __expf(-z_val));
        oh[i] = normed * gate;
    }
}
