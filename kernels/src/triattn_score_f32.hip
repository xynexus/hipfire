// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

// TriAttention scoring over an F32 post-RoPE K cache. `rope_layout` selects
// the complex pair convention used by the source model and CASK accumulator:
//   0 = adjacent/interleaved pairs (2f, 2f+1)
//   1 = half-split pairs (f, f + head_dim/2)

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_f32(
    const float* __restrict__ k_cache,
    const float* __restrict__ centers,
    float* __restrict__ scores,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,
    int rope_layout,
    float rope_theta,
    float p_q,
    int seq_len
) {
    const int pos = blockIdx.x;
    const int h = blockIdx.y;
    if (pos >= seq_len || h >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int n_bands = head_dim / 2;
    const int n_rot_bands = n_rot / 2;
    const float* k_head = k_cache + ((size_t)pos * n_kv_heads + kv_h) * head_dim;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;
    for (int f = tid; f < n_bands; f += 32) {
        const int d0 = rope_layout == 1 ? f : 2 * f;
        const int d1 = rope_layout == 1 ? f + n_bands : 2 * f + 1;
        const float k_re = k_head[d0];
        const float k_im = k_head[d1];
        const float k_mag = sqrtf(k_re * k_re + k_im * k_im);
        const float k_phase = atan2f(k_im, k_re);

        const float* c = c_head + f * 3;
        const float c_re = c[0];
        const float c_im = c[1];
        const float c_abs = c[2];
        const float c_mag = sqrtf(c_re * c_re + c_im * c_im);
        const float c_phase = atan2f(c_im, c_re);
        float omega = 0.0f;
        if (f < n_rot_bands) {
            const float exponent = -2.0f * (float)f / (float)n_rot;
            omega = expf(exponent * logf(rope_theta));
        }
        s_trig += c_mag * k_mag * cosf(omega * p_q + c_phase - k_phase);
        float r = c_abs > 1e-20f ? c_mag / c_abs : 0.0f;
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        s_norm += (1.0f - r) * c_abs * k_mag;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s_trig += __shfl_xor(s_trig, offset);
        s_norm += __shfl_xor(s_norm, offset);
    }
    if (tid == 0) scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
}
