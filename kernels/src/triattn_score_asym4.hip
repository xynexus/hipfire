// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on an asym4 post-RoPE K cache.
//
// asym4 stores K as: [4B cnorm][head_dim/2 packed nibbles] with the same
// per-band Givens rotation asym3 uses but a 4-bit (16-centroid) codebook.
// The write-time layout runs in `n_halves = head_dim/128` passes, each
// thread owning 4 consecutive dims per pass; cos_theta / sin_theta are
// indexed by `block_tid = half * 32 + tid` with `block_tid * 2 + {0,1}`
// as the band index. Matches kv_cache_write_asym_k_givens4.hip.
//
// Output: scores[h * seq_len + pos] = S_trig + S_norm (see paper §4).
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

__device__ __forceinline__ float asym4_quant_dim(
    const unsigned char* k_head, int dim, float cnorm
) {
    const unsigned int packed = (unsigned int)k_head[4 + dim / 2];
    const int index = (packed >> (4 * (dim & 1))) & 0xF;
    return cnorm * TURBO_C4[index];
}

__device__ __forceinline__ float asym4_original_dim(
    const unsigned char* k_head,
    const float* cos_theta,
    const float* sin_theta,
    int dim,
    float cnorm
) {
    const int band = dim / 2;
    const float a = asym4_quant_dim(k_head, band * 2, cnorm);
    const float b = asym4_quant_dim(k_head, band * 2 + 1, cnorm);
    return (dim & 1) == 0
        ? cos_theta[band] * a + sin_theta[band] * b
        : -sin_theta[band] * a + cos_theta[band] * b;
}

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_asym4(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
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

    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const unsigned char* k_head = k_cache + (size_t)pos * k_bytes_per_pos + kv_h * k_bytes_per_head;

    const float cnorm = *(const float*)k_head;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    for (int f = tid; f < n_bands; f += 32) {
            const int dim0 = rope_layout == 1 ? f : 2 * f;
            const int dim1 = rope_layout == 1 ? f + n_bands : 2 * f + 1;
            const float k_re = asym4_original_dim(k_head, cos_theta, sin_theta, dim0, cnorm);
            const float k_im = asym4_original_dim(k_head, cos_theta, sin_theta, dim1, cnorm);
            const float k_mag = sqrtf(k_re * k_re + k_im * k_im);
            const float k_phase = atan2f(k_im, k_re);

            const float* c = c_head + f * 3;
            const float c_re  = c[0];
            const float c_im  = c[1];
            const float c_abs = c[2];
            const float c_mag   = sqrtf(c_re * c_re + c_im * c_im);
            const float c_phase = atan2f(c_im, c_re);

            float omega = 0.0f;
            if (f < n_rot_bands) {
                const float exponent = -2.0f * (float)f / (float)n_rot;
                omega = expf(exponent * logf(rope_theta));
            }

            const float angle = omega * p_q + c_phase - k_phase;
            s_trig += c_mag * k_mag * cosf(angle);

            float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
            if (r > 1.0f) r = 1.0f;
            s_norm += (1.0f - r) * c_abs * k_mag;
    }

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }
    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
