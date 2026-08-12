// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on an asym3 post-RoPE K cache (arXiv:2604.04921).
//
// asym3 stores K as: [4B cnorm][32 threads × 3 bytes packed 3-bit indices].
// To recover post-RoPE K per band for scoring, each thread:
//   1. Reads its 3 bytes (8 × 3-bit indices) and dequantizes via the
//      Lloyd-Max N(0,1/256) codebook (same table asym3 write used).
//   2. Multiplies by cnorm (the shared magnitude-correction scalar stored
//      at the head block head).
//   3. Inverse-applies the per-band Givens rotation that the writer used
//      to spread mass across dims — this un-mixing recovers post-RoPE
//      (complex) K at each band, so arg/norm can be read directly.
//
// Layout (matches kv_cache_write_asym_k_givens3.hip):
//   bytes_per_head = 4 + (head_dim * 3) / 8     // 100 for hd=256
//   per-thread owns 8 consecutive dims = 4 complex bands (tid*4 .. tid*4+3)
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

__device__ __forceinline__ float asym3_quant_dim(
    const unsigned char* k_head, int dim, int head_dim, float cnorm
) {
    const int bit = dim * 3;
    const int byte = bit / 8;
    const int shift = bit & 7;
    unsigned int packed = (unsigned int)k_head[4 + byte];
    const int packed_bytes = (head_dim * 3) / 8;
    if (byte + 1 < packed_bytes) packed |= (unsigned int)k_head[4 + byte + 1] << 8;
    const int index = (packed >> shift) & 0x7;
    return cnorm * TURBO_C3_256[index];
}

__device__ __forceinline__ float asym3_original_dim(
    const unsigned char* k_head,
    const float* cos_theta,
    const float* sin_theta,
    int dim,
    int head_dim,
    float cnorm
) {
    const int band = dim / 2;
    const float a = asym3_quant_dim(k_head, band * 2, head_dim, cnorm);
    const float b = asym3_quant_dim(k_head, band * 2 + 1, head_dim, cnorm);
    return (dim & 1) == 0
        ? cos_theta[band] * a + sin_theta[band] * b
        : -sin_theta[band] * a + cos_theta[band] * b;
}

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_asym3(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,       // [n_heads × n_bands × 3]
    const float* __restrict__ cos_theta,     // [n_bands] Givens angles (shared across layers)
    const float* __restrict__ sin_theta,     // [n_bands]
    float* __restrict__ scores,              // [n_heads × seq_len]
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

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const unsigned char* k_head = k_cache + (size_t)pos * k_bytes_per_pos + kv_h * k_bytes_per_head;

    // Shared per-head magnitude-restore factor.
    const float cnorm = *(const float*)k_head;

    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    for (int f = tid; f < n_bands; f += 32) {
        const int dim0 = rope_layout == 1 ? f : 2 * f;
        const int dim1 = rope_layout == 1 ? f + n_bands : 2 * f + 1;
        const float k_re = asym3_original_dim(
            k_head, cos_theta, sin_theta, dim0, head_dim, cnorm
        );
        const float k_im = asym3_original_dim(
            k_head, cos_theta, sin_theta, dim1, head_dim, cnorm
        );
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
