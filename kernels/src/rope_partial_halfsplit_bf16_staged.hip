// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

#include <hip/hip_runtime.h>

// Half-split RoPE with the BF16 pointwise staging used by PyTorch Gemma 4:
//   bf16(bf16(x * cos) + bf16(rotate_half(x) * sin)).
//
// Q/K remain F32 buffers so this composes with Hipfire's generic attention
// scratch. Every stored result is exactly representable BF16 promoted to F32.
// The trig values are rounded before multiplication because Transformers casts
// its F32 RoPE tables to the model dtype before apply_rotary_pos_emb.
static __device__ __forceinline__ float bf16_rne_f32(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    unsigned lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7fffu + lsb) & 0xffff0000u;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ void rotate_pair_bf16(
    float* values,
    int first,
    int second,
    float cos_value,
    float sin_value
) {
    float x0 = bf16_rne_f32(values[first]);
    float x1 = bf16_rne_f32(values[second]);
    float p00 = bf16_rne_f32(x0 * cos_value);
    float p01 = bf16_rne_f32(x1 * sin_value);
    float p10 = bf16_rne_f32(x1 * cos_value);
    float p11 = bf16_rne_f32(x0 * sin_value);
    values[first] = bf16_rne_f32(p00 - p01);
    values[second] = bf16_rne_f32(p10 + p11);
}

extern "C" __global__ void rope_partial_halfsplit_bf16_staged_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    int basis_dim,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rot / 2;
    int half = basis_dim / 2;
    if (i >= n_pairs) return;

    float frequency = 1.0f / powf(freq_base, (float)(2 * i) / (float)basis_dim);
    float angle = (float)(*pos_buf) * frequency;
    float cos_value = bf16_rne_f32(cosf(angle));
    float sin_value = bf16_rne_f32(sinf(angle));

    for (int head = 0; head < n_heads_q; ++head) {
        int base = head * head_dim;
        rotate_pair_bf16(q, base + i, base + i + half, cos_value, sin_value);
    }
    for (int head = 0; head < n_heads_k; ++head) {
        int base = head * head_dim;
        rotate_pair_bf16(k, base + i, base + i + half, cos_value, sin_value);
    }
}
