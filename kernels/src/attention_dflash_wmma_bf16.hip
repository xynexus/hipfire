// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// BF16 causal FlashAttention parity kernel for the Gemma 4 prefill path.
//
// Q/K/V are carried as F32 by the generic runtime, but Gemma 4's pinned
// Transformers reference materializes them as BF16 before ROCm SDPA. This
// kernel performs the same RNE conversion in-register, uses BF16 WMMA for both
// QK^T and P@V, retains the online-softmax state in F32, and rounds the final
// output through BF16. It intentionally starts as an explicit parity primitive;
// architecture code must not select it until real layer evidence passes.
//
// Layout:
//   Q   [B, n_heads,    head_dim]
//   K/V [L, n_kv_heads, head_dim]
//   out [B, n_heads,    head_dim]
//
// Grid [n_heads, ceil(B/16)], block [32]. head_dim must be a multiple of 16
// and at most 256 so the three 16-row tiles fit in LDS.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ unsigned short f32_to_bf16_rne(float value) {
    union {
        float f;
        unsigned int u;
    } bits;
    bits.f = value;
    const unsigned int exponent = bits.u & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        // Preserve infinities and quiet NaNs instead of carrying a mantissa
        // rounding overflow into the exponent.
        if (bits.u & 0x007fffffu) bits.u |= 0x00400000u;
        return (unsigned short)(bits.u >> 16);
    }
    const unsigned int lsb = (bits.u >> 16) & 1u;
    return (unsigned short)((bits.u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float bf16_to_f32(unsigned short value) {
    union {
        unsigned int u;
        float f;
    } bits;
    bits.u = ((unsigned int)value) << 16;
    return bits.f;
}

extern "C" __launch_bounds__(32, 4)
__global__ void attention_dflash_wmma_bf16_causal_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int L,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    int query_position_base,
    int kv_layout,
    int kv_window,
    int kv_batch_size
) {
    const int head = blockIdx.x;
    const int q_start = blockIdx.y * 16;
    const int tid = threadIdx.x;
    if (head >= n_heads || q_start >= B) return;

    const int kv_head = head / (n_heads / n_kv_heads);
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;
    const int lane16 = tid & 15;
    const int half_id = tid >> 4;

    extern __shared__ float sdata[];
    float* q_lds = sdata;
    float* v_lds = q_lds + 16 * head_dim;
    float* o_lds = v_lds + 16 * head_dim;
    float* scores = o_lds + 16 * head_dim;
    float* running_max = scores + 16 * 16;
    float* running_sum = running_max + 16;
    float* alpha = running_sum + 16;

    const int cols_per_half = head_dim / 2;
    const int col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        const int q_pos = q_start + row;
        const float* q_row = q_pos < B
            ? q + (long long)q_pos * q_stride + head * head_dim
            : nullptr;
        for (int d = 0; d < cols_per_half; ++d) {
            const int col = col_base + d;
            const int index = row * head_dim + col;
            q_lds[index] = q_row ? q_row[col] : 0.0f;
            o_lds[index] = 0.0f;
        }
    }
    if (tid < 16) {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int k_start = 0; k_start < L; k_start += 16) {
        if (k_start >= query_position_base + q_start + 16) continue;
        const int key_pos = k_start + lane16;
        const bool key_valid = key_pos < L;
        const int staged_batch = q_start < B ? q_start : B - 1;

        float8_t score_acc = {0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f};
        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            b16x16_t q_frag;
            b16x16_t k_frag;
            const float* q_row = q_lds + lane16 * head_dim + d0;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float key_value = !key_valid ? 0.0f : kv_layout == 0
                    ? k[(long long)key_pos * kv_stride + kv_head * head_dim + d0 + j]
                    : k[(((long long)kv_head * kv_batch_size + staged_batch) * head_dim
                         + d0 + j) * kv_window + key_pos];
                q_frag[j] = (short)f32_to_bf16_rne(q_row[j]);
                k_frag[j] = (short)f32_to_bf16_rne(key_value);
            }
            score_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                q_frag, k_frag, score_acc
            );
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = 2 * j + half_id;
            const bool visible = key_valid
                && key_pos <= query_position_base + q_start + row;
            scores[row * 16 + lane16] = visible
                ? score_acc[j] * scale
                : -INFINITY;
        }

        for (int d = col_base; d < col_base + cols_per_half; ++d) {
            v_lds[lane16 * head_dim + d] = !key_valid ? 0.0f : kv_layout == 0
                ? v[(long long)key_pos * kv_stride + kv_head * head_dim + d]
                : v[(((long long)kv_head * kv_batch_size + staged_batch) * head_dim
                     + d) * kv_window + key_pos];
        }
        __syncthreads();

        if (tid < 16) {
            float* row_scores = scores + tid * 16;
            float tile_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 16; ++j) tile_max = fmaxf(tile_max, row_scores[j]);
            const float old_max = running_max[tid];
            const float new_max = fmaxf(old_max, tile_max);
            const float correction = expf(old_max - new_max);
            float tile_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float probability = expf(row_scores[j] - new_max);
                row_scores[j] = probability;
                tile_sum += probability;
            }
            alpha[tid] = correction;
            running_sum[tid] = correction * running_sum[tid] + tile_sum;
            running_max[tid] = new_max;
        }
        __syncthreads();

        for (int index = tid; index < 16 * head_dim; index += 32) {
            o_lds[index] *= alpha[index / head_dim];
        }
        __syncthreads();

        b16x16_t probability_frag;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            probability_frag[j] = (short)f32_to_bf16_rne(
                scores[lane16 * 16 + j]
            );
        }

        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            b16x16_t value_frag;
            const int output_col = d0 + lane16;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                value_frag[j] = (short)f32_to_bf16_rne(
                    v_lds[j * head_dim + output_col]
                );
            }
            float8_t output_acc = {0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f};
            output_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                probability_frag, value_frag, output_acc
            );
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = 2 * j + half_id;
                o_lds[row * head_dim + output_col] += output_acc[j];
            }
        }
        __syncthreads();
    }

    for (int index = tid; index < 16 * head_dim; index += 32) {
        const int row = index / head_dim;
        const int col = index % head_dim;
        const int q_pos = q_start + row;
        if (q_pos < B) {
            const float value = running_sum[row] > 0.0f
                ? o_lds[index] / running_sum[row]
                : 0.0f;
            out[(long long)q_pos * q_stride + head * head_dim + col] =
                bf16_to_f32(f32_to_bf16_rne(value));
        }
    }
}
