// SPDX-License-Identifier: Apache-2.0
// hipfire — portable raw F16/BF16 routed-MoE grouped fallback.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Correctness fallback for architectures without an admitted grouped matrix
// kernel. The routing permutation still batches every active expert into
// 16-route tiles; one scalar thread computes each output row. This deliberately
// favors portability over throughput and is never selected on gfx1151, where
// the raw WMMA kernels are available.

static __device__ __forceinline__ float hipfire_bf16_to_f32(unsigned short value) {
    union {
        unsigned int bits;
        float value;
    } converted;
    converted.bits = ((unsigned int)value) << 16;
    return converted.value;
}

#define HIPFIRE_RAW_MOE_BODY(WEIGHT_T, TO_F32)                                      \
    const int output_row = (int)(blockIdx.x * blockDim.x + threadIdx.x);             \
    const int grouped_row = (int)blockIdx.y;                                         \
    if (output_row >= M || grouped_row >= m_total) return;                           \
    const int flat_slot = sorted_slot_index[grouped_row];                            \
    if (flat_slot < 0) {                                                             \
        Y_grouped[(long long)grouped_row * M + output_row] = 0.0f;                   \
        return;                                                                      \
    }                                                                                \
    const int expert_id = expert_tile_ids[grouped_row / 16];                         \
    if (expert_id < 0) {                                                             \
        Y_grouped[(long long)grouped_row * M + output_row] = 0.0f;                   \
        return;                                                                      \
    }                                                                                \
    const WEIGHT_T* weights =                                                        \
        reinterpret_cast<const WEIGHT_T*>(expert_weight_ptrs[expert_id]);            \
    const int x_row = (x_row_div > 1) ? (flat_slot / x_row_div) : flat_slot;         \
    const float* x = X_src + (long long)x_row * K;                                   \
    const WEIGHT_T* w = weights + (long long)output_row * K;                         \
    float sum = 0.0f;                                                                \
    for (int column = 0; column < K; ++column) {                                     \
        sum = fmaf(TO_F32(w[column]), x[column], sum);                               \
    }                                                                                \
    Y_grouped[(long long)grouped_row * M + output_row] = sum

static __device__ __forceinline__ float hipfire_f16_to_f32(_Float16 value) {
    return (float)value;
}

extern "C" __global__ void gemm_f16_moe_grouped_portable(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    HIPFIRE_RAW_MOE_BODY(_Float16, hipfire_f16_to_f32);
}

extern "C" __global__ void gemm_bf16_moe_grouped_portable(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    HIPFIRE_RAW_MOE_BODY(unsigned short, hipfire_bf16_to_f32);
}

#undef HIPFIRE_RAW_MOE_BODY
