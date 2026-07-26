// SPDX-License-Identifier: Apache-2.0
// hipfire — portable raw F16/BF16 weight x F32 activation GEMM.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Correctness path for architectures without wave32 WMMA (RDNA2, CDNA, and
// older supported targets). A work-item owns one [batch, output] cell and
// accumulates K in F32. Source BF16 is normally converted once to F16 before
// this path, but the BF16 entry point remains available for honest direct use.

static __device__ __forceinline__ float hipfire_raw_bf16_to_f32(unsigned short value) {
    union {
        unsigned int bits;
        float value;
    } converted;
    converted.bits = ((unsigned int)value) << 16;
    return converted.value;
}

#define HIPFIRE_RAW_GEMM_BODY(WEIGHT_T, TO_F32)                                    \
    const int output = (int)(blockIdx.x * blockDim.x + threadIdx.x);                 \
    const int batch = (int)blockIdx.y;                                                \
    if (output >= M || batch >= B) return;                                            \
    const WEIGHT_T* weight = A + (long long)output * K;                              \
    const float* input = X + (long long)batch * K;                                   \
    float sum = 0.0f;                                                                \
    for (int column = 0; column < K; ++column) {                                     \
        sum = fmaf(TO_F32(weight[column]), input[column], sum);                      \
    }                                                                                \
    Y[(long long)batch * M + output] = sum

static __device__ __forceinline__ float hipfire_raw_f16_to_f32(_Float16 value) {
    return (float)value;
}

extern "C" __global__ void gemm_f16_x_f32_portable(
    const _Float16* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int B
) {
    HIPFIRE_RAW_GEMM_BODY(_Float16, hipfire_raw_f16_to_f32);
}

extern "C" __global__ void gemm_bf16_x_f32_portable(
    const unsigned short* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int B
) {
    HIPFIRE_RAW_GEMM_BODY(unsigned short, hipfire_raw_bf16_to_f32);
}

#undef HIPFIRE_RAW_GEMM_BODY
