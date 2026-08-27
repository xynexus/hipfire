// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Plain-basis DFLASH Opus reference GEMM.
//
// The arch-20 DFLASH artifact stores the original NPU blocks interleaved:
//   oq8+      [f16 scale | 256 signed int8]                         258 B
//   oq4+      [f16 scale | 128 packed signed-int4]                  130 B
//   oq4.N+    oq4+ block | N * (u8 index-in-group, i8 replacement)
//
// These kernels consume those bytes directly. They do not repack or expand
// weights, and they never apply FWHT: DFLASH qt=45/46/47 are explicitly plain
// basis. One wave computes one (batch,row) output. Activations are rounded to
// f16 before multiplication to match the former expanded-F16 comparison path;
// accumulation and output remain f32. The global flattened weight index is
// used rather than assuming K%256==0, so row-spanning partial groups retain the
// exact converter semantics.

#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float dflash_f16_scale(const uint8_t* block) {
    const _Float16 value = *(const _Float16*)block;
    return (float)value;
}

static __device__ __forceinline__ int dflash_i4(const uint8_t* block, int pos) {
    const uint8_t packed = block[2 + (pos >> 1)];
    int q = (pos & 1) ? (packed >> 4) : (packed & 0xf);
    return (q << 28) >> 28;
}

static __device__ __forceinline__ float dflash_wave_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
    }
    return value;
}

#define DFLASH_OQ_GEMM(NAME, MODE)                                             \
__launch_bounds__(32)                                                         \
extern "C" __global__ void NAME(                                             \
    const uint8_t* __restrict__ W,                                            \
    const float*   __restrict__ X,                                            \
    float*         __restrict__ Y,                                            \
    int M, int K, int B, int block_stride                                     \
) {                                                                           \
    const int row = (int)blockIdx.x;                                          \
    const int batch = (int)blockIdx.y;                                        \
    const int lane = (int)threadIdx.x;                                        \
    if (row >= M || batch >= B) return;                                       \
    float acc = 0.0f;                                                         \
    for (int col = lane; col < K; col += 32) {                                \
        const long long flat = (long long)row * K + col;                      \
        const long long block_index = flat >> 8;                              \
        const int pos = (int)(flat & 255);                                    \
        const uint8_t* block = W + block_index * block_stride;                \
        int q;                                                                 \
        if ((MODE) == 8) {                                                    \
            q = (int)(int8_t)block[2 + pos];                                  \
        } else {                                                               \
            q = dflash_i4(block, pos);                                        \
            if ((MODE) == 5) {                                                \
                const int overlays = (block_stride - 130) >> 1;               \
                const uint8_t* table = block + 130;                           \
                /* Last duplicate wins, matching the CPU expansion oracle. */ \
                for (int entry = 0; entry < overlays; ++entry) {              \
                    if ((int)table[2 * entry] == pos)                         \
                        q = (int)(int8_t)table[2 * entry + 1];                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float scale = dflash_f16_scale(block);                          \
        const float activation = (float)(_Float16)X[(long long)batch * K + col]; \
        acc += scale * (float)q * activation;                                 \
    }                                                                          \
    acc = dflash_wave_sum(acc);                                               \
    if (lane == 0) Y[(long long)batch * M + row] = acc;                       \
}

DFLASH_OQ_GEMM(gemm_dflash_oq8_plain_ref, 8)
DFLASH_OQ_GEMM(gemm_dflash_oq4_plain_ref, 4)
DFLASH_OQ_GEMM(gemm_dflash_oq4_mixed_plain_ref, 5)

// Production launch: eight independent output rows share one 256-thread
// block. Each wave still owns one complete dot product, so arithmetic and
// ragged-row semantics remain byte-for-byte identical to the one-wave oracle,
// while block scheduling overhead is amortized across eight rows.
#define DFLASH_OQ_GEMM_8W(NAME, MODE)                                          \
__launch_bounds__(256)                                                         \
extern "C" __global__ void NAME(                                              \
    const uint8_t* __restrict__ W,                                             \
    const float*   __restrict__ X,                                             \
    float*         __restrict__ Y,                                             \
    int M, int K, int B, int block_stride                                      \
) {                                                                            \
    const int lane = (int)threadIdx.x & 31;                                    \
    const int wave = (int)threadIdx.x >> 5;                                    \
    const int row = (int)blockIdx.x * 8 + wave;                                \
    const int batch = (int)blockIdx.y;                                         \
    if (row >= M || batch >= B) return;                                        \
    float acc = 0.0f;                                                          \
    for (int col = lane; col < K; col += 32) {                                 \
        const long long flat = (long long)row * K + col;                       \
        const long long block_index = flat >> 8;                               \
        const int pos = (int)(flat & 255);                                     \
        const uint8_t* block = W + block_index * block_stride;                 \
        int q;                                                                  \
        if ((MODE) == 8) {                                                     \
            q = (int)(int8_t)block[2 + pos];                                   \
        } else {                                                                \
            q = dflash_i4(block, pos);                                         \
            if ((MODE) == 5) {                                                 \
                const int overlays = (block_stride - 130) >> 1;                \
                const uint8_t* table = block + 130;                            \
                for (int entry = 0; entry < overlays; ++entry) {               \
                    if ((int)table[2 * entry] == pos)                          \
                        q = (int)(int8_t)table[2 * entry + 1];                  \
                }                                                               \
            }                                                                   \
        }                                                                       \
        const float scale = dflash_f16_scale(block);                           \
        const float activation = (float)(_Float16)X[(long long)batch * K + col]; \
        acc += scale * (float)q * activation;                                  \
    }                                                                           \
    acc = dflash_wave_sum(acc);                                                \
    if (lane == 0) Y[(long long)batch * M + row] = acc;                        \
}

DFLASH_OQ_GEMM_8W(gemm_dflash_oq8_plain_8w, 8)
DFLASH_OQ_GEMM_8W(gemm_dflash_oq4_plain_8w, 4)
DFLASH_OQ_GEMM_8W(gemm_dflash_oq4_mixed_plain_8w, 5)

static __device__ __forceinline__ int dflash_pack_i4_4(const uint8_t* block, int pos) {
    const uint16_t packed = *(const uint16_t*)(block + 2 + (pos >> 1));
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int q = (packed >> (4 * i)) & 0xf;
        q = (q << 28) >> 28;
        result |= (q & 0xff) << (8 * i);
    }
    return result;
}

static __device__ __forceinline__ int dflash_dot4_i8(int a, int b, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#else
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#endif
}

// Production W{4,8}A8 path. Eight waves share one activation group: all 256
// threads cooperatively quantize X to signed int8 in LDS once, then each wave
// computes one output row with dot4. Mixed OQ keeps the int4 bulk dot and adds
// exact sparse-int8 corrections; duplicate overlay positions retain last-wins
// semantics. K must be G256-aligned so checkpoint blocks do not cross rows.
#define DFLASH_OQ_GEMM_DP4A_8W(NAME, MODE)                                     \
__launch_bounds__(256)                                                         \
extern "C" __global__ void NAME(                                              \
    const uint8_t* __restrict__ W,                                             \
    const float*   __restrict__ X,                                             \
    float*         __restrict__ Y,                                             \
    int M, int K, int B, int block_stride                                      \
) {                                                                            \
    __shared__ __align__(4) int8_t xq[256];                                    \
    __shared__ float wave_max[8];                                              \
    __shared__ float x_scale;                                                  \
    const int tid = (int)threadIdx.x;                                          \
    const int lane = tid & 31;                                                 \
    const int wave = tid >> 5;                                                 \
    const int row = (int)blockIdx.x * 8 + wave;                                \
    const int batch = (int)blockIdx.y;                                         \
    const bool valid = row < M && batch < B;                                   \
    const int groups = K >> 8;                                                 \
    float acc = 0.0f;                                                          \
    for (int group = 0; group < groups; ++group) {                             \
        const int col = group * 256 + tid;                                     \
        const float xv = batch < B ? X[(long long)batch * K + col] : 0.0f;    \
        float vmax = fabsf(xv);                                                \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            vmax = fmaxf(vmax, __shfl_down(vmax, offset));                     \
        if (lane == 0) wave_max[wave] = vmax;                                  \
        __syncthreads();                                                       \
        if (wave == 0) {                                                       \
            float block_max = lane < 8 ? wave_max[lane] : 0.0f;               \
            for (int offset = 16; offset > 0; offset >>= 1)                    \
                block_max = fmaxf(block_max, __shfl_down(block_max, offset));  \
            if (lane == 0) x_scale = block_max > 0.0f ? block_max / 127.0f : 1.0f; \
        }                                                                      \
        __syncthreads();                                                       \
        int qx = __float2int_rn(xv / x_scale);                                 \
        qx = qx > 127 ? 127 : (qx < -127 ? -127 : qx);                        \
        xq[tid] = (int8_t)qx;                                                  \
        __syncthreads();                                                       \
        if (valid) {                                                           \
            const uint8_t* block = W +                                        \
                ((long long)row * groups + group) * block_stride;             \
            const int pos = lane * 8;                                         \
            const int x_a = *(const int*)(xq + pos);                           \
            const int x_b = *(const int*)(xq + pos + 4);                       \
            int w_a, w_b;                                                      \
            if ((MODE) == 8) {                                                \
                w_a = *(const int*)(block + 2 + pos);                          \
                w_b = *(const int*)(block + 2 + pos + 4);                      \
            } else {                                                           \
                w_a = dflash_pack_i4_4(block, pos);                            \
                w_b = dflash_pack_i4_4(block, pos + 4);                        \
            }                                                                  \
            int isum = dflash_dot4_i8(w_a, x_a, 0);                            \
            isum = dflash_dot4_i8(w_b, x_b, isum);                             \
            if ((MODE) == 5) {                                                \
                const int overlays = (block_stride - 130) >> 1;               \
                const uint8_t* table = block + 130;                            \
                for (int entry = lane; entry < overlays; entry += 32) {       \
                    const int overlay_pos = (int)table[2 * entry];             \
                    bool shadowed = false;                                     \
                    for (int later = entry + 1; later < overlays; ++later)     \
                        shadowed |= table[2 * later] == overlay_pos;           \
                    if (!shadowed) {                                           \
                        const int base_q = dflash_i4(block, overlay_pos);       \
                        const int replacement = (int)(int8_t)table[2 * entry + 1]; \
                        isum += (replacement - base_q) * (int)xq[overlay_pos]; \
                    }                                                          \
                }                                                              \
            }                                                                  \
            const float w_scale = dflash_f16_scale(block);                    \
            acc += (float)isum * w_scale * x_scale;                           \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    acc = dflash_wave_sum(acc);                                                \
    if (valid && lane == 0) Y[(long long)batch * M + row] = acc;              \
}

DFLASH_OQ_GEMM_DP4A_8W(gemm_dflash_oq8_plain_dp4a_8w, 8)
DFLASH_OQ_GEMM_DP4A_8W(gemm_dflash_oq4_plain_dp4a_8w, 4)
DFLASH_OQ_GEMM_DP4A_8W(gemm_dflash_oq4_mixed_plain_dp4a_8w, 5)

// Production activation staging. The original DP4A kernel above quantizes one
// G256 activation group independently in every eight-output-row block. That
// repeats identical work ceil(M/8) times for each projection. Stage X once per
// (batch, group) into reusable scratch instead. The reduction and rounding are
// intentionally identical to DFLASH_OQ_GEMM_DP4A_8W so this is a scheduling
// change, not a quantization-policy change.
__launch_bounds__(256)
extern "C" __global__ void quantize_dflash_act_g256(
    const float* __restrict__ X,
    int8_t* __restrict__ XQ,
    float* __restrict__ XS,
    int K,
    int B
) {
    __shared__ float wave_max[8];
    __shared__ float x_scale;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int group = (int)blockIdx.x;
    const int batch = (int)blockIdx.y;
    const int groups = K >> 8;
    if (group >= groups || batch >= B) return;

    const int col = group * 256 + tid;
    const float xv = X[(long long)batch * K + col];
    float vmax = fabsf(xv);
    for (int offset = 16; offset > 0; offset >>= 1)
        vmax = fmaxf(vmax, __shfl_down(vmax, offset));
    if (lane == 0) wave_max[wave] = vmax;
    __syncthreads();
    if (wave == 0) {
        float block_max = lane < 8 ? wave_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            block_max = fmaxf(block_max, __shfl_down(block_max, offset));
        if (lane == 0)
            x_scale = block_max > 0.0f ? block_max / 127.0f : 1.0f;
    }
    __syncthreads();
    int qx = __float2int_rn(xv / x_scale);
    qx = qx > 127 ? 127 : (qx < -127 ? -127 : qx);
    XQ[(long long)batch * K + col] = (int8_t)qx;
    if (tid == 0) XS[(long long)batch * groups + group] = x_scale;
}

// Eight output rows per block consume the staged G256 activations. No LDS or
// block barriers remain in the weight kernel; each wave owns one output row.
// Mixed OQ still applies sparse int8 last-write-wins corrections over the int4
// bulk dot product.
#define DFLASH_OQ_GEMM_DP4A_STAGED_8W(NAME, MODE)                              \
__launch_bounds__(256)                                                         \
extern "C" __global__ void NAME(                                              \
    const uint8_t* __restrict__ W,                                             \
    const int8_t*  __restrict__ XQ,                                            \
    const float*   __restrict__ XS,                                            \
    float*         __restrict__ Y,                                             \
    int M, int K, int B, int block_stride                                      \
) {                                                                            \
    const int lane = (int)threadIdx.x & 31;                                    \
    const int wave = (int)threadIdx.x >> 5;                                    \
    const int row = (int)blockIdx.x * 8 + wave;                                \
    const int batch = (int)blockIdx.y;                                         \
    if (row >= M || batch >= B) return;                                        \
    const int groups = K >> 8;                                                 \
    float acc = 0.0f;                                                          \
    for (int group = 0; group < groups; ++group) {                             \
        const uint8_t* block = W +                                             \
            ((long long)row * groups + group) * block_stride;                  \
        const int8_t* xq = XQ + (long long)batch * K + group * 256;            \
        const int pos = lane * 8;                                              \
        const int x_a = *(const int*)(xq + pos);                               \
        const int x_b = *(const int*)(xq + pos + 4);                           \
        int w_a, w_b;                                                          \
        if ((MODE) == 8) {                                                     \
            w_a = *(const int*)(block + 2 + pos);                              \
            w_b = *(const int*)(block + 2 + pos + 4);                          \
        } else {                                                               \
            w_a = dflash_pack_i4_4(block, pos);                                \
            w_b = dflash_pack_i4_4(block, pos + 4);                            \
        }                                                                      \
        int isum = dflash_dot4_i8(w_a, x_a, 0);                                \
        isum = dflash_dot4_i8(w_b, x_b, isum);                                 \
        if ((MODE) == 5) {                                                     \
            const int overlays = (block_stride - 130) >> 1;                   \
            const uint8_t* table = block + 130;                               \
            for (int entry = lane; entry < overlays; entry += 32) {           \
                const int overlay_pos = (int)table[2 * entry];                 \
                bool shadowed = false;                                        \
                for (int later = entry + 1; later < overlays; ++later)         \
                    shadowed |= table[2 * later] == overlay_pos;               \
                if (!shadowed) {                                              \
                    const int base_q = dflash_i4(block, overlay_pos);          \
                    const int replacement = (int)(int8_t)table[2 * entry + 1]; \
                    isum += (replacement - base_q) * (int)xq[overlay_pos];     \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float w_scale = dflash_f16_scale(block);                         \
        const float x_scale = XS[(long long)batch * groups + group];           \
        acc += (float)isum * w_scale * x_scale;                                \
    }                                                                          \
    acc = dflash_wave_sum(acc);                                                \
    if (lane == 0) Y[(long long)batch * M + row] = acc;                        \
}

DFLASH_OQ_GEMM_DP4A_STAGED_8W(gemm_dflash_oq8_plain_dp4a_staged_8w, 8)
DFLASH_OQ_GEMM_DP4A_STAGED_8W(gemm_dflash_oq4_plain_dp4a_staged_8w, 4)
DFLASH_OQ_GEMM_DP4A_STAGED_8W(gemm_dflash_oq4_mixed_plain_dp4a_staged_8w, 5)
