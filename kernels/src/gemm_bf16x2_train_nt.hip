// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Split-precision ("2xbf16") training GEMM (NT), F32 in / F32 out — near-f32
// accuracy ON the WMMA matrix cores.
//
// Each f32 operand is split into a bf16 hi + bf16 lo residual:
//     v_hi = round_bf16(v);   v_lo = round_bf16(v - v_hi)
// and the product is accumulated (f32) from three WMMA passes, dropping the
// negligible lo*lo term (~2^-16 relative):
//     A*X ≈ A_hi*X_hi + A_hi*X_lo + A_lo*X_hi
// Two 8-bit bf16 halves capture ~16 mantissa bits (vs f32's 24, bf16's 8), so
// the result is far more accurate than a single bf16 pass — enough for the
// precision-sensitive vocab-head logits that feed the softmax/CE loss — while
// still running on the matrix cores (3 WMMA passes, still ~5-10x the scalar f32
// path). bf16 (not f16) is the base so the hi term keeps full f32 range.
//
// Computes Y[B,M] = X[B,K]·A[M,K]^T. Map onto the training linear forward
// Y[Mtok,Nout] = X[Mtok,K]·W[Nout,K]^T as A=W (M=Nout), X=X (B=Mtok). Fragment/
// output layout matches gemm_bf16c_train_nt. gfx1103/gfx1151 wave32, ZERO LDS.
// acc[j] at lane t maps to (row = 2*j + (t>>4), col = t & 15). K may be any
// length (tail zero-masked). This is the LDS-free reference; the M-heavy LDS
// variant (gemm_bf16x2_train_nt_gfx1151_m128) stages the split activation.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// F32 -> BF16, round-to-nearest-even.
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(b >> 16); // NaN
    unsigned int bias = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + bias) >> 16);
}
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
// Split f32 v into bf16 hi + bf16 lo (residual). Returns hi, writes lo.
__device__ __forceinline__ unsigned short split_bf16(float v, unsigned short* lo) {
    unsigned short hi = f32_to_bf16_rne(v);
    *lo = f32_to_bf16_rne(v - bf16_to_f32(hi));
    return hi;
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16x2_train_nt(
    const float* __restrict__ A,   // [M, K] F32
    const float* __restrict__ X,   // [B, K] F32
    float*       __restrict__ Y,   // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    int my_m = row_start + lane;
    int my_b = batch_start + lane;
    int safe_m = (my_m < M) ? my_m : (M - 1);
    int safe_b = (my_b < B) ? my_b : (B - 1);

    const float* w_row = A + (long long)safe_m * K;
    const float* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 16) {
        b16x16_t a_hi, a_lo, x_hi, x_lo;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int kk = k + i;
            float av = (kk < K) ? w_row[kk] : 0.0f;
            float xv = (kk < K) ? x_row[kk] : 0.0f;
            unsigned short al, xl;
            a_hi[i] = (short)split_bf16(av, &al);
            a_lo[i] = (short)al;
            x_hi[i] = (short)split_bf16(xv, &xl);
            x_lo[i] = (short)xl;
        }
        // A*X ≈ hi*hi + hi*lo + lo*hi  (drop lo*lo).
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_hi, x_hi, acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_hi, x_lo, acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_lo, x_hi, acc);
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

// gfx1151 large-M split-precision variant (mirrors gemm_bf16_x_bf16_wmma_gfx1151
// _m128). One block computes 128 output rows x 16 batch rows with 8 wave32
// warps; the 16-row activation tile is split (hi+lo) and staged in LDS once per
// K_BLOCK, then reused by all 8 warps — the bandwidth win for the M-heavy vocab
// head (huge n_out, small batch). Weight rows are streamed from global and split
// in-register. 3 WMMA passes per K-tile, same near-f32 accuracy as the LDS-free
// kernel above.
#define M_TILE         128
#define N_TILE         16
#define M_PER_WARP     16
#define K_BLOCK        256
#define SLOTS_PER_WARP 2

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_bf16x2_train_nt_gfx1151_m128(
    const float* __restrict__ A,   // [M, K] F32
    const float* __restrict__ X,   // [B, K] F32
    float*       __restrict__ Y,   // [B, M] F32
    int M, int K, int B
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = blockIdx.x * M_TILE;
    const int batch_start = blockIdx.y * N_TILE;

    if (row_start >= M || batch_start >= B) return;

    __shared__ unsigned short lds_xhi[N_TILE * K_BLOCK];
    __shared__ unsigned short lds_xlo[N_TILE * K_BLOCK];

    const int warp_row_start = row_start + warp_id * M_PER_WARP;
    const int my_row = warp_row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const float* w_row = A + (long long)safe_row * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k_base = 0; k_base < K; k_base += K_BLOCK) {
        // Cooperatively load + split X[16, K_BLOCK] into LDS (hi + lo).
        const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 4);
        const int k_off = (lane & 15) * 16;
        const int batch_row = batch_start + slot_in_block;
        b16x16_t xhi_frag, xlo_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float xv = 0.0f;
            int kk = k_base + k_off + i;
            if (batch_row < B && kk < K) {
                xv = X[(long long)batch_row * K + kk];
            }
            unsigned short lo;
            xhi_frag[i] = (short)split_bf16(xv, &lo);
            xlo_frag[i] = (short)lo;
        }
        *(b16x16_t*)(lds_xhi + slot_in_block * K_BLOCK + k_off) = xhi_frag;
        *(b16x16_t*)(lds_xlo + slot_in_block * K_BLOCK + k_off) = xlo_frag;

        __syncthreads();

        const int group_k = ((K - k_base) < K_BLOCK) ? (K - k_base) : K_BLOCK;
        for (int kt = 0; kt < group_k; kt += 16) {
            b16x16_t a_hi, a_lo;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float av = 0.0f;
                int kk = k_base + kt + i;
                if (kk < K) av = w_row[kk];
                unsigned short al;
                a_hi[i] = (short)split_bf16(av, &al);
                a_lo[i] = (short)al;
            }
            b16x16_t b_hi = *(const b16x16_t*)(lds_xhi + lane16 * K_BLOCK + kt);
            b16x16_t b_lo = *(const b16x16_t*)(lds_xlo + lane16 * K_BLOCK + kt);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_hi, b_hi, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_hi, b_lo, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_lo, b_hi, acc);
        }

        __syncthreads();
    }

    const int out_col = batch_start + lane16;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = warp_row_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#undef M_TILE
#undef N_TILE
#undef M_PER_WARP
#undef K_BLOCK
#undef SLOTS_PER_WARP
