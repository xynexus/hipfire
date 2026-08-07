// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated GEMM: BF16 weight x BF16 input -> F32 output.
// gfx1100/gfx1151 wave32 path, matching gemm_f16_x_f16_wmma's output
// layout:
//   A : [M, K] BF16 row-major (raw uint16_t BF16 payload)
//   X : [B, K] BF16 row-major
//   Y : [B, M] F32 row-major

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_x_bf16_wmma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*               __restrict__ Y,
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

    const unsigned short* w_row = A + (long long)safe_m * K;
    const unsigned short* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 16) {
        b16x16_t a_frag = *(const b16x16_t*)(w_row + k);
        b16x16_t x_frag = *(const b16x16_t*)(x_row + k);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, x_frag, acc);
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

// gfx1151 large-M BF16 GEMM variant.
//
// Computes the same layout as gemm_bf16_x_bf16_wmma:
//   A [M, K] BF16, X [B, K] BF16, Y [B, M] F32.
//
// One block computes 128 output rows x 16 batch rows using 8 wave32 warps.
// The 16-row X tile is staged once per K_BLOCK and reused by all 8 warps,
// reducing activation reloads and launch overhead for LM-head-sized M.
#define M_TILE         128
#define N_TILE         16
#define WARPS          8
#define M_PER_WARP     16
#define K_BLOCK        256
#define SLOTS_PER_WARP 2

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_bf16_x_bf16_wmma_gfx1151_m128(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*               __restrict__ Y,
    int M, int K, int B
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = blockIdx.x * M_TILE;
    const int batch_start = blockIdx.y * N_TILE;

    if (row_start >= M || batch_start >= B) return;

    __shared__ unsigned short lds_x[N_TILE * K_BLOCK];

    const int warp_row_start = row_start + warp_id * M_PER_WARP;
    const int my_row = warp_row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const unsigned short* w_row = A + (long long)safe_row * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const b16x16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int k_base = 0; k_base < K; k_base += K_BLOCK) {
        const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 4);
        const int k_off = (lane & 15) * 16;
        const int batch_row = batch_start + slot_in_block;
        b16x16_t x_frag = zero_b;
        if (batch_row < B && k_base + k_off < K) {
            x_frag = *(const b16x16_t*)(X + (long long)batch_row * K + k_base + k_off);
        }
        *(b16x16_t*)(lds_x + slot_in_block * K_BLOCK + k_off) = x_frag;

        __syncthreads();

        const int group_k = ((K - k_base) < K_BLOCK) ? (K - k_base) : K_BLOCK;
        for (int kt = 0; kt < group_k; kt += 16) {
            b16x16_t a_frag = *(const b16x16_t*)(w_row + k_base + kt);
            b16x16_t b_frag = *(const b16x16_t*)(lds_x + lane16 * K_BLOCK + kt);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
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
#undef WARPS
#undef M_PER_WARP
#undef K_BLOCK
#undef SLOTS_PER_WARP

// gfx1151 N-heavy BF16 GEMM variant.
//
// Same layout as the two above: A [M, K] BF16, X [B, K] BF16, Y [B, M] F32.
//
// `gemm_bf16_x_bf16_wmma_gfx1151_m128` gives each warp ONE 16x16 output tile and
// one accumulator, so every A fragment it loads feeds exactly one WMMA op. With
// N_TILE=16 a prefill window of 256 columns is 16 column-blocks, and each of them
// re-reads the whole of A: A traffic is M*K*2 * (B/16). Measured, that kernel
// runs 3.3 TFLOPS on gfx1151 (1200 calls x 837 us for a 2048-token chunk), about
// 10-13% of peak, and it is 61% of a KLD reference build's GPU time.
//
// Here each warp keeps NH_NSUB accumulators and computes a 16 x (16*NH_NSUB)
// output strip, so ONE A fragment feeds NH_NSUB WMMA ops. A traffic drops by
// NH_NSUB and arithmetic intensity rises by it. This is the "N-heavy" shape from
// the gfx1151 iu4 GEMM tuning notes, which reached ~50% of peak there.
//
// K_BLOCK is halved to 128 to keep the wider X tile in LDS:
// 64 columns x 128 k x 2 B = 16 KB.
#define NH_M_TILE     128
#define NH_N_TILE      64
#define NH_M_PER_WARP  16
#define NH_K_BLOCK    128
#define NH_NSUB         4   // NH_N_TILE / 16 accumulators per warp

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_bf16_x_bf16_wmma_gfx1151_nheavy(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*               __restrict__ Y,
    int M, int K, int B
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start   = blockIdx.x * NH_M_TILE;
    const int batch_start = blockIdx.y * NH_N_TILE;
    if (row_start >= M || batch_start >= B) return;

    __shared__ unsigned short lds_x[NH_N_TILE * NH_K_BLOCK];

    const int warp_row_start = row_start + warp_id * NH_M_PER_WARP;
    const int my_row = warp_row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const unsigned short* w_row = A + (long long)safe_row * K;

    float8_t acc[NH_NSUB];
    #pragma unroll
    for (int j = 0; j < NH_NSUB; ++j) {
        acc[j] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    const b16x16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int k_base = 0; k_base < K; k_base += NH_K_BLOCK) {
        // 64 slots x 8 k-fragments = 512 fragments, two per thread.
        for (int f = tid; f < NH_N_TILE * (NH_K_BLOCK / 16); f += 256) {
            const int slot = f / (NH_K_BLOCK / 16);
            const int ko   = (f % (NH_K_BLOCK / 16)) * 16;
            const int batch_row = batch_start + slot;
            b16x16_t v = zero_b;
            if (batch_row < B && k_base + ko < K) {
                v = *(const b16x16_t*)(X + (long long)batch_row * K + k_base + ko);
            }
            *(b16x16_t*)(lds_x + slot * NH_K_BLOCK + ko) = v;
        }
        __syncthreads();

        const int group_k = ((K - k_base) < NH_K_BLOCK) ? (K - k_base) : NH_K_BLOCK;
        for (int kt = 0; kt < group_k; kt += 16) {
            // Loaded ONCE, spent across NH_NSUB output subtiles — the point.
            b16x16_t a_frag = *(const b16x16_t*)(w_row + k_base + kt);
            #pragma unroll
            for (int j = 0; j < NH_NSUB; ++j) {
                b16x16_t b_frag =
                    *(const b16x16_t*)(lds_x + (j * 16 + lane16) * NH_K_BLOCK + kt);
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc[j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < NH_NSUB; ++j) {
        const int out_col = batch_start + j * 16 + lane16;
        if (out_col < B) {
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                int out_row = warp_row_start + 2 * q + (lane >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[j][q];
                }
            }
        }
    }
}

#undef NH_M_TILE
#undef NH_N_TILE
#undef NH_M_PER_WARP
#undef NH_K_BLOCK
#undef NH_NSUB
