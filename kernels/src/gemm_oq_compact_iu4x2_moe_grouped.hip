// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Grouped compact-MoE GEMM on the INTEGER matrix cores: the inner loop of the
// dense `gemm_oq_compact_iu4x2_wmma`, one 16-slot expert tile per workgroup.
//
// WHY: the routed experts of Qwen3.6-35B-A3B were 61% of its prefill kernel time
// on `gemm_oq_compact_moe_grouped_f32` (+ `_narrow`), which does the dot products
// on VALU at ~2.5 TFLOPS, while the dense iu4x2 GEMM beside it runs ~34 int8 TOPS.
// The f16 `gemm_oq_compact_moe_grouped_wmma` sibling is one 32-thread wave per
// 16x16 tile with no shared staging and measured SLOWER than the f32 kernel.
//
// Shape: grid.y walks the 16-slot tiles of `expert_tile_ids` (one expert each, -1
// = padding), grid.x walks blocks of 16*OQMG_MW weight rows; each wave owns 16
// rows. Per 256-group the tile's 16 int8 activation rows are staged ONCE into LDS
// as the hi (signed) / lo (unsigned) int4 digit planes the dense kernel uses, and
// raw, for the overlay. Every wave then runs 16 k-tiles x 2 iu4 WMMAs against
// its rows, recombines 16*hi + lo exactly in i32, adds the group's sparse overlay
// (val * xq[idx], also i32), and scales by the row's f16 scale and the slot's
// activation scale.
//
// NUMERICS: W4A8 -- activations int8 per 256-group (`quantize_act_oq8`), exactly
// as the dense prefill GEMMs already run. Not bit-exact with the decode GEMV the
// f32 kernel mirrors; MoE path 2 runs only at >= 64 rows, so DFlash verify
// (B <= 16) never reaches it. parity_gemm_oq_compact_moe_grouped checks it
// against an oracle over the SAME quantized activations (integer path exact) and
// against the f32 one (the quantization gap).
//
//   W  : per expert, compact SPLIT-PLANE blocks: nibbles [M][ng][128], then side
//        [M][ng][block_stride-128] = f16 scale, then n_ov (u8 idx, i8 val) pairs
//   Xq : [rows, K] int8   Xs : [rows, K/256] f32   (row = flat / x_row_div)
//   Y  : [m_total, M] f32, slot-major like the f32 kernel

// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

#ifndef OQMG_MW
#define OQMG_MW 8
#endif
#define OQMG_TILE 16
#define OQMG_PAD  4

static __device__ __forceinline__ int32x2_t oqmg_load(const uint8_t* p) {
    return (int32x2_t){(int32_t)*(const uint32_t*)(p + 0),
                       (int32_t)*(const uint32_t*)(p + 4)};
}

// Pack the low nibble of each of 4 bytes into the low 16 bits, in order.
// Input must already be masked to 0x0f0f0f0f.
static __device__ __forceinline__ uint32_t oqmg_pack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}

__launch_bounds__(32 * OQMG_MW, 1)
extern "C" __global__ void gemm_oq_compact_iu4x2_moe_grouped(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const int8_t*             __restrict__ Xq,
    const float*              __restrict__ Xs,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int block_stride,
    // Non-null (gate_up, x_row_div == 1): write each slot's rows straight to
    // their per-(token, k_rank) gate / up rows -- `moe_gate_up_unscatter_k8`'s
    // job, done here so Y_grouped is never written and read back.
    float*                    __restrict__ Y_gate,
    float*                    __restrict__ Y_up
) {
    const int tile_y = blockIdx.y;
    const int expert = expert_tile_ids[tile_y];
    // Workgroup-uniform, so it may return before the barriers below.
    if (expert < 0) return;

    const int tid   = threadIdx.x;
    const int wave  = tid >> 5;
    const int wlane = tid & 31;
    const int lane  = wlane & 15;  // weight row (A) and slot column (B, output)
    const int half  = wlane >> 4;
    const int row_start  = (blockIdx.x * OQMG_MW + wave) * 16;
    const int slot_start = tile_y * OQMG_TILE;
    const int ng = K >> 8;
    const int side_stride = block_stride - 128;
    const int n_ov = (side_stride - 2) >> 1;
    const uint8_t* __restrict__ W =
        reinterpret_cast<const uint8_t*>(expert_weight_ptrs[expert]);
    const long long side_base = (long long)M * ng * 128;
    const int a_row  = row_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);

    __shared__ int xrow_s[OQMG_TILE];
    __shared__ float xs_s[OQMG_TILE];
    __shared__ uint8_t xh[OQMG_TILE * (128 + OQMG_PAD)];
    __shared__ uint8_t xl[OQMG_TILE * (128 + OQMG_PAD)];
    __shared__ int8_t  xr8[OQMG_TILE * (256 + OQMG_PAD)];
    const int lds_stride = 128 + OQMG_PAD;
    const int raw_stride = 256 + OQMG_PAD;

    if (tid < OQMG_TILE) {
        const int slot = slot_start + tid;
        int xr = -1;
        if (slot < m_total) {
            const int flat = sorted_slot_index[slot];
            if (flat >= 0) xr = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
        xrow_s[tid] = xr;
    }

    float8_t facc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < ng; ++g) {
        // The previous group's LDS is read out (and, first time round, xrow_s is
        // visible) before anything is overwritten.
        __syncthreads();
        // 16 columns x 32 output dwords; one dword = 8 activations.
        for (int i = tid; i < OQMG_TILE * 32; i += blockDim.x) {
            const int c = i >> 5;
            const int d = i & 31;
            const int xr = xrow_s[c];
            uint32_t a0 = 0, a1 = 0;  // an empty slot stages zeros, never stored
            if (xr >= 0) {
                const long long off = (long long)xr * K + (long long)g * 256 + d * 8;
                a0 = *(const uint32_t*)(Xq + off);
                a1 = *(const uint32_t*)(Xq + off + 4);
            }
            const uint32_t hp = oqmg_pack4((a0 >> 4) & 0x0f0f0f0fu)
                              | (oqmg_pack4((a1 >> 4) & 0x0f0f0f0fu) << 16);
            const uint32_t lp = oqmg_pack4(a0 & 0x0f0f0f0fu)
                              | (oqmg_pack4(a1 & 0x0f0f0f0fu) << 16);
            *(uint32_t*)(xh + c * lds_stride + d * 4) = hp;
            *(uint32_t*)(xl + c * lds_stride + d * 4) = lp;
            *(uint32_t*)(xr8 + c * raw_stride + d * 8) = a0;
            *(uint32_t*)(xr8 + c * raw_stride + d * 8 + 4) = a1;
        }
        if (tid < OQMG_TILE) {
            const int xr = xrow_s[tid];
            xs_s[tid] = (xr >= 0) ? Xs[(long long)xr * ng + g] : 0.f;
        }
        __syncthreads();

        // Wave-uniform: no barrier inside.
        if (row_start < M) {
            const uint8_t* nib = W + ((long long)safe_a * ng + g) * 128;
            int32x8_t ihi = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t ilo = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kt = 0; kt < 16; ++kt) {
                // One weight load, both digit passes.
                const int32x2_t a = oqmg_load(nib + kt * 8);
                const int off = lane * lds_stride + kt * 8;
                const int32x2_t bh = oqmg_load(xh + off);
                const int32x2_t bl = oqmg_load(xl + off);
                ihi = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, bh, ihi, false);
                ilo = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, false, bl, ilo, false);
            }
            const float sx = xs_s[lane];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int m_row = row_start + 2 * j + half;
                if (m_row < M) {
                    const uint8_t* sb = W + side_base
                        + ((long long)m_row * ng + g) * (long long)side_stride;
                    const float sw = (float)(*(const _Float16*)sb);
                    int v = 16 * ihi[j] + ilo[j];
                    for (int e = 0; e < n_ov; ++e) {
                        const int idx = (int)sb[2 + 2 * e];
                        const int val = (int)(int8_t)sb[3 + 2 * e];
                        v += val * (int)xr8[lane * raw_stride + idx];
                    }
                    facc[j] += (float)v * sw * sx;
                }
            }
        }
    }

    const int xr = xrow_s[lane];
    if (row_start >= M || xr < 0) return;
    const int slot = slot_start + lane;
    if (Y_gate != nullptr) {
        // xr is the slot's flat (token * K_TOP + k_rank) index here.
        const int mi = M >> 1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            if (m_row < mi) Y_gate[(long long)xr * mi + m_row] = facc[j];
            else if (m_row < M) Y_up[(long long)xr * mi + (m_row - mi)] = facc[j];
        }
        return;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m_row = row_start + 2 * j + half;
        if (m_row < M) Y_grouped[(long long)slot * M + m_row] = facc[j];
    }
}
