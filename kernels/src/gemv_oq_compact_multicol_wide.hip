// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// WIDE-LOAD twin of `gemv_oq_compact_multicol`. Same contract, same output
// layout, same exact-integer-per-group scaling; what changes is the shape of
// BOTH memory streams.
//
// The narrow kernel is issue-bound, by its own diagnosis: per group-round only
// about 4 of ~32 memory instructions carry weights, so ~12% of issue slots move
// weight traffic and it sustains ~63 GB/s of a ~250 GB/s part. Widening the
// WEIGHT load alone does not fix that -- activations scale with weights, so the
// ratio is unchanged. Both sides have to widen together:
//
//   narrow: lane owns  8 weights   1 dword weight  + BC*2 dword activation
//   wide:   lane owns 32 weights   1 dwordx4 "     + BC*2 dwordx4 "
//
// Same instruction count per group-round, 4x the weights. That is the whole
// idea, and it is the shape `gemv_oq_compact_grouped_v3` already uses to reach
// ~239 GB/s on this same weight layout at B=1.
//
// Lane split follows v3: 8 lanes per group, so one wave32 covers FOUR groups per
// round and `ng % 4 == 0` is required (K % 1024 == 0). Callers fall back to the
// narrow kernel otherwise.
//
//   W  : compact SPLIT-PLANE blocks (all nibbles, then all [f16 scale][table])
//   Xq : [B, K] int8   Xs : [B, K/256] f32   Y : [B, M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCMW_MAX_B 16

static __device__ __forceinline__ int oqcmw_dot4_i8(int a, int b, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#else
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#endif
}

// Decode one nibble dword into two packed int8x4, MSB-safe sign extension.
// Identical to the narrow kernel's decode so the two agree bit-for-bit on the
// weight values.
static __device__ __forceinline__ void oqcmw_decode(uint32_t w, uint32_t* p0, uint32_t* p1) {
    const uint32_t lo4 = w & 0x0F0F0F0Fu;
    const uint32_t hi4 = (w >> 4) & 0x0F0F0F0Fu;
    uint32_t a = __builtin_amdgcn_perm(hi4, lo4, 0x05010400u);
    uint32_t b = __builtin_amdgcn_perm(hi4, lo4, 0x07030602u);
    a |= ((a >> 3) & 0x01010101u) * 0xF0u;
    b |= ((b >> 3) & 0x01010101u) * 0xF0u;
    *p0 = a;
    *p1 = b;
}

template <int BC, int RW>
static __device__ __forceinline__ void oqcmw_body(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ Xq,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int block_stride
) {
    const int lane = threadIdx.x & 31;
    const int wpb  = blockDim.x >> 5;
    const int wid  = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int nw   = gridDim.x * wpb;
    const int ng   = K / 256;
    const int n_ov = (block_stride - 130) >> 1;

    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * ng * 128LL;

    const int gir = lane >> 3;   // 0..3 — which of the four groups this round
    const int lig = lane & 7;    // 0..7 — which 32-weight chunk of that group
    const int elt = lig * 32;    // weight/activation offset inside the group

    for (int row0 = wid * RW; row0 < M; row0 += nw * RW) {
        const uint8_t* n_row[RW];
        const uint8_t* s_row[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (row0 + r < M) ? (row0 + r) : (M - 1);
            n_row[r] = W + (long long)row * ng * 128LL;
            s_row[r] = W + side_base + (long long)row * ng * (long long)side_stride;
        }

        float facc[RW][BC];
        #pragma unroll
        for (int r = 0; r < RW; ++r)
            #pragma unroll
            for (int b = 0; b < BC; ++b) facc[r][b] = 0.0f;

        for (int g0 = 0; g0 < ng; g0 += 4) {
            const int gi = g0 + gir;

            // 16 bytes = this lane's 32 weights, 16-byte aligned inside a
            // 128-byte-aligned row.
            uint32_t wv[RW][4];
            #pragma unroll
            for (int r = 0; r < RW; ++r)
                __builtin_memcpy(wv[r], n_row[r] + (long long)gi * 128 + lig * 16, 16);

            uint32_t p[RW][8];
            #pragma unroll
            for (int r = 0; r < RW; ++r)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    oqcmw_decode(wv[r][j], &p[r][2 * j], &p[r][2 * j + 1]);

            float sw[RW];
            const uint8_t* table[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
                const uint8_t* sb = s_row[r] + (long long)gi * side_stride;
                table[r] = sb + 2;
                sw[r] = (float)(*(const _Float16*)sb);
            }

            // Overlay: 8 lanes per group, so lane `lig` owns slot `lig`, exactly
            // as v3 does. Branchless — the inactive lanes multiply by zero.
            const bool has_ov = lig < n_ov;
            int oidx[RW], oval[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
                const int oe = has_ov ? lig : 0;
                oidx[r] = (int)table[r][2 * oe];
                oval[r] = has_ov ? (int)(int8_t)table[r][2 * oe + 1] : 0;
            }

            float sx[BC];
            #pragma unroll
            for (int b = 0; b < BC; ++b) sx[b] = Xs[(long long)b * ng + gi];

            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                const int8_t* xq = Xq + (long long)b * K + (long long)gi * 256;
                // 32 activations for this lane's 32 weights: two dwordx4.
                uint32_t xv[8];
                __builtin_memcpy(&xv[0], xq + elt, 16);
                __builtin_memcpy(&xv[4], xq + elt + 16, 16);
                #pragma unroll
                for (int r = 0; r < RW; ++r) {
                    int acc = 0;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc = oqcmw_dot4_i8((int)p[r][j], (int)xv[j], acc);
                    acc += oval[r] * (int)xq[oidx[r]];
                    for (int e = lig + 8; e < n_ov; e += 8) {
                        acc += (int)(int8_t)table[r][2 * e + 1] * (int)xq[(int)table[r][2 * e]];
                    }
                    facc[r][b] += (float)acc * sw[r] * sx[b];
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < RW; ++r)
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    facc[r][b] += __shfl_xor(facc[r][b], off);
            }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
                if (row0 + r >= M) break;
                #pragma unroll
                for (int b = 0; b < BC; ++b)
                    Y[(long long)b * M + row0 + r] = facc[r][b];
            }
        }
    }
}

#define OQCMW_ENTRY(N)                                                         \
    __launch_bounds__(256) extern "C" __global__                               \
    void gemv_oq_compact_multicol_w##N(                                        \
        const uint8_t* __restrict__ W, const int8_t* __restrict__ Xq,          \
        const float* __restrict__ Xs, float* __restrict__ Y,                   \
        int M, int K, int B, int block_stride                                  \
    ) { (void)B; oqcmw_body<N, 3>(W, Xq, Xs, Y, M, K, block_stride); }

OQCMW_ENTRY(1)  OQCMW_ENTRY(2)  OQCMW_ENTRY(3)  OQCMW_ENTRY(4)
OQCMW_ENTRY(5)  OQCMW_ENTRY(6)  OQCMW_ENTRY(7)  OQCMW_ENTRY(8)
OQCMW_ENTRY(9)  OQCMW_ENTRY(10) OQCMW_ENTRY(11) OQCMW_ENTRY(12)
OQCMW_ENTRY(13) OQCMW_ENTRY(14) OQCMW_ENTRY(15) OQCMW_ENTRY(16)
