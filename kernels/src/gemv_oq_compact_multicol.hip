// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus SMALL-BATCH GEMM: Y[b, m] = sum_k W[m,k] * Xq[b,k].
//
// `gemm_oq_compact_grouped_wmma` tiles B by 16, so at spec-decode's verify width
// (B = K+1, typically 8) it computes a 16-wide WMMA tile for 8 useful columns and
// has no N to amortize the int4 decode + overlay scan over. Measured on
// Qwen3.8-27B oq4.25++, a B=8 verify costs 64 us per draft token against plain
// decode's 66 — verify is barely cheaper per token than just decoding, which is
// why spec-decode loses on this family.
//
// This reads each weight row ONCE, exactly like the decode GEMV, and accumulates
// B columns against it: one weight sweep for all B tokens instead of one sweep
// per token. Deliberately the SIMPLEST correct shape — ONE group per wave round,
// 32 lanes x 8 weights = 256 — so there is no lane-group split and no ng
// divisibility requirement.
//
//   W  : compact SPLIT-PLANE blocks (all nibbles, then all [f16 scale][table])
//   Xq : [B, K] int8   Xs : [B, K/256] f32   Y : [B, M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCM_MAX_B 16

// V_DOT4 with the same arch guard `gemm_dflash_oq_plain_ref.hip` uses: gfx11xx /
// gfx12xx expose it as sudot4 with explicit per-operand signedness, older parts
// as sdot4. gfx1151 has NO `dot1-insts`, so a bare `__builtin_amdgcn_sdot4`
// fails to compile here — keep the guard rather than calling it directly.
static __device__ __forceinline__ int oqcm_dot4_i8(int a, int b, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#else
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#endif
}

// B IS A COMPILE-TIME PARAMETER. With a runtime B the compiler cannot unroll the
// column loop, so `facc[b]` becomes an INDEXED register access — `s_mov_b32 m0`
// plus a `v_movrels_b32`/`v_movreld_b32` read-modify-write per column — and each
// iteration also ends up with its own `s_waitcnt lgkmcnt(0)` waiting on that
// column's `Xs` scalar load. Measured, the column loop cost ~0.070 ms/column
// against a ~0.207 ms fixed part that is just the weight stream (45.2 MiB at
// 233 GB/s = 0.19 ms), i.e. the per-column overhead — not arithmetic — was the
// whole addressable cost. Specializing on B unrolls it into distinct registers,
// removes m0 entirely, and lets one waitcnt cover all B scale loads.
//
// The dispatch asserts 1 <= B <= 16, so every reachable B has a specialization
// and there is no runtime-B fallback to fall into.
// RW = ROWS PER WAVE. The activations for a group are identical for every row,
// but a one-row-per-wave loop re-loads them for each row it owns: per row per
// group there is ONE weight dword against 3*B activation loads, so at B=8,
// ng=20 a row costs 20 weight loads and 480 activation loads — 24:1. The kernel
// is bound by memory-instruction ISSUE on activations that are already cache
// hits, not by weight bandwidth. Holding RW rows in flight shares each
// activation load across RW rows and divides that 24:1 down by RW.
template <int BC, int RW>
static __device__ __forceinline__ void oqcm_body(
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

    for (int row0 = wid * RW; row0 < M; row0 += nw * RW) {
        const uint8_t* n_row[RW];
        const uint8_t* s_row[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) {
            // Clamp instead of branching: an out-of-range row reads valid bytes
            // and its result is simply not stored at the end.
            const int row = (row0 + r < M) ? (row0 + r) : (M - 1);
            n_row[r] = W + (long long)row * ng * 128LL;
            s_row[r] = W + side_base + (long long)row * ng * (long long)side_stride;
        }

        float facc[RW][BC];
        #pragma unroll
        for (int r = 0; r < RW; ++r)
            #pragma unroll
            for (int b = 0; b < BC; ++b) facc[r][b] = 0.0f;

        for (int g = 0; g < ng; ++g) {
            const uint8_t* table[RW];
            float sw[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
                const uint8_t* sb = s_row[r] + (long long)g * side_stride;
                table[r] = sb + 2;
                sw[r] = (float)(*(const _Float16*)sb);
            }

            // 4 bytes = this lane's 8 weights. 32 lanes x 8 = 256 = one group.
            uint32_t w4[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r)
                __builtin_memcpy(&w4[r], n_row[r] + (long long)g * 128 + lane * 4, 4);

            // Decode the 8 nibbles ONCE per group into two packed int8x4 for
            // V_DOT4, instead of 8 scalar shift/mask/sign-extends re-done for
            // every column. v_perm_b32 does the lo/hi interleave in one
            // instruction: q order is [lo.b0, hi.b0, lo.b1, hi.b1, ...].
            uint32_t p0[RW], p1[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
            const uint32_t lo4 = w4[r] & 0x0F0F0F0Fu;
            const uint32_t hi4 = (w4[r] >> 4) & 0x0F0F0F0Fu;
            p0[r] = __builtin_amdgcn_perm(hi4, lo4, 0x05010400u);
            p1[r] = __builtin_amdgcn_perm(hi4, lo4, 0x07030602u);
            // Sign-extend 4->8 bits per byte by OR, not by subtract: each byte
            // holds 0..15, so `x - 16` would BORROW into the next byte. Setting
            // bits 4-7 where bit 3 is set is borrow-free (1 * 0xF0 = 240 < 256,
            // so the multiply cannot carry across byte lanes either).
            p0[r] |= ((p0[r] >> 3) & 0x01010101u) * 0xF0u;
            p1[r] |= ((p1[r] >> 3) & 0x01010101u) * 0xF0u;
            }

            // Overlays REPLACE the bulk nibble and the loader ZEROED it, so a
            // correction is val*x[idx] and needs no lane's registers. Lane `lane`
            // takes entry `lane`; the wave reduction sums them wherever they land.
            const int has_ov = (lane < n_ov);
            const int oe     = has_ov ? lane : 0;
            int oidx[RW], oval[RW];
            #pragma unroll
            for (int r = 0; r < RW; ++r) {
                oidx[r] = (int)table[r][2 * oe];
                oval[r] = has_ov ? (int)(int8_t)table[r][2 * oe + 1] : 0;
            }

            // Issue all BC scale loads together so ONE waitcnt covers them,
            // instead of one drain per column inside the loop below.
            float sx[BC];
            #pragma unroll
            for (int b = 0; b < BC; ++b) sx[b] = Xs[(long long)b * ng + g];

            const int elt = lane * 8;
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                const int8_t* xq = Xq + (long long)b * K + (long long)g * 256;
                uint32_t x0, x1;
                __builtin_memcpy(&x0, xq + elt, 4);
                __builtin_memcpy(&x1, xq + elt + 4, 4);
                #pragma unroll
                for (int r = 0; r < RW; ++r) {
                int acc = oqcm_dot4_i8((int)p0[r], (int)x0, 0);
                acc = oqcm_dot4_i8((int)p1[r], (int)x1, acc);
                acc += oval[r] * (int)xq[oidx[r]];
                // Tables deeper than one entry per lane (32 here) never occur in
                // shipped artifacts, but must not be silently dropped.
                for (int e = lane + 32; e < n_ov; e += 32) {
                    acc += (int)(int8_t)table[r][2 * e + 1] * (int)xq[(int)table[r][2 * e]];
                }
                // Reduce the group's dot to an EXACT INTEGER before scaling, so
                // this is BIT-IDENTICAL to the WMMA GEMM rather than merely close.
                // Integer addition is associative, so the lane split does not
                // change the sum; scaling per-lane partials in f32 would. The
                // WMMA path accumulates the whole 256-wide group in int32 and
                // then does one `+= (float)iacc * sw * sx` per group, and this
                // now matches that order exactly.
                facc[r][b] += (float)acc * sw[r] * sx[b];
                }
            }
        }

        // ONE reduction per row instead of one per (group, column).
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

#define OQCM_ENTRY(N)                                                          \
    __launch_bounds__(256) extern "C" __global__                               \
    void gemv_oq_compact_multicol_b##N(                                        \
        const uint8_t* __restrict__ W, const int8_t* __restrict__ Xq,          \
        const float* __restrict__ Xs, float* __restrict__ Y,                   \
        int M, int K, int B, int block_stride                                  \
    ) { (void)B; oqcm_body<N, (N <= 8 ? 4 : 2)>(W, Xq, Xs, Y, M, K, block_stride); }

OQCM_ENTRY(1)  OQCM_ENTRY(2)  OQCM_ENTRY(3)  OQCM_ENTRY(4)
OQCM_ENTRY(5)  OQCM_ENTRY(6)  OQCM_ENTRY(7)  OQCM_ENTRY(8)
OQCM_ENTRY(9)  OQCM_ENTRY(10) OQCM_ENTRY(11) OQCM_ENTRY(12)
OQCM_ENTRY(13) OQCM_ENTRY(14) OQCM_ENTRY(15) OQCM_ENTRY(16)
