// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Sparse overlay correction for the compact W4A4 path.
//
// `gemm_oq_compact_iu4_wmma` computes the BULK term only: it feeds the block's
// int4 nibbles to the matrix core, and an overlay's int8 value does not fit an
// int4 lane. The loader ZEROES the bulk nibble under every overlay entry, so
// those positions contribute exactly 0 there and the missing term is exactly
//
//   Y[b,m] += Σ_g sw[m,g]·sx[b,g] · Σ_e val[m,g,e] · x4[b, g·G + idx[m,g,e]]
//
// which is the same identity the decode GEMV already uses.
//
// THREE LAYOUTS MEASURED. All three are far too expensive, and the one shipped
// here is merely the least bad. At N_out=3 over G=256 this is ~1.2% of the MACs
// of the GEMM it corrects; getting near that ratio is purely a memory-layout
// problem and none of these solves it.
//
//   layout                                  cost vs the iu4 GEMM   net speedup
//   lanes over B-COLUMNS (this one)                90 - 203%        0.80 - 1.45x
//   lanes over ROWS, b split across grid.y        450 - 930%        0.22 - 0.57x
//   lanes over B-COLUMNS + activation in LDS      340 - 460%        0.38 - 0.60x
//
// Why each fails:
//
//  * lanes over B-COLUMNS. The overlay table is wave-uniform and read exactly
//    once per (row, group) — good. But a gather's 32 lanes read
//    `x[b, g*G+idx]` for 32 different b, an activation row apart (K/2 = 2560 B
//    at K=5120), so every gather touches 32 cache lines. 267M of them.
//  * lanes over ROWS fixes exactly that — 32 indices then land in one 128-byte
//    group window — but rows must be split across grid.y b-chunks, and each
//    chunk RE-READS the whole overlay table: 2.8 MB becomes 89 MB at B=256.
//    Trading 32x on the gather for 32x on the table is a net loss.
//  * staging the activation group in LDS keeps the table read once AND makes the
//    gather an LDS read, which should have been the answer. It is not: the
//    per-group stage of 32 KB with two barriers costs more than the scattered
//    gather it removes, because there is only ONE row of reuse per wave to
//    amortise it over.
//
// **So W4A4 on compact is BLOCKED on this pass, not on the GEMM.** The GEMM is
// 2.1-3.2x faster than the iu8 path; the correction gives all of that back and
// more. The question worth answering before another layout attempt is whether
// the overlay is needed AT ALL under W4A4 — dropping it makes oq4.25++ behave as
// plain oq4 for this path and deletes the problem outright, and that is a
// QUALITY measurement, not a kernel one.
//
//   W : compact SPLIT-PLANE blocks   X : [B, K/2] packed signed int4
//   Xs: [B, K/G] f32                 Y : [B, M] f32, ACCUMULATED IN PLACE

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef OQCO_MAX_OV
#define OQCO_MAX_OV 16
#endif

static __device__ __forceinline__ float oqco_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_overlay_correct(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int wave  = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int row   = blockIdx.x * waves + wave;
    if (row >= M) return;

    const int n_groups = K / group;
    const int nib_bytes = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const int n_ov_all = (side_stride - 2) >> 1;
    const int n_ov = (n_ov_all < OQCO_MAX_OV) ? n_ov_all : OQCO_MAX_OV;
    const long long side_base = (long long)M * n_groups * nib_bytes;
    const uint8_t* s_row = W + side_base + (long long)row * n_groups * side_stride;
    const int x_stride = K / 2;

    // Each lane owns a strided slice of the b-columns and keeps its partials in
    // registers across ALL groups, so Y is touched once at the end rather than
    // read-modify-written per group.
    for (int b0 = lane; b0 < B; b0 += 32) {
        const uint8_t* xrow = X + (long long)b0 * x_stride;
        float acc = 0.0f;
        for (int g = 0; g < n_groups; ++g) {
            const uint8_t* sb = s_row + (long long)g * side_stride;
            const uint8_t* table = sb + 2;
            int isum = 0;
            for (int e = 0; e < n_ov; ++e) {
                const int idx = (int)table[2 * e];
                const int val = (int)(int8_t)table[2 * e + 1];
                const uint8_t packed = xrow[(g * group + idx) >> 1];
                const int q = (idx & 1) ? (((int)(packed >> 4) << 28) >> 28)
                                        : (((int)(packed & 0xf) << 28) >> 28);
                isum += val * q;
            }
            acc += (float)isum * oqco_f16(sb) * Xs[(long long)b0 * n_groups + g];
        }
        Y[(long long)b0 * M + row] += acc;
    }
}
