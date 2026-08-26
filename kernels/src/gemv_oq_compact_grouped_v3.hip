// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus decode GEMV v3 (batch=1): WIDE-LOAD twin of
// `gemv_oq_compact_grouped_v2`. Identical math and the same overlay contract
// (duplicate indices resolved and the bulk nibble zeroed at load by
// `normalize_compact_overlays`); what changes is the shape of the weight read.
//
// v2 gives each group 16 lanes, so a lane loads 8 nibble bytes — a `dwordx2`.
// The pure-read stream this kernel is chasing moves 250 GB/s with `dwordx4`,
// and ablations said the residual gap is weight-load SHAPE rather than bytes,
// ALU (2.5%) or activation traffic (~0 at one wave per workgroup). So v3 gives
// each group 8 lanes and a `dwordx4`: 16 bytes = 32 weights per lane. One wave32
// then covers FOUR groups per round instead of two, which halves the weight VMEM
// instruction count per byte moved and quadruples the groups in flight.
//
// Requires group == 256 and ng % 4 == 0 (K % 1024 == 0); callers fall back to v2
// otherwise. Qwen3.8-27B's K values (5120, 6144, 17408) all qualify.
//
// Numerics are v2's exactly: inline int4 dequant + sparse int8 overlay against
// the FULL-PRECISION f32 activation (W4A16 at decode).
//
//   W : compact blocks, [M, K/256] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCV3_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV3_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// One 32-bit word = 8 nibbles = 8 weights, against two float4 of activation.
// The word parameter is `ww`, NOT `w`: a parameter named `w` would make the
// `(xa).w` component selector expand to the caller's argument.
#define OQCV3_WORD(ww, xa, xb)                                                 \
    ( (float)OQCV3_LO(ww, 0) * (xa).x + (float)OQCV3_HI(ww, 0) * (xa).y        \
    + (float)OQCV3_LO(ww, 1) * (xa).z + (float)OQCV3_HI(ww, 1) * (xa).w        \
    + (float)OQCV3_LO(ww, 2) * (xb).x + (float)OQCV3_HI(ww, 2) * (xb).y        \
    + (float)OQCV3_LO(ww, 3) * (xb).z + (float)OQCV3_HI(ww, 3) * (xb).w )

// PERSISTENT GRID: a fixed, wide launch whose waves grid-stride over output
// rows, rather than one 32-thread workgroup per row. Dispatching a tiny
// workgroup per row costs real bandwidth at these row counts — lm_head alone is
// 248320 of them. The same synthetic sweep that chose the split layout prices
// the dispatch shape too (examples/bench_oq_layout):
//
//     flat dwordx4 stream (upper bound)        243.2 GB/s   100.0%
//     one workgroup per row (split layout)     234.5         96.4%
//     persistent 2048 x 256                    239.0         98.3%
//
// The reduction still happens within one wave, so a wave still owns a whole row
// and nothing crosses waves.
__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_grouped_v3(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    const int lane = threadIdx.x & 31;          // wave32
    const int wpb  = blockDim.x >> 5;           // waves per workgroup
    const int wid  = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int nw   = gridDim.x * wpb;
    const int ng   = K / 256;       // groups (multiple of 4)
    const int gir  = lane >> 3;     // 0..3 : which of the 4 groups this round
    const int lig  = lane & 7;      // 0..7 : 32 weights within the group
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 32;      // weight/activation offset within the group

    for (int row = wid; row < M; row += nw) {

    // SPLIT PLANES: this tensor holds ALL nibble groups first, then all
    // [f16 scale][overlay table] side records — see
    // `hipfire_runtime::oq8_arch::split_compact_planes`. Same bytes, same 4.25
    // bits; what it buys is that a nibble row starts at ng*(group/2), a multiple
    // of 128 at G=256, so EVERY row is cache-line aligned. The interleaved
    // 136-byte block left only every fourth row aligned, worth ~3.7% of achieved
    // bandwidth (examples/bench_oq_layout: 226.4 -> 234.8 GB/s).
        const int side_stride = block_stride - 128;
        const uint8_t* n_row = W + (long long)row * ng * 128LL;
        const uint8_t* s_row = W + (long long)M * ng * 128LL
                                 + (long long)row * ng * (long long)side_stride;

        float facc = 0.0f;
        for (int r = 0; r < ng; r += 4) {
            const int gi = r + gir;
            const uint8_t* sb = s_row + (long long)gi * side_stride;

            const float sw = (float)(*(const _Float16*)sb);

            // 16 bytes = 32 weights per lane, now 16-byte aligned within a
            // 128-byte-aligned row.
            uint32_t wv[4];
            __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 16, 16);

            const int xoff = gi * 256 + elt;
            const float4 x0 = *(const float4*)(X + xoff);
            const float4 x1 = *(const float4*)(X + xoff + 4);
            const float4 x2 = *(const float4*)(X + xoff + 8);
            const float4 x3 = *(const float4*)(X + xoff + 12);
            const float4 x4 = *(const float4*)(X + xoff + 16);
            const float4 x5 = *(const float4*)(X + xoff + 20);
            const float4 x6 = *(const float4*)(X + xoff + 24);
            const float4 x7 = *(const float4*)(X + xoff + 28);

            float s = OQCV3_WORD(wv[0], x0, x1) + OQCV3_WORD(wv[1], x2, x3)
                    + OQCV3_WORD(wv[2], x4, x5) + OQCV3_WORD(wv[3], x6, x7);

            // Overlays REPLACE the bulk nibble, and the loader ZEROES that nibble, so
            // a correction is just val*x[idx] and depends on no lane's registers.
            // BRANCHLESS ON PURPOSE: the `lig < n_ov` guard used to be a divergent
            // branch, and that branch — not the table loads, not the scattered
            // gather, not alignment — was what the overlay actually cost. Every lane
            // now issues the same loads and the inactive ones multiply by zero,
            // worth 3.2% of decode (73.17 -> 70.83 ms/token on Qwen3.8-27B).
            // Load-merging into one ushort (+1%), 8-byte block alignment (-0.5%),
            // K-depth 2 (+14.6%) and pipelining the table one group ahead (+2.7%)
            // were all tried against this same cost and all LOST; the branch was the
            // thing. Entries beyond one per lane fall to the residual loop, whose
            // bound is wave-uniform.
            const uint8_t* table = sb + 2;
            {
                const int e   = (lig < n_ov) ? lig : 0;
                const int idx = (int)table[2 * e];
                const int val = (lig < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
                s += (float)val * X[gi * 256 + idx];
            }
            for (int e = lig + 8; e < n_ov; e += 8) {
                const int idx = (int)table[2 * e];
                const int val = (int)(int8_t)table[2 * e + 1];
                s += (float)val * X[gi * 256 + idx];
            }

            facc += s * sw;
        }

        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
        if (lane == 0) Y[row] = facc;
    }
}

// SPLIT-K twin of the kernel above, for SMALL M.
//
// v3 gives each row one wave, so the wave count IS M. That is plenty for
// lm_head (M=248320) but starves the per-layer projections. Measured on
// Qwen3.5-35B-A3B--oq4.25++ decode, from one kernel trace — note rows 2 and 3
// move the SAME 4.7 MB and differ only in shape:
//
//     M=248320 K=2048  16384 waves (clamped)   1204 us   225 GB/s
//     M=8192   K=2048   8192 waves   9.4 MB      49.9    189
//     M=4096   K=2048   4096 waves   4.7 MB      29.1    162
//     M=2048   K=4096   2048 waves   4.7 MB      44.6    106   <-- same bytes
//     M=512    K=2048    512 waves   0.6 MB       6.1     97
//
// Identical bytes at half the waves costs 1.53x the time, so this is
// PARALLELISM, not bandwidth: too few waves, each walking a serial chain of ng
// groups with a dependent accumulate.
//
// So: SPLIT waves cooperate on one row, each taking a contiguous ng/SPLIT slice
// of the groups, then one LDS reduction folds the partials. Wave count becomes
// M*SPLIT and each chain shortens by SPLIT, which is exactly the axis the table
// above is monotonic in.
//
// Requires (ng / SPLIT) % 4 == 0, because a wave still consumes 4 groups per
// round (8 lanes x dwordx4). The caller picks SPLIT and falls back to the
// one-wave-per-row kernel when no split divides cleanly.
//
// Numerics: each wave reduces its own slice with the same 32-lane __shfl_down,
// then the slices are summed low-to-high. Against the unsplit kernel that is a
// different summation ASSOCIATION over groups (the same class of difference as
// the wide-load port), not a different computation.
__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_grouped_v3_splitk(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride, int split
) {
    const int lane = threadIdx.x & 31;
    const int wig  = (int)(threadIdx.x >> 5);      // wave index in workgroup, 0..7
    const int rpb  = 8 / split;                    // rows per workgroup
    const int rib  = wig / split;                  // which row this wave serves
    const int sub  = wig % split;                  // which K-slice
    const int nrow = gridDim.x * rpb;
    const int ng   = K / 256;
    const int gir  = lane >> 3;
    const int lig  = lane & 7;
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 32;
    const int gpw  = ng / split;                   // groups per wave, multiple of 4
    const int g0   = sub * gpw;

    __shared__ float part[8];

    for (int row = blockIdx.x * rpb + rib; row < M; row += nrow) {
        const int side_stride = block_stride - 128;
        const uint8_t* n_row = W + (long long)row * ng * 128LL;
        const uint8_t* s_row = W + (long long)M * ng * 128LL
                                 + (long long)row * ng * (long long)side_stride;

        float facc = 0.0f;
        for (int r = 0; r < gpw; r += 4) {
            const int gi = g0 + r + gir;
            const uint8_t* sb = s_row + (long long)gi * side_stride;
            const float sw = (float)(*(const _Float16*)sb);

            uint32_t wv[4];
            __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 16, 16);

            const int xoff = gi * 256 + elt;
            const float4 x0 = *(const float4*)(X + xoff);
            const float4 x1 = *(const float4*)(X + xoff + 4);
            const float4 x2 = *(const float4*)(X + xoff + 8);
            const float4 x3 = *(const float4*)(X + xoff + 12);
            const float4 x4 = *(const float4*)(X + xoff + 16);
            const float4 x5 = *(const float4*)(X + xoff + 20);
            const float4 x6 = *(const float4*)(X + xoff + 24);
            const float4 x7 = *(const float4*)(X + xoff + 28);

            float sdot = OQCV3_WORD(wv[0], x0, x1) + OQCV3_WORD(wv[1], x2, x3)
                       + OQCV3_WORD(wv[2], x4, x5) + OQCV3_WORD(wv[3], x6, x7);

            const uint8_t* table = sb + 2;
            {
                const int e   = (lig < n_ov) ? lig : 0;
                const int idx = (int)table[2 * e];
                const int val = (lig < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
                sdot += (float)val * X[gi * 256 + idx];
            }
            for (int e = lig + 8; e < n_ov; e += 8) {
                const int idx = (int)table[2 * e];
                const int val = (int)(int8_t)table[2 * e + 1];
                sdot += (float)val * X[gi * 256 + idx];
            }

            facc += sdot * sw;
        }

        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
        if (lane == 0) part[wig] = facc;
        __syncthreads();
        if (sub == 0 && lane == 0) {
            float t = 0.0f;
            for (int i = 0; i < split; ++i) t += part[rib * split + i];
            Y[row] = t;
        }
        __syncthreads();
    }
}
