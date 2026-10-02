// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Right-looking blocked Cholesky: the trailing-submatrix update.
//
//     A[i, j] -= sum_{t < jb} L[i, j0+t] * L[j, j0+t]     for i >= j >= j0+jb
//
// This is where essentially all of the O(k^3) lives. The panel factorization and
// TRSM are O(k * jb^2) and stay on the host in f64, where they are cheap and
// where precision actually matters — an ill-conditioned Hessian breaks down in
// the panel, not in the trailing update.
//
// MEASURED motivation: faer's CPU Cholesky achieves ~20 GFLOP/s here with all 32
// threads (it is NOT misconfigured — probed, and forcing Par::rayon changes
// nothing). Cholesky scales badly at these sizes because the panel serialises,
// so ~400s of a ~600s quantize sits in this routine.
//
// Only the LOWER triangle is touched: j > i returns immediately. Writing the
// upper half would be wasted work and would break the symmetry the next panel
// download assumes.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void chol_syrk_trailing(
    float* __restrict__ a,   // [k, k] row-major, updated in place
    int k, int j0, int jb
) {
    const int start = j0 + jb;
    const int i = start + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= k || j >= k || j > i) return;

    const float* li = a + (long long)i * k + j0;
    const float* lj = a + (long long)j * k + j0;
    float acc = 0.0f;
    for (int t = 0; t < jb; ++t) acc += li[t] * lj[t];
    a[(long long)i * k + j] -= acc;
}

// LDS-tiled variant of the same update.
//
// The scalar kernel above reads 2*jb floats to produce ONE output: at jb=128
// that is 1 KB of traffic for 128 FLOP, an arithmetic intensity of 0.125
// FLOP/byte. On a ~250 GB/s part that caps it near 31 GF/s no matter how much
// arithmetic the CU can issue — the update is bandwidth-starved, not
// compute-bound, and every f32 FMA in it is already free.
//
// Here each block computes a 64x64 output tile with 256 threads (4x4 outputs
// per thread) over BK=16 depth chunks staged in LDS. The panel rows are read
// once per tile instead of once per output element, taking intensity to
// ~16 FLOP/byte — two orders of magnitude — which is what makes the kernel
// compute-bound and, only then, makes precision-emulation tricks (splitting f32
// into f16 pairs for WMMA) worth considering.
//
// Numerically this is the SAME f32 arithmetic. Results differ from the scalar
// kernel only by summation order (reassociation over the BK chunks), not by
// precision class.
#define CHOL_TM 64
#define CHOL_TN 64
#define CHOL_BK 16

extern "C" __global__ __launch_bounds__(256) void chol_syrk_trailing_tiled(
    float* __restrict__ a,   // [k, k] row-major, updated in place
    int k, int j0, int jb
) {
    const int start = j0 + jb;
    const int ti0 = start + (int)blockIdx.y * CHOL_TM;   // tile's first row
    const int tj0 = start + (int)blockIdx.x * CHOL_TN;   // tile's first column
    if (ti0 >= k || tj0 >= k) return;
    // Only the lower triangle is written, so a tile entirely above the diagonal
    // has nothing to do. Diagonal tiles still need the per-element j<=i mask.
    if (tj0 > ti0 + CHOL_TM - 1) return;

    // +1 padding: the inner loop reads As[t][ty*4+u] with ty consecutive across
    // threads, so an unpadded [BK][64] stride would put those in one bank.
    __shared__ float As[CHOL_BK][CHOL_TM + 1];
    __shared__ float Bs[CHOL_BK][CHOL_TN + 1];

    const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
    const int ty = tid >> 4;          // 0..15 -> rows  ty*4 .. ty*4+3
    const int tx = tid & 15;          // 0..15 -> cols  tx*4 .. tx*4+3

    float acc[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) acc[u][v] = 0.0f;

    for (int t0 = 0; t0 < jb; t0 += CHOL_BK) {
        // Stage both panel slabs. Each thread takes 4 elements that are
        // CONTIGUOUS in t (4 divides BK), so a wave walks 16 consecutive floats
        // of a row before moving on — coalesced within the row.
        for (int e = tid * 4; e < CHOL_TM * CHOL_BK; e += 256 * 4) {
            const int r = e / CHOL_BK;
            const int t = e % CHOL_BK;
            const int gi = ti0 + r;
            const int gj = tj0 + r;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int tt = t + u;
                const int depth = t0 + tt;
                As[tt][r] = (gi < k && depth < jb)
                                ? a[(long long)gi * k + j0 + depth] : 0.0f;
                Bs[tt][r] = (gj < k && depth < jb)
                                ? a[(long long)gj * k + j0 + depth] : 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < CHOL_BK; ++t) {
            float av[4], bv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) av[u] = As[t][ty * 4 + u];
#pragma unroll
            for (int v = 0; v < 4; ++v) bv[v] = Bs[t][tx * 4 + v];
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[u][v] += av[u] * bv[v];
        }
        __syncthreads();
    }

#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int i = ti0 + ty * 4 + u;
        if (i >= k) continue;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int j = tj0 + tx * 4 + v;
            if (j < k && j <= i) a[(long long)i * k + j] -= acc[u][v];
        }
    }
}

// Same update at a 128x128 tile, 8x8 outputs per thread.
//
// Tile size is the lever on the DOMINANT traffic term. Over a whole
// factorization the trailing read-modify-write costs ~8k^3/(6*jb) bytes, but the
// panel slabs are re-read once per tile: ~k^3/(TM*3/4) bytes. At k=12288,
// jb=128 that is ~19 GB of trailing traffic against ~39 GB of panel re-reads
// with TM=64 — so doubling the tile halves the larger half. LDS goes to
// 2*16*129*4 = 16.5 KB of the 64 KB, still leaving room for concurrent blocks.
//
// The cost is register pressure: acc[8][8] is 64 VGPRs against 16 at TM=64.
// gfx1151 has 1536, so this is affordable where it would not be on a part with
// a smaller file.
#define CHOL_TM2 128
#define CHOL_TN2 128
#define CHOL_BK2 16

extern "C" __global__ __launch_bounds__(256) void chol_syrk_trailing_tiled128(
    float* __restrict__ a,
    int k, int j0, int jb
) {
    const int start = j0 + jb;
    const int ti0 = start + (int)blockIdx.y * CHOL_TM2;
    const int tj0 = start + (int)blockIdx.x * CHOL_TN2;
    if (ti0 >= k || tj0 >= k) return;
    if (tj0 > ti0 + CHOL_TM2 - 1) return;

    __shared__ float As[CHOL_BK2][CHOL_TM2 + 1];
    __shared__ float Bs[CHOL_BK2][CHOL_TN2 + 1];

    const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
    const int ty = tid >> 4;   // 0..15 -> rows ty*8 .. ty*8+7
    const int tx = tid & 15;   // 0..15 -> cols tx*8 .. tx*8+7

    float acc[8][8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
#pragma unroll
        for (int v = 0; v < 8; ++v) acc[u][v] = 0.0f;

    for (int t0 = 0; t0 < jb; t0 += CHOL_BK2) {
        // 128*16 = 2048 elements per side, 8 per thread, contiguous in t.
        for (int e = tid * 8; e < CHOL_TM2 * CHOL_BK2; e += 256 * 8) {
            const int r = e / CHOL_BK2;
            const int t = e % CHOL_BK2;
            const int gi = ti0 + r;
            const int gj = tj0 + r;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int tt = t + u;
                const int depth = t0 + tt;
                As[tt][r] = (gi < k && depth < jb)
                                ? a[(long long)gi * k + j0 + depth] : 0.0f;
                Bs[tt][r] = (gj < k && depth < jb)
                                ? a[(long long)gj * k + j0 + depth] : 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < CHOL_BK2; ++t) {
            float av[8], bv[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) av[u] = As[t][ty * 8 + u];
#pragma unroll
            for (int v = 0; v < 8; ++v) bv[v] = Bs[t][tx * 8 + v];
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[u][v] += av[u] * bv[v];
        }
        __syncthreads();
    }

#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int i = ti0 + ty * 8 + u;
        if (i >= k) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int j = tj0 + tx * 8 + v;
            if (j < k && j <= i) a[(long long)i * k + j] -= acc[u][v];
        }
    }
}

// Gather the panel (rows j0..k, columns j0..j0+jb) into a contiguous
// [rows, jb] buffer. Without this the host must pull the whole [k, k] matrix
// back every block just to read a k x jb slice — at k=8192 that is 256 MB per
// block against the ~4 MB actually needed, and it is why the first draft
// plateaued at 27 GF/s.
extern "C" __global__ __launch_bounds__(256) void chol_panel_gather(
    const float* __restrict__ a, float* __restrict__ out,
    int k, int j0, int jb, int rows
) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (c >= jb || r >= rows) return;
    out[(long long)r * jb + c] = a[(long long)(j0 + r) * k + j0 + c];
}

// Scatter the factored panel back, LOWER triangle only (c <= r within the
// diagonal block; every column for rows below it). Writing the upper half would
// destroy the symmetry the next gather relies on.
extern "C" __global__ __launch_bounds__(256) void chol_panel_scatter(
    float* __restrict__ a, const float* __restrict__ src,
    int k, int j0, int jb, int rows
) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (c >= jb || r >= rows || c > r) return;
    a[(long long)(j0 + r) * k + j0 + c] = src[(long long)r * jb + c];
}
