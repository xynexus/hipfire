// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Sparse overlay correction, K-MAJOR activation variant.
//
// `gemv_oq_compact_overlay_correct` documents three layouts, all far too
// expensive, and names the reason for the best of them exactly: "a gather's 32
// lanes read `x[b, g*G+idx]` for 32 different b, an activation row apart
// (K/2 = 2560 B at K=5120), so every gather touches 32 cache lines."
//
// All three kept the activation in its native [B, K/2] b-major layout. That is
// the whole problem: b is the axis the lanes vary along, and it is the OUTER
// axis of the array. Transposing the activation to K-major makes the same
// gather 32 CONTIGUOUS bytes — one cache line instead of 32 — without changing
// a single arithmetic operation.
//
// The transpose is cheap and amortises: the activation is quantized once per
// layer and shared across every projection that reads it (gate+up, q/k/v), and
// it is only B*K bytes (1.3 MB at B=256, K=5120) against the 44.6 MB of weights
// the GEMM streams.
//
//   XT : [K, B] int8, one sign-extended value per byte (not packed)
//   XsT: [n_groups, B] f32
//   W  : compact SPLIT-PLANE blocks
//   Y  : [B, M] f32, ACCUMULATED IN PLACE
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef OQCO_MAX_OV
#define OQCO_MAX_OV 16
#endif

static __device__ __forceinline__ float oqcot_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

// [B, K/2] packed signed int4  ->  [K, B] int8, and [B, ng] f32 -> [ng, B] f32.
// One byte per activation: the correction gathers single values, so packing
// buys nothing here and costs an unpack per gather.
__launch_bounds__(256) extern "C" __global__ void oq_compact_x4_transpose(
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    int8_t*        __restrict__ XT,
    float*         __restrict__ XsT,
    int B, int K, int n_groups
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (b >= B || k >= K) return;
    const uint8_t packed = X[(long long)b * (K / 2) + (k >> 1)];
    const int q = (k & 1) ? (((int)(packed >> 4) << 28) >> 28)
                          : (((int)(packed & 0xf) << 28) >> 28);
    XT[(long long)k * B + b] = (int8_t)q;          // contiguous across lanes
    if (k < n_groups) XsT[(long long)k * B + b] = Xs[(long long)b * n_groups + k];
}

// int8 twin of the above, for the exact W4A8 (iu4x2) path whose activation is
// already one byte per element. Same [K, B] / [ng, B] output layout.
__launch_bounds__(256) extern "C" __global__ void oq_compact_x8_transpose(
    const int8_t* __restrict__ X,
    const float*  __restrict__ Xs,
    int8_t*       __restrict__ XT,
    float*        __restrict__ XsT,
    int B, int K, int n_groups
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (b >= B || k >= K) return;
    XT[(long long)k * B + b] = X[(long long)b * K + k];
    if (k < n_groups) XsT[(long long)k * B + b] = Xs[(long long)b * n_groups + k];
}

__launch_bounds__(256) extern "C" __global__ void oq_compact_overlay_correct_t(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ XT,
    const float*   __restrict__ XsT,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride,
    int b_off                                   // first b column this launch owns
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

    // B-SLICE IN REGISTERS, TABLE READ ONCE, GATHER AS A DWORD.
    //
    // Three things the b-major original gets wrong, in order of cost:
    //   1. the activation is [B, K/2], so 32 lanes reading 32 different b touch
    //      32 cache lines (its own comment says so). K-major fixes that.
    //   2. the overlay table is read INSIDE the b0 loop -- B/32 times over, and
    //      idx/val are wave-uniform so it is redundant scalar work too.
    //   3. a byte gather puts 32 lanes x 1 B = 32 B into a 64 B line, half wasted.
    //      Giving each lane FOUR CONSECUTIVE b makes it a dword gather: 32 lanes
    //      x 4 B = 128 contiguous bytes, two full lines, nothing wasted.
    //
    // XT rows are B-aligned and the block base is a multiple of 4, so the dword
    // load is always aligned.
    // MAXQ dword blocks per lane x 128 b per block = 512 b per launch. `nblk`
    // MUST be clamped to MAXQ: it indexes the acc[]/isum[] REGISTER arrays, and
    // letting it run to (B+127)>>7 on a B > 512 batch indexes them out of bounds
    // -- observed as a GPU memory fault in this kernel at
    // HIPFIRE_PREFILL_MAX_BATCH=1024. The caller tiles b in 512-wide launches
    // and passes the tile base in `b_off`; XT/XsT/Y are still indexed with the
    // FULL B stride, since XT is [K, B] k-major and a b-tile is not contiguous.
    constexpr int MAXQ = 4;
    const int b_rem = B - b_off;
    int nblk = (b_rem + 127) >> 7;
    if (nblk > MAXQ) nblk = MAXQ;
    float acc[MAXQ][4];
    #pragma unroll
    for (int q = 0; q < MAXQ; ++q)
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc[q][t] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const uint8_t* sb = s_row + (long long)g * side_stride;
        const uint8_t* table = sb + 2;
        const int kbase = g * group;
        const float sw = oqcot_f16(sb);
        int isum[MAXQ][4];
        #pragma unroll
        for (int q = 0; q < MAXQ; ++q)
            #pragma unroll
            for (int t = 0; t < 4; ++t) isum[q][t] = 0;

        // Every q loop below is bounded by the COMPILE-TIME MAXQ and predicated,
        // never by the runtime `nblk`. With a runtime bound, acc[q][t] and
        // isum[q][t] cannot be assigned static registers, so LLVM falls back to
        // M0-relative indexed register moves -- the hot loop came out at 144
        // instructions carrying 7 MAC-ish ops (20.6 instructions per MAC), with
        // 24 v_movrels_b32 / v_movreld_b32 plus 33 v_mov and 77 SALU of pure
        // index bookkeeping. Predicating against MAXQ keeps q constant in each
        // unrolled copy, which is what puts the accumulators in real registers.
        #pragma unroll
        for (int e = 0; e < OQCO_MAX_OV; ++e) {
            if (e >= n_ov) break;
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            const int8_t* xr = XT + (long long)(kbase + idx) * B;
            #pragma unroll
            for (int q = 0; q < MAXQ; ++q) {
                const int b0 = b_off + (q << 7) + (lane << 2);
                const uint32_t w4 = (q < nblk && b0 < B) ? *(const uint32_t*)(xr + b0) : 0u;
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    isum[q][t] += val * (int)(int8_t)((w4 >> (t * 8)) & 0xff);
            }
        }
        const float* xsg = XsT + (long long)g * B;
        #pragma unroll
        for (int q = 0; q < MAXQ; ++q) {
            const int b0 = b_off + (q << 7) + (lane << 2);
            if (q >= nblk || b0 >= B) continue;
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[q][t] += (float)isum[q][t] * sw * xsg[b0 + t];
        }
    }
    #pragma unroll
    for (int q = 0; q < MAXQ; ++q) {
        const int b0 = b_off + (q << 7) + (lane << 2);
        if (q >= nblk || b0 >= B) continue;
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            Y[(long long)(b0 + t) * M + row] += acc[q][t];
    }
}

// ---------------------------------------------------------------------------
// ROW-COALESCED WRITE VARIANT.
//
// `oq_compact_overlay_correct_t` above gets the GATHER right and the STORE
// badly wrong. Ablation on gate/up M=17408 K=5120 B=512 (corr_T_ms):
//
//   baseline                       1.657
//   gather replaced by a constant  1.174     gather = 0.48 ms (29%)
//   Y store predicated off         0.555     STORE  = 1.10 ms (67%)
//   neither                        0.324
//
// Y is [B, M] f32, so it is contiguous in ROWS. A wave there owns ONE row and
// 128 b, so its 128 stores land in 128 distinct cache lines using 4 bytes of
// each -- 32x write amplification, on a read-modify-write.
//
// Coalescing needs consecutive ROWS in consecutive lanes, which is the opposite
// of what the gather wants (consecutive b in consecutive lanes, which is why XT
// is k-major at all). Both are satisfiable: keep the gather exactly as it is,
// accumulate into LDS, and transpose there before the store. A workgroup owns
// ROWS=32 consecutive rows so one store instruction covers 32 floats = 128 B =
// a full line.
//
//   grid  [ceil(M/32), ceil(B/128)]   block 256 (8 wave32, 4 rows per wave)
//   LDS   32 x 129 f32 = 16.5 kB      (+1 pad: a [32][128] stride is 128 dwords,
//                                      which is 0 mod 64 -- every lane on ONE
//                                      bank, a 32-way conflict on the readback)
#define OQCO_ROWS 32
#define OQCO_BB   128

__launch_bounds__(256) extern "C" __global__ void oq_compact_overlay_correct_tr(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ XT,
    const float*   __restrict__ XsT,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    __shared__ float acc_s[OQCO_ROWS][OQCO_BB + 1];

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * OQCO_ROWS;
    const int b0   = blockIdx.y * OQCO_BB + (lane << 2);

    const int n_groups = K / group;
    const int nib_bytes = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const int n_ov_all = (side_stride - 2) >> 1;
    const int n_ov = (n_ov_all < OQCO_MAX_OV) ? n_ov_all : OQCO_MAX_OV;
    const long long side_base = (long long)M * n_groups * nib_bytes;

    #pragma unroll
    for (int r = 0; r < OQCO_ROWS / 8; ++r) {
        const int rl  = wid * (OQCO_ROWS / 8) + r;
        const int row = row0 + rl;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (row < M && b0 < B) {
            const uint8_t* s_row = W + side_base
                                 + (long long)row * n_groups * (long long)side_stride;
            for (int g = 0; g < n_groups; ++g) {
                const uint8_t* sb = s_row + (long long)g * side_stride;
                const uint8_t* table = sb + 2;
                const float sw = oqcot_f16(sb);
                const int kbase = g * group;
                int isum[4] = {0, 0, 0, 0};
                #pragma unroll
                for (int e = 0; e < OQCO_MAX_OV; ++e) {
                    if (e >= n_ov) break;
                    const int idx = (int)table[2 * e];
                    const int val = (int)(int8_t)table[2 * e + 1];
                    const uint32_t w4 =
                        *(const uint32_t*)(XT + (long long)(kbase + idx) * B + b0);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                        isum[t] += val * (int)(int8_t)((w4 >> (t * 8)) & 0xff);
                }
                const float* xsg = XsT + (long long)g * B;
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    acc[t] += (float)isum[t] * sw * xsg[b0 + t];
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc_s[rl][(lane << 2) + t] = acc[t];
    }
    __syncthreads();

    // Transposed drain: consecutive threads take consecutive ROWS, so each
    // 32-thread run writes 32 contiguous floats of Y -- one full 128 B line.
    for (int i = threadIdx.x; i < OQCO_ROWS * OQCO_BB; i += 256) {
        const int rr = i & (OQCO_ROWS - 1);
        const int bl = i >> 5;
        const int row = row0 + rr;
        const int b = blockIdx.y * OQCO_BB + bl;
        if (row < M && b < B) Y[(long long)b * M + row] += acc_s[rr][bl];
    }
}

// ---------------------------------------------------------------------------
// SMALL-B ROW-COALESCED VARIANT.
//
// `_tr` maps a wave's 32 lanes across 128 b (4 per lane) and walks its rows one
// at a time. At a decode step's width (25..64 rows) most of those lanes have no
// b: at B=33 nine of 32 are live, so the kernel is a serial per-row walk over
// n_groups x n_ov dependent gathers with 72% of each wave masked off — ~0.18 ms
// on gate/up where its traffic is ~8 MB (~40 us).
//
// Same gather, same per-element order (group-major, overlay entries in table
// order), so the result is bit-identical to `_tr`; only the lane mapping moves.
// A wave takes 8 rows at once, 4 lanes per row x 4 b per lane = 16 b.
//
//   grid  [ceil(M/64), ceil(B/16)]   block 256 (8 wave32, 8 rows per wave)
//   LDS   64 x 17 f32 = 4.3 kB
#define OQCOS_ROWS 64
#define OQCOS_BB   16

__launch_bounds__(256) extern "C" __global__ void oq_compact_overlay_correct_trs(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ XT,
    const float*   __restrict__ XsT,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    __shared__ float acc_s[OQCOS_ROWS][OQCOS_BB + 1];

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rl   = wid * 8 + (lane >> 2);
    const int row  = blockIdx.x * OQCOS_ROWS + rl;
    const int bl0  = (lane & 3) << 2;
    const int b0   = blockIdx.y * OQCOS_BB + bl0;

    const int n_groups = K / group;
    const int nib_bytes = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const int n_ov_all = (side_stride - 2) >> 1;
    const int n_ov = (n_ov_all < OQCO_MAX_OV) ? n_ov_all : OQCO_MAX_OV;
    const long long side_base = (long long)M * n_groups * nib_bytes;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < M && b0 < B) {
        const uint8_t* s_row = W + side_base
                             + (long long)row * n_groups * (long long)side_stride;
        for (int g = 0; g < n_groups; ++g) {
            const uint8_t* sb = s_row + (long long)g * side_stride;
            const uint8_t* table = sb + 2;
            const float sw = oqcot_f16(sb);
            const int kbase = g * group;
            int isum[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int e = 0; e < OQCO_MAX_OV; ++e) {
                if (e >= n_ov) break;
                const int idx = (int)table[2 * e];
                const int val = (int)(int8_t)table[2 * e + 1];
                const uint32_t w4 =
                    *(const uint32_t*)(XT + (long long)(kbase + idx) * B + b0);
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    isum[t] += val * (int)(int8_t)((w4 >> (t * 8)) & 0xff);
            }
            const float* xsg = XsT + (long long)g * B;
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[t] += (float)isum[t] * sw * xsg[b0 + t];
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) acc_s[rl][bl0 + t] = acc[t];
    __syncthreads();

    // Consecutive threads take consecutive ROWS: 64 contiguous floats of Y.
    for (int i = threadIdx.x; i < OQCOS_ROWS * OQCOS_BB; i += 256) {
        const int rr = i & (OQCOS_ROWS - 1);
        const int bl = i >> 6;
        const int r = blockIdx.x * OQCOS_ROWS + rr;
        const int b = blockIdx.y * OQCOS_BB + bl;
        if (r < M && b < B) Y[(long long)b * M + r] += acc_s[rr][bl];
    }
}

// ---------------------------------------------------------------------------
// TILED x8 transpose: [B, K] int8 -> [K, B], [B, ng] f32 -> [ng, B].
//
// `oq_compact_x8_transpose` above maps a thread to one (b, k) byte with b
// fastest, so its writes are contiguous but every read is a lone byte K apart
// from its neighbour's. At prefill width that is the most expensive kernel in
// the whole forward: B=7584, K=17408 (down_proj) moved 132 MB in 143 ms
// (~0.9 GB/s), twice the GEMM it feeds; over a 41K-token prefill the transpose
// was 35% of all GPU time, more than the GEMMs.
//
// A 64x64-byte tile through LDS: each thread reads 16 contiguous bytes of a
// row (4 threads per 64-byte row segment) and writes 16 bytes of a transposed
// row from LDS, so both global streams are line-contiguous. Same values in the
// same places as the naive kernel — a pure layout change.
//
//   grid  [ceil(K/64), ceil(B/64)]   block 256
#define OQXT_T 64

__launch_bounds__(256) extern "C" __global__ void oq_compact_x8_transpose_tiled(
    const int8_t* __restrict__ X,
    const float*  __restrict__ Xs,
    int8_t*       __restrict__ XT,
    float*        __restrict__ XsT,
    int B, int K, int n_groups
) {
    __shared__ int8_t tile[OQXT_T][OQXT_T + 4];
    const int k0 = blockIdx.x * OQXT_T;
    const int b0 = blockIdx.y * OQXT_T;
    const int t = threadIdx.x;
    const int r = t >> 2;              // 0..63
    const int c = (t & 3) << 4;        // 0, 16, 32, 48

    // Load: row b0+r, bytes k0+c .. k0+c+15.
    {
        const int b = b0 + r;
        if (b < B && k0 + c + 15 < K && ((((long long)b * K + k0 + c) & 15) == 0)) {
            const int4 v = *(const int4*)(X + (long long)b * K + k0 + c);
            const int8_t* pv = (const int8_t*)&v;
            #pragma unroll
            for (int i = 0; i < 16; ++i) tile[r][c + i] = pv[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int k = k0 + c + i;
                tile[r][c + i] = (b < B && k < K) ? X[(long long)b * K + k] : (int8_t)0;
            }
        }
    }
    __syncthreads();
    // Store: XT row k0+r, columns b0+c .. b0+c+15 (from tile column r).
    {
        const int k = k0 + r;
        if (k < K) {
            int8_t* dst = XT + (long long)k * B + b0 + c;
            if (b0 + c + 15 < B && ((((long long)k * B + b0 + c) & 15) == 0)) {
                int4 v;
                int8_t* pv = (int8_t*)&v;
                #pragma unroll
                for (int i = 0; i < 16; ++i) pv[i] = tile[c + i][r];
                *(int4*)dst = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (b0 + c + i < B) dst[i] = tile[c + i][r];
            }
        }
    }
    // Scales ride the first column of tiles: [B, ng] -> [ng, B], tiny.
    if (blockIdx.x == 0) {
        for (int i = t; i < OQXT_T * n_groups; i += 256) {
            const int g = i / OQXT_T;
            const int b = b0 + (i - g * OQXT_T);
            if (b < B) XsT[(long long)g * B + b] = Xs[(long long)b * n_groups + g];
        }
    }
}
