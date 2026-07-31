// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// LDS-staged, double-buffered, register-super-tiled variant of
// gemm_oq4_grouped_wmma (Opus Quant W4A4 core). Same contract, same bit-exact
// per-group f32 accumulation order — this is a pure throughput optimization of
// that kernel. See kernels/src/gemm_oq4_grouped_wmma.hip for the algorithm and
// kernels/src/gemm_iu4_i32_wmma_lds.hip for the LDS-tiling template this borrows.
//
//   W  : [M, K/2] packed signed int4  (byte = k_even | k_odd<<4, two's-comp)
//   Ws : [M, K/group] f32  per-group weight scales
//   X  : [B, K/2] packed signed int4
//   Xs : [B, K/group] f32  per-group activation scales
//   Y  : [B, M] f32  (or bf16/uint16 for the _bf16out entry)
//
//   Y[b,m] = Σ_g scale_w[m,g]·scale_x[b,g] · Σ_{k∈g} q_w[m,k]·q_x[b,k]
//
// Why this beats the zero-LDS original: the original is one wave32 wave per
// 16×16 tile with (a) NO operand reuse — every block re-reads its W rows and X
// cols straight from DRAM, (b) ONE serial WMMA accumulation chain (matrix-core
// latency exposed), (c) no prefetch. This kernel:
//   • 8 wave32 waves / block computing a BM=64 × BN=128 block tile → each staged
//     operand is reused across the whole tile (LDS), killing the redundant DRAM
//     traffic that made the original ~8× off its roofline.
//   • WMt×WNt = 2×2 register super-tiling → 4 INDEPENDENT WMMA chains per wave
//     (breaks the serial iacc dependency, keeps the matrix core busy).
//   • BK=64 K-strips staged in double-buffered LDS with a register-staged
//     global→reg prefetch of the next strip DURING compute (RDNA has no cp.async).
//   • Per-256-group i32→f32 rescale flushed at group boundaries in ascending g
//     order — identical arithmetic to the original ⇒ BIT-EXACT (parity example
//     parity_gemm_oq4_grouped_wmma_lds).
//
// Requires K % 64 == 0, group % 64 == 0, group % 16 == 0, K % group == 0
// (Oq4G256 guarantees group=256, so all hold). gfx1103/RDNA3 wave32; gated to
// WMMA archs by the dispatch (same as the original).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));

#define W32(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)

#define BK 64                    // K-strip depth (int4 elements)
#define WARPS_M 2
#define WARPS_N 4
#define NWARP (WARPS_M * WARPS_N) // 8 wave32 waves
#define BLOCK (NWARP * 32)        // 256 threads
#define WMt 2                     // M sub-tiles per wave
#define WNt 2                     // N sub-tiles per wave
#define BM (WARPS_M * WMt * 16)   // 64
#define BN (WARPS_N * WNt * 16)   // 128

template <bool BF16>
static __device__ __forceinline__ void gemm_oq4_grouped_wmma_lds_body(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    void*          __restrict__ Yv,
    int M, int K, int B, int group
) {
    __shared__ uint8_t s_A[2][BM * (BK / 2)];
    __shared__ uint8_t s_X[2][BN * (BK / 2)];

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;            // 0..7
    const int wtid = tid & 31;            // wave-local lane 0..31
    const int lane = wtid & 15;           // 0..15 (mirrored WMMA input row)
    const int wm   = wid / WARPS_N;       // 0..1
    const int wn   = wid % WARPS_N;       // 0..3
    const int hrow = wtid >> 4;           // 0/1 output row-half

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    const int st = K / 2;                 // packed bytes per row
    const int n_groups = K / group;
    const int nstrip = K / BK;
    const int spg = group / BK;           // K-strips per group (=4 for group 256)

    const int UA  = (BK / 2) / 4;         // u32 per source row-strip (=8)
    const int NA  = BM * UA / BLOCK;      // As u32 / thread (=2)
    const int NXv = BN * UA / 4 / BLOCK;  // Xs i32x4 / thread (=1)

    i32x8 iacc[WMt][WNt];
    float facc[WMt][WNt][8];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            iacc[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int e = 0; e < 8; e++) facc[i][j][e] = 0.f;
        }

    uint32_t rA[NA];
    i32x4    rX[NXv];

#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { \
        int idx = tid + l * BLOCK, row = idx / UA, c = idx % UA; \
        int r = bm + row < M ? bm + row : M - 1; \
        rA[l] = *(const uint32_t*)(W + (long long)r * st + (kk) / 2 + c * 4); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { \
        int q = tid + l * BLOCK, base = q * 4, row = base / UA, c = base % UA; \
        int r = bn + row < B ? bn + row : B - 1; \
        rX[l] = *(const i32x4*)(X + (long long)r * st + (kk) / 2 + c * 4); }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = tid + l * BLOCK; ((uint32_t*)s_A[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = tid + l * BLOCK; ((i32x4*)s_X[bf])[q] = rX[l]; }

    { LOADG(0); STS(0); }
    __syncthreads();

    for (int k = 0; k < nstrip; k++) {
        const int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++)
                af[i] = *(const i32x2*)(s_A[buf] + ((wm * WMt + i) * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++)
                xf[j] = *(const i32x2*)(s_X[buf] + ((wn * WNt + j) * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++)
                    iacc[i][j] = W32(af[i], xf[j], iacc[i][j]);
        }

        // Group boundary: rescale the i32 group partial to f32 and accumulate.
        // Same (row,col)->scale map and same (iacc*sw)*sx order as the original
        // kernel, in ascending group order ⇒ bit-exact f32 result.
        if ((k + 1) % spg == 0) {
            const int g = k / spg;
            float sw[WMt][8];
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int r = bm + (wm * WMt + i) * 16 + 2 * e + hrow;
                    int rs = r < M ? r : M - 1;
                    sw[i][e] = Ws[(long long)rs * n_groups + g];
                }
            float sx[WNt];
#pragma unroll
            for (int j = 0; j < WNt; j++) {
                int c = bn + (wn * WNt + j) * 16 + lane;
                int cs = c < B ? c : B - 1;
                sx[j] = Xs[(long long)cs * n_groups + g];
            }
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) {
#pragma unroll
                    for (int e = 0; e < 8; e++)
                        facc[i][j][e] += (float)iacc[i][j][e] * sw[i][e] * sx[j];
                    iacc[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
                }
        }

        if (k + 1 < nstrip) { STS((k + 1) & 1); }
        __syncthreads();
    }
#undef LOADG
#undef STS

    float*    Yf = (float*)Yv;
    uint16_t* Yb = (uint16_t*)Yv;
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = bm + (wm * WMt + i) * 16 + 2 * e + hrow;
                int c = bn + (wn * WNt + j) * 16 + lane;
                if (r < M && c < B) {
                    const float v = facc[i][j][e];
                    if constexpr (BF16) {
                        Yb[(long long)c * M + r] = (uint16_t)(__float_as_uint(v) >> 16);
                    } else {
                        Yf[(long long)c * M + r] = v;
                    }
                }
            }
}

__launch_bounds__(BLOCK, 2)
extern "C" __global__ void gemm_oq4_grouped_wmma_lds(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group
) {
    gemm_oq4_grouped_wmma_lds_body<false>(W, Ws, X, Xs, (void*)Y, M, K, B, group);
}

__launch_bounds__(BLOCK, 2)
extern "C" __global__ void gemm_oq4_grouped_wmma_lds_bf16out(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    uint16_t*      __restrict__ Y,
    int M, int K, int B, int group
) {
    gemm_oq4_grouped_wmma_lds_body<true>(W, Ws, X, Xs, (void*)Y, M, K, B, group);
}
