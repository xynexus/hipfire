// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// EXPERIMENT (not production): gemm_bf16_tiled_wmma_lds + a runtime `extra` count
// of throwaway VALU FMAs interleaved with each WMMA. Purpose: measure the DiT
// bf16 GEMM's *free ALU headroom*. On RDNA3 the WMMA runs on the vector ALU; the
// kernel sits at ~10% of peak (compute-bound roofline but stalled), so the VALU
// is mostly idle. If wall-time is flat as `extra` rises (up to some knee), those
// FMAs hide in the stalls = free compute a QTIP/codebook decode or a correction
// branch (dequant, low-rank add, error-feedback) can spend at ~zero warm-step
// cost. The knee (extra-FMAs/WMMA before time rises) is that budget.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short   __attribute__((ext_vector_type(16))) b16x16_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
typedef int32_t __attribute__((ext_vector_type(4)))  i32x4;

#define BK 64
#define WARPS_M 2
#define WARPS_N 4
#define NWARP (WARPS_M * WARPS_N)
#define BLOCK (NWARP * 32)
#define WMt 2
#define WNt 2
#define BM (WARPS_M * WMt * 16)
#define BN (WARPS_N * WNt * 16)

static __device__ __forceinline__ float bf16_to_f32(short s) {
    return __uint_as_float(((uint32_t)(uint16_t)s) << 16);
}

__launch_bounds__(BLOCK, 2)
extern "C" __global__ void bench_bf16_lds_freealu(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*                __restrict__ Y,
    int M, int K, int B, int extra
) {
    __shared__ unsigned short s_A[2][BM * BK];
    __shared__ unsigned short s_X[2][BN * BK];
    // Small codebook in LDS (256 bf16 entries) — the `extra` ops below gather from
    // it with a DATA-DEPENDENT, serially-chained index, mimicking a QTIP/LUT
    // decode (serial trellis + LDS gather), which hides less cleanly than a pure
    // register FMA. This is the realistic decode-shape budget.
    __shared__ unsigned short s_cb[256];

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int wtid = tid & 31;
    const int lane = wtid & 15;
    const int wm   = wid / WARPS_N;
    const int wn   = wid % WARPS_N;
    const int hrow = wtid >> 4;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    const int nstrip = K / BK;

    const int UA  = BK / 2;
    const int UXv = BK / 8;
    const int NA  = BM * UA / BLOCK;
    const int NXv = BN * UXv / BLOCK;

    float8_t acc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++)
            acc[i][j] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    uint32_t rA[NA];
    i32x4    rX[NXv];
    float junk = 0.f; // throwaway accumulator (kept alive, ~0 effect on output)

#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { \
        int idx = tid + l * BLOCK, row = idx / UA, c = idx % UA; \
        int r = bm + row < M ? bm + row : M - 1; \
        rA[l] = *(const uint32_t*)(A + (long long)r * K + (kk) + c * 2); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { \
        int q = tid + l * BLOCK, row = q / UXv, c = q % UXv; \
        int r = bn + row < B ? bn + row : B - 1; \
        rX[l] = *(const i32x4*)(X + (long long)r * K + (kk) + c * 8); }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = tid + l * BLOCK; ((uint32_t*)s_A[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = tid + l * BLOCK; ((i32x4*)s_X[bf])[q] = rX[l]; }

    if (tid < 256) s_cb[tid] = (unsigned short)(0x3c00 + tid); // arbitrary codebook
    { LOADG(0); STS(0); }
    __syncthreads();

    for (int k = 0; k < nstrip; k++) {
        const int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            b16x16_t af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++)
                af[i] = *(const b16x16_t*)(&s_A[buf][((wm * WMt + i) * 16 + lane) * BK + s * 16]);
#pragma unroll
            for (int j = 0; j < WNt; j++)
                xf[j] = *(const b16x16_t*)(&s_X[buf][((wn * WNt + j) * 16 + lane) * BK + s * 16]);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], xf[j], acc[i][j]);
                    // `extra` throwaway VALU FMAs per WMMA on fragment-derived
                    // scalars (not DCE'd; competes for VALU issue).
                    float a0 = bf16_to_f32(af[i][0]);
                    float x0 = bf16_to_f32(xf[j][0]);
                    unsigned int idx = __float_as_uint(a0) ^ __float_as_uint(x0);
                    for (int e = 0; e < extra; e++) {
                        // serial, data-dependent LDS-gather + fma (trellis/LUT shape)
                        idx = (idx * 2654435761u + __float_as_uint(junk)) & 255u;
                        junk = fmaf(junk, bf16_to_f32(s_cb[idx]), x0);
                    }
                }
        }
        if (k + 1 < nstrip) { STS((k + 1) & 1); }
        __syncthreads();
    }
#undef LOADG
#undef STS

#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = bm + (wm * WMt + i) * 16 + 2 * e + hrow;
                int c = bn + (wn * WNt + j) * 16 + lane;
                if (r < M && c < B) {
                    Y[(long long)c * M + r] = acc[i][j][e] + junk * 1e-30f;
                }
            }
}
