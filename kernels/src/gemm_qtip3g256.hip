// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// QTIP-3 GEMM: the batched form of `gemv_qtip3g256`.
//
// WHY THIS EXISTS. `Qtip3G256` had a GEMV and no GEMM, and it is absent from
// `tables/gemm_table.rs`, so every batched consumer — prefill, KLD scoring,
// speculative verify — degenerated into a per-token GEMV loop. Measured on
// Qwen3.5-0.8B: prefill 238 tok/s against 2128 for an oq4.25++ body of the same
// model, a 9x gap, and ttft 117.8 ms against 13.2. It is also why MTP
// speculative decode loses on a trellis trunk: a K-token verify costs K full
// weight sweeps.
//
// The fix is the same one `gemm_bf16l3_xf32` applies to the LUT3 head, and this
// kernel deliberately mirrors its shape (one block per row, wave32, an
// `NT`-wide column tile, `col_base`/`n_cols` for tiling): decode each weight
// ONCE and reuse it across NT activation columns. Weight traffic drops by NT
// while the decode ALU cost is paid once instead of NT times — and QTIP decode
// is pure ALU (a 1MAD hash, no table), so it is exactly the cost that was being
// needlessly repeated.
//
// Block layout, decode, and the FWHT-rotated frame are IDENTICAL to
// `gemv_qtip3g256` — see that file for the format contract. The state
// recurrence is reproduced here verbatim rather than shared through a macro,
// matching the convention already used by the `_residual` variant next to it:
// the decode is the part a reader must be able to check line-by-line against
// `qtip::decode_group_bits`.

#include <hip/hip_runtime.h>

#define QTIP3_MEAN       (-0.0071028476f)
#define QTIP3_INV_STD    (0.9959345134f)
#define QTIP3_STATE_MASK 0xFFFu
#define QTIP3_NT         32      // activation columns per pass

__device__ __forceinline__ float qtip3_gemm_decode(unsigned int state) {
    unsigned int x = state & 0xFFFFFFFFu;
    x = x * 34038481u + 76625530u;
    unsigned int bs = (x & 0xFFu) + ((x >> 8) & 0xFFu)
                    + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    float raw = ((float)bs - 510.0f) * (1.0f / 147.800537109375f);
    return (raw - QTIP3_MEAN) * QTIP3_INV_STD;
}

__launch_bounds__(32)
extern "C" __global__ void gemm_qtip3g256(
    const char*  __restrict__ A,   // [M, K] Qtip3G256, 100 B per 256-group
    const float* __restrict__ x,   // [N, K] F32 row-major (FWHT-rotated)
    float*       __restrict__ y,   // [N, M] F32 row-major
    int M, int K, int N, int col_base, int n_cols
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 100;

    const int nc = n_cols < QTIP3_NT ? n_cols : QTIP3_NT;

    float sum[QTIP3_NT];
    #pragma unroll
    for (int c = 0; c < QTIP3_NT; ++c) sum[c] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 100;
        const float scale = *reinterpret_cast<const float*>(gptr);
        const unsigned char* data = (const unsigned char*)(gptr + 4);

        const int boff = tid * 3;
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int s_m3 = 0u, s_m2 = 0u, s_m1 = 0u;
        if (tid > 0) {
            const int pboff = (tid - 1) * 3;
            unsigned int ppk = (unsigned int)data[pboff]
                             | ((unsigned int)data[pboff + 1] << 8)
                             | ((unsigned int)data[pboff + 2] << 16);
            s_m3 = (ppk >> 15) & 7u;
            s_m2 = (ppk >> 18) & 7u;
            s_m1 = (ppk >> 21) & 7u;
        }

        unsigned int q0 = pk & 7u,        q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u, q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u, q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u, q7 = (pk >> 21) & 7u;

        #define QST(a,b,c,d) ((((a)<<9)|((b)<<6)|((c)<<3)|(d)) & QTIP3_STATE_MASK)
        // Decode ONCE per weight — this is the whole point of the kernel.
        const float w0 = scale * qtip3_gemm_decode(QST(s_m3, s_m2, s_m1, q0));
        const float w1 = scale * qtip3_gemm_decode(QST(s_m2, s_m1, q0,  q1));
        const float w2 = scale * qtip3_gemm_decode(QST(s_m1, q0,  q1,  q2));
        const float w3 = scale * qtip3_gemm_decode(QST(q0,   q1,  q2,  q3));
        const float w4 = scale * qtip3_gemm_decode(QST(q1,   q2,  q3,  q4));
        const float w5 = scale * qtip3_gemm_decode(QST(q2,   q3,  q4,  q5));
        const float w6 = scale * qtip3_gemm_decode(QST(q3,   q4,  q5,  q6));
        const float w7 = scale * qtip3_gemm_decode(QST(q4,   q5,  q6,  q7));
        #undef QST

        const int base_idx = g * 256 + tid * 8;
        for (int c = 0; c < nc; ++c) {
            const float* __restrict__ xc = x + (long long)(col_base + c) * K + base_idx;
            sum[c] += w0 * xc[0] + w1 * xc[1] + w2 * xc[2] + w3 * xc[3]
                    + w4 * xc[4] + w5 * xc[5] + w6 * xc[6] + w7 * xc[7];
        }
    }

    for (int c = 0; c < nc; ++c) {
        float v = sum[c];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) y[(long long)(col_base + c) * M + row] = v;
    }
}

// Residual variant: y += A·x, for the projections that write straight back into
// the residual stream (o_proj / out_proj / down_proj). Identical decode and
// batching; only the final store accumulates. Duplicated verbatim rather than
// shared through a macro, matching `gemv_qtip3g256_residual` next door — the
// decode is the part a reader checks line-by-line against qtip::decode_group_bits.
__launch_bounds__(32)
extern "C" __global__ void gemm_qtip3g256_residual(
    const char*  __restrict__ A,   // [M, K] Qtip3G256, 100 B per 256-group
    const float* __restrict__ x,   // [N, K] F32 row-major (FWHT-rotated)
    float*       __restrict__ y,   // [N, M] F32 row-major, PRE-ADD value
    int M, int K, int N, int col_base, int n_cols
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 100;

    const int nc = n_cols < QTIP3_NT ? n_cols : QTIP3_NT;

    float sum[QTIP3_NT];
    #pragma unroll
    for (int c = 0; c < QTIP3_NT; ++c) sum[c] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 100;
        const float scale = *reinterpret_cast<const float*>(gptr);
        const unsigned char* data = (const unsigned char*)(gptr + 4);

        const int boff = tid * 3;
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int s_m3 = 0u, s_m2 = 0u, s_m1 = 0u;
        if (tid > 0) {
            const int pboff = (tid - 1) * 3;
            unsigned int ppk = (unsigned int)data[pboff]
                             | ((unsigned int)data[pboff + 1] << 8)
                             | ((unsigned int)data[pboff + 2] << 16);
            s_m3 = (ppk >> 15) & 7u;
            s_m2 = (ppk >> 18) & 7u;
            s_m1 = (ppk >> 21) & 7u;
        }

        unsigned int q0 = pk & 7u,        q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u, q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u, q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u, q7 = (pk >> 21) & 7u;

        #define QST(a,b,c,d) ((((a)<<9)|((b)<<6)|((c)<<3)|(d)) & QTIP3_STATE_MASK)
        // Decode ONCE per weight — this is the whole point of the kernel.
        const float w0 = scale * qtip3_gemm_decode(QST(s_m3, s_m2, s_m1, q0));
        const float w1 = scale * qtip3_gemm_decode(QST(s_m2, s_m1, q0,  q1));
        const float w2 = scale * qtip3_gemm_decode(QST(s_m1, q0,  q1,  q2));
        const float w3 = scale * qtip3_gemm_decode(QST(q0,   q1,  q2,  q3));
        const float w4 = scale * qtip3_gemm_decode(QST(q1,   q2,  q3,  q4));
        const float w5 = scale * qtip3_gemm_decode(QST(q2,   q3,  q4,  q5));
        const float w6 = scale * qtip3_gemm_decode(QST(q3,   q4,  q5,  q6));
        const float w7 = scale * qtip3_gemm_decode(QST(q4,   q5,  q6,  q7));
        #undef QST

        const int base_idx = g * 256 + tid * 8;
        for (int c = 0; c < nc; ++c) {
            const float* __restrict__ xc = x + (long long)(col_base + c) * K + base_idx;
            sum[c] += w0 * xc[0] + w1 * xc[1] + w2 * xc[2] + w3 * xc[3]
                    + w4 * xc[4] + w5 * xc[5] + w6 * xc[6] + w7 * xc[7];
        }
    }

    for (int c = 0; c < nc; ++c) {
        float v = sum[c];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) y[(long long)(col_base + c) * M + row] += v;
    }
}
