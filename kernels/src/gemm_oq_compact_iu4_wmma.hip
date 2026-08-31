// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus W4A4: the iu4 twin of `gemm_oq_compact_grouped_wmma`.
//
// The compact block already stores its bulk weights as signed int4 nibbles
// packed `byte = k_even | k_odd<<4` — byte-for-byte what
// v_wmma_i32_16x16x16_iu4 wants for its A operand. So this kernel feeds the
// nibble plane to the matrix core RAW: no sign-extend, no unpack, no decode at
// all. The iu8 twin spends ~6% of its time on that decode purely to widen data
// the hardware could already have consumed.
//
// Two wins, and the smaller one is the MAC rate:
//
//   * iu4 WMMA is 1.876x iu8 on gfx1151 (99,237 vs 52,900 GOPS, measured by
//     examples/probe_gfx1151_iu4_wmma). But the iu8 twin only reaches ~25% of
//     the iu8 peak, so this alone would buy little.
//   * int4 ACTIVATIONS halve the activation traffic, and that traffic is what
//     the iu8 twin is actually bound by — ablating its activation loads was
//     worth +70%, against +6% for the decode and +12% for the overlay scan.
//
// THE OVERLAY IS NOT APPLIED HERE. A compact block carries N_out sparse int8
// corrections that REPLACE a bulk nibble, and an int8 value does not fit an int4
// lane. The loader zeroes the bulk nibble under every overlay, so those
// positions contribute exactly 0 to this kernel and the correction is a separate
// sparse pass adding `val * x[idx]` — the same identity the decode GEMV already
// uses. At N_out=3 per 256 that pass is ~1.2% of the MACs of this one.
//
//   W : compact SPLIT-PLANE blocks (all nibble groups, then [f16 scale][table])
//   X : [B, K/2] packed signed int4    Xs: [B, K/G] f32
//   Y : [B, M] f32
//
// Tiling mirrors the iu8 twin: OQC_NB b-tiles per workgroup so the weight tile
// is read once, and OQC_MW row-tiles (one per WAVE) so those waves share the
// same activation columns through L1.

// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

#ifndef OQC4_NB
#define OQC4_NB 8
#endif
#ifndef OQC4_MW
#define OQC4_MW 16
#endif

// Stage the activation tile in LDS: the OQC4_MW waves in this workgroup all
// consume the SAME OQC4_NB b-tiles, so without staging each wave re-issues the
// identical global loads and the kernel stays activation-bound (ablating those
// loads measured 2-3x: gate/up 20.58 -> 39.73 TOPS).
//
// One group's tile is OQC4_NB*16 columns x (group/2) bytes = 16 KB at NB=8,
// G=256. PADDED by 4 bytes per column: at an unpadded 128-byte stride every lane
// lands on the same LDS bank (128 B = 32 dwords, and there are 32 banks), which
// is a 32-way conflict. 132 bytes = 33 dwords walks the banks instead.
#ifndef OQC4_LDS
#define OQC4_LDS 1
#endif
#define OQC4_COLS   (OQC4_NB * 16)
#define OQC4_LDS_PAD 4

static __device__ __forceinline__ int32x2_t oqc4_load_u4x16(const uint8_t* p) {
    return (int32x2_t){(int32_t)*(const uint32_t*)(p + 0),
                       (int32_t)*(const uint32_t*)(p + 4)};
}

static __device__ __forceinline__ float oqc4_f16_scale(const uint8_t* block) {
    const _Float16 value = *(const _Float16*)block;
    return (float)value;
}

__launch_bounds__(32 * OQC4_MW, 1)
extern "C" __global__ void gemm_oq_compact_iu4_wmma(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int tid   = threadIdx.x;
    const int wave  = tid >> 5;
    const int wlane = tid & 31;
    const int lane  = wlane & 15;
    const int half  = wlane >> 4;
    const int row_start   = (blockIdx.x * OQC4_MW + wave) * 16;
    const int batch_start = blockIdx.y * (16 * OQC4_NB);
    // Only the workgroup-UNIFORM condition may early-return: `batch_start`
    // depends solely on blockIdx.y, so every wave leaves together. `row_start`
    // varies per wave, and returning on it would desert the __syncthreads()
    // below and hang the rest of the workgroup. Out-of-range rows instead ride
    // the existing `safe_a` clamp and are dropped by the `m_row < M` store guard.
    if (batch_start >= B) return;

    const int a_row  = row_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;
    const int nib_bytes   = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const long long side_base = (long long)M * n_groups * nib_bytes;
    const int x_stride = K / 2;   // packed int4 row stride, HALF the iu8 path

    const uint8_t* x_prow[OQC4_NB];
    int b_col_nb[OQC4_NB];
    float8_t facc[OQC4_NB];
    // Per-b-tile ACTIVATION-SCALE pointers, hoisted out of the group loop. The
    // index `b_col * n_groups + g` only varies in `g` across groups, so what was
    // a 64-bit multiply per (group, b-tile) becomes one multiply per b-tile for
    // the whole kernel and a `[g]` index inside. Same for the weight scales
    // below. Those multiplies are a large share of the residual non-WMMA VALU:
    // the measured 3.7-per-WMMA is well above what the staging, addressing and
    // rescale arithmetic alone account for.
    const float* xs_p[OQC4_NB];
    #pragma unroll
    for (int nb = 0; nb < OQC4_NB; ++nb) {
        const int x_row = batch_start + nb * 16 + lane;
        b_col_nb[nb] = x_row;
        const int safe_x = (x_row < B) ? x_row : (B - 1);
        x_prow[nb] = X + (long long)safe_x * x_stride;
        xs_p[nb] = Xs + (long long)safe_x * n_groups;
        facc[nb] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    // Per-accumulator-row WEIGHT-SCALE pointers, likewise hoisted; only the
    // `g * side_stride` step varies per group.
    const uint8_t* sw_p[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m_row = row_start + 2 * j + half;
        const int safe_m = (m_row < M) ? m_row : (M - 1);
        sw_p[j] = W + side_base
                + (long long)safe_m * n_groups * (long long)side_stride;
    }

    // DOUBLE-BUFFERED activation stage. The single-buffered form needed TWO
    // barriers per group — one before the overwrite so no wave is still reading
    // group g-1, one after so no wave runs ahead of the fill — and those
    // barriers were the binding constraint, not instruction count: ablating them
    // measured +19% on gate/up and +45% on `down` (which has 68 groups, so 136
    // barriers).
    //
    // Staging g+1 into the alternate buffer while computing from g removes the
    // first barrier entirely: the writes and the reads target different memory,
    // so only ONE barrier per group is needed, at the end.
    __shared__ uint8_t xt[2][OQC4_COLS * (128 + OQC4_LDS_PAD)];
    const int lds_stride = nib_bytes + OQC4_LDS_PAD;
    const int dwords_per_col = nib_bytes >> 2;
    // `group` is 128 or 256, so this is 16 or 32 — a power of two, hence a
    // shift/mask and not a DIVISION. AMD integer divide expands to ~20 VALU and
    // this runs once per staged dword.
    const int dpc_shift = __builtin_ctz(dwords_per_col);
    const int dpc_mask  = dwords_per_col - 1;
    const int stage_total = OQC4_COLS * dwords_per_col;

#define OQC4_STAGE(G, BUF)                                                     \
    do {                                                                       \
        for (int i = tid; i < stage_total; i += blockDim.x) {                  \
            const int col = i >> dpc_shift;                                    \
            const int d   = i & dpc_mask;                                      \
            const int brow = batch_start + col;                                \
            const int sb = (brow < B) ? brow : (B - 1);                        \
            const uint32_t v = *(const uint32_t*)(                             \
                X + (long long)sb * x_stride + (G) * nib_bytes + d * 4);       \
            *(uint32_t*)(xt[BUF] + col * lds_stride + d * 4) = v;              \
        }                                                                      \
    } while (0)

    OQC4_STAGE(0, 0);
    __syncthreads();

    for (int g = 0; g < n_groups; ++g) {
        const int cur = g & 1;
        // Fill the NEXT group's buffer while this one is consumed below.
        if (g + 1 < n_groups) OQC4_STAGE(g + 1, cur ^ 1);

        const uint8_t* nibbles = W + ((long long)safe_a * n_groups + g) * nib_bytes;

        int32x8_t iacc[OQC4_NB];
        #pragma unroll
        for (int nb = 0; nb < OQC4_NB; ++nb) {
            iacc[nb] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
        }
        // Specialised on G=256 so `tiles_per_group` (16) and `lds_stride` (132)
        // are COMPILE-TIME: the k-loop fully unrolls and every LDS offset becomes
        // an immediate rather than an add per (k-tile, b-tile).
#define OQC4_KLOOP(TPG, STRIDE)                                                \
        for (int kt = 0; kt < (TPG); ++kt) {                                   \
            const int32x2_t a = oqc4_load_u4x16(nibbles + kt * 8);             \
            _Pragma("unroll")                                                  \
            for (int nb = 0; nb < OQC4_NB; ++nb) {                             \
                const int32x2_t b = oqc4_load_u4x16(                           \
                    xt[cur] + (nb * 16 + lane) * (STRIDE) + kt * 8);           \
                iacc[nb] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(         \
                    true, a, true, b, iacc[nb], false);                        \
            }                                                                  \
        }
        if (group == 256) {
            #pragma unroll
            OQC4_KLOOP(16, 128 + OQC4_LDS_PAD)
        } else {
            OQC4_KLOOP(tiles_per_group, lds_stride)
        }
#undef OQC4_KLOOP

        const int g_off = g * side_stride;
        float sw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            sw[j] = (m_row < M) ? oqc4_f16_scale(sw_p[j] + g_off) : 0.f;
        }
        #pragma unroll
        for (int nb = 0; nb < OQC4_NB; ++nb) {
            const float sx = (b_col_nb[nb] < B) ? xs_p[nb][g] : 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                facc[nb][j] += (float)iacc[nb][j] * sw[j] * sx;
            }
        }
        // The ONE barrier: the next iteration flips `cur`, so this must cover
        // both "everyone finished reading xt[cur]" and "the g+1 fill completed".
        __syncthreads();
    }
#undef OQC4_STAGE

    #pragma unroll
    for (int nb = 0; nb < OQC4_NB; ++nb) {
        const int b_col = b_col_nb[nb];
        if (b_col >= B) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            if (m_row < M) {
                Y[(long long)b_col * M + m_row] = facc[nb][j];
            }
        }
    }
}
