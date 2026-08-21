// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant compact-resident W8A8 GEMM: the int8×int8 core of
// gemm_oq8_grouped_wmma, but reading OqPlusCompact (qt=36) blocks DIRECTLY
// instead of a pre-expanded dense int8 plane.
//
// Why: oq4.25++ is ~4.25 bits/weight on disk, but the host loader
// (`oqplus_compact_to_oq8_combined`) unpacks it to one int8 per weight before
// upload, so it costs 8 bits/weight resident — roughly 2x the VRAM the format
// exists to avoid. Expansion belongs in the kernel. The decode itself is not new
// work: `gemm_dflash_oq_plain_ref.hip` MODE 5 already consumes exactly these
// bytes, but those kernels are plain-basis and W4A16 (activations rounded to
// f16). This one keeps canonical FWHT-rotated W8A8 semantics — the same
// int8 activations `quantize_act_oq8` produces — so it is bit-comparable with
// the expanded path rather than a different numeric regime.
//
// On-disk block, one per (row, group) of 256 weights:
//
//   [ f16 scale | G/2 packed signed-int4 | N_out * (u8 idx_in_group, i8 val) ]
//     2 bytes      G/2 bytes                2*N_out bytes
//   block_stride = (2 + G/2) + 2*N_out    (N_out is implicit from the stride)
//
// G is the `group` argument, so this serves OqCompactG256 (G=256, header 130)
// and OqCompactG128 (G=128, header 66) from one body. The u8 index caps G at
// 256; larger groups would need a wider index (see
// docs/experiments/2026-08-06-oq-compact-group-size.md).
//
// Weight at group position p: low nibble of byte[2 + p/2] when p is even, high
// nibble when odd, sign-extended from 4 bits; then any overlay entry whose idx
// equals p replaces it. Table order matters — LAST duplicate wins, matching
// `oqplus_compact_to_oq8_combined`'s sequential `combined[dst + idx] = val`.
//
//   W : compact blocks, [M, K/G] blocks of block_stride bytes
//   X : [B, K] int8 (signed)   row stride = K
//   Xs: [B, K/G] f32 per-group activation scales
//   Y : [B, M] f32
//
// Y[b,m] = Σ_g scale_w[m,g]·scale_x[b,g] · Σ_{p∈g} q_w[m,p]·q_x[b,p]
//
// There is no separate weight-scale plane: the per-group weight scale is the
// block's leading f16. Requires K % G == 0 and G % 16 == 0. gfx1103 (RDNA3)
// wave32, register-tiled, ZERO LDS — mirrors gemm_oq8_grouped_wmma so the two
// stay comparable.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

// Overlay entries held in registers per group. oq4.25++ carries 3 and oq4.5++
// carries 7, so 8 covers every shipped format. Blocks with more overlays than
// this are rejected host-side by the dispatch contract, never silently clipped.
//
// This was 32, and the extra headroom was NOT free: `ov_idx`/`ov_val` are
// indexed by a runtime bound, so oversizing them costs registers (and spill)
// rather than sitting unused. Measured on gate/up M=17408 K=5120 B=256:
//
//   cap 32   9.41 TOPS      cap 16  10.24      cap 8   10.22      cap 4  10.97
//   overlay scan removed entirely: 10.88 — i.e. by cap 4 the SCAN itself is
//   free and everything the cap cost was register pressure, not comparisons.
//
// The cliff is between 16 and 32, so 16 is chosen over 8: it costs nothing
// against 8 and keeps `parity_gemm_oq_compact`'s N_out=16 cell, which covers a
// representable format even though no shipped artifact uses more than 7.
#define OQC_MAX_OVERLAYS 16

// B-tiles handled by ONE workgroup. The weight tile — nibble loads, int4 decode
// and the overlay scan — is done ONCE and reused across all of them.
//
// Without this the grid is [M/16, B/16] and every b-tile re-reads the whole
// weight tile: FETCH_SIZE measured 566 GB for a 128-token prefill that needs
// 13.3 GB, i.e. 42x redundant, against the decode GEMV's 1.05x on the same
// weights. That redundancy — not the decode cost, and not the WMMA rate — is why
// this kernel ran at ~3% of int8 peak.
#define OQC_NB 8

// ROW-TILES per workgroup, one per wave. Each wave owns a different 16-row tile
// but they all consume the SAME OQC_NB b-tiles of X, so the activation reads are
// issued by 8 waves on one CU against identical addresses and are served from L1
// instead of going to memory 8 separate times.
//
// This is the other half of the redundancy story. OQC_NB fixed the WEIGHT
// re-read; the ACTIVATION re-read was still 16:1 — a 16-row x 128-col workgroup
// reads 128*K bytes of X against 8*K of weights, and with M/16 = 1088 row-tiles
// X was fetched ~1088 times: ~1.4 GB of X traffic against 45 MB of weights.
// Measured: removing the activation loads entirely was worth +70%, far more than
// the nibble decode (+6%) or the overlay scan (+12%).
//
// No LDS and no barrier: the waves share through cache, so a wave whose row-tile
// is past M can still early-return without deadlocking the others.
#ifndef OQC_MW
#define OQC_MW 16
#endif

static __device__ __forceinline__ float oqc_f16_scale(const uint8_t* block) {
    const _Float16 value = *(const _Float16*)block;
    return (float)value;
}

// Sign-extend a 4-bit code held in the low nibble.
static __device__ __forceinline__ int oqc_sext4(int q) {
    return (q << 28) >> 28;
}

// Fewer resident blocks than the old (32, 8): OQC_NB accumulators cost VGPRs,
// and the kernel is redundancy-bound rather than occupancy-bound, so trading
// occupancy for 1/OQC_NB of the weight traffic is the right side of the deal.
__launch_bounds__(32 * OQC_MW, 1)
extern "C" __global__ void gemm_oq_compact_grouped_wmma(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int tid   = threadIdx.x;
    const int wave  = tid >> 5;
    const int wlane = tid & 31;
    const int lane  = wlane & 15;
    // Half-of-wave selector for the C-fragment row mapping. Was `tid >> 4`,
    // which is only correct while the workgroup is a single wave.
    const int half  = wlane >> 4;
    const int row_start   = (blockIdx.x * OQC_MW + wave) * 16;
    const int batch_start = blockIdx.y * (16 * OQC_NB);
    if (row_start >= M || batch_start >= B) return;

    const int a_row = row_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;
    // Header = f16 scale + G/2 nibble bytes; the rest is the overlay table.
    const int header = 2 + (group >> 1);
    const int overlays = (block_stride - header) >> 1;

    const int8_t* x_prow[OQC_NB];
    int b_col_nb[OQC_NB];
    float8_t facc[OQC_NB];
    #pragma unroll
    for (int nb = 0; nb < OQC_NB; ++nb) {
        const int x_row = batch_start + nb * 16 + lane;
        b_col_nb[nb] = x_row;
        const int safe_x = (x_row < B) ? x_row : (B - 1);
        x_prow[nb] = X + (long long)safe_x * K;
        facc[nb] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // SPLIT PLANES: all nibble groups first, then all [f16 scale][overlay table]
    // side records — see `hipfire_runtime::oq8_arch::split_compact_planes`.
    const int nib_bytes   = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const long long side_base = (long long)M * n_groups * nib_bytes;
    for (int g = 0; g < n_groups; ++g) {
        const long long bi = (long long)safe_a * n_groups + g;
        const uint8_t* nibbles = W + bi * nib_bytes;
        const uint8_t* sb      = W + side_base + bi * side_stride;
        const uint8_t* table   = sb + 2;

        // Hoist this group's overlay table into registers once, rather than
        // re-reading it inside every one of the 16 K-tiles.
        uint8_t ov_idx[OQC_MAX_OVERLAYS];
        int8_t  ov_val[OQC_MAX_OVERLAYS];
        const int n_ov = (overlays < OQC_MAX_OVERLAYS) ? overlays : OQC_MAX_OVERLAYS;
        for (int e = 0; e < n_ov; ++e) {
            ov_idx[e] = table[2 * e];
            ov_val[e] = (int8_t)table[2 * e + 1];
        }

        int32x8_t iacc[OQC_NB];
        #pragma unroll
        for (int nb = 0; nb < OQC_NB; ++nb) {
            iacc[nb] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
        }
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            // 16 group positions [base, base+16) live in 8 packed bytes.
            const int base = kt * 16;
            // ONE 8-byte load, not eight 1-byte loads. Lanes are strided by
            // `nib_bytes` (a whole weight row apart), so byte-granular reads put
            // every single load on its own cache line — measured at ~3% of int8
            // WMMA peak. The split-plane layout makes the nibble plane 8-byte
            // aligned (`nib_bytes` = group/2 = 128, `base >> 1` = kt*8), so this
            // window is aligned and the ISA takes it as a dwordx2.
            uint32_t pk[2];
            __builtin_memcpy(pk, nibbles + (base >> 1), 8);
            int8_t q[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint8_t packed = (uint8_t)(pk[i >> 2] >> (8 * (i & 3)));
                q[2 * i]     = (int8_t)oqc_sext4(packed & 0xf);
                q[2 * i + 1] = (int8_t)oqc_sext4(packed >> 4);
            }
            // Sequential application preserves last-duplicate-wins.
            for (int e = 0; e < n_ov; ++e) {
                const int idx = (int)ov_idx[e];
                if ((idx & ~15) == base) {
                    q[idx & 15] = ov_val[e];
                }
            }

            // ONE decoded weight tile, OQC_NB WMMAs against it.
            const int32x4_t a = *(const int32x4_t*)q;
            #pragma unroll
            for (int nb = 0; nb < OQC_NB; ++nb) {
                const int32x4_t bv =
                    *(const int32x4_t*)(x_prow[nb] + g * group + base);
                iacc[nb] =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bv, iacc[nb], false);
            }
        }

        // Per-group rescale. The WEIGHT scale depends only on (m_row, g), so it
        // is fetched once and reused across every b-tile; the ACTIVATION scale is
        // per b-col.
        float sw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            sw[j] = (m_row < M)
                ? oqc_f16_scale(W + side_base
                      + ((long long)m_row * n_groups + g) * (long long)side_stride)
                : 0.f;
        }
        #pragma unroll
        for (int nb = 0; nb < OQC_NB; ++nb) {
            const int b_col = b_col_nb[nb];
            const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                facc[nb][j] += (float)iacc[nb][j] * sw[j] * sx;
            }
        }
    }

    #pragma unroll
    for (int nb = 0; nb < OQC_NB; ++nb) {
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
