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
// carries 7; the cap keeps the scan in registers. Blocks with more overlays than
// this are rejected host-side by the dispatch contract, never silently clipped.
#define OQC_MAX_OVERLAYS 32

static __device__ __forceinline__ float oqc_f16_scale(const uint8_t* block) {
    const _Float16 value = *(const _Float16*)block;
    return (float)value;
}

// Sign-extend a 4-bit code held in the low nibble.
static __device__ __forceinline__ int oqc_sext4(int q) {
    return (q << 28) >> 28;
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_oq_compact_grouped_wmma(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ X,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    if (row_start >= M || batch_start >= B) return;

    const int a_row = row_start + lane;
    const int x_row = batch_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int safe_x = (x_row < B) ? x_row : (B - 1);
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;
    // Header = f16 scale + G/2 nibble bytes; the rest is the overlay table.
    const int header = 2 + (group >> 1);
    const int overlays = (block_stride - header) >> 1;

    const int8_t* x_prow = X + (long long)safe_x * K;

    float8_t facc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < n_groups; ++g) {
        const uint8_t* blk =
            W + ((long long)safe_a * n_groups + g) * (long long)block_stride;
        const uint8_t* nibbles = blk + 2;
        const uint8_t* table   = blk + header;

        // Hoist this group's overlay table into registers once, rather than
        // re-reading it inside every one of the 16 K-tiles.
        uint8_t ov_idx[OQC_MAX_OVERLAYS];
        int8_t  ov_val[OQC_MAX_OVERLAYS];
        const int n_ov = (overlays < OQC_MAX_OVERLAYS) ? overlays : OQC_MAX_OVERLAYS;
        for (int e = 0; e < n_ov; ++e) {
            ov_idx[e] = table[2 * e];
            ov_val[e] = (int8_t)table[2 * e + 1];
        }

        int32x8_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            // 16 group positions [base, base+16) live in 8 packed bytes.
            const int base = kt * 16;
            int8_t q[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint8_t packed = nibbles[(base >> 1) + i];
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

            const int32x4_t a = *(const int32x4_t*)q;
            const int32x4_t b = *(const int32x4_t*)(x_prow + g * group + base);
            iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, iacc, false);
        }

        // Per-group rescale. Activation scale is per (b-col) from the Xs plane;
        // the weight scale is the block's own f16, and each lane owns 8 distinct
        // m-rows (acc[j]) so each needs its own row's block scale.
        const int b_col = batch_start + lane;
        const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + (tid >> 4);
            float sw = 0.f;
            if (m_row < M) {
                const uint8_t* jblk =
                    W + ((long long)m_row * n_groups + g) * (long long)block_stride;
                sw = oqc_f16_scale(jblk);
            }
            facc[j] += (float)iacc[j] * sw * sx;
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m_row = row_start + 2 * j + (tid >> 4);
        const int b_col = batch_start + lane;
        if (m_row < M && b_col < B) {
            Y[(long long)b_col * M + m_row] = facc[j];
        }
    }
}
