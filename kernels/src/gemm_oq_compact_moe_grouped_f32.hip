// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped compact-MoE GEMM that keeps **f32 activations**.
//
// WHY THIS EXISTS, given gemm_oq_compact_moe_grouped_wmma already does grouped
// MoE: that kernel converts activations to f16 (`ensure_fp16_x`, because WMMA
// needs f16 inputs). Measured against a reference by
// parity_gemm_oq_compact_moe_grouped, it is NUMERICALLY FINE — ~3e-4 of both an
// f64 oracle and the decode GEMV, flat across shapes, i.e. exactly the f16
// floor. But 3e-4 is not zero, and DFlash verify has to commit the tokens AR
// decode would, so a verify routed through it forfeits spec-decode's guarantee.
// It is also, measured on Qwen3.5-35B-A3B--oq4.25++, SLOWER than the indexed
// GEMV it would replace (prefill 203.86 vs 251.69 tok/s).
//
// The amortization win does not need f16. Decode on this model is
// weight-bandwidth-bound, not compute-bound (lm_head reaches 225 GB/s of a 250
// GB/s ceiling while the MoE kernels sit at 103-147), so WMMA's throughput is
// not the resource in short supply — expert weight TRAFFIC is. The indexed GEMV
// re-reads an expert row once per token routed to it; at a 16-slot tile that is
// 16x the traffic this kernel needs.
//
// So: hoist the weight read out of the slot loop and keep everything else
// byte-for-byte what `oqc_moe_row_dot_wide` does.
//
//   for each round of 4 groups:
//       load scale + dwordx4 nibbles + overlay ONCE
//       for each of the 16 slots in the tile:
//           accumulate that slot's dot with the SAME loaded weights
//
// BIT-EXACTNESS is the point, and it is structural rather than hoped for: lane
// l owns group `l>>3` and 32-weight chunk `l&7`, the per-slot accumulator takes
// `facc += sdot * sw` in the same round order r = 0,4,8..., the accumulator is
// split in halves to match the reference's fold (see below), and the tail is the
// same 32-lane __shfl_down(16,8,4,2,1). Nothing about the arithmetic depends on
// how many slots share the load, so each slot's result is the GEMV's result.
// parity_gemm_oq_compact_moe_grouped checks this arm as STRICT EQUALITY, and it
// holds: 0 bit-mismatches across every shape tried, up to 2,097,152 values.
//
// Measured against the indexed GEMV it replaces (same harness, 50 iters):
//
//     M=512  K=1024 nexp=8  batch=16    0.094 -> 0.068 ms   1.38x
//     M=1024 K=2048 nexp=16 batch=32    0.788 -> 0.403 ms   1.96x
//     M=1024 K=2048 nexp=32 batch=128   4.504 -> 1.722 ms   2.62x
//     M=1024 K=2048 nexp=64 batch=256  11.042 -> 3.315 ms   3.33x
//
// The speedup grows with batch because that is what deepens the reuse: more
// slots per tile means the same weight read serves more tokens.
//
// TWO LANE SHAPES, because the reference has two. Routed gate_up is K=2048
// (ng=8) and takes the WIDE path (8 lanes/group, dwordx4) like
// `oqc_moe_row_dot_wide`; routed down_proj is K=512 (ng=2), fails ng % 4, and
// the reference falls back to the NARROW path (32 lanes/group, one dword, 8
// weights per lane) in `oqc_moe_row_dot`. Both are mirrored here, as two entry
// points the dispatcher picks between by K (`gemm_oq_compact_moe_grouped_f32`
// for the wide shape, `..._f32_narrow` for the narrow one), because a grouped
// kernel that only covered gate_up would be useless: enabling path-2 routes
// BOTH projections through it. Two kernels rather than one with a uniform
// branch, so each gets its own register budget.
//
// The narrow kernel also uses a SINGLE accumulator, matching its reference —
// splitk applies to gate_up only.

// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCF_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCF_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// `ww` not `w`: a parameter named `w` makes `(xa).w` bind to the argument.
#define OQCF_WORD(ww, xa, xb)                                                  \
    ( (float)OQCF_LO(ww, 0) * (xa).x + (float)OQCF_HI(ww, 0) * (xa).y          \
    + (float)OQCF_LO(ww, 1) * (xa).z + (float)OQCF_HI(ww, 1) * (xa).w          \
    + (float)OQCF_LO(ww, 2) * (xb).x + (float)OQCF_HI(ww, 2) * (xb).y          \
    + (float)OQCF_LO(ww, 3) * (xb).z + (float)OQCF_HI(ww, 3) * (xb).w )

#define OQCF_TILE 16   // slots per tile — matches expert_tile_ids granularity
#define OQCF_WAVES 8   // waves per workgroup (256 threads)

__launch_bounds__(256)
extern "C" __global__ void gemm_oq_compact_moe_grouped_f32(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const float*              __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int block_stride
) {
    const int tile_y = blockIdx.y;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int wig = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * OQCF_WAVES + wig;
    if (row >= M) return;

    const int slot_start = tile_y * OQCF_TILE;
    const int ng = K / 256;
    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const int elt = lig * 32;

    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_weight_ptrs[expert_id]);
    const uint8_t* __restrict__ n_row = A + (long long)row * ng * 128LL;
    const uint8_t* __restrict__ s_row = A + (long long)M * ng * 128LL
                                          + (long long)row * ng * (long long)side_stride;

    // Resolve this tile's slots to activation rows once.
    const float* xb[OQCF_TILE];
    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) {
        const int slot = slot_start + s;
        int xrow = -1;
        if (slot < m_total) {
            const int flat = sorted_slot_index[slot];
            if (flat >= 0) xrow = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
        xb[s] = (xrow >= 0) ? (X_src + (long long)xrow * K) : nullptr;
    }

    // TWO HALF-ACCUMULATORS, to mirror the reference's FOLD ORDER.
    //
    // `gemv_oq_compact_moe_gate_up_k8_indexed_batched_splitk` gives each row two
    // waves — wave 0 takes groups [0, ng/2), wave 1 the rest — reduces each
    // wave separately and then sums the two partials through LDS. A single
    // accumulator over all ng groups is the same arithmetic in a different
    // order, and lands ~1 ulp away.
    //
    // Splitting the accumulator the same way makes this kernel BIT-EXACT
    // against it. It also stays exact against the UNSPLIT wide GEMV (which the
    // dispatcher uses when ng % 8 != 0): the second half is then empty, reduces
    // to 0.0f, and `v + 0.0f` is bitwise identity.
    const int half_groups = ng >> 1;
    float facc[OQCF_TILE][2];
    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) {
        facc[s][0] = 0.0f;
        facc[s][1] = 0.0f;
    }

    for (int r = 0; r < ng; r += 4) {
        const int half = (half_groups > 0 && r >= half_groups) ? 1 : 0;
        const int gi = r + gir;
        const uint8_t* sb = s_row + (long long)gi * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        // THE POINT: this weight read is hoisted out of the slot loop below.
        uint32_t wv[4];
        __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 16, 16);
        const uint8_t* table = sb + 2;

        const int xoff = gi * 256 + elt;
        const int ov_e = (lig < n_ov) ? lig : 0;
        const int ov_idx = (int)table[2 * ov_e];
        const int ov_val = (lig < n_ov) ? (int)(int8_t)table[2 * ov_e + 1] : 0;

        #pragma unroll
        for (int s = 0; s < OQCF_TILE; s++) {
            const float* xp = xb[s];
            if (xp == nullptr) continue;
            const float4 x0 = *(const float4*)(xp + xoff);
            const float4 x1 = *(const float4*)(xp + xoff + 4);
            const float4 x2 = *(const float4*)(xp + xoff + 8);
            const float4 x3 = *(const float4*)(xp + xoff + 12);
            const float4 x4 = *(const float4*)(xp + xoff + 16);
            const float4 x5 = *(const float4*)(xp + xoff + 20);
            const float4 x6 = *(const float4*)(xp + xoff + 24);
            const float4 x7 = *(const float4*)(xp + xoff + 28);

            float sd = OQCF_WORD(wv[0], x0, x1) + OQCF_WORD(wv[1], x2, x3)
                     + OQCF_WORD(wv[2], x4, x5) + OQCF_WORD(wv[3], x6, x7);

            // Branchless, same as the GEMV: the divergent guard, not the
            // gather, is what the overlay costs.
            sd += (float)ov_val * xp[gi * 256 + ov_idx];
            for (int e = lig + 8; e < n_ov; e += 8) {
                const int idx = (int)table[2 * e];
                const int val = (int)(int8_t)table[2 * e + 1];
                sd += (float)val * xp[gi * 256 + idx];
            }
            facc[s][half] += sd * sw;
        }
    }

    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) {
        float v0 = facc[s][0];
        float v1 = facc[s][1];
        for (int off = 16; off > 0; off >>= 1) v0 += __shfl_down(v0, off);
        for (int off = 16; off > 0; off >>= 1) v1 += __shfl_down(v1, off);
        const float v = v0 + v1;
        const int slot = slot_start + s;
        if (lane == 0 && slot < m_total && xb[s] != nullptr) {
            Y_grouped[(long long)slot * M + row] = v;
        }
    }
}

#define OQCN_NB 4   // narrow kernel: slots whose loads are issued together

// ── NARROW: mirrors `oqc_moe_row_dot` (32 lanes own one group, 8 weights a
// lane) and, like it, keeps a single accumulator. Routed down_proj (K=512,
// ng=2) fails the wide path's ng % 4, so it lands here. ──
//
// LOADS IN BATCHES. A narrow lane's 8 activations are 32 contiguous bytes, so
// a wave's load is 1 KB of consecutive lines and the vector cache serves many
// at once -- the limit is how many the wave has in flight. Written as one slot
// at a time with a branch per slot (empty slot, overlay entries past the
// first), each slot's loads were waited out before the next slot's issued.
// Here an empty slot reads row 0 and is never stored, entries past the first
// (none at oq4.25) take a separate instantiation, and OQCN_NB slots' loads are
// issued before any of them is used. Same terms, same order, same folds: still
// BIT-EXACT. Measured with parity_gemm_oq_compact_moe_grouped 2048 512 64 512
// (35B-A3B down, 4096 slots): 6.40 -> 3.57 ms. Batches of 4 beat 8 and 16
// (4.21, 4.92 ms), which spend registers the occupancy needs.
//
// The WIDE kernel does not take the same change: a wide lane reads its own
// 128-byte line, so each float4 load touches 32 lines, and the texture unit is
// already busy for ~99% of the kernel. More slots in flight measured slower
// there (6.6 -> 7.8-9.4 ms).
template <bool EXTRAS>
__device__ __forceinline__ void oqcn_rows(
    const uint8_t* n_row, const uint8_t* s_row, const float* const* xb,
    int ng, int side_stride, int n_ov, int lane, float* facc
) {
    const int nbase_off = lane * 8;
    for (int g = 0; g < ng; g++) {
        const uint8_t* sb = s_row + (long long)g * side_stride;
        const float sw = (float)(*(const _Float16*)sb);
        uint32_t wv;
        __builtin_memcpy(&wv, n_row + (long long)g * 128 + lane * 4, 4);
        float wf[8];   // decoded once per group, not once per slot
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wf[2 * j] = (float)OQCF_LO(wv, j);
            wf[2 * j + 1] = (float)OQCF_HI(wv, j);
        }
        const uint8_t* table = sb + 2;
        const int e0 = (lane < n_ov) ? lane : 0;
        const int i0 = (int)table[2 * e0];
        const int v0 = (lane < n_ov) ? (int)(int8_t)table[2 * e0 + 1] : 0;
        const int base = g * 256 + nbase_off;

        #pragma unroll
        for (int s0 = 0; s0 < OQCF_TILE; s0 += OQCN_NB) {
            float4 xv[OQCN_NB][2];
            float xo[OQCN_NB];
            #pragma unroll
            for (int j = 0; j < OQCN_NB; j++) {
                const float* xp = xb[s0 + j];
                xv[j][0] = *(const float4*)(xp + base);
                xv[j][1] = *(const float4*)(xp + base + 4);
                xo[j] = xp[g * 256 + i0];
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int j = 0; j < OQCN_NB; j++) {
                float sd = wf[0] * xv[j][0].x + wf[1] * xv[j][0].y
                         + wf[2] * xv[j][0].z + wf[3] * xv[j][0].w
                         + wf[4] * xv[j][1].x + wf[5] * xv[j][1].y
                         + wf[6] * xv[j][1].z + wf[7] * xv[j][1].w;
                sd += (float)v0 * xo[j];
                if (EXTRAS) {
                    const float* xp = xb[s0 + j];
                    for (int e = lane + 32; e < n_ov; e += 32) {
                        const int idx = (int)table[2 * e];
                        const int val = (int)(int8_t)table[2 * e + 1];
                        sd += (float)val * xp[g * 256 + idx];
                    }
                }
                facc[s0 + j] += sd * sw;
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }
}

__launch_bounds__(256)
extern "C" __global__ void gemm_oq_compact_moe_grouped_f32_narrow(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const float*              __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int block_stride
) {
    const int tile_y = blockIdx.y;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int wig = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * OQCF_WAVES + wig;
    if (row >= M) return;

    const int slot_start = tile_y * OQCF_TILE;
    const int ng = K / 256;
    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;
    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_weight_ptrs[expert_id]);
    const uint8_t* n_row = A + (long long)row * ng * 128LL;
    const uint8_t* s_row = A + (long long)M * ng * 128LL + (long long)row * ng * (long long)side_stride;
    // An empty slot reads row 0 and is never stored, so the slot loop carries
    // no branch.
    const float* xb[OQCF_TILE];
    bool sv[OQCF_TILE];
    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) {
        const int slot = slot_start + s;
        int xrow = -1;
        if (slot < m_total) {
            const int flat = sorted_slot_index[slot];
            if (flat >= 0) xrow = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
        sv[s] = xrow >= 0;
        xb[s] = X_src + (long long)(xrow >= 0 ? xrow : 0) * K;
    }

    float facc[OQCF_TILE];
    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) facc[s] = 0.0f;
    // Past 32 overlay entries a lane has a second one to add.
    if (n_ov > 32) oqcn_rows<true>(n_row, s_row, xb, ng, side_stride, n_ov, lane, facc);
    else           oqcn_rows<false>(n_row, s_row, xb, ng, side_stride, n_ov, lane, facc);

    #pragma unroll
    for (int s = 0; s < OQCF_TILE; s++) {
        float v = facc[s];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
        const int slot = slot_start + s;
        if (lane == 0 && slot < m_total && sv[s]) {
            Y_grouped[(long long)slot * M + row] = v;
        }
    }
}
