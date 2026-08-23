// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident indexed MoE decode GEMVs — the compact twins of
// gemv_oq8g256_moe_{gate_up_k8_indexed, gate_up_k8_indexed_batched,
// down_k8_indexed_batched_expanded}.
//
// WHY: routed experts were the last tensors still EXPANDED to one int8 per
// weight at load (`load_moe_expert` -> `oqplus_compact_to_moe_oq8_blocks`),
// because these three GEMVs were the only readers of that layout and no compact
// sibling existed. Dense tensors have stayed compact-resident by default since
// `oq8_arch_load` learned `split_compact_planes`. The expansion costs 1.80x on
// the dominant tensor class of a MoE model, and it compounds with the driver's
// 2 MiB GTT rounding (see `weight_pager::gtt_alloc_cost`) into 3.5x: the 122B
// occupies ~137 GiB resident for a 63.9 GiB artifact, which is why it does not
// load in 124 GiB of unified memory.
//
// LAYOUT (split planes, produced by `split_compact_planes`, identical to what
// the dense compact GEMVs read -- one layout, two call shapes):
//
//     nibbles : A + row*ng*128                      ng = K/256
//     side    : A + M*ng*128 + row*ng*side_stride   side_stride = block_stride-128
//     record  : [f16 scale][N_out x (u8 idx, i8 val)]
//
// Split rather than interleaved because a nibble row then starts at a multiple
// of 128 and EVERY row is cache-line aligned; the interleaved 136-byte block
// left only every fourth row aligned, measured at ~3.7% of achieved bandwidth
// (examples/bench_oq_layout, 226.4 -> 234.8 GB/s). The Oq8 blocks these replace
// are worse still: payload at 260g+4 is 4-byte-misaligned on EVERY group, which
// is why that kernel reads its weights one byte at a time.
//
// NUMERICS: W4A16 at decode -- inline int4 dequant plus the sparse int8 overlay,
// against the full-precision f32 activation, exactly as
// `gemv_oq_compact_grouped_v3` does it. The overlay REPLACES the bulk nibble and
// the loader zeroes that nibble (`normalize_compact_overlays`), so a correction
// is just val*x[idx] and depends on no other lane.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCM_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCM_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// One wave32 owns one output row. Lane L takes bytes [4L, 4L+4) of each group's
// 128 nibble bytes -- so the wave reads exactly 128 contiguous bytes per group,
// one full cache line, perfectly coalesced. Byte j of the loaded word carries
// weights 2j (low nibble) and 2j+1 (high), matching OQCV3_WORD's ordering.
//
// Returns the row dot product in LANE 0 only.
__device__ __forceinline__ float oqc_moe_row_dot(
    const uint8_t* __restrict__ A,
    const float* __restrict__ x,
    int row, int M, int K, int block_stride, int lane
) {
    const int ng = K / 256;
    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;

    const uint8_t* __restrict__ n_row = A + (long long)row * ng * 128LL;
    const uint8_t* __restrict__ s_row = A + (long long)M * ng * 128LL
                                          + (long long)row * ng * (long long)side_stride;

    float acc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const uint8_t* sb = s_row + (long long)g * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        uint32_t wv;
        __builtin_memcpy(&wv, n_row + (long long)g * 128 + lane * 4, 4);

        const int base = g * 256 + lane * 8;
        float s = (float)OQCM_LO(wv, 0) * x[base + 0]
                + (float)OQCM_HI(wv, 0) * x[base + 1]
                + (float)OQCM_LO(wv, 1) * x[base + 2]
                + (float)OQCM_HI(wv, 1) * x[base + 3]
                + (float)OQCM_LO(wv, 2) * x[base + 4]
                + (float)OQCM_HI(wv, 2) * x[base + 5]
                + (float)OQCM_LO(wv, 3) * x[base + 6]
                + (float)OQCM_HI(wv, 3) * x[base + 7];

        // BRANCHLESS, deliberately: on the dense sibling the divergent
        // `lane < n_ov` guard -- not the table load, not the scattered gather --
        // was what the overlay actually cost (3.2% of decode). Every lane issues
        // the same loads; inactive ones multiply by zero.
        const uint8_t* table = sb + 2;
        {
            const int e   = (lane < n_ov) ? lane : 0;
            const int idx = (int)table[2 * e];
            const int val = (lane < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
            s += (float)val * x[g * 256 + idx];
        }
        // N_out > 32 never occurs in practice; the bound is wave-uniform so this
        // costs one predicated compare when it does not.
        for (int e = lane + 32; e < n_ov; e += 32) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            s += (float)val * x[g * 256 + idx];
        }

        acc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    return acc;
}

// Grid (M, K_TOP, 1), block = 32. `x_per_slot`: 1 selects [K_TOP x K] (one
// rotated basis per expert, which per-expert AWQ scales require); 0 a shared
// [K] row. See gemv_oq8g256_moe_gate_up_indexed.hip's preamble.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq_compact_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K] or [K_TOP x K]
    float* __restrict__ y_gate,                          // [K_TOP x MI]
    float* __restrict__ y_up,                            // [K_TOP x MI]
    int M, int K, int x_per_slot, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int lane = threadIdx.x;

    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x_row = x + (long long)(x_per_slot ? krank : 0) * K;

    const float acc = oqc_moe_row_dot(A, x_row, row, M, K, block_stride, lane);

    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[krank * mi + row] = acc;
        else          y_up[krank * mi + (row - mi)] = acc;
    }
}

// Grid (M, K_TOP, N), block = 32.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq_compact_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N x K_TOP]
    const float* __restrict__ x,                    // [N x K] or [N x K_TOP x K]
    float* __restrict__ y_gate,                          // [N x K_TOP x MI]
    float* __restrict__ y_up,                            // [N x K_TOP x MI]
    int M, int K, int K_TOP, int x_per_slot, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = threadIdx.x;

    const uint8_t* __restrict__ A = reinterpret_cast<const uint8_t*>(
        expert_ptrs[topk_indices[bid * K_TOP + krank]]);
    const float* __restrict__ x_row =
        x + (x_per_slot ? ((long long)bid * K_TOP + krank) : (long long)bid) * K;

    const float acc = oqc_moe_row_dot(A, x_row, row, M, K, block_stride, lane);

    if (lane == 0) {
        const int mi = M >> 1;
        const long long slot = (long long)bid * K_TOP + krank;
        if (row < mi) y_gate[slot * mi + row] = acc;
        else          y_up[slot * mi + (row - mi)] = acc;
    }
}

// Atomic-free: writes each (token, krank) result to expert_outputs; reuse
// moe_down_combine_k8_batched to fold into the residual.
// Grid (M, K_TOP, N), block = 32.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq_compact_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N x K_TOP]
    const float* __restrict__ rot_batch,                 // [N x K_TOP x K]
    float*       __restrict__ expert_outputs,            // [N x K_TOP x M]
    int M, int K, int K_TOP, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = threadIdx.x;

    const uint8_t* __restrict__ A = reinterpret_cast<const uint8_t*>(
        expert_ptrs[topk_indices[bid * K_TOP + krank]]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const float acc = oqc_moe_row_dot(A, x, row, M, K, block_stride, lane);

    if (lane == 0) {
        expert_outputs[((long long)bid * K_TOP + krank) * (long long)M + row] = acc;
    }
}
