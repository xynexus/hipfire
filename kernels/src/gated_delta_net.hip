// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FP CONTRACTION IS PINNED. HIP defaults to -ffp-contract=fast, which lets the
// compiler fuse `x += y*z` ACROSS statements wherever scheduling suits it. The
// pairwise reduction below creates many such sites, and the choice came out
// differently in the routed kernel than in the linear one running
// character-identical source -- 1713/3072 byte-exact on the f16 trio, diverging
// on the FIRST token, uniformly across every row, head and column. These
// kernels are required to agree bit-for-bit, so the contraction cannot be left
// to the scheduler.
#pragma clang fp contract(on)

#define HD 128
#define TILE_ROWS 4

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;  // which row tile (0..HD/TILE_ROWS-1)
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    // 8-LANE ROW GROUPS. The wave is 4 groups of 8 lanes: group `g` owns row
    // `row_start+g`, lane-in-group `l` owns columns [16l, 16l+16). The previous
    // layout gave each lane 4 columns of ONE row and walked the 4 rows serially,
    // so every row paid two 32-lane tree reductions (5 shuffle steps each plus a
    // broadcast) back to back -- 40+ shuffles per token on the critical path.
    // Ablating just those reductions cut t=24 by 45%, making them the single
    // largest per-token term.
    //
    // ⚠️ THIS BLOCK IS REPLICATED VERBATIM in gated_delta_net.hip,
    // gated_delta_net_f32_tree.hip and gated_delta_net_f32_routed_batch_seq.hip.
    // The three MUST agree bit-for-bit -- tree replay reproduces the linear path,
    // and a mismatch commits a state sequential decode never reaches, collapsing
    // spec-decode acceptance. Change all three together and re-run
    // examples/test_gated_delta_net_tree_f32.
    const int g = tid >> 3;
    const int l = tid & 7;
    const int col = l * 16;
    const int row_in_tile = g;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    // Tile of S in LDS: TILE_ROWS × 128 floats = 4KB
    __shared__ float S_tile[TILE_ROWS * HD];

    // Load tile from global → LDS
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = S_global[i];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];


        // --- identical f32 GDN step: see the lane-group note above ---
        float sreg[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) sreg[i] = S_tile[row_in_tile * HD + col + i];

        float kk[16], qq[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) { kk[i] = k_t[col + i]; qq[i] = q_t[col + i]; }

        // PAIRWISE, not sequential. A flat `kv += sreg[i]*kk[i]` chain over 16
        // terms is measurably less accurate than the layout it replaced (which
        // summed 4 per lane then reduced across 32 lanes as a tree): it took the
        // f64-reference state error 3.00e-7 -> 3.41e-7, and that was enough to
        // move spec-decode acceptance (tau 6.917 -> 6.308 on one prompt, 2.414
        // -> 1.857 on another). Tree-summing the 16 restores the old depth.
        float part[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) part[i] = sreg[i] * kk[i];
        #pragma unroll
        for (int w = 8; w > 0; w >>= 1)
            #pragma unroll
            for (int i = 0; i < w; i++) part[i] += part[i + w];
        float kv = part[0];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) kv += __shfl_down(kv, o, 8);
        kv = __shfl(kv, 0, 8);

        const int row = row_start + row_in_tile;
        const float delta = (v_t[row] - alpha * kv) * beta_v;

        #pragma unroll
        for (int i = 0; i < 16; i++) sreg[i] = alpha * sreg[i] + kk[i] * delta;
        #pragma unroll
        for (int i = 0; i < 16; i++) part[i] = sreg[i] * qq[i];
        #pragma unroll
        for (int w = 8; w > 0; w >>= 1)
            #pragma unroll
            for (int i = 0; i < w; i++) part[i] += part[i + w];
        float out_v = part[0];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) out_v += __shfl_down(out_v, o, 8);

        #pragma unroll
        for (int i = 0; i < 16; i++) S_tile[row_in_tile * HD + col + i] = sreg[i];

        // One lane per group stores its own row; the 4 rows are contiguous, so
        // this is a single coalesced segment instead of four single-lane
        // 4-byte scattered writes.
        if (l == 0)
            output[t * stride + h * HD + row] = out_v;
        // --- end identical step ---
    }

    // Store tile back
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = S_tile[i];
}
