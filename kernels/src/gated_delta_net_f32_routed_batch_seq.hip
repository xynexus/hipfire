// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 128
#define TILE_ROWS 4

// Routed FP32 Gated Delta Net recurrence for independent request sessions.
//
// Grid: [n_heads, HD/TILE_ROWS, n_sessions]; block [32].
//
// Each block owns one session/head/S-row-tile. It scans the flattened
// round-major prefix rows and applies only rows whose row_session_indices[b]
// match blockIdx.z. This is intentionally O(n_sessions * n_rows) for the first
// implementation: it preserves exact per-session recurrence order while
// removing the correctness bug of sharing one DeltaNet state across sessions.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_routed_batch_seq(
    const float* __restrict__ q,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ k,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ v,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ gate,                 // [batch_rows * n_heads]
    const float* __restrict__ beta,                 // [batch_rows * n_heads]
    const uint64_t* __restrict__ state_ptrs,        // [n_sessions * ptr_layer_stride]
    const int* __restrict__ row_session_indices,    // [batch_rows]
    float* __restrict__ output,                     // [batch_rows * n_heads * HD]
    int ptr_layer_stride,
    int delta_layer_index,
    int batch_rows,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    const int session_idx = blockIdx.z;
    if (h >= n_heads || head_dim != HD) return;

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

    __shared__ float S_tile[TILE_ROWS * HD];

    float* state = reinterpret_cast<float*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = S_global[i];
    }

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];


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
            output[b * stride + h * HD + row] = out_v;
        // --- end identical step ---
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_global[i] = S_tile[i];
    }
}
