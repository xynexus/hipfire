// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

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
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    // Tile of S in LDS: TILE_ROWS × 128 floats = 4KB
    __shared__ float S_tile[TILE_ROWS * HD];

    // Load tile from global → LDS
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = S_global[i];

    float out_row[TILE_ROWS];
    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        float q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            // Stash rather than store. The per-row store was a SINGLE-LANE
            // 4-byte scattered write, four per token per block, each its own
            // memory transaction against a 32-byte granule. Ablating it was
            // worth 40% of t=24 -- the second-largest term after the shuffle
            // reductions.
            out_row[r] = out_v;
        }
        // One coalesced store for the tile's 4 CONTIGUOUS rows. Pure store
        // shaping: the arithmetic and its ORDER are untouched, so the f32 GDN
        // kernels stay byte-exact with each other -- which is exactly what the
        // 8-lane-group reduction rewrite could not preserve.
        if (tid == 0) {
            float* dst = output + t * stride + h * HD + row_start;
            #pragma unroll
            for (int r = 0; r < TILE_ROWS; r++) dst[r] = out_row[r];
        }
    }

    // Store tile back
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = S_tile[i];
}
