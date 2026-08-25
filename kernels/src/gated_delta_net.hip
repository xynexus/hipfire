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

    // Load tile from global → LDS. LDS exists ONLY to redistribute: the global
    // load is coalesced (lane `tid` takes elements tid, tid+32, ...) while the
    // compute wants lane `tid` to own columns [4*tid, 4*tid+4) of every row.
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = S_global[i];
    __syncthreads();

    // Then hoist this lane's slice into REGISTERS for the whole token loop.
    // Nothing is shared across lanes inside the loop, so the LDS round-trip the
    // original did every token -- 4 rows x (4 reads + 4 writes) per token -- was
    // pure overhead. 16 VGPRs buys it back.
    float sreg[TILE_ROWS][4];
    #pragma unroll
    for (int r = 0; r < TILE_ROWS; r++) {
        const float* sr0 = S_tile + r * HD + col;
        sreg[r][0] = sr0[0]; sreg[r][1] = sr0[1];
        sreg[r][2] = sr0[2]; sreg[r][3] = sr0[3];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        float q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        #pragma unroll
        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float s0 = sreg[r][0], s1 = sreg[r][1], s2 = sreg[r][2], s3 = sreg[r][3];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sreg[r][0] = s0; sreg[r][1] = s1; sreg[r][2] = s2; sreg[r][3] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Registers → LDS → coalesced global, mirroring the load.
    #pragma unroll
    for (int r = 0; r < TILE_ROWS; r++) {
        float* sr0 = S_tile + r * HD + col;
        sr0[0] = sreg[r][0]; sr0[1] = sreg[r][1];
        sr0[2] = sreg[r][2]; sr0[3] = sreg[r][3];
    }
    __syncthreads();
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = S_tile[i];
}
