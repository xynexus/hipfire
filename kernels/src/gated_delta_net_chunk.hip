// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 128
#define TILE_ROWS 4
// Chunk width. Bounded by the pair-kernel's LDS staging of K and Q
// (2 * CMAX * HD floats = 16 KB at 16) and by the C^2 scalar work in the
// triangular solve. Longer inputs are split into back-to-back chunks by the
// dispatcher, which still collapses the serial depth by this factor.
#define CMAX 16

// Chunkwise-parallel gated DeltaNet. Semantics are pinned by
// `hipfire_rdna::gdn_chunk::gdn_chunked_f64`, which is checked against the
// token-at-a-time recurrence in that module's tests; read the derivation there.
//
// The recurrence these two kernels replace is serial in `t`: every token needs
// the state the previous token produced. That is correct and unavoidable for
// decode, but it makes a BATCHED prefill of N tokens cost N serial decodes,
// which is why speculative verify does not amortize on a DeltaNet-heavy stack.
// Here the chunk's tokens are resolved together: the only serial step left is a
// C x C triangular solve over scalars, with no head_dim work inside it.

// Pass 1 — the token-pair scalars, which do not depend on the state row.
//
//   kk[t][s] = <k_s, k_t>   (s <  t)   feeds the triangular solve
//   kq[t][s] = <k_s, q_t>   (s <= t)   feeds the output
//   lcum[t]  = sum_{r<=t} gate_r       cumulative LOG gate
//
// Hoisted out of the main kernel because they are shared by all HD/TILE_ROWS
// row-tiles of a head; recomputing them per tile would repeat this work 32x.
extern "C" __global__ void gdn_chunk_pairs(
    const float* __restrict__ q,      // [n_tokens, n_heads, HD]
    const float* __restrict__ k,      // [n_tokens, n_heads, HD]
    const float* __restrict__ gate,   // [n_tokens, n_heads]
    float* __restrict__ kk,           // [n_heads, c, c]
    float* __restrict__ kq,           // [n_heads, c, c]
    float* __restrict__ lcum,         // [n_heads, c]
    int t0,
    int c,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_heads || head_dim != HD || c > CMAX) return;

    __shared__ float Kl[CMAX * HD];
    __shared__ float Ql[CMAX * HD];

    const int stride = n_heads * HD;
    for (int i = threadIdx.x; i < c * HD; i += blockDim.x) {
        const int t = i / HD, d = i % HD;
        const size_t src = (size_t)(t0 + t) * stride + (size_t)h * HD + d;
        Kl[i] = k[src];
        Ql[i] = q[src];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int t = 0; t < c; t++) {
            acc += gate[(size_t)(t0 + t) * n_heads + h];
            lcum[(size_t)h * c + t] = acc;
        }
    }

    float* kk_h = kk + (size_t)h * c * c;
    float* kq_h = kq + (size_t)h * c * c;
    for (int p = threadIdx.x; p < c * c; p += blockDim.x) {
        const int t = p / c, s = p % c;
        float dkk = 0.0f, dkq = 0.0f;
        if (s <= t) {
            const float* ks = Kl + s * HD;
            const float* kt = Kl + t * HD;
            const float* qt = Ql + t * HD;
            for (int d = 0; d < HD; d++) {
                dkk += ks[d] * kt[d];
                dkq += ks[d] * qt[d];
            }
            if (s == t) dkk = 0.0f;  // A is STRICTLY lower triangular
        }
        kk_h[p] = dkk;
        kq_h[p] = dkq;
    }
}

// Pass 2 — the state rows. Grid [n_heads, HD/TILE_ROWS], block [32], one wave
// per row-tile, exactly the shape the serial kernel uses. Everything below is
// row-local given pass 1's scalars, which is what makes the tiling legal.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_chunk(
    const float* __restrict__ q,      // [n_tokens, n_heads, HD]
    const float* __restrict__ k,      // [n_tokens, n_heads, HD]
    const float* __restrict__ v,      // [n_tokens, n_heads, HD]
    const float* __restrict__ beta,   // [n_tokens, n_heads]
    const float* __restrict__ kk,     // [n_heads, c, c]
    const float* __restrict__ kq,     // [n_heads, c, c]
    const float* __restrict__ lcum,   // [n_heads, c]
    float* __restrict__ state,        // [n_heads, HD, HD]
    float* __restrict__ output,       // [n_tokens, n_heads, HD]
    int t0,
    int c,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads || head_dim != HD || c > CMAX) return;

    const int tid = threadIdx.x;
    const int col = tid * 4;               // 4 contiguous cols per lane, 32 lanes
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];
    __shared__ float U[CMAX * TILE_ROWS];  // the solved pseudo-values, per row
    __shared__ float L[CMAX];
    __shared__ float Beta[CMAX];

    float* S_global = state + (size_t)h * HD * HD + (size_t)row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) S_tile[i] = S_global[i];
    for (int t = tid; t < c; t += 32) {
        L[t] = lcum[(size_t)h * c + t];
        Beta[t] = beta[(size_t)(t0 + t) * n_heads + h];
    }
    __syncthreads();

    const float* kk_h = kk + (size_t)h * c * c;
    const float* kq_h = kq + (size_t)h * c * c;

    // Step A: B[t][r] = beta_t (v_t[r] - exp(L_t) <S_0[r], k_t>).
    // The C reductions here are INDEPENDENT across t — that is the whole point.
    // In the serial kernel this same dot product sits on the critical path once
    // per token, each waiting on the previous token's state write.
    for (int t = 0; t < c; t++) {
        const size_t base = (size_t)(t0 + t) * stride + (size_t)h * HD;
        const float k0 = k[base + col], k1 = k[base + col + 1];
        const float k2 = k[base + col + 2], k3 = k[base + col + 3];
        const float e_t = __expf(L[t]);
        for (int r = 0; r < TILE_ROWS; r++) {
            const float* sr = S_tile + r * HD + col;
            float kv = sr[0] * k0 + sr[1] * k1 + sr[2] * k2 + sr[3] * k3;
            for (int o = 16; o > 0; o >>= 1) kv += __shfl_down(kv, o);
            if (tid == 0) {
                U[t * TILE_ROWS + r] = Beta[t] * (v[base + row_start + r] - e_t * kv);
            }
        }
    }
    __syncthreads();

    // Step B: forward substitution for (I + A) U = B, A[t][s] = beta_t
    // exp(L_t - L_s) <k_s, k_t>. Serial in t, but it is C^2 * TILE_ROWS scalar
    // FMAs with no head_dim factor — 1024 of them at c=16 — against the C
    // dependent HD-wide state updates it removes.
    if (tid < TILE_ROWS) {
        for (int t = 1; t < c; t++) {
            float acc = U[t * TILE_ROWS + tid];
            for (int s = 0; s < t; s++) {
                const float a = Beta[t] * __expf(L[t] - L[s]) * kk_h[t * c + s];
                acc -= a * U[s * TILE_ROWS + tid];
            }
            U[t * TILE_ROWS + tid] = acc;
        }
    }
    __syncthreads();

    // Step C: o_t = exp(L_t) <S_0[r], q_t> + sum_{s<=t} exp(L_t - L_s) <k_s,q_t> U_s.
    // Reads the ENTRY state, so these reductions are independent across t too.
    for (int t = 0; t < c; t++) {
        const size_t base = (size_t)(t0 + t) * stride + (size_t)h * HD;
        const float q0 = q[base + col], q1 = q[base + col + 1];
        const float q2 = q[base + col + 2], q3 = q[base + col + 3];
        const float e_t = __expf(L[t]);
        for (int r = 0; r < TILE_ROWS; r++) {
            const float* sr = S_tile + r * HD + col;
            float sq = sr[0] * q0 + sr[1] * q1 + sr[2] * q2 + sr[3] * q3;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_down(sq, o);
            if (tid == 0) {
                float acc = e_t * sq;
                for (int s = 0; s <= t; s++) {
                    acc += __expf(L[t] - L[s]) * kq_h[t * c + s] * U[s * TILE_ROWS + r];
                }
                output[base + row_start + r] = acc;
            }
        }
    }

    // Step D: S_C = exp(L_C) S_0 + sum_t exp(L_C - L_t) U_t k_t^T. One decay
    // scaling and C independent rank-1 accumulations, against the serial
    // kernel's C dependent ones.
    __syncthreads();
    const float e_c = __expf(L[c - 1]);
    for (int r = 0; r < TILE_ROWS; r++) {
        float* sr = S_tile + r * HD + col;
        float s0 = sr[0] * e_c, s1 = sr[1] * e_c, s2 = sr[2] * e_c, s3 = sr[3] * e_c;
        for (int t = 0; t < c; t++) {
            const size_t base = (size_t)(t0 + t) * stride + (size_t)h * HD;
            const float w = __expf(L[c - 1] - L[t]) * U[t * TILE_ROWS + r];
            s0 += w * k[base + col];
            s1 += w * k[base + col + 1];
            s2 += w * k[base + col + 2];
            s3 += w * k[base + col + 3];
        }
        sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;
    }
    __syncthreads();
    for (int i = tid; i < TILE_ROWS * HD; i += 32) S_global[i] = S_tile[i];
}
