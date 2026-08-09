// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// Linear DeltaNet recurrence with the persistent S state STORED as f16.
//
// This is the sanctioned half-precision state path. qwen35/state.rs names FP16
// (or a purpose-built codec) as the only acceptable alternatives to FP32 and
// rules Q8 out entirely; this is the FP16 half of that, and the reason the Q8
// state kernels can go.
//
// Storage is f16, arithmetic is FP32. The tile is widened to f32 in LDS on
// load, the whole recurrence runs in f32 registers, and it narrows once on
// store. So f16 costs exactly ONE rounding per token per element — at the
// store — and never compounds inside a token's math.
//
// Why that is categorically safer than the Q8 state it replaces:
//   - f16 carries ~11 bits of mantissa with a per-element exponent. Q8 carried
//     8 bits against a per-ROW absmax scale, so a row with one large element
//     crushed everything else in that row. DeltaNet's S rows are exactly that
//     shape.
//   - No scales tensor, so no `s_scales` to keep in sync (and no chance of a
//     caller passing scales that a FP32 state never allocated — the class of
//     bug that shows up as a GPU memory fault rather than an error).
//   - No stochastic rounding, so no PRNG. The Q8 requant seed is what leaked
//     execution history into target numerics (issue #17), and it is also why Q8
//     needed the `s_ef_residual` error-feedback accumulator that
//     `DeltaNetSnapshot` never saved (issue #22). Neither exists here: plain
//     round-to-nearest is stateless, so spec-decode rollback is trivially
//     lossless.
//
// Memory: exactly half the FP32 state. state.rs sizes FP32 at ~149 MB for
// 27B/122B-A10B and 186 MB for 397B-A17B per sequence, doubled while
// spec-decoding — so this is the lever for concurrent serving of 27B-class
// models, which is the case that motivated wanting a codec at all.
//
// Lane mapping matches `gated_delta_net_f32` (`col = tid * 4`) so the two are
// directly comparable element-for-element; the only difference between them is
// the state's storage format.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f16(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    _Float16* __restrict__ state,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB, FP32 working copy

    _Float16* S_global = state + h * HD * HD + row_start * HD;

    // Widen f16 -> f32 once for the whole batch. Every token in this call then
    // works from the f32 tile, so a multi-token batch pays ONE narrowing at the
    // end rather than one per token.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = (float)S_global[i];

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
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Narrow back to f16. Round-to-nearest, no stochastic term — deterministic,
    // so repeated runs on identical input are bit-identical and a spec-decode
    // rollback restores exactly what it saved.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (_Float16)S_tile[i];
}
