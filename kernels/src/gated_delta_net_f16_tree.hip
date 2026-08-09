// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// FP16-tape tree-aware DeltaNet replay — the memory-halved sibling of
// `gated_delta_net_f32_tree`.
//
// Only the TAPE is f16. The pre-block state (`s_init`) stays FP32 and the
// recurrence runs entirely in FP32 registers; f16 appears solely as the storage
// format for the per-token tape slots. That split is the point:
//
//   - The tape is what scales with tree WIDTH (n_tokens x n_heads x HD x HD),
//     so it is the allocation worth halving. At HD=128 an f32 tape costs 64 KiB
//     per head per token; f16 makes that 32 KiB.
//   - The persistent DeltaNet state is per-SEQUENCE and is the numerically
//     sensitive one — qwen35/state.rs is explicit that its precision is not a
//     free parameter. Leaving it FP32 keeps this kernel orthogonal to that
//     policy: nothing here changes what the model carries between tokens.
//
// Why f16 and not the retired Q8 tape: f16 keeps ~11 bits of mantissa with no
// scales, no per-row absmax, and no stochastic rounding. The Q8 tape needed all
// three, and its PRNG seed is what leaked execution history into target
// numerics (issue #17). A straight f16 store has no such state.
//
// Accuracy: each tape round-trip costs one f16 rounding (~2^-11 relative). That
// error enters only through the parent read, so it accumulates with tree DEPTH,
// not with the number of siblings. For spec-decode drafts — a handful of tokens
// deep — that is far below the acceptance threshold. It is NOT byte-exact
// against the linear FP32 kernel, unlike `gated_delta_net_f32_tree`; use the f32
// tree kernel when exactness against the linear reference is what you are
// testing.
//
// Semantics are otherwise identical to the f32 tree kernel: DFS order, read the
// parent's slot (or `s_init` at a root), persist-write the post-update tile so
// children see the parent rather than the previous sibling.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f16_tree(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ s_init,
    _Float16* __restrict__ s_tape,
    const int* __restrict__ parent_indices,
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

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB, always FP32

    const int head_tile_offset = h * HD * HD + row_start * HD;
    const int token_stride     = n_heads * HD * HD;

    for (int t = 0; t < n_tokens; t++) {
        const int parent = parent_indices[t];
        if (parent >= 0) {
            const _Float16* src = s_tape + (size_t)parent * token_stride + head_tile_offset;
            for (int i = tid; i < TILE_ROWS * HD; i += 32)
                S_tile[i] = (float)src[i];
        } else {
            // Root reads the FP32 pre-block state directly — no f16 round-trip
            // on the way in, so a depth-1 token is exact against f32.
            const float* src = s_init + head_tile_offset;
            for (int i = tid; i < TILE_ROWS * HD; i += 32)
                S_tile[i] = src[i];
        }

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

        // Persist-write, narrowing to f16 exactly once per token.
        _Float16* dst = s_tape + (size_t)t * token_stride + head_tile_offset;
        for (int i = tid; i < TILE_ROWS * HD; i += 32)
            dst[i] = (_Float16)S_tile[i];
    }
}
