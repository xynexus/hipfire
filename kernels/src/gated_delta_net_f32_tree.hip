// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// FP32 tree-aware DeltaNet replay — the FP32 counterpart of
// `gated_delta_net_q8_tree`.
//
// Why it exists: the tree path was Q8-ONLY, so DDTree spec-decode could not run
// under the standing policy that DeltaNet state must never be Q8 (see
// qwen35/state.rs). That left tree replay reachable only by violating the
// policy. This kernel closes that gap, so the Q8 tree kernel can be retired
// without taking DDTree with it.
//
// Semantics are identical to the Q8 tree kernel: tokens are processed in
// tree-DFS order, and each token's S-state is read from either (a) its in-block
// parent's slot in the tape, or (b) the pre-block initial state when
// `parent_indices[t] < 0` (root). After each token the updated S-tile is written
// back to `s_tape[t]`, so a later token whose parent is `t` reads the parent's
// POST-update S rather than the previous sibling's — the persist-write pattern
// that stops sibling pollution from corrupting a tree verify.
//
// What it does NOT inherit from the Q8 variant, and why that matters:
//   - No dequant/requant per token. The Q8 tree kernel round-trips the whole
//     S-tile through int8 at EVERY token, accumulating rounding noise along the
//     tree depth rather than once per block. Here the tape is the state.
//   - No stochastic rounding, so no PRNG seed. Issue #17 was a Q8 seed leaking
//     execution history into target numerics; with an exact tape there is
//     nothing to seed.
//   - No `s_scales` tape.
//
// Cost: the tape is 4x the Q8 tape (n_tokens x n_heads x HD x HD x 4B). For a
// wide tree that is the dominant scratch allocation, which is what the f16
// sibling exists to halve.
//
// Lane mapping is `col = tid * 4` (contiguous), matching the FP32 LINEAR kernel
// in gated_delta_net.hip rather than the Q8 family's stride-32 layout. That is
// deliberate: it makes a spine topology (parent_indices = [-1, 0, 1, 2, ...])
// produce output byte-exact with `gated_delta_net_f32` called n_tokens=1 N times
// in sequence, which is the same reference guarantee the Q8 tree kernel offers
// against its own linear variant. Summation order is what fixes the low bits, so
// the mapping has to match the kernel you want to be exact against.
//
// `s_init` is READ-ONLY (pre-block snapshot). The caller replays the accepted
// spine linearly after acceptance to advance the persistent state, the same
// pattern the tree-conv1d kernel uses.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_tree(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ s_init,
    float* __restrict__ s_tape,
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

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB

    const int head_tile_offset = h * HD * HD + row_start * HD;
    const int token_stride     = n_heads * HD * HD;

    for (int t = 0; t < n_tokens; t++) {
        const int parent = parent_indices[t];
        const float* s_src = (parent >= 0)
            ? (s_tape + (size_t)parent * token_stride + head_tile_offset)
            : (s_init + head_tile_offset);

        for (int i = tid; i < TILE_ROWS * HD; i += 32)
            S_tile[i] = s_src[i];

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

        // Persist-write: token t's POST-update tile becomes readable by its
        // children. Exact — the tape IS the state, so there is no requant step
        // and no noise to accumulate down the tree.
        float* dst = s_tape + (size_t)t * token_stride + head_tile_offset;
        for (int i = tid; i < TILE_ROWS * HD; i += 32)
            dst[i] = S_tile[i];
    }
}
