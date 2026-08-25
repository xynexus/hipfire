// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// FP16 tree-aware DeltaNet replay — the tree half of the f16 state path, and
// the memory-halved sibling of `gated_delta_net_f32_tree`.
//
// Both the pre-block state (`s_init`) and the tape are f16, matching
// `gated_delta_net_f16`: under FP16 state the persistent S matrix IS f16, so a
// tree replay that read an f32 init would not describe any real configuration.
// Storage is f16 throughout; arithmetic is FP32 — the tile widens into LDS on
// load and narrows once per token on the persist-write, so f16 never enters the
// recurrence itself.
//
// Halving matters twice over here. The tape scales with tree WIDTH
// (n_tokens x n_heads x HD x HD) and is usually the dominant scratch — at
// HD=128 an f32 tape is 64 KiB per head per token, f16 makes it 32 KiB — while
// the state it hangs off is per-SEQUENCE and doubled during spec-decode.
//
// Why f16 and not the retired Q8 tape: f16 keeps ~11 bits of mantissa with a
// per-element exponent, no scales, no per-row absmax, and no stochastic
// rounding. Q8 needed all of those, and its PRNG seed is what leaked execution
// history into target numerics (issue #17). A straight f16 store is stateless.
//
// Accuracy: one f16 rounding per tape round-trip (~2^-11 relative), entering
// only through the parent read, so it accumulates with tree DEPTH rather than
// sibling count — a spec-decode draft is a handful of tokens deep. It is NOT
// byte-exact against the linear FP32 kernel; use `gated_delta_net_f32_tree`
// when exactness against an FP32 reference is the thing under test. Against the
// f16 LINEAR kernel on a spine it is exact, which is the parity that actually
// characterises this kernel.
//
// Semantics are otherwise identical to the f32 tree kernel: DFS order, read the
// parent's slot (or `s_init` at a root), persist-write the post-update tile so
// children see the parent rather than the previous sibling.
// Deterministic dithered narrowing — the SAME function, and below the same
// index derivation, as `gated_delta_net_f16.hip`. That identity is the point:
// tree replay must reproduce, bit for bit, the state sequential decode would
// have produced, or accepting a drafted token commits a state the non-spec path
// would never have reached and spec-decode stops being lossless. A different
// seed here would be worse than no dither at all.
__device__ __forceinline__ _Float16 f32_to_f16_dither(float x, unsigned int idx) {
    unsigned int u = __float_as_uint(x);
    unsigned int r = u ^ (idx * 2654435761u);
    r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
    unsigned int dithered = (u + (r & 0x1FFFu)) & 0xFFFFE000u;
    return (_Float16)__uint_as_float(dithered);
}

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f16_tree(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const _Float16* __restrict__ s_init,
    _Float16* __restrict__ s_tape,
    const int* __restrict__ parent_indices,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    int stochastic_round
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
        // Root reads the pre-block state, children read their parent's tape
        // slot. Both are f16 under FP16 state, so the widening is the same.
        const _Float16* src = (parent >= 0)
            ? (s_tape + (size_t)parent * token_stride + head_tile_offset)
            : (s_init + head_tile_offset);
        for (int i = tid; i < TILE_ROWS * HD; i += 32)
            S_tile[i] = (float)src[i];

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

        // Persist-write, narrowing to f16 exactly once per token. The dither
        // index is `(row_start * HD + i) ^ (h << 19)` — byte-for-byte the live
        // kernel's derivation, and deliberately independent of `t`, so the tape
        // slot for a token holds exactly what the live path would have stored.
        _Float16* dst = s_tape + (size_t)t * token_stride + head_tile_offset;
        for (int i = tid; i < TILE_ROWS * HD; i += 32)
            dst[i] = stochastic_round
                ? f32_to_f16_dither(S_tile[i],
                                    (unsigned int)(row_start * HD + i) ^ ((unsigned int)h << 19))
                : (_Float16)S_tile[i];
    }
}
