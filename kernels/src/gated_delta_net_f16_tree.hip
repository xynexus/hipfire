// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FP CONTRACTION IS PINNED. HIP defaults to -ffp-contract=fast, which lets the
// compiler fuse `x += y*z` ACROSS statements wherever scheduling suits it. The
// pairwise reduction below creates many such sites, and the choice came out
// differently in the routed kernel than in the linear one running
// character-identical source -- 1713/3072 byte-exact, diverging on the FIRST
// token, uniformly across every row, head and column. These kernels are
// required to agree bit-for-bit, so the contraction cannot be left to the
// scheduler.
#pragma clang fp contract(on)

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
    // 8-LANE ROW GROUPS. The wave is 4 groups of 8 lanes: group `g` owns row
    // `row_start+g`, lane-in-group `l` owns columns [16l, 16l+16). The previous
    // layout gave each lane 4 columns of ONE row and walked the 4 rows serially,
    // so every row paid two 32-lane tree reductions (5 shuffle steps each plus a
    // broadcast) back to back -- 40+ shuffles per token on the critical path.
    // Ablating just those reductions cut t=24 by 45%, making them the single
    // largest per-token term.
    //
    // ⚠️ THIS BLOCK IS REPLICATED VERBATIM in gated_delta_net_f16.hip,
    // gated_delta_net_f16_tree.hip and gated_delta_net_f16_routed_batch_seq.hip.
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


        // --- identical f16 GDN step: see the lane-group note above ---
        float sreg[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) sreg[i] = S_tile[row_in_tile * HD + col + i];

        float kk[16], qq[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) { kk[i] = k_t[col + i]; qq[i] = q_t[col + i]; }

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
