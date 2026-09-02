// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

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

// Routed FP16-state Gated Delta Net recurrence for independent request sessions
// — the f16 counterpart of `gated_delta_net_f32_routed_batch_seq`.
//
// Grid: [n_heads, HD/TILE_ROWS, n_sessions]; block [32].
//
// Each block owns one session/head/S-row-tile and scans the flattened
// round-major rows, applying only those whose `row_session_indices[b]` matches
// its `blockIdx.z`. That preserves exact per-session recurrence order while
// sessions share one launch.
//
// This is the entry point that MATTERS for f16. State is per-SEQUENCE, so its
// cost scales with concurrency, and cross-session batching is precisely the
// high-concurrency case: `state_ptrs` holds one state per session, so halving
// each one halves the whole fleet's resident state. On 27B that is ~149 MB ->
// ~75 MB per session, doubled again while spec-decoding.
//
// Storage f16, arithmetic FP32, identical to the linear f16 kernel: the tile
// widens into LDS once, the recurrence runs in f32 registers, and it narrows
// once on store — so a session pays ONE rounding per launch, not one per row.
//
// The pointers in `state_ptrs` therefore address f16 buffers (2 bytes/element),
// HALF the f32 stride. A caller that allocated f32 state and reached this kernel
// would read every element at the wrong offset — which is why the state's
// precision and the kernel choice are decided together at the call site rather
// than independently.
//
// Lane mapping is `col = tid * 4`, matching both the FP32 routed kernel and the
// linear f16 kernel, so per-session results stay comparable element-for-element
// across all four variants.
// Deterministic dithered narrowing — see gated_delta_net_f16.hip for why the
// dither hashes the value's own bits rather than using a counter or carried RNG
// (the store stays a pure function of its input, so spec-decode snapshots still
// restore exactly what they saved).
__device__ __forceinline__ _Float16 f32_to_f16_dither(float x, unsigned int idx) {
    unsigned int u = __float_as_uint(x);
    unsigned int r = u ^ (idx * 2654435761u);
    r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
    unsigned int dithered = (u + (r & 0x1FFFu)) & 0xFFFFE000u;
    return (_Float16)__uint_as_float(dithered);
}

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f16_routed_batch_seq(
    const float* __restrict__ q,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ k,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ v,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ gate,                 // [batch_rows * n_heads]
    const float* __restrict__ beta,                 // [batch_rows * n_heads]
    const uint64_t* __restrict__ state_ptrs,        // [n_sessions * ptr_layer_stride], -> f16
    const int* __restrict__ row_session_indices,    // [batch_rows]
    float* __restrict__ output,                     // [batch_rows * n_heads * HD]
    int ptr_layer_stride,
    int delta_layer_index,
    int batch_rows,
    int n_heads,
    int head_dim
    ,int stochastic_round
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

    __shared__ float S_tile[TILE_ROWS * HD];  // FP32 working copy

    _Float16* state = reinterpret_cast<_Float16*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    _Float16* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = (float)S_global[i];
    }

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];


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

        // Narrow ONCE PER TOKEN, in place, using the derivation now shared by all
        // three f16 GDN kernels: `(row_start * HD + tile_flat) ^ (h << 19)`.
        //
        // Storing the raw f32 accumulator and narrowing only at the end of the
        // call made an N-token launch disagree with N single-token launches, so
        // batched prefill and the speculative verify advanced the state
        // differently from per-token decode. Measured on 64 tokens at head_dim
        // 128: one 64-token launch differed from 64 single-token launches in
        // EVERY state element, worst |rel| 1.989. The FP32 kernel was already
        // bit-exact across the same schedules. See
        // docs/bugs/2026-09-02-fp16-deltanet-recurrence-is-not-chunk-invariant.md
        //
        // Fewer roundings is the better approximation of an FP32 reference, and
        // that is what this used to optimise for. But prefill and decode must
        // agree with EACH OTHER, and speculative.rs already pays this price on
        // the rollback replay path for the same reason.
        //
        // The OUTPUT above is computed from the un-narrowed `sreg`, matching a
        // single-token call, which narrows only on store. The final store is then
        // an exact conversion, so n_tokens == 1 is unchanged.
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const unsigned int nidx =
                (unsigned int)(row_start * HD + row_in_tile * HD + col + i) ^ ((unsigned int)h << 19);
            const _Float16 nv =
                stochastic_round ? f32_to_f16_dither(sreg[i], nidx) : (_Float16)sreg[i];
            sreg[i] = (float)nv;
            S_tile[row_in_tile * HD + col + i] = sreg[i];
        }

        // One lane per group stores its own row; the 4 rows are contiguous, so
        // this is a single coalesced segment instead of four single-lane
        // 4-byte scattered writes.
        if (l == 0)
            output[b * stride + h * HD + row] = out_v;
        // --- end identical step ---
    }

    // Dither index UNIFIED with gated_delta_net_f16.hip and _f16_tree.hip:
    // `(row_start * HD + tile_flat) ^ (h << 19)`.
    //
    // This used to read `i ^ (blockIdx.x << 19) ^ (blockIdx.y << 7)`. With
    // `h == blockIdx.x` and `tile == blockIdx.y` that is `i ^ (h << 19) ^
    // (tile << 7)`, which coincides with the linear derivation only at
    // `tile == 0`: linear ADDS `row_start * HD` (= tile * 512) into the index
    // where this XORed `tile << 7` (= tile * 128). So the same element of the
    // same head could be dithered differently by the two kernels, and the
    // header three lines up requires them to agree bit-for-bit.
    //
    // The session (`blockIdx.z`) is deliberately NOT in the index: each session
    // has its own state buffer, and per-session linear replay — which the routed
    // kernel must reproduce — has no session term to match.
    // Exact conversion, NOT a second rounding: the per-token narrowing above
    // already left every element f16-representable. Rounding again here would
    // double-round the final token and break the equivalence.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (_Float16)S_tile[i];
}
