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
// Deterministic dithered narrowing, f32 -> f16.
//
// Round-to-nearest on a RECURRENT accumulator is biased: the same value always
// rounds the same way, so the error does not cancel across steps and the state
// drifts. Measured on a 35B-A3B, FP16-vs-FP32 state divergence grew 13x over
// 2.5x the tokens — superlinear, i.e. compounding rather than a fixed storage
// cost.
//
// The dither is a hash of the VALUE'S OWN BITS and the element index, not a
// counter or an RNG carried across calls. That matters: the result stays a pure
// function of the input, so a spec-decode snapshot still restores exactly what
// it saved — the determinism this path was built for (see the dispatcher doc,
// which notes the Q8 path's stochastic rounding was removed for precisely that
// reason). Because an accumulator's value changes every step, the dither changes
// with it, which is what breaks the correlated bias.
//
// Adding the dither into the 13 mantissa bits f16 discards and then truncating
// makes the narrowing exact, so there is no second rounding.
__device__ __forceinline__ _Float16 f32_to_f16_dither(float x, unsigned int idx) {
    unsigned int u = __float_as_uint(x);
    unsigned int r = u ^ (idx * 2654435761u);
    r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
    unsigned int dithered = (u + (r & 0x1FFFu)) & 0xFFFFE000u;
    return (_Float16)__uint_as_float(dithered);
}

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
            output[t * stride + h * HD + row] = out_v;
        // --- end identical step ---
    }

    // Narrow back to f16. Round-to-nearest, no stochastic term — deterministic,
    // so repeated runs on identical input are bit-identical and a spec-decode
    // rollback restores exactly what it saved.
    // Exact conversion, NOT a second rounding: the per-token narrowing above
    // already left every element f16-representable. Rounding again here would
    // double-round the final token and break the equivalence.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (_Float16)S_tile[i];
}
