// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident indexed MoE decode GEMVs — the compact twins of
// gemv_oq8g256_moe_{gate_up_k8_indexed, gate_up_k8_indexed_batched,
// down_k8_indexed_batched_expanded}.
//
// WHY: routed experts were the last tensors still EXPANDED to one int8 per
// weight at load (`load_moe_expert` -> `oqplus_compact_to_moe_oq8_blocks`),
// because these three GEMVs were the only readers of that layout and no compact
// sibling existed. Dense tensors have stayed compact-resident by default since
// `oq8_arch_load` learned `split_compact_planes`. The expansion costs 1.80x on
// the dominant tensor class of a MoE model, and it compounds with the driver's
// 2 MiB GTT rounding (see `weight_pager::gtt_alloc_cost`) into 3.5x: the 122B
// occupies ~137 GiB resident for a 63.9 GiB artifact, which is why it does not
// load in 124 GiB of unified memory.
//
// LAYOUT (split planes, produced by `split_compact_planes`, identical to what
// the dense compact GEMVs read -- one layout, two call shapes):
//
//     nibbles : A + row*ng*128                      ng = K/256
//     side    : A + M*ng*128 + row*ng*side_stride   side_stride = block_stride-128
//     record  : [f16 scale][N_out x (u8 idx, i8 val)]
//
// Split rather than interleaved because a nibble row then starts at a multiple
// of 128 and EVERY row is cache-line aligned; the interleaved 136-byte block
// left only every fourth row aligned, measured at ~3.7% of achieved bandwidth
// (examples/bench_oq_layout, 226.4 -> 234.8 GB/s). The Oq8 blocks these replace
// are worse still: payload at 260g+4 is 4-byte-misaligned on EVERY group, which
// is why that kernel reads its weights one byte at a time.
//
// NUMERICS: W4A16 at decode -- inline int4 dequant plus the sparse int8 overlay,
// against the full-precision f32 activation, exactly as
// `gemv_oq_compact_grouped_v3` does it. The overlay REPLACES the bulk nibble and
// the loader zeroes that nibble (`normalize_compact_overlays`), so a correction
// is just val*x[idx] and depends on no other lane.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Oq8G256 interleaved block: [f32 scale | 256 int8]. Present here because a
// mixed layer routes its promoted experts through these kernels too.
#define OQCM_OQ8_BLK 260

#define OQCM_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCM_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// One wave32 owns one output row. Lane L takes bytes [4L, 4L+4) of each group's
// 128 nibble bytes -- so the wave reads exactly 128 contiguous bytes per group,
// one full cache line, perfectly coalesced. Byte j of the loaded word carries
// weights 2j (low nibble) and 2j+1 (high), matching OQCV3_WORD's ordering.
//
// Returns the row dot product in LANE 0 only.
__device__ __forceinline__ float oqc_moe_row_dot(
    const uint8_t* __restrict__ A,
    const float* __restrict__ x,
    int row, int M, int K, int block_stride, int lane
) {
    const int ng = K / 256;

    // MIXED-PRECISION LAYERS. `block_stride` arrives from a PER-EXPERT table,
    // and 0 means this expert was promoted to Oq8G256 while its neighbours
    // stayed compact -- so one launch has to serve both layouts.
    //
    // 0 as the sentinel, not 260: a compact stride is `130 + 2*N_out`, and 260
    // satisfies that at N_out=65, so a stride cannot identify its own layout.
    //
    // The branch is WORKGROUP-UNIFORM -- a whole workgroup shares one expert_id
    // (it comes from blockIdx.y/z, never from the lane) -- so it costs a scalar
    // compare, not divergence.
    //
    // The arithmetic below is the Oq8 kernel's, term for term and in its order,
    // so an Oq8 expert routed through here is BIT-IDENTICAL to routing it
    // through `gemv_oq8g256_moe_*`.
    if (block_stride == 0) {
        const char* __restrict__ row_ptr =
            (const char*)A + (long long)row * ng * OQCM_OQ8_BLK;
        const int boff = lane * 8;
        float acc8 = 0.0f;
        for (int g = 0; g < ng; ++g) {
            const char* gp = row_ptr + (long long)g * OQCM_OQ8_BLK;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            const signed char* w8 = (const signed char*)(gp + 4 + boff);
            const int base = g * 256 + boff;
            float gsum = 0.0f;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                gsum += (float)w8[n] * x[base + n];
            }
            acc8 += sc * gsum;
        }
        for (int off = 16; off > 0; off >>= 1) acc8 += __shfl_down(acc8, off);
        return acc8;
    }

    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;

    const uint8_t* __restrict__ n_row = A + (long long)row * ng * 128LL;
    const uint8_t* __restrict__ s_row = A + (long long)M * ng * 128LL
                                          + (long long)row * ng * (long long)side_stride;

    float acc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const uint8_t* sb = s_row + (long long)g * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        uint32_t wv;
        __builtin_memcpy(&wv, n_row + (long long)g * 128 + lane * 4, 4);

        const int base = g * 256 + lane * 8;
        float s = (float)OQCM_LO(wv, 0) * x[base + 0]
                + (float)OQCM_HI(wv, 0) * x[base + 1]
                + (float)OQCM_LO(wv, 1) * x[base + 2]
                + (float)OQCM_HI(wv, 1) * x[base + 3]
                + (float)OQCM_LO(wv, 2) * x[base + 4]
                + (float)OQCM_HI(wv, 2) * x[base + 5]
                + (float)OQCM_LO(wv, 3) * x[base + 6]
                + (float)OQCM_HI(wv, 3) * x[base + 7];

        // BRANCHLESS, deliberately: on the dense sibling the divergent
        // `lane < n_ov` guard -- not the table load, not the scattered gather --
        // was what the overlay actually cost (3.2% of decode). Every lane issues
        // the same loads; inactive ones multiply by zero.
        const uint8_t* table = sb + 2;
        {
            const int e   = (lane < n_ov) ? lane : 0;
            const int idx = (int)table[2 * e];
            const int val = (lane < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
            s += (float)val * x[g * 256 + idx];
        }
        // N_out > 32 never occurs in practice; the bound is wave-uniform so this
        // costs one predicated compare when it does not.
        for (int e = lane + 32; e < n_ov; e += 32) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            s += (float)val * x[g * 256 + idx];
        }

        acc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    return acc;
}

// Grid (M, K_TOP, 1), block = 32. `x_per_slot`: 1 selects [K_TOP x K] (one
// rotated basis per expert, which per-expert AWQ scales require); 0 a shared
// [K] row. See gemv_oq8g256_moe_gate_up_indexed.hip's preamble.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq_compact_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [k_top]
    const int* __restrict__ expert_strides,              // [n_exp], 0 = Oq8
    const float* __restrict__ x,                         // [K] or [K_TOP x K]
    float* __restrict__ y_gate,                          // [K_TOP x MI]
    float* __restrict__ y_up,                            // [K_TOP x MI]
    int M, int K, int x_per_slot
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int lane = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)(x_per_slot ? krank : 0) * K;

    const float acc =
        oqc_moe_row_dot(A, x_row, row, M, K, expert_strides[expert_id], lane);

    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[krank * mi + row] = acc;
        else          y_up[krank * mi + (row - mi)] = acc;
    }
}

// One 32-bit word = 8 nibbles = 8 weights, against two float4 of activation.
// `ww` not `w`, so the `(xa).w` selector cannot capture the caller's argument.
#define OQCM_WORD(ww, xa, xb)                                                  \
    ( (float)OQCM_LO(ww, 0) * (xa).x + (float)OQCM_HI(ww, 0) * (xa).y          \
    + (float)OQCM_LO(ww, 1) * (xa).z + (float)OQCM_HI(ww, 1) * (xa).w          \
    + (float)OQCM_LO(ww, 2) * (xb).x + (float)OQCM_HI(ww, 2) * (xb).y          \
    + (float)OQCM_LO(ww, 3) * (xb).z + (float)OQCM_HI(ww, 3) * (xb).w )

// WIDE-LOAD row dot: the `gemv_oq_compact_grouped_v3` weight-read shape, ported
// to the indexed-MoE kernels.
//
// `oqc_moe_row_dot` above gives each group all 32 lanes, so a lane loads 4
// nibble bytes — a plain dword. v3's note prices exactly that gap: the pure-read
// stream moves 243 GB/s with `dwordx4`, and ablations there found the residual
// gap is weight-load SHAPE, not bytes, ALU or activation traffic. Measured here
// on Qwen3.5-35B-A3B--oq4.25++ decode, in ONE trace: the routed-expert kernels
// ran at ~104 GB/s while grouped_v3 hit ~225 GB/s on lm_head — same GPU, same
// compact dtype, so the difference is this loop, not the hardware.
//
// So: 8 lanes per group and a dwordx4 (16 bytes = 32 weights per lane). One
// wave32 covers FOUR groups per round instead of one, quartering the weight VMEM
// instruction count per byte moved.
//
// Needs ng % 4 == 0; callers keep `oqc_moe_row_dot` otherwise. Routed gate_up
// has K = 2048 (ng = 8) and qualifies; down_proj has K = 512 (ng = 2) and does
// not — it would need v2's 16-lane dwordx2 shape instead.
//
// The Oq8 promoted-expert layout (block_stride == 0) is NOT handled here; that
// branch stays on the narrow path so its bit-identical-to-gemv_oq8g256 contract
// is untouched. Both selectors are workgroup-uniform (one expert per workgroup).
//
// NUMERICS ARE NOT BIT-IDENTICAL to the narrow path, and that is inherent, not
// a bug: same int4 dequant, same sparse int8 overlay, same f32 activation, but a
// different SUMMATION ASSOCIATION. Narrow gives one group all 32 lanes, so a
// lane's partial is 8 weights of one group; wide gives a lane 32 weights of one
// group and puts 4 groups in a wave. Both reduce with the same 32-lane
// __shfl_down; the order of the adds differs.
//
// It is the association `gemv_oq_compact_grouped_v3` already uses for every
// dense projection and for lm_head on this model, so this makes the routed
// experts agree with the rest of the forward rather than diverge from it.
//
// Observed effect at greedy decode: output tracks the narrow path until a
// near-tie and can then take the other branch. Evidence it is a tie-flip and not
// corruption: AR decode of humaneval_0 is **60/60 tokens IDENTICAL** to the
// narrow path, while the prose prompt splits at token 5 onto a loop attractor
// that several unrelated perturbations of this model also land on.
//
// ⚠️ DO NOT try to validate this kernel with `hipfire-eval --battery
// perplexity`. It does not exercise it. PROVEN by deliberately corrupting this
// function to return `facc * 0.5f` and rebuilding the daemon: perplexity stayed
// bit-identical at ppl 2.8994 / nll 1.0645. That battery is prefill/batch
// shaped; these are the batch=1 DECODE gemv kernels. An earlier revision of this
// comment, and the commit that introduced it, cited exactly that perplexity pair
// as proof of equivalence -- it proved nothing. Compare DECODE TOKENS on a
// confident prompt instead.
__device__ __forceinline__ float oqc_moe_row_dot_wide(
    const uint8_t* __restrict__ A,
    const float* __restrict__ x,
    int row, int M, int K, int block_stride, int lane
) {
    const int ng = K / 256;
    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;
    const int gir = lane >> 3;   // 0..3 : which of the four groups this round
    const int lig = lane & 7;    // 0..7 : which 32-weight chunk within it
    const int elt = lig * 32;

    const uint8_t* __restrict__ n_row = A + (long long)row * ng * 128LL;
    const uint8_t* __restrict__ s_row = A + (long long)M * ng * 128LL
                                          + (long long)row * ng * (long long)side_stride;

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 4) {
        const int gi = r + gir;
        const uint8_t* sb = s_row + (long long)gi * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        uint32_t wv[4];
        __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 16, 16);

        const int xoff = gi * 256 + elt;
        const float4 x0 = *(const float4*)(x + xoff);
        const float4 x1 = *(const float4*)(x + xoff + 4);
        const float4 x2 = *(const float4*)(x + xoff + 8);
        const float4 x3 = *(const float4*)(x + xoff + 12);
        const float4 x4 = *(const float4*)(x + xoff + 16);
        const float4 x5 = *(const float4*)(x + xoff + 20);
        const float4 x6 = *(const float4*)(x + xoff + 24);
        const float4 x7 = *(const float4*)(x + xoff + 28);

        float s = OQCM_WORD(wv[0], x0, x1) + OQCM_WORD(wv[1], x2, x3)
                + OQCM_WORD(wv[2], x4, x5) + OQCM_WORD(wv[3], x6, x7);

        // Branchless for the same reason as the narrow path and v3: the
        // divergent guard, not the gather, was what the overlay cost.
        const uint8_t* table = sb + 2;
        {
            const int e   = (lig < n_ov) ? lig : 0;
            const int idx = (int)table[2 * e];
            const int val = (lig < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
            s += (float)val * x[gi * 256 + idx];
        }
        for (int e = lig + 8; e < n_ov; e += 8) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            s += (float)val * x[gi * 256 + idx];
        }

        facc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    return facc;
}

// K-SLICED variant of `oqc_moe_row_dot_wide`: this wave handles only groups
// [g0, g0+gpw), so SPLIT waves can share one output row.
//
// Why: routed gate_up has ng = 8, and the wide path eats 4 groups per round, so
// each wave walks TWO dependent rounds. The dense split-K shape reaching 195
// GB/s has the same 8192 waves but only ONE round. Measured here: gate_up 63.5
// us vs the dense 8192-wave shape at 48.3 us for comparable bytes (8.6 vs 9.4
// MB) -- the extra round is the difference, not bandwidth.
//
// Ruled out first, so nobody re-runs them: occupancy (1-wave workgroups -> 8
// waves/workgroup changed nothing), load width (dwordx2 on down_proj bought
// 0.3%), and expert SCATTER -- forcing the 8 routed experts to be consecutive
// allocations, same byte volume, moved 63.5 -> 59.2 us, only 6.8%. So a
// per-layer expert-buffer consolidation is NOT worth its blast radius.
__device__ __forceinline__ float oqc_moe_row_dot_wide_slice(
    const uint8_t* __restrict__ A,
    const float* __restrict__ x,
    int row, int M, int K, int block_stride, int lane, int g0, int gpw
) {
    const int ng = K / 256;
    const int side_stride = block_stride - 128;
    const int n_ov = (block_stride - 130) >> 1;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const int elt = lig * 32;

    const uint8_t* __restrict__ n_row = A + (long long)row * ng * 128LL;
    const uint8_t* __restrict__ s_row = A + (long long)M * ng * 128LL
                                          + (long long)row * ng * (long long)side_stride;

    float facc = 0.0f;
    for (int r = 0; r < gpw; r += 4) {
        const int gi = g0 + r + gir;
        const uint8_t* sb = s_row + (long long)gi * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        uint32_t wv[4];
        __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 16, 16);

        const int xoff = gi * 256 + elt;
        const float4 x0 = *(const float4*)(x + xoff);
        const float4 x1 = *(const float4*)(x + xoff + 4);
        const float4 x2 = *(const float4*)(x + xoff + 8);
        const float4 x3 = *(const float4*)(x + xoff + 12);
        const float4 x4 = *(const float4*)(x + xoff + 16);
        const float4 x5 = *(const float4*)(x + xoff + 20);
        const float4 x6 = *(const float4*)(x + xoff + 24);
        const float4 x7 = *(const float4*)(x + xoff + 28);

        float sd = OQCM_WORD(wv[0], x0, x1) + OQCM_WORD(wv[1], x2, x3)
                 + OQCM_WORD(wv[2], x4, x5) + OQCM_WORD(wv[3], x6, x7);

        const uint8_t* table = sb + 2;
        {
            const int e   = (lig < n_ov) ? lig : 0;
            const int idx = (int)table[2 * e];
            const int val = (lig < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
            sd += (float)val * x[gi * 256 + idx];
        }
        for (int e = lig + 8; e < n_ov; e += 8) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            sd += (float)val * x[gi * 256 + idx];
        }

        facc += sd * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    return facc;
}

// SPLIT-K gate_up: OQCM_GU_SPLIT waves share one output row, 8/SPLIT rows per
// 256-thread workgroup. Grid (ceil(M/rows_per_block), K_TOP, N).
#define OQCM_GU_SPLIT 2
#define OQCM_GU_ROWS  (8 / OQCM_GU_SPLIT)

__launch_bounds__(256)
extern "C" __global__ void gemv_oq_compact_moe_gate_up_k8_indexed_batched_splitk(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const int* __restrict__ expert_strides,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K, int K_TOP, int x_per_slot
) {
    const int wig = (int)(threadIdx.x >> 5);
    const int rib = wig / OQCM_GU_SPLIT;
    const int sub = wig % OQCM_GU_SPLIT;
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * OQCM_GU_ROWS + rib;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;

    __shared__ float part[8];

    if (row < M) {
        const int expert_id = topk_indices[bid * K_TOP + krank];
        const uint8_t* __restrict__ A =
            reinterpret_cast<const uint8_t*>(expert_ptrs[expert_id]);
        const float* __restrict__ x_row =
            x + (x_per_slot ? ((long long)bid * K_TOP + krank) : (long long)bid) * K;
        const int e_stride = expert_strides[expert_id];
        const int gpw = (K / 256) / OQCM_GU_SPLIT;
        // MIXED PRECISION: 0 means this expert was promoted to Oq8G256 while its
        // neighbours stayed compact, and one launch has to serve both. The
        // compact slice helper cannot read that layout, so for a promoted expert
        // ONE wave computes the whole row the narrow way and the other waves
        // contribute zero. Workgroup-uniform (one expert per workgroup), so it
        // costs a scalar compare and no divergence.
        const float acc = (e_stride == 0)
            ? ((sub == 0) ? oqc_moe_row_dot(A, x_row, row, M, K, 0, lane) : 0.0f)
            : oqc_moe_row_dot_wide_slice(A, x_row, row, M, K, e_stride, lane,
                                         sub * gpw, gpw);
        if (lane == 0) part[wig] = acc;
    }
    __syncthreads();
    if (row < M && sub == 0 && lane == 0) {
        float t = 0.0f;
        for (int i = 0; i < OQCM_GU_SPLIT; ++i) t += part[rib * OQCM_GU_SPLIT + i];
        const int mi = M >> 1;
        const long long slot = (long long)bid * K_TOP + krank;
        if (row < mi) y_gate[slot * mi + row] = t;
        else          y_up[slot * mi + (row - mi)] = t;
    }
}

// ROWS/BLOCK: one wave per output row// ROWS/BLOCK: one wave per output row, OQCM_ROWS_PER_BLOCK waves per workgroup.
//
// These launched as block=32 -- a SINGLE wave per workgroup, no LDS -- with
// __launch_bounds__(32, 16). On RDNA the per-CU WORKGROUP-slot limit then binds
// before the wave-slot limit does, so 16 workgroups x 1 wave = 16 waves/CU was
// the ceiling, and a memory-bound GEMV has nothing left to hide DRAM latency
// with. Measured on Qwen3.5-35B-A3B--oq4.25++ decode: the routed-expert kernels
// ran at ~104 GB/s while `gemv_oq_compact_grouped_v3` hit 225 GB/s (90% of the
// 250 GB/s ceiling) on lm_head in the SAME trace -- same GPU, same dtype, so
// 104 was a launch-geometry artefact, not a bandwidth wall.
//
// Packing 8 rows into a 256-thread workgroup makes waves-per-workgroup rather
// than workgroup slots set occupancy. The per-row math is untouched:
// `oqc_moe_row_dot` reduces with __shfl_down over 32 lanes, uses no LDS and no
// blockDim, so each wave still reduces independently and results stay
// BIT-IDENTICAL. `expert_id` still comes from blockIdx.y/z, so the
// workgroup-uniform branch in that helper stays workgroup-uniform.
#define OQCM_ROWS_PER_BLOCK 8
#define OQCM_BLOCK_THREADS (OQCM_ROWS_PER_BLOCK * 32)

// Grid (ceil(M/OQCM_ROWS_PER_BLOCK), K_TOP, N), block = OQCM_BLOCK_THREADS.
__launch_bounds__(OQCM_BLOCK_THREADS, 4)
extern "C" __global__ void gemv_oq_compact_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N x K_TOP]
    const int* __restrict__ expert_strides,              // [n_exp], 0 = Oq8
    const float* __restrict__ x,                    // [N x K] or [N x K_TOP x K]
    float* __restrict__ y_gate,                          // [N x K_TOP x MI]
    float* __restrict__ y_up,                            // [N x K_TOP x MI]
    int M, int K, int K_TOP, int x_per_slot
) {
    const int row = blockIdx.x * OQCM_ROWS_PER_BLOCK + (int)(threadIdx.x >> 5);
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = (int)(threadIdx.x & 31);

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row =
        x + (x_per_slot ? ((long long)bid * K_TOP + krank) : (long long)bid) * K;

    // Workgroup-uniform: one expert per workgroup, so both terms are scalar.
    const int e_stride = expert_strides[expert_id];
    const float acc = (e_stride != 0 && ((K / 256) & 3) == 0)
        ? oqc_moe_row_dot_wide(A, x_row, row, M, K, e_stride, lane)
        : oqc_moe_row_dot(A, x_row, row, M, K, e_stride, lane);

    if (lane == 0) {
        const int mi = M >> 1;
        const long long slot = (long long)bid * K_TOP + krank;
        if (row < mi) y_gate[slot * mi + row] = acc;
        else          y_up[slot * mi + (row - mi)] = acc;
    }
}

// Atomic-free: writes each (token, krank) result to expert_outputs; reuse
// moe_down_combine_k8_batched to fold into the residual.
// Grid (ceil(M/OQCM_ROWS_PER_BLOCK), K_TOP, N), block = OQCM_BLOCK_THREADS.
// Same rows-per-block change as gate_up above; see the note there.
__launch_bounds__(OQCM_BLOCK_THREADS, 4)
extern "C" __global__ void gemv_oq_compact_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N x K_TOP]
    const int*   __restrict__ expert_strides,            // [n_exp], 0 = Oq8
    const float* __restrict__ rot_batch,                 // [N x K_TOP x K]
    float*       __restrict__ expert_outputs,            // [N x K_TOP x M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x * OQCM_ROWS_PER_BLOCK + (int)(threadIdx.x >> 5);
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = (int)(threadIdx.x & 31);

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const uint8_t* __restrict__ A =
        reinterpret_cast<const uint8_t*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const float acc =
        oqc_moe_row_dot(A, x, row, M, K, expert_strides[expert_id], lane);

    if (lane == 0) {
        expert_outputs[((long long)bid * K_TOP + krank) * (long long)M + row] = acc;
    }
}
