// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Decode / verify attention over KVarN storage on WMMA (RDNA3 wave32), split
// over the context and shared across each KV head's query group.
//
// The tile path (attention_flash_kvarn_tile_batched + reduce) runs one workgroup
// per (query head, 128-token tile, row): a KV head's K records and V rows are
// dequantized and read once per query head sharing it (6x on Qwen3.8-27B's 24/4
// heads, 8x on Qwen3.6-35B-A3B's 16/2) and once per DFlash verify row, so its cost
// grows with heads x rows x context.
//
// Here the M = G x R query rows of one KV head (G query heads, R rows of one
// session, row-major: m = r*G + g) are the WMMA rows. One workgroup owns 16 of
// them -- a decode step's whole query group -- for one context split; it walks
// the split in 16-token sub-tiles, dequantizing each K/V sub-tile ONCE into LDS
// for all its rows, with the staging and prefetch of attention_prefill_kvarn_wmma.
// The four waves split the head dims: wave w forms S over dims [64w, 64w+64)
// (four WMMAs), the partial S tiles are summed through LDS, every wave runs the
// same online softmax, and wave w accumulates O for its own 64 dims. With one
// wave per 16 rows instead (the prefill layout), a decode step's single row tile
// left three waves at the barrier while one ran all 32 WMMAs and the softmax --
// measured as the largest part of the kernel.
// Each split writes per-row (max, sum, unnormalized O) partials in the tile
// path's layout, with the split as the "tile", so the same
// attention_flash_asym_reduce_batched combines them. The max is stored in
// natural-log units (the softmax runs on exp2 of log2(e)-scaled scores), which is
// what the reduce's expf expects.
//
// The split length is derived HERE, from the workgroup's own context: the
// gridDim.y splits divide it into whole 16-token sub-tiles (not 128-token blocks,
// so a short context still spreads over workgroups). Sized on the host from the
// launch's max_ctx_len, a hipGraph-captured decode (which bakes the cache's
// physical capacity there) would leave most splits past the context and the few
// live ones walking long ranges. Every split writes its partial -- an empty one
// as sum 0, which the reduce skips -- so the reduce always merges gridDim.y.
//
// Masking: causal per row from `positions` (token <= position), or, with a tree
// bias, every token below block_start + block_cols plus the additive bias on
// the block's columns -- the tile path's semantics.
//
// Two entries over one body: attention_decode_kvarn_wmma (one session's rows,
// grid [n_kv_heads, n_splits, ceil(G*R / 16)]) and
// attention_decode_kvarn_wmma_routed (independent sessions' rows through the
// routed pointer tables, grid [n_kv_heads, n_splits, rows]). Block: [128].
// Dynamic LDS: see the dispatcher. Specialized for head_dim 256; KV_BITS compiled
// in like the prefill kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#ifndef KV_BITS
#define KV_BITS 4
#endif
namespace {
constexpr int HD = 256;       // head_dim
constexpr int MQ = 16;        // query rows per workgroup (one WMMA row tile)
constexpr int DCW = HD / 16 / 4;  // 16-dim chunks per wave (4 waves split HD)
constexpr int NT = 16;        // KV tokens per sub-tile
constexpr int GROUP = 128;    // KVarN block / record columns
constexpr int QS = HD + 8;    // Q_lds row stride (halfs)
constexpr int KS = HD + 8;    // K_lds row stride (token-major)
constexpr int VS = NT + 8;    // V_T row stride (dim-major)
// V_T row R starts at R*VS + (R/16)*VG halfs. A staging store writes one dword
// to 16 rows 16 apart (one per 16-dim group); at a plain VS stride those rows all
// map to one LDS bank -- a 16-way conflict on every store, measured as the
// largest part of the staging. Offsetting each group by two dwords spreads them
// over 32 banks; a group's rows stay 8-byte aligned for the operand reads.
constexpr int VG = 4;
constexpr int PS = NT + 8;    // per-wave P scratch row stride
}  // namespace

__device__ __forceinline__ float row16_max(float v) {
#define XM(M) { int s_ = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 | (M), 0xf, 0xf, true); \
                v = fmaxf(v, __builtin_bit_cast(float, s_)); }
    XM(1) XM(2) XM(4) XM(8)
#undef XM
    return v;
}
__device__ __forceinline__ float row16_sum(float v) {
#define XM(M) { int s_ = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 | (M), 0xf, 0xf, true); \
                v += __builtin_bit_cast(float, s_); }
    XM(1) XM(2) XM(4) XM(8)
#undef XM
    return v;
}

// The kernel body: `R` rows of one session (q, positions and partials point at
// row 0 of them), the workgroup's 16-row group starting at query row m0.
__device__ __forceinline__ void decode_kvarn_body(
    const float* __restrict__ q,               // [R x n_heads x HD]
    const uint8_t* __restrict__ records,       // KVarN K records, block-tiled
    const void* __restrict__ window,           // [GROUP x kv_dim] f32 or f16
    const unsigned char* __restrict__ v_cache, // Q8_0, [max_seq x n_kv_heads*(HD/32)*34]
    float* __restrict__ partials,              // [R x n_heads x n_splits x (2+HD)]
    const int* __restrict__ positions,         // [R] absolute position of each row
    const float* __restrict__ tree_bias,       // optional [R x block_cols]
    int R,
    int n_heads,
    int n_kv_heads,
    float scale_attn,
    int n_full_blocks,
    int rec_bytes,
    int window_f16,
    int block_start,                           // tree mode only
    int block_cols,                            // tree mode only
    int m0
) {
    const int kv_h = blockIdx.x;
    const int split = blockIdx.y;
    const int n_splits = gridDim.y;
    const int G = n_heads / n_kv_heads;
    const int M = G * R;
    if (kv_h >= n_kv_heads || m0 >= M) return;
    const int rows_here = min(MQ, M - m0);
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int kv_dim = n_kv_heads * HD;
    const bool tree = (tree_bias != nullptr);

    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    _Float16* Q_lds = (_Float16*)smem;              // MQ x QS
    _Float16* K_lds = Q_lds + MQ * QS;              // NT x KS
    _Float16* V_T = K_lds + NT * KS;                // HD x VS
    _Float16* P_lds = V_T + HD * VS + (HD / 16) * VG;  // 4 waves x 16 x PS
    float* S_x = (float*)(P_lds + 4 * 16 * PS);     // 4 waves x 32 lanes x 8
    int* s_maxpos = (int*)(S_x + 4 * 32 * 8);

    if (tid == 0) *s_maxpos = -1;
    __syncthreads();

    const float LOG2E = 1.4426950408889634f;
    const float qscale = scale_attn * LOG2E;
    for (int i = tid; i < MQ * HD; i += 128) {
        const int ml = i / HD;
        const int d = i - ml * HD;
        const int m = m0 + ml;
        float v = 0.0f;
        if (ml < rows_here) {
            const int r = m / G;
            const int h = kv_h * G + (m - r * G);
            v = q[((size_t)r * n_heads + h) * HD + d] * qscale;
        }
        Q_lds[ml * QS + d] = (_Float16)v;
    }
    if (tid < rows_here) {
        const int r = (m0 + tid) / G;
        atomicMax(s_maxpos, tree ? (block_start + block_cols - 1) : positions[r]);
    }
    __syncthreads();
    const int kv_len = *s_maxpos + 1;
    // Splits are whole sub-tiles, not whole KVarN blocks: a short context (all
    // window, a block or two) still spreads over workgroups.
    const int kv_subs = (kv_len + NT - 1) / NT;
    const int split_len = (kv_subs + n_splits - 1) / n_splits * NT;
    const int t_begin = split * split_len;
    if (t_begin >= kv_len) {
        // Past this workgroup's context: an empty partial (sum 0) per row.
        if (tid < rows_here) {
            const int m = m0 + tid;
            const int r = m / G;
            const int h = kv_h * G + (m - r * G);
            float* p = partials + (((size_t)r * n_heads + h) * n_splits + split) * (2 + HD);
            p[0] = -1e30f;
            p[1] = 0.0f;
        }
        return;
    }
    const int t_end = min(t_begin + split_len, kv_len);

    // This lane's 8 rows in the C layout: row 2j + half_id of the 16.
    int rrow[8], rpos[8];
    float m_row[8], l_row[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int ml = 2 * j + half_id;
        const bool live = ml < rows_here;
        const int r = live ? (m0 + ml) / G : 0;
        rrow[j] = r;
        rpos[j] = !live ? -1 : (tree ? (block_start + block_cols - 1) : positions[r]);
        m_row[j] = -INFINITY;
        l_row[j] = 0.0f;
    }
    // This wave's output dims: chunks dc0 .. dc0+DCW-1 of 16.
    const int dc0 = wave * DCW;
    float8_t O[DCW];
#pragma unroll
    for (int c = 0; c < DCW; c++) O[c] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};

    constexpr int cpb_shift = (KV_BITS == 8) ? 0 : ((KV_BITS == 4) ? 1 : 2);
    constexpr int codemask = (1 << KV_BITS) - 1;
    constexpr int tpw = 32 / KV_BITS;
    constexpr int wpd = NT / tpw;
    const int off_scale = (HD * GROUP) >> cpb_shift;
    const int off_zp = off_scale + HD * 2;
    const int off_scol = off_zp + HD * 2;
    const int bph = HD / 32;
    const int v_row_stride = n_kv_heads * bph * 34;

    // Software pipeline and thread mapping: as attention_prefill_kvarn_wmma.
    constexpr int DPT = HD * wpd / 128;
    typedef _Float16 __attribute__((ext_vector_type(DPT))) hvec_t;
    const int k_w = tid % wpd;
    const int k_d0 = (tid / wpd) * DPT;
    typedef _Float16 __attribute__((ext_vector_type(tpw))) scvec_t;
    uint32_t pk_word[DPT];
    hvec_t pk_sa, pk_za;
    scvec_t pk_sc;
    // The recent window (the trailing partial block) rides the same pipeline: a
    // thread's DPT dims x tpw tokens as raw words (f32: one per value, f16: a pair
    // per word). Loaded where it was used, element by element, it cost ~6 us a
    // sub-tile -- 4.6x the tile path on a short, window-only context.
    typedef uint32_t __attribute__((ext_vector_type(DPT))) wvec_t;
    uint32_t pw[tpw * DPT];
    const int v_hh = tid & 1;
    const int v_b = (tid >> 1) & (bph - 1);
    const int v_tp = tid >> 4;
    uint32_t pv[2][4];
    uint16_t pv_s[2];
    bool pv_live[2] = {false, false};
    auto prefetch = [&](int kt_n) {
        const int blk_n = kt_n / GROUP;
        if (kt_n < t_end && blk_n < n_full_blocks) {
            const uint8_t* rec = records + (size_t)(blk_n * n_kv_heads + kv_h) * rec_bytes;
            const _Float16* scale_abs = (const _Float16*)(rec + off_scale);
            const _Float16* zp_abs = (const _Float16*)(rec + off_zp);
            const _Float16* s_col = (const _Float16*)(rec + off_scol);
            const int t_in = kt_n - blk_n * GROUP;
            const int t0 = t_in + k_w * tpw;
#pragma unroll
            for (int n = 0; n < DPT; n++)
                pk_word[n] = *(const uint32_t*)(rec + (((k_d0 + n) * GROUP + t0) >> cpb_shift));
            pk_sa = *(const hvec_t*)(scale_abs + k_d0);
            pk_za = *(const hvec_t*)(zp_abs + k_d0);
            pk_sc = *(const scvec_t*)(s_col + t0);
        } else if (kt_n < t_end) {
            const int win_base = n_full_blocks * GROUP;
#pragma unroll
            for (int tt = 0; tt < tpw; tt++) {
                const int tok = kt_n + k_w * tpw + tt;
                wvec_t w = {};
                if (tok < t_end) {
                    const size_t off = (size_t)(tok - win_base) * kv_dim + kv_h * HD + k_d0;
                    if (window_f16) {
                        typedef uint32_t __attribute__((ext_vector_type(DPT / 2))) w2_t;
                        const w2_t h = *(const w2_t*)((const _Float16*)window + off);
#pragma unroll
                        for (int n = 0; n < DPT / 2; n++) w[n] = h[n];
                    } else {
                        w = *(const wvec_t*)((const float*)window + off);
                    }
                }
#pragma unroll
                for (int n = 0; n < DPT; n++) pw[tt * DPT + n] = w[n];
            }
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int tok = kt_n + 2 * v_tp + u;
            pv_live[u] = tok < t_end;
            if (pv_live[u]) {
                const uint16_t* base = (const uint16_t*)(
                    v_cache + (size_t)tok * v_row_stride + (size_t)(kv_h * bph + v_b) * 34);
                pv_s[u] = base[0];
#pragma unroll
                for (int e = 0; e < 4; e++)
                    pv[u][e] = (uint32_t)base[1 + v_hh * 8 + 2 * e]
                             | ((uint32_t)base[2 + v_hh * 8 + 2 * e] << 16);
            }
        }
    };
    prefetch(t_begin);

    const _Float16* qrow = Q_lds + half * QS;
    const _Float16* krow = K_lds + half * KS;
    _Float16* prow = P_lds + wave * 16 * PS;
    for (int kt = t_begin; kt < t_end; kt += NT) {
        // ── Stage K (token-major f16) ──
        const int blk = kt / GROUP;
        if (blk < n_full_blocks) {
#pragma unroll
            for (int tt = 0; tt < tpw; tt++) {
                const float sc = (float)pk_sc[tt];
                hvec_t v;
#pragma unroll
                for (int n = 0; n < DPT; n++) {
                    const float code = (float)((pk_word[n] >> (tt * KV_BITS)) & codemask);
                    v[n] = (_Float16)((code * (float)pk_sa[n] + (float)pk_za[n]) * sc);
                }
                *(hvec_t*)(K_lds + (k_w * tpw + tt) * KS + k_d0) = v;
            }
        } else {
#pragma unroll
            for (int tt = 0; tt < tpw; tt++) {
                hvec_t v;
                if (window_f16) {
#pragma unroll
                    for (int n = 0; n < DPT; n++)
                        v[n] = __builtin_bit_cast(
                            _Float16, (uint16_t)(pw[tt * DPT + (n >> 1)] >> (16 * (n & 1))));
                } else {
#pragma unroll
                    for (int n = 0; n < DPT; n++)
                        v[n] = (_Float16)__builtin_bit_cast(float, pw[tt * DPT + n]);
                }
                *(hvec_t*)(K_lds + (k_w * tpw + tt) * KS + k_d0) = v;
            }
        }
        // ── Stage V (Q8_0 -> f16, dim-major) ──
        {
            const float s0 = pv_live[0] ? (float)__builtin_bit_cast(_Float16, pv_s[0]) : 0.0f;
            const float s1 = pv_live[1] ? (float)__builtin_bit_cast(_Float16, pv_s[1]) : 0.0f;
            _Float16* vt = V_T + (v_b * 32 + v_hh * 16) * VS + (2 * v_b + v_hh) * VG + 2 * v_tp;
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const signed char q0 = (signed char)(pv[0][e >> 2] >> (8 * (e & 3)));
                const signed char q1 = (signed char)(pv[1][e >> 2] >> (8 * (e & 3)));
                typedef _Float16 __attribute__((ext_vector_type(2))) h2_t;
                *(h2_t*)(vt + e * VS) = (h2_t){(_Float16)(s0 * (float)q0), (_Float16)(s1 * (float)q1)};
            }
        }
        prefetch(kt + NT);
        __syncthreads();

        // ── This wave's share of S = Q K^T: dims [64w, 64w+64) ──
        {
            float8_t sp = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int c = 0; c < DCW; c++) {
                const int dc = dc0 + c;
                half16_t a, b;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    a[j] = qrow[dc * 16 + j];
                    b[j] = krow[dc * 16 + j];
                }
                sp = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sp);
            }
            *(float8_t*)(S_x + (wave * 32 + lane) * 8) = sp;
        }
        __syncthreads();
        float8_t s = *(const float8_t*)(S_x + lane * 8);
#pragma unroll
        for (int w = 1; w < 4; w++) s += *(const float8_t*)(S_x + (w * 32 + lane) * 8);

        // ── Mask + bias + online softmax (log2 domain), identical in every wave ──
        const int tok = kt + half;
        float alpha[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = (tok < t_end && tok <= rpos[j]) ? s[j] : -INFINITY;
            if (tree && tok >= block_start && x != -INFINITY)
                x += tree_bias[(size_t)rrow[j] * block_cols + (tok - block_start)] * LOG2E;
            const float m_new = fmaxf(m_row[j], row16_max(x));
            const bool none = (m_new == -INFINITY);
            const float a_ = none ? 1.0f : exp2f(m_row[j] - m_new);
            const float p = none ? 0.0f : exp2f(x - m_new);
            l_row[j] = l_row[j] * a_ + row16_sum(p);
            m_row[j] = m_new;
            alpha[j] = a_;
            prow[(2 * j + half_id) * PS + half] = (_Float16)p;
        }
        // P is this wave's own scratch: a wave-level fence, not a barrier.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // ── O[:, this wave's dims] = alpha * O + P V ──
        half16_t pa;
#pragma unroll
        for (int j = 0; j < 16; j++) pa[j] = prow[half * PS + j];
        bool rescale = false;
#pragma unroll
        for (int j = 0; j < 8; j++) rescale |= (alpha[j] != 1.0f);
        if (rescale) {
#pragma unroll
            for (int c = 0; c < DCW; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) O[c][j] *= alpha[j];
        }
#pragma unroll
        for (int c = 0; c < DCW; c++) {
            typedef _Float16 __attribute__((ext_vector_type(4))) h4_t;
            const h4_t* vrow =
                (const h4_t*)(V_T + ((dc0 + c) * 16 + half) * VS + (dc0 + c) * VG);
            const h4_t v0 = vrow[0], v1 = vrow[1], v2 = vrow[2], v3 = vrow[3];
            const half16_t vb = {v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3],
                                 v2[0], v2[1], v2[2], v2[3], v3[0], v3[1], v3[2], v3[3]};
            O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[c]);
        }
        __syncthreads();   // K/V/S_x are restaged next iteration
    }

    // ── Partials: (max in natural-log units, sum, unnormalized O) per row ──
    const float LN2 = 0.6931471805599453f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int ml = 2 * j + half_id;
        if (ml >= rows_here) continue;
        const int m = m0 + ml;
        const int r = m / G;
        const int h = kv_h * G + (m - r * G);
        float* p = partials + (((size_t)r * n_heads + h) * n_splits + split) * (2 + HD);
        if (wave == 0 && half == 0) {
            p[0] = (m_row[j] == -INFINITY) ? -1e30f : m_row[j] * LN2;
            p[1] = l_row[j];
        }
#pragma unroll
        for (int c = 0; c < DCW; c++) p[2 + (dc0 + c) * 16 + half] = O[c][j];
    }
}

// One session's rows (decode, DFlash verify, DDTree with a tree bias).
// Grid: [n_kv_heads, n_splits, ceil(G*R / 16)].
extern "C" __launch_bounds__(128, 1) __global__ void attention_decode_kvarn_wmma(
    const float* __restrict__ q, const uint8_t* __restrict__ records,
    const void* __restrict__ window, const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials, const int* __restrict__ positions,
    const float* __restrict__ tree_bias, int R, int n_heads, int n_kv_heads, float scale_attn,
    int n_full_blocks, int rec_bytes, int window_f16, int block_start, int block_cols) {
    decode_kvarn_body(q, records, window, v_cache, partials, positions, tree_bias, R, n_heads,
                      n_kv_heads, scale_attn, n_full_blocks, rec_bytes, window_f16, block_start,
                      block_cols, blockIdx.z * MQ);
}

// Rows of independent sessions (multi-session decode): row b's caches come from
// the session-major pointer tables (as attention_kvarn_routed_batched), its
// records cover the full blocks below its own position. q / positions / the
// tables are indexed by the absolute row b = row_offset + blockIdx.z; partials by
// the launch-local row. Grid: [n_kv_heads, n_splits, rows].
extern "C" __launch_bounds__(128, 1) __global__ void attention_decode_kvarn_wmma_routed(
    const float* __restrict__ q, const uint64_t* __restrict__ rec_ptrs,
    const uint64_t* __restrict__ win_ptrs, const uint64_t* __restrict__ v_ptrs,
    float* __restrict__ partials, const int* __restrict__ row_session_indices,
    const int* __restrict__ positions, int ptr_layer_stride, int layer_index, int n_heads,
    int n_kv_heads, float scale_attn, int rec_bytes, int window_f16, int row_offset) {
    const int b = row_offset + (int)blockIdx.z;
    const size_t slot = (size_t)row_session_indices[b] * ptr_layer_stride + layer_index;
    const uint8_t* records = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(rec_ptrs[slot]));
    const void* window = reinterpret_cast<const void*>(static_cast<uintptr_t>(win_ptrs[slot]));
    const unsigned char* v_cache =
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(v_ptrs[slot]));
    decode_kvarn_body(q + (size_t)b * n_heads * HD, records, window, v_cache,
                      partials + (size_t)blockIdx.z * n_heads * gridDim.y * (2 + HD),
                      positions + b, nullptr, 1, n_heads, n_kv_heads, scale_attn,
                      (positions[b] + 1) / GROUP, rec_bytes, window_f16, 0, 0, 0);
}
