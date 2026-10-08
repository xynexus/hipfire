// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Causal prefill attention over KVarN storage, on WMMA (RDNA3 wave32).
//
// The prefill path used to run the DECODE tile kernel once per query row
// (`attention_flash_kvarn_tile_batched`, grid [heads, kv_tiles, rows]): every row
// re-read and re-dequantized every K/V tile of its prefix on the VALU, then a
// reduce pass merged per-tile partials. Profiled on a cold 8.3K-token prompt
// (Qwen3.8-27B): 42% of the prefill, ~0.7 TFLOPS, and quadratic in context.
//
// Here one workgroup owns 64 query rows of one head (4 waves x 16 rows) and walks
// the KV once:
//   - Q (f32) is staged to LDS as f16, pre-multiplied by scale * log2(e) so the
//     softmax runs on exp2.
//   - K/V stream in 16-token sub-tiles. K comes from the 4-bit (2/8) variance-
//     normalized records, dequantized ONCE per sub-tile for all 64 rows —
//     deq = (code * scale_abs[d] + zp_abs[d]) * s_col[t] — or from the f32/f16
//     recent window for the trailing partial block; V is Q8_0, staged
//     transposed (dim-major) so the P.V B-operand is a contiguous LDS read.
//   - S = Q.K^T and O += P.V on v_wmma_f32_16x16x16_f16, online softmax per row
//     in registers (row stats live in the WMMA C layout: lane t holds rows
//     2j + (t>>4) for column t&15, so a row reduction is a 16-lane DPP row op).
//   - Causal mask per row from `positions`; the walk stops at the block's
//     largest position. O stays in registers and is written normalized — no
//     partials buffer, no reduce pass.
//
// Specialized for head_dim 256 (Qwen3.5/3.6/3.8 full-attention layers). The
// dispatcher routes here only for head_dim == 256, no tree bias, gfx11 WMMA.
//
// Record layout (== kvarn_quantize_tile), GROUP = 128 cols:
//   [ head_dim*GROUP*bits/8 ] codes, channel-major rows, LSB-first
//   [ head_dim*2 ] scale_abs fp16   [ head_dim*2 ] zp_abs fp16   [ GROUP*2 ] s_col fp16
//
// Grid: [n_heads, ceil(B / 64)]. Block: [128]. Dynamic LDS: see LDS_BYTES.

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
constexpr int MQ = 64;        // query rows per workgroup (4 waves x 16)
constexpr int NT = 16;        // KV tokens per sub-tile
constexpr int GROUP = 128;    // KVarN block / record columns
constexpr int QS = HD + 8;    // Q_lds row stride (halfs); +8 staggers banks
constexpr int KS = HD + 8;    // K_lds row stride (token-major)
constexpr int VS = NT + 8;    // V_T row stride (dim-major)
constexpr int PS = NT + 8;    // per-wave P scratch row stride
}  // namespace

// Reductions over the 16 lanes of a DPP row (lanes 0-15 / 16-31): exactly the
// lanes that share a WMMA C-layout row.
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

extern "C" __launch_bounds__(128, 1) __global__ void attention_prefill_kvarn_wmma(
    const float* __restrict__ q,               // [B x n_heads x HD]
    const uint8_t* __restrict__ records,       // KVarN K records, block-tiled
    const void* __restrict__ window,           // [GROUP x kv_dim] f32 or f16
    const unsigned char* __restrict__ v_cache, // Q8_0, [max_seq x n_kv_heads*(HD/32)*34]
    float* __restrict__ out,                   // [B x n_heads x HD]
    const int* __restrict__ positions,         // [B] absolute position of each row
    int B,
    int n_heads,
    int n_kv_heads,
    float scale_attn,
    int n_full_blocks,
    int rec_bytes,
    int bits,
    int window_f16
) {
    const int h = blockIdx.x;
    const int q_start = blockIdx.y * MQ;
    if (h >= n_heads || q_start >= B) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int q_dim = n_heads * HD;
    const int kv_dim = n_kv_heads * HD;

    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    _Float16* Q_lds = (_Float16*)smem;          // MQ x QS
    _Float16* K_lds = Q_lds + MQ * QS;          // NT x KS
    _Float16* V_T = K_lds + NT * KS;            // HD x VS
    _Float16* P_lds = V_T + HD * VS;            // 4 waves x 16 x PS
    int* s_maxpos = (int*)(P_lds + 4 * 16 * PS);

    if (tid == 0) *s_maxpos = -1;
    __syncthreads();

    // Stage Q (f16, pre-scaled for exp2) and find the block's last position.
    const float qscale = scale_attn * 1.4426950408889634f;
    for (int i = tid; i < MQ * HD; i += 128) {
        const int r = i / HD;
        const int d = i - r * HD;
        const int gq = q_start + r;
        const float v = (gq < B) ? q[(size_t)gq * q_dim + h * HD + d] * qscale : 0.0f;
        Q_lds[r * QS + d] = (_Float16)v;
    }
    if (tid < MQ && q_start + tid < B) atomicMax(s_maxpos, positions[q_start + tid]);
    __syncthreads();
    const int kv_len = *s_maxpos + 1;

    // This lane's 8 rows in the C layout: row 2j + half_id of the wave's 16.
    int rpos[8];
    float m_row[8], l_row[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int gq = q_start + wave * 16 + 2 * j + half_id;
        rpos[j] = (gq < B) ? positions[gq] : -1;
        m_row[j] = -INFINITY;
        l_row[j] = 0.0f;
    }
    float8_t O[HD / 16];
#pragma unroll
    for (int dc = 0; dc < HD / 16; dc++) O[dc] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};

    // KVarN record geometry.
    // K precision is a compile-time constant (KV_BITS; the dispatcher builds one
    // variant per width) so the prefetch registers below are sized exactly: sized
    // for the widest case at runtime they took the kernel to 256 VGPRs and 69
    // spilled, slower than no prefetch at all.
    (void)bits;
    constexpr int cpb_shift = (KV_BITS == 8) ? 0 : ((KV_BITS == 4) ? 1 : 2);  // log2(8/bits)
    constexpr int codemask = (1 << KV_BITS) - 1;
    constexpr int tpw = 32 / KV_BITS;             // codes per dword
    constexpr int wpd = NT / tpw;                 // dwords per dim per sub-tile (>= 1)
    const int off_scale = (HD * GROUP) >> cpb_shift;
    const int off_zp = off_scale + HD * 2;
    const int off_scol = off_zp + HD * 2;
    // Q8_0 V geometry.
    const int bph = HD / 32;
    const int v_row_stride = n_kv_heads * bph * 34;

    // ── Software pipeline: sub-tile kt+NT's raw K/V is loaded into registers
    // while sub-tile kt computes, so the global latency -- measured at about
    // half the staging time when it sat in front of every sub-tile -- overlaps
    // the WMMA work instead. A thread's K items are always the same channels d
    // and the same token offset w (128 is a multiple of wpd), so a sub-tile's
    // raw K is KMAX code words, their channels' scale/zero-point and the tpw
    // per-token scales; its V item is one 34-byte Q8_0 block. The window tail
    // (the last partial block) is loaded where it is used, as before.
    // K: a thread owns DPT CONSECUTIVE channels d0.. at token offset w, so each
    // token's DPT values leave in one vector LDS store (8 B at 4 bits) instead of
    // DPT scattered 2-byte ones. V: a thread owns two tokens x 16 dims, so each
    // dim's pair leaves as one 4-byte store into the dim-major V_T.
    constexpr int DPT = HD * wpd / 128;           // K channels per thread: 4 / 2 / 8
    typedef _Float16 __attribute__((ext_vector_type(DPT))) hvec_t;
    const int k_w = tid % wpd;
    const int k_d0 = (tid / wpd) * DPT;
    // Packed: the halves ride two to a register (vector loads), not one each --
    // unpacked they held the kernel at 256 VGPRs with spills.
    typedef _Float16 __attribute__((ext_vector_type(tpw))) scvec_t;
    uint32_t pk_word[DPT];
    hvec_t pk_sa, pk_za;
    scvec_t pk_sc;
    const int v_hh = tid & 1;                     // which 16 dims of the block
    const int v_b = (tid >> 1) & (bph - 1);       // 32-dim Q8_0 block
    const int v_tp = tid >> 4;                    // token pair (2*v_tp, 2*v_tp+1)
    uint32_t pv[2][4];                            // per token: 16 int8, 4 to a register
    uint16_t pv_s[2];                             // per token: the block's f16 scale
    bool pv_live[2] = {false, false};
    auto prefetch = [&](int kt_n) {
        const int blk_n = kt_n / GROUP;
        if (kt_n < kv_len && blk_n < n_full_blocks) {
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
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int tok = kt_n + 2 * v_tp + u;
            pv_live[u] = tok < kv_len;
            if (pv_live[u]) {
                const uint16_t* base = (const uint16_t*)(
                    v_cache + (size_t)tok * v_row_stride + (size_t)(kv_h * bph + v_b) * 34);
                pv_s[u] = base[0];
                // 2-byte aligned (34-byte blocks), so 16-bit loads, paired up.
#pragma unroll
                for (int e = 0; e < 4; e++)
                    pv[u][e] = (uint32_t)base[1 + v_hh * 8 + 2 * e]
                             | ((uint32_t)base[2 + v_hh * 8 + 2 * e] << 16);
            }
        }
    };
    prefetch(0);

    for (int kt = 0; kt < kv_len; kt += NT) {
        // ── Stage K (token-major f16) ──
        const int blk = kt / GROUP;               // NT divides GROUP: one block
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
            const int win_base = n_full_blocks * GROUP;
            for (int i = tid; i < NT * HD; i += 128) {
                const int t = i / HD;
                const int d = i - t * HD;
                const int tok = kt + t;
                float v = 0.0f;
                if (tok < kv_len) {
                    const size_t off = (size_t)(tok - win_base) * kv_dim + kv_h * HD + d;
                    v = window_f16 ? (float)((const _Float16*)window)[off]
                                   : ((const float*)window)[off];
                }
                K_lds[t * KS + d] = (_Float16)v;
            }
        }
        // ── Stage V (Q8_0 -> f16, dim-major), from the prefetched block ──
        {
            const float s0 = pv_live[0] ? (float)__builtin_bit_cast(_Float16, pv_s[0]) : 0.0f;
            const float s1 = pv_live[1] ? (float)__builtin_bit_cast(_Float16, pv_s[1]) : 0.0f;
            _Float16* vt = V_T + (v_b * 32 + v_hh * 16) * VS + 2 * v_tp;
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const signed char q0 = (signed char)(pv[0][e >> 2] >> (8 * (e & 3)));
                const signed char q1 = (signed char)(pv[1][e >> 2] >> (8 * (e & 3)));
                typedef _Float16 __attribute__((ext_vector_type(2))) h2_t;
                // A dead token's scale is 0, so its column is written as 0 (q is
                // whatever the register held; 0 * finite stays 0 -- and the
                // buffer is never loaded for it, so q is a stale finite byte).
                *(h2_t*)(vt + e * VS) = (h2_t){(_Float16)(s0 * (float)q0), (_Float16)(s1 * (float)q1)};
            }
        }
        // The next sub-tile's loads go out now; they land during this one's math.
        prefetch(kt + NT);
        __syncthreads();

        // ── S = Q K^T for this wave's 16 rows x NT tokens ──
        float8_t s = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
        const _Float16* qrow = Q_lds + (wave * 16 + half) * QS;
        const _Float16* krow = K_lds + half * KS;
#pragma unroll
        for (int dc = 0; dc < HD / 16; dc++) {
            half16_t a, b;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                a[j] = qrow[dc * 16 + j];
                b[j] = krow[dc * 16 + j];
            }
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
        }

        // ── Causal mask + online softmax (log2 domain), P to scratch ──
        const int tok = kt + half;
        float alpha[8];
        _Float16* prow = P_lds + wave * 16 * PS;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float x = (tok <= rpos[j]) ? s[j] : -INFINITY;
            const float m_new = fmaxf(m_row[j], row16_max(x));
            // A row with nothing visible yet keeps m = -inf; guard -inf - -inf.
            const bool none = (m_new == -INFINITY);
            const float a_ = none ? 1.0f : exp2f(m_row[j] - m_new);
            const float p = none ? 0.0f : exp2f(x - m_new);
            l_row[j] = l_row[j] * a_ + row16_sum(p);
            m_row[j] = m_new;
            alpha[j] = a_;
            prow[(2 * j + half_id) * PS + half] = (_Float16)p;
        }
        __syncthreads();

        // ── O = alpha * O + P V ──
        half16_t pa;
#pragma unroll
        for (int j = 0; j < 16; j++) pa[j] = prow[half * PS + j];
        // alpha is exactly 1 for a row whose running max did not move -- the
        // common case once it settles -- and then O needs no rescale.
        bool rescale = false;
#pragma unroll
        for (int j = 0; j < 8; j++) rescale |= (alpha[j] != 1.0f);
        if (rescale) {
#pragma unroll
            for (int dc = 0; dc < HD / 16; dc++)
#pragma unroll
                for (int j = 0; j < 8; j++) O[dc][j] *= alpha[j];
        }
#pragma unroll
        for (int dc = 0; dc < HD / 16; dc++) {
            half16_t vb;
            const _Float16* vrow = V_T + (dc * 16 + half) * VS;
#pragma unroll
            for (int j = 0; j < 16; j++) vb[j] = vrow[j];
            O[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[dc]);
        }
        __syncthreads();   // K/V/P are restaged next iteration
    }

    // ── Normalize and write ──
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int gq = q_start + wave * 16 + 2 * j + half_id;
        if (gq >= B) continue;
        const float inv_l = (l_row[j] > 0.0f) ? 1.0f / l_row[j] : 0.0f;
        float* orow = out + (size_t)gq * q_dim + h * HD;
#pragma unroll
        for (int dc = 0; dc < HD / 16; dc++) orow[dc * 16 + half] = O[dc][j] * inv_l;
    }
}
