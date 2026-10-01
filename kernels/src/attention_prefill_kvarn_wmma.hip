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
    const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);  // log2(8/bits)
    const int codemask = (1 << bits) - 1;
    const int tpw = 32 / bits;                    // codes per dword
    const int wpd = NT / tpw;                     // dwords per dim per sub-tile (>= 1)
    const int off_scale = (HD * GROUP) >> cpb_shift;
    const int off_zp = off_scale + HD * 2;
    const int off_scol = off_zp + HD * 2;
    // Q8_0 V geometry.
    const int bph = HD / 32;
    const int v_row_stride = n_kv_heads * bph * 34;

    for (int kt = 0; kt < kv_len; kt += NT) {
        // ── Stage K (token-major f16) ──
        const int blk = kt / GROUP;               // NT divides GROUP: one block
        if (blk < n_full_blocks) {
            const uint8_t* rec = records + (size_t)(blk * n_kv_heads + kv_h) * rec_bytes;
            const __half* scale_abs = (const __half*)(rec + off_scale);
            const __half* zp_abs = (const __half*)(rec + off_zp);
            const __half* s_col = (const __half*)(rec + off_scol);
            const int t_in = kt - blk * GROUP;
            for (int i = tid; i < HD * wpd; i += 128) {
                const int d = i / wpd;
                const int w = i - d * wpd;
                const int t0 = t_in + w * tpw;    // token within the block
                const int idx = d * GROUP + t0;   // multiple of tpw: dword aligned
                const uint32_t word = *(const uint32_t*)(rec + (idx >> cpb_shift));
                const float sa = __half2float(scale_abs[d]);
                const float za = __half2float(zp_abs[d]);
                for (int tt = 0; tt < tpw; tt++) {
                    const float code = (float)((word >> (tt * bits)) & codemask);
                    const float val = (code * sa + za) * __half2float(s_col[t0 + tt]);
                    K_lds[(w * tpw + tt) * KS + d] = (_Float16)val;
                }
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
        // ── Stage V (Q8_0 -> f16, dim-major) ──
        for (int i = tid; i < NT * bph; i += 128) {
            const int t = i / bph;
            const int b = i - t * bph;
            const int tok = kt + t;
            if (tok < kv_len) {
                const unsigned char* base =
                    v_cache + (size_t)tok * v_row_stride + (size_t)(kv_h * bph + b) * 34;
                const float s = (float)*(const _Float16*)base;
                const signed char* vq = (const signed char*)(base + 2);
#pragma unroll
                for (int e = 0; e < 32; e++) V_T[(b * 32 + e) * VS + t] = (_Float16)(s * (float)vq[e]);
            } else {
#pragma unroll
                for (int e = 0; e < 32; e++) V_T[(b * 32 + e) * VS + t] = (_Float16)0.0f;
            }
        }
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
#pragma unroll
        for (int dc = 0; dc < HD / 16; dc++) {
#pragma unroll
            for (int j = 0; j < 8; j++) O[dc][j] *= alpha[j];
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
