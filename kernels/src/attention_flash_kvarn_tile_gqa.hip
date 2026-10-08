// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// KVarN flash-decode tile, one workgroup per (KV HEAD, tile, row) serving all G
// query heads of the group.
//
// `attention_flash_kvarn_tile_batched` runs one workgroup per (QUERY head, tile,
// row), so every KV head's record tile and Q8 V rows are dequantized and read G
// times (6 on Qwen3.8-27B, 8 on Qwen3.6-35B-A3B). At depth that is the decode cost:
// a 27B token at 13.3K context spent ~0.64 ms per full-attention layer in that
// kernel + its reduce against ~0.09 ms of actual K/V traffic. Here each K code
// dword and each V row is loaded once and applied to all G heads; the per-head
// math, the partials layout and the reduce (attention_flash_asym_reduce_batched)
// are unchanged.
//
// Q.K keeps the tile kernel's dequant fold, per head g:
//   partial_g(t) = s_col[t] * ( sum_i code[i,t] * (q_g[i]*sa[i]) + sum_i q_g[i]*za[i] )
// so the inner loop is one code extract and G FMAs per element.
//
// head_dim 256 only (8 dims x 32 lanes), K code bits 4 or 8 (TPW = 32/bits <= 8
// tokens per dword); anything else stays on the tile kernel.
//
// Grid: [n_kv_heads, max_tiles, sub_batch]. Block: [32 * KVARN_GQA_NW].
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef KVARN_GQA_NW
#define KVARN_GQA_NW 4
#endif

__device__ __forceinline__ float gqa_sum_dpp(float v) {
#define OQ_XM(M) { int s_ = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 | (M), 0xf, 0xf, true); \
                   v += __builtin_bit_cast(float, s_); }
    OQ_XM(1) OQ_XM(2) OQ_XM(4) OQ_XM(8)
#undef OQ_XM
    int s_ = __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v), 0u, 0u, false, false);
    return v + __builtin_bit_cast(float, s_);
}
__device__ __forceinline__ float gqa_max_dpp(float v) {
#define OQ_XM(M) { int s_ = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 | (M), 0xf, 0xf, true); \
                   v = fmaxf(v, __builtin_bit_cast(float, s_)); }
    OQ_XM(1) OQ_XM(2) OQ_XM(4) OQ_XM(8)
#undef OQ_XM
    int s_ = __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v), 0u, 0u, false, false);
    return fmaxf(v, __builtin_bit_cast(float, s_));
}

template <int G, int NW>
__device__ __forceinline__ void kvarn_tile_gqa_body(
    const float* __restrict__ q,               // [sub_batch x n_heads x 256]
    const uint8_t* __restrict__ records,
    const float* __restrict__ window,
    const unsigned char* __restrict__ v_cache, // Q8_0
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, float scale_attn, int tile_size, int max_tiles,
    int batch_offset, int n_full_blocks, int rec_bytes, int bits, int window_f16) {
    constexpr int HD = 256, DPT = 8;
    const int kh = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int wv = threadIdx.x >> 5;
    const int global_bid = batch_offset + local_bid;
    const int seq_len = positions[global_bid] + 1;
    const int group = tile_size;
    const int tile_start = tile_id * group;
    if (tile_start >= seq_len) return;
    const int tile_len = min(tile_start + group, seq_len) - tile_start;
    const int h0 = kh * G;
    const int q_dim = n_heads * HD;
    const int kv_dim = n_kv_heads * HD;
    const int d0 = lane * DPT;
    const int cpb_shift = (bits == 8) ? 0 : 1;          // 4 or 8 bits only
    const int codemask = (1 << bits) - 1;

    // LDS: [G][group] scores | [G][NW] wave slots | [NW][HD] V accumulators
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* wred = scores + G * group;
    float* dacc = wred + G * NW;

    const float* qrow = q + (size_t)local_bid * q_dim + (size_t)h0 * HD;

    // ── Phase A: Q.K for all G heads ──
    if (tile_id < n_full_blocks) {
        const uint8_t* rec = records + (size_t)(tile_id * n_kv_heads + kh) * rec_bytes;
        const int cpb = 8 / bits;
        const int off_scale = (HD * group + cpb - 1) / cpb;
        const int off_zp = off_scale + HD * 2;
        const int off_scol = off_zp + HD * 2;
        const __half* scale_abs = (const __half*)(rec + off_scale);
        const __half* zp_abs = (const __half*)(rec + off_zp);
        const __half* s_col = (const __half*)(rec + off_scol);
        float qa[G][DPT];
        float c_local[G];
        #pragma unroll
        for (int g = 0; g < G; g++) {
            c_local[g] = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPT; i++) {
                const float mq = qrow[g * HD + d0 + i];
                qa[g][i] = mq * __half2float(scale_abs[d0 + i]);
                c_local[g] += mq * __half2float(zp_abs[d0 + i]);
            }
        }
        const int TPW = 32 / bits;                     // 8 at bits=4, 4 at bits=8
        for (int t_base = wv * TPW; t_base < tile_len; t_base += NW * TPW) {
            const int nt = min(TPW, tile_len - t_base);
            float part[G][8];
            #pragma unroll
            for (int g = 0; g < G; g++)
                #pragma unroll
                for (int tt = 0; tt < 8; tt++) part[g][tt] = c_local[g];
            #pragma unroll
            for (int i = 0; i < DPT; i++) {
                const int idx = (d0 + i) * group + t_base;
                const uint32_t w4 = *(const uint32_t*)(rec + (idx >> cpb_shift));
                #pragma unroll
                for (int tt = 0; tt < 8; tt++) {
                    if (tt < TPW) {
                        const float code = (float)((w4 >> (tt * bits)) & codemask);
                        #pragma unroll
                        for (int g = 0; g < G; g++) part[g][tt] += code * qa[g][i];
                    }
                }
            }
            for (int tt = 0; tt < nt; tt++) {
                const float sc = __half2float(s_col[t_base + tt]) * scale_attn;
                #pragma unroll
                for (int g = 0; g < G; g++) {
                    const float s = gqa_sum_dpp(part[g][tt]) * sc;
                    if (lane == 0) scores[g * group + t_base + tt] = s;
                }
            }
        }
    } else {
        // Trailing partial block from the recent window (f32 or f16).
        float mq[G][DPT];
        #pragma unroll
        for (int g = 0; g < G; g++)
            #pragma unroll
            for (int i = 0; i < DPT; i++) mq[g][i] = qrow[g * HD + d0 + i];
        const int win_base = n_full_blocks * group;
        for (int t_local = wv; t_local < tile_len; t_local += NW) {
            const int t = tile_start + t_local;
            float k[DPT];
            if (window_f16) {
                const _Float16* kt = reinterpret_cast<const _Float16*>(window)
                    + (size_t)(t - win_base) * kv_dim + kh * HD + d0;
                #pragma unroll
                for (int i = 0; i < DPT; i++) k[i] = (float)kt[i];
            } else {
                const float* kt = window + (size_t)(t - win_base) * kv_dim + kh * HD + d0;
                #pragma unroll
                for (int i = 0; i < DPT; i++) k[i] = kt[i];
            }
            #pragma unroll
            for (int g = 0; g < G; g++) {
                float p = 0.0f;
                #pragma unroll
                for (int i = 0; i < DPT; i++) p += mq[g][i] * k[i];
                const float s = gqa_sum_dpp(p) * scale_attn;
                if (lane == 0) scores[g * group + t_local] = s;
            }
        }
    }
    __syncthreads();

    // ── Phase B/C: per-head tile max and exp-sum (one token per thread) ──
    const int nthr = 32 * NW;
    float tmax[G], tsum[G];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float m = -1e30f;
        for (int i = threadIdx.x; i < tile_len; i += nthr) m = fmaxf(m, scores[g * group + i]);
        m = gqa_max_dpp(m);
        if (lane == 0) wred[g * NW + wv] = m;
    }
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float m = wred[g * NW];
        #pragma unroll
        for (int w = 1; w < NW; w++) m = fmaxf(m, wred[g * NW + w]);
        tmax[g] = m;
    }
    __syncthreads();                                   // wred reused below
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < tile_len; i += nthr) {
            const float e = expf(scores[g * group + i] - tmax[g]);
            scores[g * group + i] = e;
            s += e;
        }
        s = gqa_sum_dpp(s);
        if (lane == 0) wred[g * NW + wv] = s;
    }
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; w++) s += wred[g * NW + w];
        tsum[g] = s;
    }

    // ── Phase D: V (Q8_0), each row loaded once for all G heads ──
    const int blocks_per_head = HD / 32;
    const int v_row_stride = n_kv_heads * blocks_per_head * 34;
    const size_t vh_off = (size_t)(kh * blocks_per_head + d0 / 32) * 34;
    const int bj0 = d0 % 32;
    float acc[G][DPT];
    #pragma unroll
    for (int g = 0; g < G; g++)
        #pragma unroll
        for (int i = 0; i < DPT; i++) acc[g][i] = 0.0f;
    for (int t_local = wv; t_local < tile_len; t_local += NW) {
        const unsigned char* vbh = v_cache + (size_t)(tile_start + t_local) * v_row_stride + vh_off;
        const float vs = (float)*((const _Float16*)vbh);
        const signed char* vq = (const signed char*)(vbh + 2 + bj0);
        float v[DPT];
        #pragma unroll
        for (int i = 0; i < DPT; i++) v[i] = vs * (float)vq[i];
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const float w = scores[g * group + t_local];
            #pragma unroll
            for (int i = 0; i < DPT; i++) acc[g][i] += w * v[i];
        }
    }

    // Cross-wave V sum, one head at a time through a [NW][HD] buffer.
    #pragma unroll
    for (int g = 0; g < G; g++) {
        #pragma unroll
        for (int i = 0; i < DPT; i++) dacc[wv * HD + d0 + i] = acc[g][i];
        __syncthreads();
        float* p = partials + ((size_t)local_bid * n_heads + h0 + g) * max_tiles * (2 + HD)
                            + (size_t)tile_id * (2 + HD);
        if (threadIdx.x == 0) {
            p[0] = tmax[g];
            p[1] = tsum[g];
        }
        if (wv == 0) {
            #pragma unroll
            for (int i = 0; i < DPT; i++) {
                float a = dacc[d0 + i];
                #pragma unroll
                for (int w = 1; w < NW; w++) a += dacc[w * HD + d0 + i];
                p[2 + d0 + i] = a;
            }
        }
        __syncthreads();
    }
}

#define KVARN_TILE_GQA(G)                                                                       \
    extern "C" __launch_bounds__(32 * KVARN_GQA_NW, 1) __global__ void                          \
    attention_flash_kvarn_tile_gqa##G(                                                          \
        const float* __restrict__ q, const uint8_t* __restrict__ records,                       \
        const float* __restrict__ window, const unsigned char* __restrict__ v_cache,            \
        float* __restrict__ partials, const int* __restrict__ positions, int n_heads,           \
        int n_kv_heads, float scale_attn, int tile_size, int max_tiles, int batch_offset,       \
        int n_full_blocks, int rec_bytes, int bits, int window_f16) {                           \
        kvarn_tile_gqa_body<G, KVARN_GQA_NW>(q, records, window, v_cache, partials, positions,  \
            n_heads, n_kv_heads, scale_attn, tile_size, max_tiles, batch_offset, n_full_blocks, \
            rec_bytes, bits, window_f16);                                                       \
    }
KVARN_TILE_GQA(2)
KVARN_TILE_GQA(3)
KVARN_TILE_GQA(4)
KVARN_TILE_GQA(5)
KVARN_TILE_GQA(6)
KVARN_TILE_GQA(7)
KVARN_TILE_GQA(8)
