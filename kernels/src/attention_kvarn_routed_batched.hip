// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Routed batched causal attention with a KVarN K cache + Q8_0 V (microbatching).
// Each query row selects its independent request-session caches from session-major
// pointer tables (mirrors attention_q8_0_routed_batched), with three tables:
//   records = rec_ptrs[row_session * ptr_layer_stride + layer_index]  (4-bit K blocks)
//   window  = win_ptrs[row_session * ptr_layer_stride + layer_index]  (f32 recent block)
//   v_cache = v_ptrs  [row_session * ptr_layer_stride + layer_index]  (Q8_0 V)
//
// K is dequantized in place (no f16 shadow): the 128-token KVarN blocks align with
// the causal sequence, so token t<n_full*GROUP comes from record block t/GROUP and
// the trailing partial block (t>=n_full*GROUP) comes from the f32 window. Each row's
// n_full_blocks is derived from its own positions[b]. The context is walked in
// `chunk`-position pieces with an online softmax, so LDS is chunk-sized and any
// context fits.
//
// Record layout (== kvarn_quantize_tile / pack_kvarn_tile), channel-major rows:
//   [ ceil(head_dim*GROUP/cpb) ] q codes | [head_dim*2] scale_abs f16 | [head_dim*2] zp_abs f16 | [GROUP*2] s_col f16
// with cpb = 8/bits codes per byte (bits in {2,4,8}); must match
// KvCache::kvarn_k_record_bytes_bits, which is what the flush writes.
// Dequant K[d,t] = (q*scale_abs[d] + zp_abs[d]) * s_col[t-in-block].
//
// attention_kvarn_routed_batched_gqa<G>: grid [n_kv_heads, rows, 1], one workgroup
// per KV head serving its G query heads. attention_kvarn_routed_batched_chunked:
// grid [n_heads, rows, 1], the fallback for G > 8. `row_offset` gives the absolute
// row of blockIdx.y == 0.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ── Chunked (streaming-softmax), one workgroup per query head ──────────────
// Running max m and sum l per head: each chunk's scores go to a chunk-sized LDS
// buffer, and the accumulated output is rescaled by exp(m_old - m_new) before the
// chunk's V is added. Fallback for query groups the GQA kernel below can't take.
//
// Each thread owns output dims d = tid, tid + nthreads, ...; KVARN_CHUNK_MAX_D
// bounds how many (head_dim 512 over 256 threads is 2).
#define KVARN_CHUNK_MAX_D 4

__device__ __forceinline__ float kvarn_routed_qk(
    const float* q_shared, const uint8_t* records, const float* window, int window_f16,
    int t, int win_base, int group, int n_kv_heads, int kv_h, int kv_dim, int head_dim,
    int rec_bytes, int bits, int cpb_shift, int cpb_mask, int codemask,
    int off_scale, int off_zp, int off_scol
) {
    float dot = 0.0f;
    if (t < win_base) {
        const int block = t / group;
        const int t_local = t - block * group;
        const uint8_t* rec = records + (size_t)(block * n_kv_heads + kv_h) * rec_bytes;
        const __half* scale_abs = (const __half*)(rec + off_scale);
        const __half* zp_abs = (const __half*)(rec + off_zp);
        const __half* s_col = (const __half*)(rec + off_scol);
        const float sc = __half2float(s_col[t_local]);
        for (int d = 0; d < head_dim; d++) {
            const int idx = d * group + t_local;
            const uint8_t byte = rec[idx >> cpb_shift];
            const float qv = (float)((byte >> (bits * (idx & cpb_mask))) & codemask);
            const float k = (qv * __half2float(scale_abs[d]) + __half2float(zp_abs[d])) * sc;
            dot += q_shared[d] * k;
        }
    } else if (window_f16) {
        const _Float16* kt = reinterpret_cast<const _Float16*>(window)
            + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) dot += q_shared[d] * (float)kt[d];
    } else {
        const float* kt = window + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) dot += q_shared[d] * kt[d];
    }
    return dot;
}

extern "C" __global__ void attention_kvarn_routed_batched_chunked(
    const float* __restrict__ q,
    const uint64_t* __restrict__ rec_ptrs,
    const uint64_t* __restrict__ win_ptrs,
    const uint64_t* __restrict__ v_ptrs,
    float* __restrict__ out,
    const int* __restrict__ row_session_indices,
    const int* __restrict__ positions,
    int ptr_layer_stride,
    int layer_index,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int rec_bytes,
    int group,
    int window_f16,
    int bits,
    int row_offset,
    int chunk
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y + row_offset;
    if (h >= n_heads) return;

    const int session_idx = row_session_indices[b];
    const size_t slot = (size_t)session_idx * ptr_layer_stride + layer_index;
    const uint8_t* records = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(rec_ptrs[slot]));
    const float* window = reinterpret_cast<const float*>(static_cast<uintptr_t>(win_ptrs[slot]));
    const unsigned char* v_cache = reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(v_ptrs[slot]));

    const int seq_len = positions[b] + 1;
    const int win_base = (seq_len / group) * group;
    extern __shared__ float sdata[];

    const int kv_h = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int q_stride = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float* q_head = q + (long long)b * q_stride + h * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int cpb = 8 / bits;
    const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);
    const int cpb_mask = cpb - 1;
    const int codemask = (1 << bits) - 1;
    const int off_scale = (head_dim * group + cpb - 1) / cpb;
    const int off_zp = off_scale + head_dim * 2;
    const int off_scol = off_zp + head_dim * 2;

    float* scores = sdata;                    // [chunk]
    float* workspace = sdata + chunk;         // [nthreads]
    float* q_shared = workspace + nthreads;   // [head_dim]

    for (int d = tid; d < head_dim; d += nthreads) q_shared[d] = q_head[d];
    __syncthreads();

    float m = -INFINITY;   // running max over every chunk so far
    float l = 0.0f;        // running sum of exp(s - m)
    float acc[KVARN_CHUNK_MAX_D];
    for (int k = 0; k < KVARN_CHUNK_MAX_D; k++) acc[k] = 0.0f;

    for (int c0 = 0; c0 < seq_len; c0 += chunk) {
        const int n = min(chunk, seq_len - c0);

        float local_max = -INFINITY;
        for (int i = tid; i < n; i += nthreads) {
            const float s = kvarn_routed_qk(q_shared, records, window, window_f16, c0 + i,
                win_base, group, n_kv_heads, kv_h, kv_dim, head_dim, rec_bytes, bits,
                cpb_shift, cpb_mask, codemask, off_scale, off_zp, off_scol) * scale_attn;
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
        workspace[tid] = local_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
            __syncthreads();
        }
        const float m_new = fmaxf(m, workspace[0]);
        const float alpha = expf(m - m_new);   // 0 on the first chunk (m = -inf)
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < n; i += nthreads) {
            const float e = expf(scores[i] - m_new);
            scores[i] = e;
            local_sum += e;
        }
        workspace[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] += workspace[tid + s];
            __syncthreads();
        }
        l = l * alpha + workspace[0];
        m = m_new;

        int k = 0;
        for (int d = tid; d < head_dim && k < KVARN_CHUNK_MAX_D; d += nthreads, k++) {
            const int bi = d / 32;
            const int bj = d % 32;
            float val = 0.0f;
            for (int i = 0; i < n; i++) {
                const unsigned char* blk = v_cache + (size_t)(c0 + i) * total_blocks_per_pos * 34
                                          + (kv_head_block_start + bi) * 34;
                const float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
                val += scores[i] * vd;
            }
            acc[k] = acc[k] * alpha + val;
        }
        __syncthreads();   // scores/workspace are reused by the next chunk
    }

    float* out_head = out + (long long)b * q_stride + h * head_dim;
    int k = 0;
    for (int d = tid; d < head_dim && k < KVARN_CHUNK_MAX_D; d += nthreads, k++) {
        out_head[d] = acc[k] / l;
    }
}

// ── GQA-grouped chunked variant (the one the dispatch runs) ────────────────
// Per (row, query head), the G query heads sharing a KV head each stream that
// head's whole K records and V rows — 6x the traffic on the 27B (24 q / 4 kv);
// with many sessions the working set leaves the MALL and that traffic is the
// bound. This launches per (row, KV head) instead: each K code is dequantized
// once and dotted with all G query heads, each V value loaded once and added into
// all G outputs. Same online softmax per head as the kernel above — the outputs
// are bit-identical to it.
#define KVARN_GQA_MAX_D 2   // output dims per thread (head_dim <= 2*blockDim)

template <int G>
__device__ __forceinline__ void kvarn_routed_gqa_impl(
    const float* __restrict__ q,
    const uint64_t* __restrict__ rec_ptrs,
    const uint64_t* __restrict__ win_ptrs,
    const uint64_t* __restrict__ v_ptrs,
    float* __restrict__ out,
    const int* __restrict__ row_session_indices,
    const int* __restrict__ positions,
    int ptr_layer_stride,
    int layer_index,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int rec_bytes,
    int group,
    int window_f16,
    int bits,
    int row_offset,
    int chunk
) {
    const int kv_h = blockIdx.x;
    const int b = blockIdx.y + row_offset;
    if (kv_h >= n_kv_heads) return;

    const int session_idx = row_session_indices[b];
    const size_t slot = (size_t)session_idx * ptr_layer_stride + layer_index;
    const uint8_t* records = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(rec_ptrs[slot]));
    const float* window = reinterpret_cast<const float*>(static_cast<uintptr_t>(win_ptrs[slot]));
    const unsigned char* v_cache = reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(v_ptrs[slot]));

    const int seq_len = positions[b] + 1;
    const int win_base = (seq_len / group) * group;
    extern __shared__ float sdata[];

    const int h0 = kv_h * G;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int q_stride = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int cpb = 8 / bits;
    const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);
    const int cpb_mask = cpb - 1;
    const int codemask = (1 << bits) - 1;
    const int off_scale = (head_dim * group + cpb - 1) / cpb;
    const int off_zp = off_scale + head_dim * 2;
    const int off_scol = off_zp + head_dim * 2;

    float* scores = sdata;                          // [G][chunk]
    float* workspace = scores + G * chunk;          // [G][nthreads]
    float* q_shared = workspace + G * nthreads;     // [G][head_dim]

    for (int i = tid; i < G * head_dim; i += nthreads) {
        const int g = i / head_dim;
        q_shared[i] = q[(long long)b * q_stride + (h0 + g) * head_dim + (i - g * head_dim)];
    }
    __syncthreads();

    float m[G], l[G];
    float acc[G][KVARN_GQA_MAX_D];
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (int k = 0; k < KVARN_GQA_MAX_D; k++) acc[g][k] = 0.0f;
    }

    for (int c0 = 0; c0 < seq_len; c0 += chunk) {
        const int n = min(chunk, seq_len - c0);

        float local_max[G];
        for (int g = 0; g < G; g++) local_max[g] = -INFINITY;
        for (int i = tid; i < n; i += nthreads) {
            const int t = c0 + i;
            float dot[G];
            for (int g = 0; g < G; g++) dot[g] = 0.0f;
            if (t < win_base) {
                const int block = t / group;
                const int t_local = t - block * group;
                const uint8_t* rec = records + (size_t)(block * n_kv_heads + kv_h) * rec_bytes;
                const __half* scale_abs = (const __half*)(rec + off_scale);
                const __half* zp_abs = (const __half*)(rec + off_zp);
                const __half* s_col = (const __half*)(rec + off_scol);
                const float sc = __half2float(s_col[t_local]);
                for (int d = 0; d < head_dim; d++) {
                    const int idx = d * group + t_local;
                    const uint8_t byte = rec[idx >> cpb_shift];
                    const float qv = (float)((byte >> (bits * (idx & cpb_mask))) & codemask);
                    const float k = (qv * __half2float(scale_abs[d]) + __half2float(zp_abs[d])) * sc;
                    for (int g = 0; g < G; g++) dot[g] += q_shared[g * head_dim + d] * k;
                }
            } else if (window_f16) {
                const _Float16* kt = reinterpret_cast<const _Float16*>(window)
                    + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    const float k = (float)kt[d];
                    for (int g = 0; g < G; g++) dot[g] += q_shared[g * head_dim + d] * k;
                }
            } else {
                const float* kt = window + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    const float k = kt[d];
                    for (int g = 0; g < G; g++) dot[g] += q_shared[g * head_dim + d] * k;
                }
            }
            for (int g = 0; g < G; g++) {
                const float s = dot[g] * scale_attn;
                scores[g * chunk + i] = s;
                local_max[g] = fmaxf(local_max[g], s);
            }
        }
        for (int g = 0; g < G; g++) workspace[g * nthreads + tid] = local_max[g];
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (int g = 0; g < G; g++)
                    workspace[g * nthreads + tid] =
                        fmaxf(workspace[g * nthreads + tid], workspace[g * nthreads + tid + s]);
            }
            __syncthreads();
        }
        float m_new[G], alpha[G];
        for (int g = 0; g < G; g++) {
            m_new[g] = fmaxf(m[g], workspace[g * nthreads]);
            alpha[g] = expf(m[g] - m_new[g]);
        }
        __syncthreads();

        float local_sum[G];
        for (int g = 0; g < G; g++) local_sum[g] = 0.0f;
        for (int i = tid; i < n; i += nthreads) {
            for (int g = 0; g < G; g++) {
                const float e = expf(scores[g * chunk + i] - m_new[g]);
                scores[g * chunk + i] = e;
                local_sum[g] += e;
            }
        }
        for (int g = 0; g < G; g++) workspace[g * nthreads + tid] = local_sum[g];
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (int g = 0; g < G; g++)
                    workspace[g * nthreads + tid] += workspace[g * nthreads + tid + s];
            }
            __syncthreads();
        }
        for (int g = 0; g < G; g++) {
            l[g] = l[g] * alpha[g] + workspace[g * nthreads];
            m[g] = m_new[g];
        }

        int k = 0;
        for (int d = tid; d < head_dim && k < KVARN_GQA_MAX_D; d += nthreads, k++) {
            const int bi = d / 32;
            const int bj = d % 32;
            float val[G];
            for (int g = 0; g < G; g++) val[g] = 0.0f;
            for (int i = 0; i < n; i++) {
                const unsigned char* blk = v_cache + (size_t)(c0 + i) * total_blocks_per_pos * 34
                                          + (kv_head_block_start + bi) * 34;
                const float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
                for (int g = 0; g < G; g++) val[g] += scores[g * chunk + i] * vd;
            }
            for (int g = 0; g < G; g++) acc[g][k] = acc[g][k] * alpha[g] + val[g];
        }
        __syncthreads();
    }

    int k = 0;
    for (int d = tid; d < head_dim && k < KVARN_GQA_MAX_D; d += nthreads, k++) {
        for (int g = 0; g < G; g++) {
            out[(long long)b * q_stride + (h0 + g) * head_dim + d] = acc[g][k] / l[g];
        }
    }
}

// G (query heads per KV head) is a template parameter so the per-head arrays
// stay in registers; the dispatch picks the instantiation by n_heads/n_kv_heads.
#define KVARN_GQA_ARGS \
    const float* __restrict__ q, const uint64_t* __restrict__ rec_ptrs, \
    const uint64_t* __restrict__ win_ptrs, const uint64_t* __restrict__ v_ptrs, \
    float* __restrict__ out, const int* __restrict__ row_session_indices, \
    const int* __restrict__ positions, int ptr_layer_stride, int layer_index, int n_heads, \
    int n_kv_heads, int head_dim, int max_seq, float scale_attn, int rec_bytes, int group, \
    int window_f16, int bits, int row_offset, int chunk
#define KVARN_GQA_PASS \
    q, rec_ptrs, win_ptrs, v_ptrs, out, row_session_indices, positions, ptr_layer_stride, \
    layer_index, n_heads, n_kv_heads, head_dim, max_seq, scale_attn, rec_bytes, group, \
    window_f16, bits, row_offset, chunk
#define KVARN_GQA_KERNEL(NG) \
    extern "C" __global__ void attention_kvarn_routed_batched_gqa##NG(KVARN_GQA_ARGS) { \
        kvarn_routed_gqa_impl<NG>(KVARN_GQA_PASS); \
    }
KVARN_GQA_KERNEL(1)
KVARN_GQA_KERNEL(2)
KVARN_GQA_KERNEL(3)
KVARN_GQA_KERNEL(4)
KVARN_GQA_KERNEL(5)
KVARN_GQA_KERNEL(6)
KVARN_GQA_KERNEL(7)
KVARN_GQA_KERNEL(8)
