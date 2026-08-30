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
// n_full_blocks is derived from its own positions[b]. Scalar per-token flash with
// scores[seq_len] in LDS (same shape/LDS as the q8 routed kernel; same context cap).
//
// Record layout (== kvarn_quantize_tile / pack_kvarn_tile), channel-major rows:
//   [ ceil(head_dim*GROUP/cpb) ] q codes | [head_dim*2] scale_abs f16 | [head_dim*2] zp_abs f16 | [GROUP*2] s_col f16
// with cpb = 8/bits codes per byte (bits in {2,4,8}); must match
// KvCache::kvarn_k_record_bytes_bits, which is what the flush writes.
// Dequant K[d,t] = (q*scale_abs[d] + zp_abs[d]) * s_col[t-in-block].
//
// Grid: [n_heads, rows_in_this_launch, 1]; `row_offset` gives the absolute row
// of blockIdx.y == 0.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__ void attention_kvarn_routed_batched(
    const float* __restrict__ q,                    // [B, n_heads, head_dim]
    const uint64_t* __restrict__ rec_ptrs,          // KVarN records, session-major
    const uint64_t* __restrict__ win_ptrs,          // f32 windows, session-major
    const uint64_t* __restrict__ v_ptrs,            // Q8_0 V, session-major
    float* __restrict__ out,                        // [B, n_heads, head_dim]
    const int* __restrict__ row_session_indices,    // [B]
    const int* __restrict__ positions,              // [B]
    int ptr_layer_stride,
    int layer_index,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int rec_bytes,
    int group,
    // 1 = the window holds f16, 0 = f32.
    int window_f16,
    // K code width in bits: 2, 4 or 8. Sets both the record STRIDE and the
    // unpack. This used to be assumed to be 4 — correct only by coincidence at
    // the default `kvarn`/`kvarn4`, and a stride + unpack mismatch at kvarn2 and
    // kvarn8, which the flush writes at the cache's real width.
    int bits,
    // Absolute index of the first row in this launch. The fused prefill writes
    // the window in SEGMENTS (it is a `group`-slot ring, so a run of rows that
    // fills it must be drained before the next run wraps it) and now attends per
    // segment, so `blockIdx.y` is an offset into the segment while q/out/positions
    // stay indexed by ABSOLUTE row. 0 for a whole-batch launch.
    int row_offset
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
    const int n_full = seq_len / group;
    const int win_base = n_full * group;
    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int q_stride = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float* q_head = q + (long long)b * q_stride + h * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    // Same derivation as attention_flash_kvarn_tile_batched: `bits` is a runtime
    // argument, so give the compiler the shift/mask form rather than a divide.
    const int cpb = 8 / bits;                                        // codes per byte
    const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);   // log2(cpb)
    const int cpb_mask = cpb - 1;
    const int codemask = (1 << bits) - 1;
    const int off_scale = (head_dim * group + cpb - 1) / cpb;
    const int off_zp = off_scale + head_dim * 2;
    const int off_scol = off_zp + head_dim * 2;

    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    for (int d = tid; d < head_dim; d += nthreads) q_shared[d] = q_head[d];
    __syncthreads();

    // Phase A: Q·K — KVarN dequant (records) or f32 window per token.
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
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
                const float qv =
                    (float)((byte >> (bits * (idx & cpb_mask))) & codemask);
                const float k = (qv * __half2float(scale_abs[d]) + __half2float(zp_abs[d])) * sc;
                dot += q_shared[d] * k;
            }
        } else {
            // f16 or f32 window — the branch is on a kernel argument, so it is
            // workgroup-uniform. The accumulator stays f32 either way; only the
            // STORAGE narrows. See KvCache::kvarn_window_dtype.
            if (window_f16) {
                const _Float16* kt = reinterpret_cast<const _Float16*>(window)
                    + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) dot += q_shared[d] * (float)kt[d];
            } else {
                const float* kt = window + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) dot += q_shared[d] * kt[d];
            }
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    const float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    const float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) scores[t] /= sum_val;
    __syncthreads();

    // Phase D: V-weighted accumulation (Q8_0) — identical to the q8 routed kernel.
    float* out_head = out + (long long)b * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* blk = v_cache + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}
