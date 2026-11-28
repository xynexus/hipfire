// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Sparse-attention (QSA) slot gather from a KVarN cache.
//
// QSA attends a SELECTED, non-contiguous set of positions, so the KVarN read
// path that rebuilds a full f16 shadow of K every step (`kvarn_attend`) is the
// wrong shape here: it would dequantise the whole history to read ~2K slots.
// This dequantises exactly the selected slots into the same f32
// `[n_kv, n_slots, head_dim]` layout `qsa_gather_kv` produces, so the
// attention kernel downstream is unchanged.
//
//   K: positions in a COMPLETED 128-token block come from that block's KVarN
//      record (`kvarn_quantize_tile` layout: tile = block*n_kv + head,
//      channel-major [head_dim x group]); positions in the OPEN block come from
//      the f32 recent window (`[group, n_kv*head_dim]`, slot = pos % group).
//      deq = (q*scale_abs[ch] + zp_abs[ch]) * s_col[tok]  (== kvarn_dequant_tile)
//   V: Q8_0 by position (`kv_cache_write_q8_0` layout:
//      [pos][n_kv*head_dim/32 blocks x (f16 scale, 32 x i8)]).
//
// Grid: [n_slots, n_kv]. Block: [128].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__ void qsa_gather_kvarn(
    const uint8_t* __restrict__ recs,    // [n_blocks, n_kv, record_bytes]
    const float*   __restrict__ window,  // [group, n_kv*head_dim]
    const uint8_t* __restrict__ vq8,     // [max_seq, n_kv*head_dim/32, 34]
    const int*     __restrict__ slot_idx,// [n_slots]
    float*         __restrict__ out_k,   // [n_kv, n_slots, head_dim]
    float*         __restrict__ out_v,   // [n_kv, n_slots, head_dim]
    int n_kv, int n_slots, int head_dim, int group,
    int record_bytes, int bits, int n_full_blocks
) {
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= n_slots || kv >= n_kv) return;
    const int pos = slot_idx[s];
    const int blk = pos / group;
    const int tok = pos - blk * group;
    const int kv_dim = n_kv * head_dim;
    const long long dst = ((long long)kv * n_slots + s) * head_dim;

    if (blk < n_full_blocks) {
        const uint8_t* rec = recs + ((long long)blk * n_kv + kv) * record_bytes;
        const int cpb = 8 / bits;
        const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);
        const int cpb_mask = cpb - 1;
        const int mask = (1 << bits) - 1;
        const int n = head_dim * group;
        const int qbytes = (n + cpb - 1) / cpb;
        const __half* scale_abs = (const __half*)(rec + qbytes);
        const __half* zp_abs = (const __half*)(rec + qbytes + head_dim * 2);
        const __half* s_col = (const __half*)(rec + qbytes + head_dim * 4);
        const float sc = __half2float(s_col[tok]);
        for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
            const int i = ch * group + tok;
            const uint8_t byte = rec[i >> cpb_shift];
            const float q = (float)((byte >> ((i & cpb_mask) * bits)) & mask);
            out_k[dst + ch] =
                (q * __half2float(scale_abs[ch]) + __half2float(zp_abs[ch])) * sc;
        }
    } else {
        const float* w = window + (long long)tok * kv_dim + kv * head_dim;
        for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) out_k[dst + ch] = w[ch];
    }

    const uint8_t* vrow = vq8 + (long long)pos * (kv_dim / 32) * 34;
    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        const int e = kv * head_dim + ch;
        const uint8_t* blkp = vrow + (e >> 5) * 34;
        const float scale = __half2float(*(const __half*)blkp);
        out_v[dst + ch] = scale * (float)(int8_t)blkp[2 + (e & 31)];
    }
}
