// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-token symmetric-absmax int8 quantizer for the hierarchical KV hot ring
// (8-bit hot tier, docs/plans/2026-07-15-8bit-hot-ring-kv-hier.md Phase 1).
// Quantizes ONE token's KV vector `src` [n_kv_heads × head_dim] (head-major) into
// a head-major slot-major ring at `slot`: per (head) absmax → scale = amax/127,
// int8 codes. Mirrors `kv_cache_write_q8` but writes head-major SEPARATE planes
// (codes + scale) so the ring layout matches the f16 ring it replaces (stride hb),
// and the query/K are FWHT-rotated (centered) so symmetric absmax ≈ affine.
//
// grid = [n_kv_heads, 1, 1]; block = [min(256, head_dim), 1, 1]; shared = block*4.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_hot_quant_q8(
    signed char* __restrict__ codes, // out [n_kv_heads * hb * head_dim]
    float* __restrict__ scales,      // out [n_kv_heads * hb]
    const float* __restrict__ src,   // in  [n_kv_heads * head_dim] (one token)
    int slot,
    int hb,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const float* hs = src + (size_t)h * head_dim;

    extern __shared__ float smem[];
    float lmax = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        lmax = fmaxf(lmax, fabsf(hs[i]));
    }
    smem[tid] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    const float amax = smem[0];
    const float scale = amax / 127.0f;
    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    const size_t sidx = (size_t)h * hb + slot;
    const size_t base = sidx * head_dim;
    if (tid == 0) scales[sidx] = scale;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        int q = __float2int_rn(hs[i] * inv);
        q = max(-127, min(127, q));
        codes[base + i] = (signed char)q;
    }
}
