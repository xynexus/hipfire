// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Dequantizer for the 8-bit KV hot ring (Phase 1 counterpart of
// kv_hot_quant_q8). Reads the head-major slot-major int8 ring (codes + per-slot
// per-head symmetric scale) and writes a head-major slot-major f16 tile of shape
// [n_kv_heads × hb × head_dim] — exactly what `attention_cold_slots` consumes with
// k_layout=2 / v_layout=2. Only the first `n_slots` live slots are written; the
// read masks by the live count, so the tail is left untouched. Shared by the
// two-tier read and by migrate (which then downloads the f16 tile and widens).
//
// grid = [n_kv_heads, n_slots, 1]; block = [min(256, head_dim), 1, 1]; shared = 0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void kv_hot_dequant_q8(
    const signed char* __restrict__ codes, // in  [n_kv_heads * hb * head_dim]
    const float* __restrict__ scales,       // in  [n_kv_heads * hb]
    __half* __restrict__ out,               // out [n_kv_heads * hb * head_dim]
    int n_slots,
    int hb,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    if (h >= n_kv_heads || s >= n_slots) return;
    const int tid = threadIdx.x;

    const size_t sidx = (size_t)h * hb + s;
    const size_t base = sidx * head_dim;
    const float scale = scales[sidx];
    for (int i = tid; i < head_dim; i += blockDim.x) {
        out[base + i] = __float2half((float)codes[base + i] * scale);
    }
}
