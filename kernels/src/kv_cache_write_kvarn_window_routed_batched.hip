// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Routed batched KVarN K-window write for independent request sessions.
//
// KVarN stores K as 4-bit variance-normalized records for every COMPLETE
// 128-token block plus an f32 "recent window" holding the trailing partial
// block. Appending a token therefore means writing its K row into the window at
// `slot = pos % group`; the block-complete flush (gather + quantize into
// records) is a separate, per-session event that fires once per `group` tokens
// and reuses `kvarn_gather_k_tiles` + `kvarn_quantize_tile`.
//
// This kernel is the per-token half, routed: each row selects its session's
// window from a session-major pointer table, exactly as
// `kv_cache_write_q8_0_routed_batched` does for V.
//
//   window = win_ptrs[row_session_indices[b] * ptr_layer_stride + layer_index]
//   window[(pos % group) * kv_dim + i] = src[b * kv_dim + i]
//
// The layout is fixed by the READER: `attention_kvarn_routed_batched` indexes
// `window + (t - n_full*group) * kv_dim + kv_h*head_dim`, and `kvarn_attend`'s
// single-session write does `slot * kv_dim * 4` bytes. Token-major [group,
// kv_dim] f32. Do not "improve" this to channel-major — the records are
// channel-major and the window is not, and the reader depends on the difference.
//
// CALLER CONTRACT: no session may advance across more than one block boundary
// within a single launch. The window physically holds `group` tokens, so a
// session writing more than `group` rows in one call would overwrite tokens the
// flush has not yet quantized. Callers split the row range at block boundaries
// (mirrors the `take`-chunked loop in `kvarn_attend`).
//
// `row_offset` exists to serve that contract without slicing buffers: a caller
// walking segments passes the SAME full-batch `src`/`row_session_indices`/
// `positions` every time and moves the window of rows it writes. Row indices are
// therefore absolute (`row_offset + blockIdx.y`), so the row tables stay aligned
// with `src` and nothing has to build offset views per segment.
//
// Grid: [ceil(kv_dim / 256), row_count, 1]; block [256, 1, 1].

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(256)
extern "C" __global__ void kv_cache_write_kvarn_window_routed_batched(
    const uint64_t* __restrict__ win_ptrs,       // [num_sessions * ptr_layer_stride]
    const float* __restrict__ src,               // [total_rows * kv_dim] token-major
    const int* __restrict__ row_session_indices, // [total_rows]
    const int* __restrict__ positions,           // [total_rows] absolute positions
    int ptr_layer_stride,
    int layer_index,
    int n_kv_heads,
    int head_dim,
    int group,
    int row_offset,                              // first row of this segment
    int row_count                                // rows in this segment
) {
    if ((int)blockIdx.y >= row_count) return;
    // Absolute row index: src and the row tables are always the FULL batch.
    const int b = row_offset + (int)blockIdx.y;

    const int kv_dim = n_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;

    const int session_idx = row_session_indices[b];
    float* window = reinterpret_cast<float*>(
        static_cast<uintptr_t>(win_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index])
    );

    // Absolute position -> slot within the trailing partial block. Matches the
    // reader's `t - n_full*group` because `n_full*group` is the block-aligned
    // base of the window's contents.
    const int slot = positions[b] % group;

    window[(size_t)slot * kv_dim + i] = src[(size_t)b * kv_dim + i];
}
