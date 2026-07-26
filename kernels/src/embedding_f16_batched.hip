// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched F16 embedding lookup. Gathers N rows of a native f16 embedding table
// and writes them as F32 into `output[n, dim]` in one launch. f16->f32 is fused
// into the gather (`__half2float` -> `v_cvt_f32_f16`, universal on all RDNA), so
// the table stays 2 B/element. token ids from a device buffer (hipGraph-safe).
extern "C" __global__ void embedding_f16_batched(
    const __half* __restrict__ table,   // [vocab × dim] f16
    float* __restrict__ output,          // [n × dim] f32 row-major
    const int* __restrict__ token_ids,   // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[batch_idx];
    const __half* row = table + (size_t)token_id * dim;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        out_row[elem] = __half2float(row[elem]);
    }
}
