// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Batched BF16 embedding lookup. Gathers N rows of a native bf16 embedding
// table and writes them as F32 into `output[n, dim]` in one launch. The
// bf16->f32 promotion is fused into the gather (`__bfloat162float`, a portable
// intrinsic the ROCm backend lowers per-arch), so the table stays 2 B/element
// in memory instead of being widened to F32. token ids come from a device
// buffer so the launch is hipGraph-captureable.
extern "C" __global__ void embedding_bf16_batched(
    const __hip_bfloat16* __restrict__ table,   // [vocab × dim] bf16
    float* __restrict__ output,                  // [n × dim] f32 row-major
    const int* __restrict__ token_ids,           // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[batch_idx];
    const __hip_bfloat16* row = table + (size_t)token_id * dim;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        out_row[elem] = __bfloat162float(row[elem]);
    }
}
