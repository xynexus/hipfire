// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Single-token BF16 embedding lookup: gather one native-bf16 row and write it
// as F32 into `output[dim]`. bf16->f32 fused into the gather (`__bfloat162float`).
extern "C" __global__ void embedding_bf16(
    const __hip_bfloat16* __restrict__ table,   // [vocab × dim] bf16
    float* __restrict__ output,                  // [dim] f32
    int token_id,
    int dim
) {
    const int tid = threadIdx.x;
    const __hip_bfloat16* row = table + (size_t)token_id * dim;
    for (int elem = tid; elem < dim; elem += blockDim.x) {
        output[elem] = __bfloat162float(row[elem]);
    }
}
