// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Single-token F16 embedding lookup: gather one native-f16 row and write it as
// F32 into `output[dim]`. f16->f32 fused into the gather (`__half2float`).
extern "C" __global__ void embedding_f16(
    const __half* __restrict__ table,   // [vocab × dim] f16
    float* __restrict__ output,          // [dim] f32
    int token_id,
    int dim
) {
    const int tid = threadIdx.x;
    const __half* row = table + (size_t)token_id * dim;
    for (int elem = tid; elem < dim; elem += blockDim.x) {
        output[elem] = __half2float(row[elem]);
    }
}
