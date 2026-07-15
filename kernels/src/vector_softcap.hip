// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Generic pointwise vector softcap. In-place operation is supported when
// input == output. No LDS or block synchronization is required.
extern "C" __global__ void vector_softcap_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    float cap
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = cap * tanhf(input[i] / cap);
}
