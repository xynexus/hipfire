// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// int8 activations -> FRAGMENT-INTERLEAVED nibble pairs.
//
// The WMMA B operand wants lane n to hold column n's 16 K values AS NIBBLES.
// From [B, K] int8 that costs a 16-byte read plus ~20 VALU ops per WMMA pair to
// split -- 29% of all instructions in a compact W4A8 GEMM, and gfx1151 cannot
// dual-issue the right shifts it needs (only V_DUAL_LSHLREV_B32 exists).
//
// Output layout: for each column, each 16-K step contributes 8 bytes of hi
// nibbles immediately followed by 8 bytes of lo. One 16-byte load then yields
// both operands with no VALU, at the same column stride and byte volume as int8
// (K/2 + K/2 == K).
//
// Two SEPARATE planes were measured and are 30% SLOWER than splitting in-kernel:
// they double the transaction count and read two streams B*K/2 apart. Locality
// beats the instruction saving. See levers.md §9.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t ailv_pack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}

// One thread per 16-K step of one column: reads 16 int8, writes 8+8 bytes.
extern "C" __global__ void act_interleave_nibbles(
    const int8_t* __restrict__ src, uint8_t* __restrict__ dst, int n, int k) {
    const int steps = k >> 4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * steps) return;
    const int col = idx / steps;
    const int st = idx - col * steps;
    const int8_t* s = src + (long long)col * k + st * 16;
    uint8_t* d = dst + (long long)col * k + st * 16;
    const uint32_t d0 = *(const uint32_t*)(s);
    const uint32_t d1 = *(const uint32_t*)(s + 4);
    const uint32_t d2 = *(const uint32_t*)(s + 8);
    const uint32_t d3 = *(const uint32_t*)(s + 12);
    // hi nibbles (signed digit) first 8 bytes, lo (unsigned) next 8
    *(uint32_t*)(d) = ailv_pack4((d0 >> 4) & 0x0f0f0f0fu)
                    | (ailv_pack4((d1 >> 4) & 0x0f0f0f0fu) << 16);
    *(uint32_t*)(d + 4) = ailv_pack4((d2 >> 4) & 0x0f0f0f0fu)
                        | (ailv_pack4((d3 >> 4) & 0x0f0f0f0fu) << 16);
    *(uint32_t*)(d + 8) = ailv_pack4(d0 & 0x0f0f0f0fu)
                        | (ailv_pack4(d1 & 0x0f0f0f0fu) << 16);
    *(uint32_t*)(d + 12) = ailv_pack4(d2 & 0x0f0f0f0fu)
                         | (ailv_pack4(d3 & 0x0f0f0f0fu) << 16);
}
