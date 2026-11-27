// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Routed-expert page-in repack, ON THE DEVICE.
//
// The weight pager reads an expert module's PORTABLE on-disk bytes and must
// turn them into the per-expert device form the GEMV/GEMM kernels read. Doing
// that on the host cost ~196 us per cold load on the 180B (37% of a page-in) and
// then uploaded the EXPANDED form (~2.4x the disk bytes). These kernels take the
// disk bytes as uploaded and write the device form in place, so the host does no
// transform and the upload carries only the disk bytes.
//
// Both are byte-for-byte ports of the host transforms they replace — the pager's
// parity check differences them against `prepare_expert_module`:
//   pager_oq4_combined   = oq4_arch::oq4_pack_arch_combined
//   pager_compact_split  = oq8_arch::normalize_compact_overlays
//                          followed by oq8_arch::split_compact_planes
//
// Neither is on a hot path in the FLOP sense (a module is a few MB); they exist
// to delete host work and upload bytes, not to be fast kernels.

#include <hip/hip_runtime.h>
#include <stdint.h>

// IEEE binary16 bits -> f32 bits, BIT-IDENTICAL to
// `hipfire_primitives::conv::f16_to_f32` — including its NaN form
// (`frac << 13 | 1`), which `__half2float` does not reproduce.
__device__ __forceinline__ uint32_t f16_bits_to_f32_bits(uint32_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x3ffu;
    if (exp == 0u) {
        if (frac == 0u) return sign << 31;
        int e = 0;
        while ((frac & 0x400u) == 0u) {
            frac <<= 1;
            e -= 1;
        }
        frac &= 0x3ffu;
        const uint32_t exp32 = (uint32_t)(127 - 15 + 1 + e);
        return (sign << 31) | (exp32 << 23) | (frac << 13);
    }
    if (exp == 31u) {
        const uint32_t f32 = frac == 0u ? 0u : ((frac << 13) | 1u);
        return (sign << 31) | (0xffu << 23) | f32;
    }
    return (sign << 31) | ((exp + 127u - 15u) << 23) | (frac << 13);
}

// Canonical Oq4G256 `[f16 scale][128 nibbles]` per 256-group, row-major, into
// the arch combined layout:
//   [split nibbles m*(k/2)] [split f32 scales m*ng] [interleaved m*ng*132]
// One 128-thread block per (row, group); thread t moves nibble byte t to both
// planes, threads 0..3 the four scale bytes to both.
extern "C" __global__ void pager_oq4_combined(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int m,
    int k
) {
    const long long ng = k / 256;
    const long long b = blockIdx.x;
    if (b >= (long long)m * ng) return;
    const int t = threadIdx.x;
    const uint8_t* s = src + b * 130;
    const long long packed = (long long)m * (k / 2);
    const long long il = packed + (long long)m * ng * 4 + b * 132;

    const uint8_t v = s[2 + t];
    // r*(k/2) + g*128 == b*128, since k/2 == ng*128.
    dst[b * 128 + t] = v;
    dst[il + 4 + t] = v;
    if (t < 4) {
        const uint32_t bits = f16_bits_to_f32_bits((uint32_t)s[0] | ((uint32_t)s[1] << 8));
        const uint8_t byte = (uint8_t)((bits >> (8 * t)) & 0xffu);
        dst[packed + b * 4 + t] = byte;
        dst[il + t] = byte;
    }
}

// OqPlusCompact (G256 or G128) interleaved blocks
// `[f16 scale][group/2 nibbles][N_out x (idx, val)]` into the device's split
// planes — ALL nibble groups, then all `[f16 scale][overlay table]` records —
// with the overlay normalisation applied on the way: every overlay entry zeroes
// the bulk nibble it sits on, and an entry superseded by a LATER entry on the
// same index has its correction zeroed (decode stays last-wins). One thread per
// block.
extern "C" __global__ void pager_compact_split(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int n_groups,
    int block_bytes,
    int group
) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_groups) return;
    const int nib = group / 2;
    const int header = 2 + nib;
    const int n_out = (block_bytes - header) / 2;
    const int side = block_bytes - nib;
    const uint8_t* s = src + b * block_bytes;
    uint8_t* dn = dst + b * nib;
    uint8_t* ds = dst + (long long)n_groups * nib + b * side;

    for (int i = 0; i < nib; ++i) dn[i] = s[2 + i];
    ds[0] = s[0];
    ds[1] = s[1];
    for (int i = 0; i < 2 * n_out; ++i) ds[2 + i] = s[header + i];

    for (int e = 0; e < n_out; ++e) {
        const int idx = s[header + 2 * e];
        if (idx >= group) continue;
        for (int e2 = e + 1; e2 < n_out; ++e2) {
            if (s[header + 2 * e2] == idx) {
                ds[2 + 2 * e + 1] = 0;
                break;
            }
        }
        dn[idx / 2] &= (idx % 2 == 0) ? 0xf0 : 0x0f;
    }
}
