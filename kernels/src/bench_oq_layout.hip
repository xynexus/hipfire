// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// SYNTHETIC: what access shape actually sustains DRAM bandwidth UNDER LOAD?
//
// The compact decode GEMV reads 14.0 GB/token at 1.05x overfetch (FETCH_SIZE)
// yet sustains ~213 GB/s against a 248.5 GB/s pure-read stream. The bytes are
// right, so the shortfall is SHAPE. These kernels vary one property at a time
// against the same total traffic so the responsible property can be named.
//
// All of them read `W` in whatever pattern they model and reduce it to one
// float per workgroup, so nothing is dead-code-eliminated and the loads cannot
// be hoisted.

#include <hip/hip_runtime.h>
#include <stdint.h>

// (1) Upper bound: flat grid-stride dwordx4 stream. What the memory system does
// when nothing is in its way.
extern "C" __global__ void bwl_stream(
    const uint8_t* __restrict__ W, float* __restrict__ Y, long long nbytes
) {
    const long long n16 = nbytes >> 4;
    const long long stride = (long long)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        uint32_t v[4];
        __builtin_memcpy(v, W + (i << 4), 16);
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
    if (threadIdx.x == 0xffff) Y[blockIdx.x] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[blockIdx.x] = 1.0f;
}

// (2) The GEMV's shape: one wave per row, ng blocks of `block_stride` bytes,
// 8 lanes per group, 16 B/lane. `block_stride` is a RUNTIME argument so 136
// (the real block, nibbles + f16 scale + overlay table) and 128 (a power of two,
// nothing else changed) can be compared with an identical instruction stream.
extern "C" __global__ void bwl_blocked(
    const uint8_t* __restrict__ W, float* __restrict__ Y,
    int M, int ng, int block_stride, int read_side
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;
    uint32_t acc = 0;
    for (int r = 0; r < ng; r += 4) {
        const uint8_t* blk = w_row + (long long)(r + gir) * block_stride;
        uint32_t v[4];
        __builtin_memcpy(v, blk + (read_side ? 2 : 0) + lig * 16, 16);
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
        if (read_side) {
            // The 2-byte scale and the 6-byte overlay table, read the way the
            // real kernel reads them: a separate small request per block.
            acc ^= (uint32_t)(*(const uint16_t*)blk);
            acc ^= (uint32_t)(*(const uint16_t*)(blk + 130 + 2 * (lig & 3)));
        }
    }
    if (lane == 0xffff) Y[row] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[row] = 1.0f;
}

// (3) The proposed fix: SPLIT PLANES. Nibbles live in one contiguous
// [M, K/2] plane, so a group is exactly 128 B and every wave request is
// 128-byte aligned and contiguous; the f16 scale + overlay table move to a
// small second plane of 8 B per block. Identical total bytes, identical bit
// budget — only the layout differs.
extern "C" __global__ void bwl_split(
    const uint8_t* __restrict__ Wn, const uint8_t* __restrict__ Ws,
    float* __restrict__ Y, int M, int ng
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const uint8_t* n_row = Wn + (long long)row * ng * 128LL;
    const uint8_t* s_row = Ws + (long long)row * ng * 8LL;
    uint32_t acc = 0;
    for (int r = 0; r < ng; r += 4) {
        const int g = r + gir;
        uint32_t v[4];
        __builtin_memcpy(v, n_row + (long long)g * 128 + lig * 16, 16);
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
        const uint8_t* sb = s_row + (long long)g * 8;
        acc ^= (uint32_t)(*(const uint16_t*)sb);
        acc ^= (uint32_t)(*(const uint16_t*)(sb + 2 + 2 * (lig & 2)));
    }
    if (lane == 0xffff) Y[row] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[row] = 1.0f;
}

// (6) ROW-SPLIT: the same reordering as (5) but inside ONE buffer. Per row the
// nibbles come first as a contiguous ng*128 run, then that row's side data as
// ng*8. Row stride is unchanged at ng*block_stride, so the allocation, the
// tensor and every dispatch signature stay exactly as they are — only the
// address arithmetic inside the kernels moves. If this matches (5), the win is
// available without a second plane.
extern "C" __global__ void bwl_rowsplit(
    const uint8_t* __restrict__ W, float* __restrict__ Y,
    int M, int ng, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const uint8_t* r0 = W + (long long)row * ng * (long long)block_stride;
    const uint8_t* sid = r0 + (long long)ng * 128;
    uint32_t acc = 0;
    for (int r = 0; r < ng; r += 4) {
        const int g = r + gir;
        uint32_t v[4];
        __builtin_memcpy(v, r0 + (long long)g * 128 + lig * 16, 16);
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
        const uint8_t* sb = sid + (long long)g * 8;
        acc ^= (uint32_t)(*(const uint16_t*)sb);
        acc ^= (uint32_t)(*(const uint16_t*)(sb + 2 + 2 * (lig & 2)));
    }
    if (lane == 0xffff) Y[row] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[row] = 1.0f;
}

// (7) TENSOR-SPLIT: one allocation, but ALL nibbles first and ALL side data
// after — not per row. This is the cheap form of (5): nibble row r sits at
// r*ng*128, so every row start is 128-byte aligned (ng*128 is a multiple of
// 128, whereas the row stride ng*136 = 2720 is not, which is why (6) failed).
// The side base is derivable from M and ng, which the kernels already have, so
// there is ONE pointer and no dispatch signature changes.
extern "C" __global__ void bwl_tsplit(
    const uint8_t* __restrict__ W, float* __restrict__ Y,
    int M, int ng, int side_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const uint8_t* n_row = W + (long long)row * ng * 128LL;
    const uint8_t* s_row = W + (long long)M * ng * 128LL
                             + (long long)row * ng * (long long)side_stride;
    uint32_t acc = 0;
    for (int r = 0; r < ng; r += 4) {
        const int g = r + gir;
        uint32_t v[4];
        __builtin_memcpy(v, n_row + (long long)g * 128 + lig * 16, 16);
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
        const uint8_t* sb = s_row + (long long)g * side_stride;
        acc ^= (uint32_t)(*(const uint16_t*)sb);
        acc ^= (uint32_t)(*(const uint16_t*)(sb + 2 + 2 * (lig & 2)));
    }
    if (lane == 0xffff) Y[row] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[row] = 1.0f;
}

// (8) PERSISTENT: same split-plane addressing as (7), but a FIXED wide grid whose
// waves grid-stride over rows, instead of one 32-thread workgroup per row. The
// flat stream in (1) reaches its number with 4096 wide workgroups; the GEMV
// dispatches one tiny workgroup per output row — 248320 of them for lm_head —
// and this asks whether that dispatch shape is itself costing bandwidth.
extern "C" __global__ void bwl_persist(
    const uint8_t* __restrict__ W, float* __restrict__ Y,
    int M, int ng, int side_stride
) {
    const int lane  = threadIdx.x & 31;
    const int wave  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int waves = (gridDim.x * blockDim.x) >> 5;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const long long side_base = (long long)M * ng * 128LL;
    uint32_t acc = 0;
    for (int row = wave; row < M; row += waves) {
        const uint8_t* n_row = W + (long long)row * ng * 128LL;
        const uint8_t* s_row = W + side_base + (long long)row * ng * (long long)side_stride;
        for (int r = 0; r < ng; r += 4) {
            const int g = r + gir;
            uint32_t v[4];
            __builtin_memcpy(v, n_row + (long long)g * 128 + lig * 16, 16);
            acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
            const uint8_t* sb = s_row + (long long)g * side_stride;
            acc ^= (uint32_t)(*(const uint16_t*)sb);
            acc ^= (uint32_t)(*(const uint16_t*)(sb + 2 + 2 * (lig & 2)));
        }
    }
    if (lane == 0xffff) Y[0] = (float)acc;
    else if (acc == 0xdeadbeefu) Y[0] = 1.0f;
}
