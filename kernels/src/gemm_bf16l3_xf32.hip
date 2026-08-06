// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// BF16L3 weight x F32 activations -> F32, batched over N columns.
//
// The batched form of gemv_bf16l3_xf32. Same plane layout, same wave-wide
// escape prefix sum, same branch on escape — the ONLY change is that the
// decoded weight is reused across up to BF16L3_NT activation columns instead of
// one.
//
// Why this exists: the GEMV form reads the entire packed head per output
// position. On an lm_head that is 367 MB per call, and a KLD reference build
// calls it once per scored position — 1023 per 2048-token chunk, 748 GB of
// weight traffic, measured at 1.99 s and 62% of all GPU time in a reference
// build (rocprofv3, gfx1151). Decoding is per weight element and independent of
// the activation, so a tile of columns amortizes both the read and the decode:
// BF16L3_NT columns per pass cuts the weight traffic by that factor.
//
// The host loops column tiles, passing `col_base`; `n_cols` is the live width of
// the final partial tile.
//
// WHERE THIS STANDS, and two falsified theories, so nobody re-runs them
// (`bench_bf16l3_vs_bf16_gemm` reproduces all of it):
//
// Against the plain-BF16 WMMA GEMM this is 0.4-0.9x at N=1 (a WIN, up to 2.6x
// on an lm_head) and 2.3-8x SLOWER at N=256. The crossover is below N=64.
//
//  * "It is weight-re-read bound." At NT=8, N=256 needs 32 passes over the whole
//    matrix while the WMMA GEMM reads its weights once. Raising NT 8 -> 32
//    quartered that traffic and bought ~10% at N=256, while making N=1 worse
//    (NT accumulators are allocated even when n_cols is 1). Not the bottleneck.
//  * "The activation loads should be contiguous." Transposing x to [K, N] so a
//    tile's columns share a cache line made every shape 15-30% SLOWER. The
//    [N, K] layout is coalesced ACROSS THE WAVE, which matters more.
//
// What is left is structural, not a constant: decode a weight tile into LDS once
// and run the same WMMA compute the BF16 path uses, so LUT3 is purely a storage
// format and prefill matches bf16 while reading 1.38x fewer bytes.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define BF16L3_BLOCK       256
#define BF16L3_LUT_STRIDE    8
#define BF16L3_CODE_STRIDE  96   // 256 codes x 3 bits
#define BF16L3_HEADER       16
#define BF16L3_ESCAPE        7
#define BF16L3_NT           32   // activation columns per pass

__device__ __forceinline__ float bf16l3_gemm_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ size_t bf16l3_gemm_align16(size_t x) {
    return (x + 15) & ~(size_t)15;
}

__launch_bounds__(32)
extern "C" __global__ void gemm_bf16l3_xf32(
    const unsigned char* __restrict__ W,   // BF16L3 payload of the [M, K] matrix
    const float*         __restrict__ x,   // [N, K] F32, row-major per column
    float*               __restrict__ y,   // [N, M] F32, row-major per column
    int M, int K, int N, int col_base, int n_cols
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;            // lane 0..31

    // Plane bases, recomputed from n exactly as the host-side Layout does.
    const size_t n        = (size_t)M * (size_t)K;
    const size_t nb       = (n + BF16L3_BLOCK - 1) / BF16L3_BLOCK;
    const size_t esc_tab  = BF16L3_HEADER;
    const size_t lut_off  = esc_tab + bf16l3_gemm_align16(4 * nb);
    const size_t mant_off = lut_off + bf16l3_gemm_align16(BF16L3_LUT_STRIDE * nb);
    const size_t code_off = mant_off + BF16L3_BLOCK * nb;
    const size_t esc_off  = code_off + BF16L3_CODE_STRIDE * nb;

    const int blocks_per_row = K / BF16L3_BLOCK;

    float sum[BF16L3_NT];
    #pragma unroll
    for (int c = 0; c < BF16L3_NT; ++c) sum[c] = 0.0f;

    for (int j = 0; j < blocks_per_row; ++j) {
        const size_t blk = (size_t)row * (size_t)blocks_per_row + (size_t)j;

        const unsigned int esc_start =
            ((const unsigned int* __restrict__)(W + esc_tab))[blk];
        const uint2 lv =
            *(const uint2* __restrict__)(W + lut_off + BF16L3_LUT_STRIDE * blk);
        const unsigned char* __restrict__ mant  = W + mant_off + BF16L3_BLOCK * blk;
        const unsigned char* __restrict__ codes = W + code_off + BF16L3_CODE_STRIDE * blk;
        const unsigned char* __restrict__ esc   = W + esc_off  + esc_start;

        // Lane t owns elements [8t, 8t+8) — exactly the 3 code bytes at 3t.
        unsigned int acc = (unsigned int)codes[3 * tid]
                         | ((unsigned int)codes[3 * tid + 1] << 8)
                         | ((unsigned int)codes[3 * tid + 2] << 16);
        const uint2 mv = *(const uint2* __restrict__)(mant + 8 * tid);

        // How many of this lane's 8 elements escaped.
        int local = 0;
        #pragma unroll
        for (int s = 0; s < 8; ++s) local += (((acc >> (3 * s)) & 7u) == BF16L3_ESCAPE);

        // Wave-wide exclusive prefix sum -> this lane's cursor into the block's
        // escape run.
        int scan = local;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up(scan, d);
            if (tid >= d) scan += up;
        }
        int cursor = scan - local;

        const unsigned int mb[2] = { mv.x, mv.y };
        const size_t k0 = (size_t)j * BF16L3_BLOCK + 8 * tid;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const unsigned int code = (acc >> (3 * s)) & 7u;
            const unsigned int lsel = (code < 4 ? lv.x : lv.y) >> (8 * (code & 3));
            const unsigned int e = (code == BF16L3_ESCAPE) ? (unsigned int)esc[cursor++]
                                                           : (lsel & 0xffu);
            const unsigned int sm = (mb[s >> 2] >> (8 * (s & 3))) & 0xffu;
            const unsigned short wq =
                (unsigned short)(((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu));
            // Decoded ONCE, spent across the column tile. This reuse is the
            // entire point of the kernel.
            const float wf = bf16l3_gemm_to_f32(wq);
            // These loads are K floats apart per column, which looks wrong and
            // is not: MEASURED, transposing x to [K, N] to make the tile
            // contiguous per thread made every shape 15-30% SLOWER. This layout
            // is coalesced ACROSS THE WAVE — adjacent lanes own k0 8 apart, so
            // their addresses are 32 bytes apart and share lines — and that
            // matters more than per-thread contiguity. Transposing trades it for
            // lanes reading 8*N floats apart.
            #pragma unroll
            for (int c = 0; c < BF16L3_NT; ++c) {
                if (c < n_cols) {
                    sum[c] += wf * x[(size_t)(col_base + c) * (size_t)K + k0 + s];
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < BF16L3_NT; ++c) {
        float v = sum[c];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
        if (tid == 0 && c < n_cols) {
            y[(size_t)(col_base + c) * (size_t)M + row] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// WMMA form: decode a LUT3 block into WMMA A-fragments, keep the compute
// identical to the plain-BF16 path.
//
// The scalar kernel above wins batch-1 and loses prefill 2.3-8x, and two
// attempts to close that by restructuring memory access failed (see the header).
// The gap is structural: the BF16 path runs 16x16x16 WMMA while this one does
// scalar FMAs. So stop competing with it and JOIN it — treat LUT3 purely as a
// storage format, decode a fragment, and hand it to the same intrinsic.
//
// Mirrors gemm_bf16_x_bf16_wmma_gfx1151_m128 exactly: 8 wave32 warps, 128 output
// rows per block, a 16-batch X tile staged in LDS per K_BLOCK and reused by all
// warps. The ONE change is that `a_frag` is decoded rather than loaded.
//
// Why the escape cursor works here: escapes are stored per block in element
// order, so the cursor is a running count from the block's start. `kt` walks
// 0..256 monotonically inside exactly one LUT3 block (K % 256 == 0 is asserted
// host-side, so a row's blocks never straddle), which is precisely the order the
// cursor needs. Resetting it at each `k_base` keeps decode and storage in step.
//
//   W : BF16L3 payload of [M, K]
//   X : [B, K] BF16 row-major
//   Y : [B, M] F32 row-major
#define L3W_M_TILE     128
#define L3W_N_TILE      16
#define L3W_M_PER_WARP  16
#define L3W_SLOTS       2

typedef short __attribute__((ext_vector_type(16))) l3_b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  l3_float8_t;

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_bf16l3_wmma_m128(
    const unsigned char*  __restrict__ W,
    const unsigned short* __restrict__ X,
    float*                __restrict__ Y,
    int M, int K, int B
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane16  = lane & 15;
    const int row_start   = blockIdx.x * L3W_M_TILE;
    const int batch_start = blockIdx.y * L3W_N_TILE;
    if (row_start >= M || batch_start >= B) return;

    __shared__ unsigned short lds_x[L3W_N_TILE * BF16L3_BLOCK];

    // Plane bases, recomputed from n exactly as the host-side Layout does.
    const size_t n        = (size_t)M * (size_t)K;
    const size_t nb       = (n + BF16L3_BLOCK - 1) / BF16L3_BLOCK;
    const size_t esc_tab  = BF16L3_HEADER;
    const size_t lut_off  = esc_tab + bf16l3_gemm_align16(4 * nb);
    const size_t mant_off = lut_off + bf16l3_gemm_align16(BF16L3_LUT_STRIDE * nb);
    const size_t code_off = mant_off + BF16L3_BLOCK * nb;
    const size_t esc_off  = code_off + BF16L3_CODE_STRIDE * nb;

    const int warp_row_start = row_start + warp_id * L3W_M_PER_WARP;
    const int my_row   = warp_row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const int blocks_per_row = K / BF16L3_BLOCK;

    l3_float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const l3_b16x16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int k_base = 0; k_base < K; k_base += BF16L3_BLOCK) {
        // Stage the X tile once per block, shared by all 8 warps.
        const int slot_in_block = warp_id * L3W_SLOTS + (lane >> 4);
        const int k_off = lane16 * 16;
        const int batch_row = batch_start + slot_in_block;
        l3_b16x16_t x_frag = zero_b;
        if (batch_row < B && k_base + k_off < K) {
            x_frag = *(const l3_b16x16_t*)(X + (long long)batch_row * K + k_base + k_off);
        }
        *(l3_b16x16_t*)(lds_x + slot_in_block * BF16L3_BLOCK + k_off) = x_frag;
        __syncthreads();

        // This lane's row-block. One LUT3 block covers exactly this k range.
        const size_t blk = (size_t)safe_row * (size_t)blocks_per_row
                         + (size_t)(k_base / BF16L3_BLOCK);
        const unsigned int esc_start =
            ((const unsigned int* __restrict__)(W + esc_tab))[blk];
        const uint2 lv =
            *(const uint2* __restrict__)(W + lut_off + BF16L3_LUT_STRIDE * blk);
        const unsigned char* __restrict__ mant  = W + mant_off + BF16L3_BLOCK * blk;
        const unsigned char* __restrict__ codes = W + code_off + BF16L3_CODE_STRIDE * blk;
        const unsigned char* __restrict__ esc   = W + esc_off  + esc_start;
        int cursor = 0;

        for (int kt = 0; kt < BF16L3_BLOCK; kt += 16) {
            // 16 codes live in 6 bytes: kt is a multiple of 8, so byte-aligned.
            const unsigned char* __restrict__ cp = codes + (kt >> 3) * 3;
            const unsigned int c0 =
                (unsigned int)cp[0] | ((unsigned int)cp[1] << 8) | ((unsigned int)cp[2] << 16);
            const unsigned int c1 =
                (unsigned int)cp[3] | ((unsigned int)cp[4] << 8) | ((unsigned int)cp[5] << 16);

            l3_b16x16_t a_frag;
            #pragma unroll
            for (int s = 0; s < 16; ++s) {
                const unsigned int cw   = (s < 8) ? c0 : c1;
                const unsigned int code = (cw >> (3 * (s & 7))) & 7u;
                const unsigned int lsel = (code < 4 ? lv.x : lv.y) >> (8 * (code & 3));
                const unsigned int e    = (code == BF16L3_ESCAPE)
                                              ? (unsigned int)esc[cursor++]
                                              : (lsel & 0xffu);
                const unsigned int sm   = (unsigned int)mant[kt + s];
                a_frag[s] = (short)(((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu));
            }

            l3_b16x16_t b_frag = *(const l3_b16x16_t*)(lds_x + lane16 * BF16L3_BLOCK + kt);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    const int out_col = batch_start + lane16;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = warp_row_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#undef L3W_M_TILE
#undef L3W_N_TILE
#undef L3W_M_PER_WARP
#undef L3W_SLOTS
