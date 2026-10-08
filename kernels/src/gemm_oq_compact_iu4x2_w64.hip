// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
//
// Compact-resident Opus EXACT W4A8 on the tuned wave64 structure.
//
// This is `gemm_oq_compact_iu4_w64`'s geometry — wave64, BK=64 K-strip,
// register-staged double buffer, b128 fragment reads — carrying the exact
// two-pass activation split of `gemm_oq_compact_iu4x2_wmma`:
//
//     x = 16 * x_hi + x_lo      (hi digit SIGNED, lo digit UNSIGNED)
//
// so one int8 activation is consumed by two iu4 WMMA passes against ONE weight
// fragment. iu4 is 16 cycles against iu8's 32 on gfx1151, so two iu4 passes cost
// the same WMMA cycles as a single iu8 pass while feeding the compact nibbles
// RAW — no unpack in the inner loop, which is what makes the compact format
// beat both oq8 (2.4x slower) and dense bf16 on this hardware.
//
// WNt is halved against the 1-pass twin (4, not 8). Two i32 accumulator sets at
// WNt=8 would be 128 VGPRs of accumulator alone before `facc`; the wave32 2-pass
// made exactly the same trade when it halved NB.
//
//   W  : compact SPLIT-PLANE blocks — nibble plane first, then [f16 scale][table]
//   X  : [B, K] int8, as `quantize_act_oq8_batched` leaves it. The two digit
//        planes are split during the LDS store: K/2 + K/2 == K, so this is the
//        same byte volume as pre-split planes and costs a few VALU ops.
//   Xs : [B, K/G] f32     Y : [B, M] f32
//
// Does NOT apply the sparse overlay; that is a separate pass, as for both twins.
// Requires K % 256 == 0 (Oq4G256 guarantees it).
// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
// hi pass: B operand SIGNED. lo pass: B operand UNSIGNED. A (weights) always signed.
#define W64H(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, true, b, c, false)
#define W64L(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, false, b, c, false)

// Pack the low nibble of each of 4 bytes into the low 16 bits, in order.
// Input must already be masked to 0x0f0f0f0f.
__device__ __forceinline__ uint32_t oqx2w64_pack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}

#define BK 64
// Wave grid. Staging is 21% of runtime at BM=64 (measured by ablation), and the
// fix is tile size, not a better copy loop: staged bytes per WMMA are
// (BM*BK/2 + BN*BK) / ((BM/16)(BN/16)(BK/16)*2), which falls as the tile grows.
// LDS was only 24 kB of the 64 kB available, so BM 64 -> 128 is nearly free.
#ifndef WARPS_M
#define WARPS_M 2
#endif
#ifndef WARPS_N
#define WARPS_N 2
#endif
#define BLOCK (WARPS_M * WARPS_N * 64)
#ifndef WMt
#define WMt 2
#endif
#ifndef WNt
#define WNt 4      // halved vs the 1-pass twin: two i32 accumulator sets.
                   // WNt=8 spills 113 VGPRs (2 waves); WNt=6 breaks the
                   // load macros (BN*UA/4 must divide BLOCK). So: 4.
#endif
#define BM (WARPS_M * WMt * 16)
#define BN (WARPS_N * WNt * 16)
#define STRIPS_PER_GROUP (256 / BK)   // 4

__launch_bounds__(BLOCK) extern "C" __global__ void gemm_oq_compact_iu4x2_w64(
    const uint8_t* __restrict__ W, const int8_t* __restrict__ X,
    const float* __restrict__ Xs, float* __restrict__ Y,
    int M, int K, int B, int block_stride
#ifdef OV_EPI
    , const int8_t* __restrict__ XT, const float* __restrict__ XsT
#endif
    ) {
    // X held INTERLEAVED in LDS exactly as it is in DRAM: per row, each 16-K step
    // is 8 bytes of hi nibbles then 8 of lo. Same bytes as separate Xhs+Xls
    // planes (BK/2 + BK/2 == BK), but staging becomes a pure copy and ONE i32x4
    // LDS read yields both WMMA operands.
    // XPAD: the interleaved row stride is BK=64 B = 16 dwords, and with 64 LDS
    // banks that puts 16 consecutive lanes on only 4 distinct banks -- a 4-way
    // conflict, twice as bad as the 32 B stride of the old split planes. Padding
    // to 80 B (20 dwords, gcd(20,64)=4) spreads them over all 16.
#define XPAD 16
    __shared__ __align__(16) uint8_t As[2][BM * (BK / 2)], Xi[2][BN * (BK + XPAD)];
    // SWIZZLE: N is the FAST grid axis, M the slow one. With M fast, the whole
    // 47 MB weight set is swept once per N-block -- 4 sweeps at B=512 against a
    // 32 MB MALL that cannot hold one, so every sweep is a fresh DRAM read.
    // With N fast, consecutive workgroups share an M-block, so the resident
    // working set is ~174 kB of weights plus the whole 2.6 MB X tensor.
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN, st = K / 2;
    const int xst = K;               // interleaved row stride (== K bytes)
    const int wid = threadIdx.x >> 6, wtid = threadIdx.x & 63, lane = wtid & 15;
    const int wm = wid / WARPS_N, wn = wid % WARPS_N, nstrip = K / BK;
    const int n_groups = K / 256;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * n_groups * 128LL;
    // X staging in 16-byte chunks: BK/16 chunks per row, BN rows. Expressing it
    // this way (rather than a fixed 32 B per thread) keeps the mapping valid for
    // any BLOCK, which is what lets WARPS_M change without touching the macros.
    const int XCPR = BK / 16;
    const int NXv = BN * XCPR / BLOCK;

    i32x4 acch[WMt][WNt], accl[WMt][WNt];
    f32x4 facc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            // acch/accl are NOT zeroed here: the first K-step of every group --
            // including the first -- seeds its WMMA from the shared `zero4`
            // below rather than from itself, so 64 v_mov_b32 per group fold go
            // away. D and C may differ in a WMMA, which is what makes this work.
            facc[i][j] = (f32x4){0.f, 0.f, 0.f, 0.f};
        }
    const i32x4 zero4 = (i32x4){0, 0, 0, 0};

    // Scale pointers hoisted: the row/col indices vary only in the group, so a
    // 64-bit multiply per (group, tile) becomes one per tile for the kernel.
    const uint8_t* swp[WMt][4];
    const float* sxp[WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const int r0 = bm + wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
            const int rs = (r0 < M) ? r0 : (M - 1);
            swp[i][g] = W + side_base
                      + (long long)rs * n_groups * (long long)side_stride;
        }
#pragma unroll
    for (int j = 0; j < WNt; j++) {
        const int c0 = bn + wn * WNt * 16 + j * 16 + lane;
        const int cs = (c0 < B) ? c0 : (B - 1);
        sxp[j] = Xs + (long long)cs * n_groups;
    }

#ifdef A_GROUP_LINE
    // Single-N-block launches (the BN=64 tile, B <= 64) read every weight exactly
    // once, so stage them a whole GROUP at a time: each row's 128-byte nibble line
    // in one non-temporal read into registers, sliced into LDS one 32-byte strip
    // per iteration. The per-strip 32 B reads relied on L2 holding the line for
    // the next three strips, and the weight stream also evicted the side table
    // (re-read every group fold): measured at B=64, gate/up pulled 99.5 MB for
    // 47.7 MB of weights+scales (~200 GB/s, bandwidth-bound on waste), down 62.7
    // for 47.7. This path pulls 48.8 / 49.0. Not for multi-N-block grids: there
    // other workgroups reuse the lines and non-temporal reads evict them (m128
    // at B=512: 97 -> 230 MB). Chunk ci = threadIdx.x + l*BLOCK is (row = ci % BM,
    // 16-byte chunk q = ci / BM), so a register slot serves one strip pair.
    static_assert((BM * 8) % BLOCK == 0, "A_GROUP_LINE needs BM*8 % BLOCK == 0");
    const int NG = BM * 8 / BLOCK;
    i32x4 rA[NG];
#define LOADA(ks) \
    if ((ks) % STRIPS_PER_GROUP == 0) { _Pragma("unroll") for (int l = 0; l < NG; l++) { int ci = threadIdx.x + l * BLOCK, row = ci % BM, q = ci / BM; int r = bm + row < M ? bm + row : M - 1; rA[l] = __builtin_nontemporal_load((const i32x4*)(W + (long long)r * st + (ks) / STRIPS_PER_GROUP * 128 + q * 16)); } }
#define STSA(bf, ks) \
    _Pragma("unroll") for (int l = 0; l < NG; l++) { int ci = threadIdx.x + l * BLOCK, row = ci % BM, q = ci / BM; \
        if ((q >> 1) == (ks) % STRIPS_PER_GROUP) *(i32x4*)(As[bf] + row * (BK / 2) + (q & 1) * 16) = rA[l]; }
#else
    const int UA = (BK / 2) / 8;   // 8-byte A chunks per row-strip
    const int NA = BM * UA / BLOCK;
    i32x2 rA[NA];
#define LOADA(ks) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK, row = idx / UA, c = idx % UA; int r = bm + row < M ? bm + row : M - 1; rA[l] = *(const i32x2*)(W + (long long)r * st + (ks) * (BK / 2) + c * 8); }
#define STSA(bf, ks) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK; ((i32x2*)As[bf])[idx] = rA[l]; }
#endif
    i32x4 rX[NXv];
#define LOADG(ks) \
    LOADA(ks) \
    /* X is pre-interleaved in DRAM: a strip is BK bytes per row, copied verbatim.
       Two i32x4 per thread covers 32 of the 64 strip bytes. */ \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { \
        int q = threadIdx.x + l * BLOCK, row = q / XCPR, ch = q % XCPR; \
        int r = bn + row < B ? bn + row : B - 1; \
        rX[l] = *(const i32x4*)((const uint8_t*)X + (long long)r * xst + (ks) * BK + ch * 16); }
#define STS(ks) \
    STSA((ks) & 1, ks) \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK; \
        int row = q / XCPR, ch = q % XCPR; \
        *(i32x4*)(Xi[(ks) & 1] + row * (BK + XPAD) + ch * 16) = rX[l]; }

    { LOADG(0); STS(0); }
    __syncthreads();
    for (int k = 0; k < nstrip; k++) {
        int buf = k & 1;
        const bool seed = (k % STRIPS_PER_GROUP) == 0;
        if (k + 1 < nstrip) { LOADG(k + 1); }
#pragma unroll
        // Every LDS read below is 128-bit at the ISA, both operands. X gets there
        // by construction (the interleave packs hi+lo of one 16-K step into 16 B);
        // A gets there for free -- a per-step weight fragment is only 8 B, but the
        // full unroll makes the s and s+1 reads provably adjacent and LLVM
        // coalesces them into ds_load_b128. Hand-writing that as an explicit
        // double-K read emits BYTE-IDENTICAL ISA, so do not bother. 20 ds_load_b128
        // per 64 v_wmma, zero narrower loads. See docs/perf/gfx1151-gemm-levers.md.
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xfh[WNt], xfl[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++) af[i] = *(const i32x2*)(As[buf] + (wm * WMt * 16 + i * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++) {
                // interleaved: hi at +0, lo at +8, one 16-byte read
                const int xo = (wn * WNt * 16 + j * 16 + lane) * (BK + XPAD) + s * 16;
                const i32x4 xv = *(const i32x4*)(Xi[buf] + xo);
                xfh[j][0] = xv[0]; xfh[j][1] = xv[1];
                xfl[j][0] = xv[2]; xfl[j][1] = xv[3];
            }
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) {
                    // ONE weight fragment, BOTH digit passes -- the whole reason
                    // two passes are not two kernels' worth of weight traffic.
                    // `s` is compile-time here and `seed` is wave-uniform, so
                    // this is one scalar branch around step 0, not a select.
                    if (s == 0 && seed) {
                        acch[i][j] = W64H(af[i], xfh[j], zero4);
                        accl[i][j] = W64L(af[i], xfl[j], zero4);
                    } else {
                        acch[i][j] = W64H(af[i], xfh[j], acch[i][j]);
                        accl[i][j] = W64L(af[i], xfl[j], accl[i][j]);
                    }
                }
        }
        // A compact group is 256 weights = STRIPS_PER_GROUP strips. Fold the i32
        // accumulator into f32 with this group's scales and reset it.
        if ((k + 1) % STRIPS_PER_GROUP == 0) {
            const int grp = k / STRIPS_PER_GROUP;
            const int g_off = grp * side_stride;
            float sw[WMt][4];
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int g = 0; g < 4; g++)
                    sw[i][g] = (float)(*(const _Float16*)(swp[i][g] + g_off));
#pragma unroll
            for (int j = 0; j < WNt; j++) {
                const float sx = sxp[j][grp];
#pragma unroll
                for (int i = 0; i < WMt; i++) {
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        // Exact radix-16 recombination in i32 before the scale.
                        const int v = 16 * acch[i][j][g] + accl[i][j][g];
                        facc[i][j][g] += (float)v * sw[i][g] * sx;
                    }
                }
            }
        }
        if (k + 1 < nstrip) { STS(k + 1); }
        __syncthreads();
    }
#undef LOADG
#undef LOADA
#undef STSA
#undef STS
#ifdef OV_EPI
    // SPARSE OVERLAY IN THE EPILOGUE (n_ov = 3 only: 8-byte side records).
    //
    // The separate pass (`oq_compact_overlay_correct_tr3`) re-reads and rewrites
    // all of Y in f32 and runs alone. Here the workgroup finishes its own tile:
    // the K-strip LDS is free once the loop ends, so the tile goes through it in
    // OVR-row slabs -- GEMM result in, the slab's overlay sum added with the same
    // lane mapping and arithmetic as `_tr3` (a row is wave-uniform, its side record
    // one scalar load, a gather an SGPR row base plus the lane's b), then one
    // row-coalesced store. Y is written once, and the gathers overlap other
    // workgroups' WMMA instead of running as their own kernel.
    //
    // Bit-identical to GEMM + `_tr3`: there Y = facc, then Y += acc with acc
    // summed over groups in order; here T = facc, then T += the same acc.
    constexpr int OVR = 32;                 // rows per slab
    constexpr int TS = BN + 1;              // +1: a 0-mod-64 stride is a 32-way conflict
    constexpr int NW = BLOCK / 64;
    constexpr int RPW = OVR / NW;           // slab rows per wave
    constexpr int BPL = BN / 64;            // b per lane
    static_assert(BLOCK == 256 && (BN == 64 || BN == 128) && BM % OVR == 0,
                  "OV_EPI: m128 / default tiles only");
    static_assert(OVR * TS * 4 <= 2 * BN * (BK + XPAD), "OV_EPI slab must fit in Xi");
    float* T = reinterpret_cast<float*>(&Xi[0][0]);
    const uint2* __restrict__ side8 = reinterpret_cast<const uint2*>(W + side_base);
    const int wv = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int bl0 = wtid * BPL;
    const int b0 = bn + bl0;
    for (int c = 0; c < BM / OVR; c++) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < WMt; i++)
#pragma unroll
            for (int j = 0; j < WNt; j++)
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    const int rl = wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
                    if (rl / OVR == c)
                        T[(rl % OVR) * TS + wn * WNt * 16 + j * 16 + lane] = facc[i][j][g];
                }
        float acc[RPW][BPL];
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
            for (int t = 0; t < BPL; t++) acc[r][t] = 0.f;
        if (b0 < B) {
            for (int g = 0; g < n_groups; g++) {
                float xs[BPL];
#pragma unroll
                for (int t = 0; t < BPL; t++) xs[t] = XsT[(long long)g * B + b0 + t];
                const int kbase = g << 8;
#pragma unroll
                for (int r = 0; r < RPW; r++) {
                    const int row = min(bm + c * OVR + wv * RPW + r, M - 1);
                    const uint2 rec = side8[(long long)row * n_groups + g];
                    const float sw = (float)__builtin_bit_cast(_Float16, (uint16_t)(rec.x & 0xffff));
                    const int idx[3] = {(int)((rec.x >> 16) & 0xff), (int)(rec.y & 0xff),
                                        (int)((rec.y >> 16) & 0xff)};
                    const int val[3] = {(int)(int8_t)(rec.x >> 24), (int)(int8_t)(rec.y >> 8),
                                        (int)(int8_t)(rec.y >> 24)};
                    uint32_t xw[3];
#pragma unroll
                    for (int e = 0; e < 3; e++) {
                        const uint8_t* xr = (const uint8_t*)XT + (long long)(kbase + idx[e]) * B
                                          + (uint32_t)b0;
                        xw[e] = (BPL == 2) ? (uint32_t)*(const uint16_t*)xr : (uint32_t)*xr;
                    }
                    int isum[BPL];
#pragma unroll
                    for (int t = 0; t < BPL; t++) isum[t] = 0;
#pragma unroll
                    for (int e = 0; e < 3; e++)
#pragma unroll
                        for (int t = 0; t < BPL; t++)
                            isum[t] += val[e] * (int)(int8_t)((xw[e] >> (t * 8)) & 0xff);
#pragma unroll
                    for (int t = 0; t < BPL; t++)
                        acc[r][t] += (float)isum[t] * sw * xs[t];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
            for (int t = 0; t < BPL; t++) T[(wv * RPW + r) * TS + bl0 + t] += acc[r][t];
        __syncthreads();
        // Row-coalesced: consecutive threads store consecutive rows of one b.
        for (int i = threadIdx.x; i < OVR * BN; i += BLOCK) {
            const int rr = i % OVR, bl = i / OVR;
            const int row = bm + c * OVR + rr, b = bn + bl;
            if (row < M && b < B) Y[(long long)b * M + row] = T[rr * TS + bl];
        }
    }
#else
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            int oc = bn + wn * WNt * 16 + j * 16 + lane;
            if (oc >= B) continue;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                int r = bm + wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
                if (r < M) Y[(long long)oc * M + r] = facc[i][j][g];
            }
        }
#endif
}
