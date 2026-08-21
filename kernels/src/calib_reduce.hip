// SPDX-License-Identifier: Apache-2.0
// hipfire — calibration activation reductions (Tier-1 native Hessian/imatrix).
//
// One indexed staging primitive plus two accumulate-in-place reductions over a
// captured activation block
// x[N, K] (row-major, F32; N = tokens this call, K = linear input dim):
//
//   calib_gather_rows_f32       dst[d+r,:] = src[sorted[s+r]/div,:]
//   calib_sumsq_reduce_f32      acc[c]   += Σ_n x[n,c]²          (imatrix / diag(H))
//   calib_hessian_outer_f32     H[i,j]   += Σ_n x[n,i]·x[n,j]    (GPTQ Hessian, K×K)
//   calib_actstats_reduce_f32   acc[4,K] += Σx, Σ|x|, Σx⁴, max|x| (activation SHAPE)
//
// Both ADD into their destination so the caller accumulates across the whole
// calibration corpus (one call per captured linear per forward). Destinations
// are caller-zeroed once at start.

#include <hip/hip_runtime.h>

// Gather a contiguous expert segment from the grouped-MoE sorted-slot
// permutation into a persistent calibration tile. `x_row_div=K_TOP` maps a
// gate/up routed slot back to its token row; `x_row_div=1` selects the flattened
// post-activation row used by down projections. The admission planner excludes
// padding, but retain the negative-slot guard so malformed work never reads src.
extern "C" __global__ void calib_gather_rows_f32(
    const float* __restrict__ src,
    const int* __restrict__ sorted_slot_index,
    float* __restrict__ dst,
    int sorted_start,
    int dst_start,
    int N,
    int K,
    int x_row_div
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * K;
    if (i >= total) return;
    int row = (int)(i / K);
    int col = (int)(i - (long long)row * K);
    int flat = sorted_slot_index[sorted_start + row];
    if (flat < 0) return;
    int src_row = flat / x_row_div;
    dst[(long long)(dst_start + row) * K + col] =
        src[(long long)src_row * K + col];
}

// acc[c] += Σ_n x[n,c]².  One thread per column; the inner loop walks rows.
// Threads in a warp cover consecutive columns → coalesced reads of x[n, c..c+31].
extern "C" __global__ void calib_sumsq_reduce_f32(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int N, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        float v = x[(long long)n * K + c];
        s += v * v;
    }
    acc[c] += s;
}

// H[i,j] += Σ_n x[n,i]·x[n,j].  Tiled GEMM of A=xᵀ (Aᵢₙ = x[n,i]) by B=x
// (Bₙⱼ = x[n,j]) → C[i,j]. TILE×TILE block computes one output tile, streaming
// the contraction dim N in TILE-row chunks through shared memory.
#define CALIB_TILE 16
extern "C" __global__ void calib_hessian_outer_f32(
    const float* __restrict__ x,
    float* __restrict__ H,
    int N, int K
) {
    __shared__ float As[CALIB_TILE][CALIB_TILE]; // As[t][i] = x[n0+t, i_tile+i]
    __shared__ float Bs[CALIB_TILE][CALIB_TILE]; // Bs[t][j] = x[n0+t, j_tile+j]

    int i = blockIdx.y * CALIB_TILE + threadIdx.y; // output row (channel i)
    int j = blockIdx.x * CALIB_TILE + threadIdx.x; // output col (channel j)

    float acc = 0.0f;
    for (int n0 = 0; n0 < N; n0 += CALIB_TILE) {
        // Cooperative load of the [CALIB_TILE rows × CALIB_TILE cols] tiles.
        int load_i = blockIdx.y * CALIB_TILE + threadIdx.x; // col index for As
        int load_j = blockIdx.x * CALIB_TILE + threadIdx.x; // col index for Bs
        int rA = n0 + threadIdx.y;
        int rB = n0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (rA < N && load_i < K) ? x[(long long)rA * K + load_i] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (rB < N && load_j < K) ? x[(long long)rB * K + load_j] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < CALIB_TILE; t++) {
            // As[t][threadIdx.y] = x[n0+t, i] ; Bs[t][threadIdx.x] = x[n0+t, j]
            acc += As[t][threadIdx.y] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }
    if (i < K && j < K) {
        H[(long long)i * K + j] += acc;
    }
}

// acc[c] += Σ_n w[n]·x[n,c]².  Per-row-weighted column sum-of-squares = the
// diagonal of the weighted Hessian (diag(H̄)). One thread per column.
extern "C" __global__ void calib_sumsq_weighted_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ acc,
    int N, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        float v = x[(long long)n * K + c];
        s += w[n] * v * v;
    }
    acc[c] += s;
}

// w[n] += (1/K) Σ_c d[n,c]².  One thread per ROW (token); the inner loop walks
// the K columns. Used by GuidedQuant calibration to turn a linear's output-grad
// block d[N,K] (∂ℓ/∂z) into the per-token Fisher weight w[n] that reweights the
// Hessian. ADDS into w (caller-zeroed) so it can accumulate if needed; for the
// per-batch use the caller passes a fresh w.
extern "C" __global__ void calib_row_meansq_f32(
    const float* __restrict__ d,
    float* __restrict__ w,
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.0f;
    for (int c = 0; c < K; c++) {
        float v = d[(long long)n * K + c];
        s += v * v;
    }
    w[n] += s / (float)K;
}

// H[i,j] += Σ_n w[n]·x[n,i]·x[n,j].  The per-row-weighted Hessian (GuidedQuant /
// empirical-Fisher): identical tiling to calib_hessian_outer_f32, but each
// rank-1 contribution x[n,:]x[n,:]ᵀ is scaled by the token weight w[n]. With
// w≡1 this reduces exactly to the unweighted outer product.
extern "C" __global__ void calib_hessian_outer_weighted_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ H,
    int N, int K
) {
    __shared__ float As[CALIB_TILE][CALIB_TILE];
    __shared__ float Bs[CALIB_TILE][CALIB_TILE];

    int i = blockIdx.y * CALIB_TILE + threadIdx.y;
    int j = blockIdx.x * CALIB_TILE + threadIdx.x;

    float acc = 0.0f;
    for (int n0 = 0; n0 < N; n0 += CALIB_TILE) {
        int load_i = blockIdx.y * CALIB_TILE + threadIdx.x;
        int load_j = blockIdx.x * CALIB_TILE + threadIdx.x;
        int rA = n0 + threadIdx.y;
        int rB = n0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (rA < N && load_i < K) ? x[(long long)rA * K + load_i] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (rB < N && load_j < K) ? x[(long long)rB * K + load_j] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < CALIB_TILE; t++) {
            int row = n0 + t;
            float wt = (row < N) ? w[row] : 0.0f;
            acc += wt * As[t][threadIdx.y] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }
    if (i < K && j < K) {
        H[(long long)i * K + j] += acc;
    }
}

// Per-channel activation SHAPE statistics, in one pass over the same staged rows.
//
// Everything the calibration captured until now was a second moment: Σx² per
// channel (imatrix) and Σxxᵀ (Hessian). Those measure channel ENERGY, which is
// what AWQ-style weight scaling wants, and they are the WRONG statistic for
// deciding anything about ACTIVATION quantization. Two channels with identical
// Σx² are indistinguishable in the imatrix and opposite in what 4-bit
// activations do to them: one whose mass is spread evenly quantizes cleanly,
// one whose mass sits in a single spike is destroyed. Second moments cannot tell
// them apart, so W4A4 had no measurement to design against.
//
// These four do, and they are all one fused pass over rows already staged:
//
//   acc[0,c] += Σ_n x[n,c]      first moment — asymmetry, i.e. whether a
//                               symmetric quantizer is even the right choice
//   acc[1,c] += Σ_n |x[n,c]|    L1; with Σx² gives a peakedness ratio that needs
//                               no fourth moment to be useful
//   acc[2,c] += Σ_n x[n,c]⁴     with Σx² gives kurtosis, THE heavy-tail signal
//   acc[3,c]  = max(acc, max|x|) range; with rms gives the crest factor, which
//                               is directly "how many bits does this channel
//                               need" — signed int4 has 15 levels, so a crest
//                               factor of 3 is comfortable and 30 clips badly
//
// f32 accumulation is deliberate even for Σx⁴, whose dynamic range is the worst
// of the four. These are corpus ESTIMATES whose sampling noise dwarfs f32
// rounding, and RDNA runs f64 at ~1:16 — per the repo's precision rule, spend
// width where the data needs it, and a calibration statistic does not.
//
// ADDs into its destination like the other two (max for the range row), so the
// caller accumulates across the corpus; the destination is caller-zeroed once.
// One thread owns a column, so the max needs no atomic.
extern "C" __global__ void calib_actstats_reduce_f32(
    const float* __restrict__ x,
    float* __restrict__ acc,   // [4, K]
    int N, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f, sa = 0.0f, s4 = 0.0f, mx = 0.0f;
    for (int n = 0; n < N; n++) {
        const float v = x[(long long)n * K + c];
        const float a = fabsf(v);
        const float v2 = v * v;
        s  += v;
        sa += a;
        s4 += v2 * v2;
        mx = fmaxf(mx, a);
    }
    acc[c]             += s;
    acc[K + c]         += sa;
    acc[2 * K + c]     += s4;
    acc[3 * K + c]      = fmaxf(acc[3 * K + c], mx);
}
