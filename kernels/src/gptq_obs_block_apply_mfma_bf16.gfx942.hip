// GPTQ Phase 2C — block-apply MFMA BF16 GEMM kernel (cast-trick variant).
//
// See docs/plans/gptq-hip-phase2-obs-kernel.md §6 (numerical equivalence
// policy). This is the BF16 cast-trick sibling of
// `gptq_obs_block_apply_mfma_f64.gfx942.hip` — the OBS rank-B GEMM stays
// numerically equivalent within the §6 codeword-divergence tolerance
// (<0.1% MQ4 codewords differ between F64 and BF16 paths) while pulling
// MFMA throughput from F64 → BF16 (~7× on gfx942).
//
// Buffers stay F64:
//   * W_residual, err_block, U   — F64 on the host side and in HBM
//   * Only the MFMA contraction is BF16
//
// The cast trick: load F64 → cast to F32 → `__float2bfloat16_rn` →
// pack into bf16x4 per lane → `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`
// → F32 accumulator → cast back to F64 at the subtract-store.
//
// Computes (subtract-accumulate, in place):
//
//   W_residual[:, perm[block_end..K]] -= err_block @ U_submatrix
//
// where:
//   err_block     : M × B   F64 (B=128, accumulated errors from current block)
//   U_submatrix   : B × N_remaining F64; logically
//                     U_submatrix[i, j] = U[block_start + i, block_end + j]
//   N_remaining   = K - block_end
//
// And the output write is permuted:
//   W_residual[m, perm[block_end + j]] -= sum_{i in [0,B)} err_block[m, i] * U_submatrix[i, j]
//
// ── MFMA tile geometry (BF16, K_inner=16 per call) ──
//
//   MFMA intrinsic   : __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
//                      (16×16 FP32 output, K_inner=16 per call)
//
//   Per-wave output  : 16×16 (256 elements / 64 lanes = 4 floats per lane,
//                      strip-major: lane l holds C[m=(l/16)*4+r, n=l%16])
//
//   Per-WG output    : 32 M × 16 N tile (2 waves stacked in M)
//
//   Grid             : ((M + 31) / 32, (N_remaining + 15) / 16, 1)
//   Block            : (128, 1, 1)  — 2 wave64 / WG
//
//   LDS B-tile       : 16 N-cols × B K-rows of U_submatrix, BF16 staged.
//                      Size: 16 × 128 × 2 B = 4 KB per WG (vs 16 KB for F64).
//
//   K-iters per WG   : ceil(B / 16) = 8 (vs 32 for F64 — 4× fewer dispatches)
//
// ── MFMA inner-loop lane layout (mirrors gemm_bf16_mfma.gfx942.hip) ──
//
//   A operand (BF16, 4 BF16 per lane = 64-bit packed):
//     lane l holds a[r] = A[m = l%16, k = (l/16)*4 + r]   for r ∈ [0,4)
//     → in our case: a[r] = err_block[my_m, kc + strip*4 + r]
//       where my_m = m_offset + (l%16) for this wave, strip = l/16
//
//   B operand (BF16, 4 BF16 per lane = 64-bit packed):
//     lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]   for r ∈ [0,4)
//     → b[r] = U_submatrix[kc + strip*4 + r, n_offset_in_tile + n_lane]
//     → from LDS: lds_b[n_lane * B + (kc + strip*4 + r)]
//
//   D output (FP32, 4 floats per lane, strip-major):
//     lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]   for r ∈ [0,4)
//
// ── Output write ──
//
//   For wave w with m_off ∈ {0, 16}:
//     gm = m_offset + (l/16)*4 + r       (4 m-rows per lane, strip-major)
//     gn = n_offset_in_tile + (l%16)     (16 n-cols spread across lanes)
//     output_col_orig = perm[block_end + gn]
//     W_residual[gm, output_col_orig] -= (double)c_acc[r]
//
//   Each (gm, gn) cell is written by exactly one lane in one wave —
//   no inter-WG races (each WG owns a unique (m_tile, n_tile)) and
//   no intra-WG races (each wave owns disjoint M-rows).
//
// ── Numerical contract ──
//
//   F64 → F32 → BF16 conversion at MFMA-feed time. BF16 has 8-bit
//   mantissa vs F64's 52-bit — the cast trick is a controlled precision
//   drop on the multiplication side. The accumulator is F32 (24-bit
//   mantissa), wider than BF16 input, so dot-product cancellation
//   within a single MFMA chunk is well-behaved. The final cast to F64
//   at the store is exact (F32 → F64 round-up).
//
//   Per plan §6 the contract is: codeword divergence rate < 0.1%
//   compared to the F64 path on real Hessians. The parity_gate test
//   `parity_gate_obs_bf16_codeword_divergence` exercises the synthetic
//   case; real-tensor validation lives on the droplet.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

// MFMA F32_16x16x16_BF16 intrinsic — operands are bit-cast __bf16x4 → short x4.
// LLVM's amdgcn intrinsic models the BF16 lane bits as i16; bit_cast is the
// no-op punning the compiler folds away. Same wrapper as
// gemm_bf16_mfma.gfx942.hip / hessian_outer_product.gfx942.hip.
__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

// Cast F64 → BF16 via F32 intermediate (round-to-nearest-even).
// gfx942 has hardware BF16 conversions; __float2bfloat16_rn is the
// supported HIP intrinsic for FP32 → BF16 RN. F64 → FP32 truncates
// via the C++ static_cast.
__device__ static inline bf16_t f64_to_bf16(double v) {
    return __float2bfloat16_rn(static_cast<float>(v));
}

#define TILE_M    32
#define TILE_N    16
#define K_INNER   16
#define BLOCK_B 128

extern "C" __global__ __launch_bounds__(128)
void gptq_obs_block_apply_mfma_bf16(
    double*       __restrict__ d_w_residual,    // [M*K] row-major  (F64)
    const double* __restrict__ d_err_block,     // [M*B] row-major  (F64)
    const double* __restrict__ d_u,             // [K*K] COLUMN-major (post-dgeam, F64)
    const int*    __restrict__ d_perm,          // [K]
    int block_start,                             // start of current block
    int block_end,                               // end of current block (exclusive)
    int k_dim,                                   // K
    int m_dim,                                   // M
    int b_dim,                                   // B (= block_end - block_start, max 128)
    int n_remaining                              // = K - block_end
) {
    // BF16-staged B-tile: 16 N-cols × B K-rows. 16 × 128 × 2 B = 4 KB per WG.
    __shared__ bf16_t lds_b[TILE_N * BLOCK_B];

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..1
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;       // 0..15 (n-col within tile, A.m, D.n)
    const int strip     = lane_in_w >> 4;       // 0..3 (k-strip / D row-strip)

    const int wave_m_off = (wave_id == 0) ? 0 : 16;
    const int m_offset   = blockIdx.x * TILE_M + wave_m_off;   // start of this wave's m-tile
    const int n_offset_in_tile = blockIdx.y * TILE_N;          // start of this WG's n-tile (in n_remaining frame)

    // ─── Cooperative LDS load of B-tile (full 16 × B BF16, cast from F64) ───
    //
    // B-tile layout: lds_b[n_in_tile * b_dim + k] = (bf16) U_submatrix[k, n_in_tile]
    //              = U[block_start + k, block_end + (n_offset_in_tile + n_in_tile)]
    //              = d_u[(block_end + n_offset_in_tile + n_in_tile) * K
    //                     + (block_start + k)]   (column-major)
    //
    // The F64 → BF16 cast happens HERE (one cast per LDS slot, amortized
    // across the 32 MFMA reads of each slot). 128 threads load 16 × b_dim
    // BF16. For B=128: 128*16 = 2048 = 16/thread.
    __syncthreads();
    int total_lds = TILE_N * b_dim;
    #pragma unroll
    for (int li = 0; li < 16; li++) {
        int idx = tid + li * 128;
        if (idx >= total_lds) break;
        int n_in_tile = idx / b_dim;       // 0..TILE_N-1
        int k_idx     = idx % b_dim;       // 0..b_dim-1
        int gn_remaining = n_offset_in_tile + n_in_tile;     // index in `n_remaining` frame
        int gn_u    = block_end + gn_remaining;              // original u-col index
        int gk_u    = block_start + k_idx;                   // original u-row index
        double v = 0.0;
        if (gn_remaining < n_remaining && gk_u < k_dim && gn_u < k_dim && k_idx < b_dim) {
            // Upper-tri: only valid when gk_u <= gn_u. Off-triangle is zero.
            if (gk_u <= gn_u) {
                v = d_u[(long long)gn_u * (long long)k_dim + (long long)gk_u];
            }
        }
        lds_b[n_in_tile * b_dim + k_idx] = f64_to_bf16(v);
    }
    __syncthreads();

    // ─── MFMA inner loop: ceil(b_dim/16) MFMA calls (for B=128: 8 calls) ───
    //
    // A operand: 4 BF16 per lane = err_block[my_m, kc + strip*4 + r] for r ∈ [0,4)
    //   where my_m = m_offset + n_lane (for this wave)
    // B operand: 4 BF16 per lane = lds_b[n_lane * b_dim + (kc + strip*4 + r)]
    // D output: 4 floats per lane = c_acc[r] for r ∈ [0,4)
    //
    // F64 err_block is cast to BF16 inside the loop (one cast per A-load).
    // err_block is loaded fresh each iteration (no LDS-stage) — it's read
    // exactly once per MFMA call per lane, so per-lane register usage stays
    // bounded and the BW cost is amortized by the 4-BF16-per-lane packing.

    const int my_m = m_offset + n_lane;
    const bool m_valid = (my_m < m_dim);

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 1
    for (int kc = 0; kc < b_dim; kc += K_INNER) {
        bf16x4 a_reg;
        bf16x4 b_reg;

        // A operand: 4 contiguous K-elements (k = kc + strip*4 + r)
        int a_k_base = kc + strip * 4;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_a = a_k_base + r;
            double a_f64 = 0.0;
            if (m_valid && k_a < b_dim) {
                a_f64 = d_err_block[(long long)my_m * (long long)b_dim + (long long)k_a];
            }
            a_reg[r] = f64_to_bf16(a_f64);
        }

        // B operand: 4 contiguous K-elements from LDS
        //   lds_b[n_lane * b_dim + (kc + strip*4 + r)]  (already BF16-cast at load)
        int k_b_base = kc + strip * 4;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_b = k_b_base + r;
            if (k_b < b_dim) {
                b_reg[r] = lds_b[n_lane * b_dim + k_b];
            } else {
                b_reg[r] = (bf16_t)0;
            }
        }

        c_acc = mfma_bf16(a_reg, b_reg, c_acc);
    }

    // ─── Output write-back: subtract-accumulate into W_residual (F64) ───
    //
    // lane l holds c_acc[r] = D[m = m_offset + (l/16)*4 + r, n = n_offset_in_tile + (l%16)]
    //
    // gm = m_offset + strip * 4 + r
    // gn_remaining = n_offset_in_tile + n_lane
    // output_col_orig = d_perm[block_end + gn_remaining]
    //
    // W_residual[gm, output_col_orig] -= (double)c_acc[r]
    //
    // F32 → F64 cast at store is exact (no rounding). The F64 W buffer is
    // the canonical residual state — only the MFMA contraction precision
    // changes vs the F64 sibling.
    //
    // Each (gm, output_col_orig) cell is written by exactly ONE lane on
    // ONE WG (each WG owns a unique (m_tile, n_tile); each wave owns
    // disjoint m-rows within the WG). No atomic needed.

    if (n_offset_in_tile + n_lane >= n_remaining) return;
    int output_col_orig = d_perm[block_end + n_offset_in_tile + n_lane];

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        if (gm < m_dim) {
            d_w_residual[(long long)gm * (long long)k_dim + (long long)output_col_orig]
                -= static_cast<double>(c_acc[r]);
        }
    }
}
