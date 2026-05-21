// GPTQ Phase 2C — block-apply MFMA F64 GEMM kernel for the cross-block
// OBS propagation step. The BLAS-3 payoff of the paper §3.2 block variant.
//
// See docs/plans/gptq-hip-phase2-obs-kernel.md §3 Kernel 3.
//
// Computes (subtract-accumulate, in place):
//
//   W_residual[:, perm[block_end..K]] -= err_block @ U_submatrix
//
// where:
//   err_block     : M × B   FP64 (B=128, accumulated errors from current block)
//   U_submatrix   : B × N_remaining FP64; logically
//                     U_submatrix[i, j] = U[block_start + i, block_end + j]
//   N_remaining   = K - block_end
//
// And the output write is permuted:
//   W_residual[m, perm[block_end + j]] -= sum_{i in [0,B)} err_block[m, i] * U_submatrix[i, j]
//
// ── MFMA tile geometry ──
//
//   MFMA intrinsic   : __builtin_amdgcn_mfma_f64_16x16x4f64
//                      (16×16 FP64 output, K_inner=4 per call)
//
//   Per-wave output  : 16×16 (256 elements / 64 lanes = 4 doubles per lane,
//                      strip-major: lane l holds C[m=(l/16)*4+r, n=l%16])
//
//   Per-WG output    : 32 M × 16 N tile (2 waves stacked in M)
//
//   Grid             : ((M + 31) / 32, (N_remaining + 15) / 16, 1)
//   Block            : (128, 1, 1)  — 2 wave64 / WG
//
//   LDS B-tile       : 16 N-cols × B K-rows of U_submatrix, loaded once
//                      per WG. Size: 16 × 128 × 8 B = 16 KB per WG.
//                      gfx942 has 64 KB LDS — comfortably fits with
//                      2 WG / CU occupancy (32 KB used, 32 KB free).
//
// ── MFMA inner-loop lane layout ──
//
//   A operand (FP64, 1 double per lane):
//     lane l holds a = A[m = l%16, k = l/16]   (l/16 ∈ {0,1,2,3})
//     → in our case: a = err_block[my_m, kc + (l/16)] where
//       my_m = m_offset + (l%16) for this wave
//
//   B operand (FP64, 1 double per lane):
//     lane l holds b = B[k = l/16, n = l%16]
//     → b = U_submatrix[kc + (l/16), n_offset_in_tile + (l%16)]
//     → from LDS: lds_b[(n_offset_in_tile + l%16) * B + (kc + l/16)]
//     (layout matched to write pattern below)
//
//   D output (FP64, 4 doubles per lane):
//     lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]  for r ∈ [0,4)
//
// ── Output write ──
//
//   For wave w with m_off ∈ {0, 16}:
//     gm = m_offset + (l/16)*4 + r       (4 m-rows per lane)
//     gn = n_offset_in_tile + (l%16)     (16 n-cols spread across lanes)
//     output_col_orig = perm[block_end + gn]
//     W_residual[gm, output_col_orig] -= c_acc[r]
//
//   Each (gm, gn) cell is written by exactly one lane in one wave —
//   no inter-WG races (each WG owns a unique (m_tile, n_tile)) and
//   no intra-WG races (each wave owns disjoint M-rows).
//
// ── Reference: mirrors gemm_hfq4g256_residual_mfma_v3.gfx942.hip
// (LDS-tile pattern) and gemm_bf16_mfma.gfx942.hip (4-double output
// layout, here with mfma_f64_16x16x4 instead of mfma_f32_16x16x16bf16_1k).

#include <hip/hip_runtime.h>

typedef double  vdouble4 __attribute__((ext_vector_type(4)));

#define TILE_M    32
#define TILE_N    16
#define K_INNER   4
#define BLOCK_B 128

extern "C" __global__ __launch_bounds__(128)
void gptq_obs_block_apply_mfma_f64(
    double*       __restrict__ d_w_residual,    // [M*K] row-major
    const double* __restrict__ d_err_block,     // [M*B] row-major
    const double* __restrict__ d_u,             // [K*K] COLUMN-major (post-dgeam)
    const int*    __restrict__ d_perm,          // [K]
    int block_start,                             // start of current block
    int block_end,                               // end of current block (exclusive)
    int k_dim,                                   // K
    int m_dim,                                   // M
    int b_dim,                                   // B (= block_end - block_start, max 128)
    int n_remaining                              // = K - block_end
) {
    __shared__ double lds_b[TILE_N * BLOCK_B];   // 16 × 128 = 2048 doubles = 16 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..1
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;       // 0..15 (n-col within tile)
    const int strip     = lane_in_w >> 4;       // 0..3 (k-inner / 4 within MFMA chunk)

    const int wave_m_off = (wave_id == 0) ? 0 : 16;
    const int m_offset   = blockIdx.x * TILE_M + wave_m_off;   // start of this wave's m-tile
    const int n_offset_in_tile = blockIdx.y * TILE_N;          // start of this WG's n-tile (in n_remaining frame)

    // ─── Cooperative LDS load of B-tile (full 16 × B doubles) ───
    //
    // B-tile layout: lds_b[n_in_tile * b_dim + k] = U_submatrix[k, n_in_tile]
    //              = U[block_start + k, block_end + (n_offset_in_tile + n_in_tile)]
    //              = d_u[d_perm... no, U_submatrix is indexed by ORIGINAL u-col,
    //                       which is `block_end + n_offset_in_tile + n_in_tile`,
    //                       NOT permuted. d_perm only enters the write side.]
    //              = d_u[(block_end + n_offset_in_tile + n_in_tile) * K
    //                     + (block_start + k)]   (column-major)
    //
    // 128 threads load 16 × b_dim doubles. For B=128: 128*16 = 2048 = 16/thread.
    __syncthreads();
    int total_lds = TILE_N * b_dim;
    int per_thread = (total_lds + 127) / 128;
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
                v = d_u[(long long)gk_u * (long long)k_dim + (long long)gn_u];
            }
        }
        lds_b[n_in_tile * b_dim + k_idx] = v;
    }
    __syncthreads();
    (void)per_thread;

    // ─── MFMA inner loop: 32 MFMA calls for B=128, K_INNER=4 per call ───
    //
    // A operand: 1 double per lane = err_block[my_m, kc + strip]
    //   where my_m = m_offset + n_lane (for this wave)
    // B operand: 1 double per lane = lds_b[(n_offset_in_tile + n_lane) * b_dim + (kc + strip)]
    //   but n_offset_in_tile is the WG-level n offset; the LDS B layout is
    //   indexed [n_in_tile, k] so the access is lds_b[n_lane * b_dim + (kc + strip)].
    // D output: 4 doubles per lane = c_acc[r] for r in [0,4)
    //   spanning D[m = (l/16)*4 + r, n = l%16]

    const int my_m = m_offset + n_lane;
    const bool m_valid = (my_m < m_dim);

    vdouble4 c_acc = {0.0, 0.0, 0.0, 0.0};

    #pragma unroll 1
    for (int kc = 0; kc < b_dim; kc += K_INNER) {
        // Bounds-check on k inner increment (guard against b_dim < BLOCK_B at last block of K).
        int k_a = kc + strip;
        double a = 0.0;
        if (m_valid && k_a < b_dim) {
            a = d_err_block[(long long)my_m * (long long)b_dim + (long long)k_a];
        }

        // B from LDS — lane l holds b = U_submatrix[kc + strip, n_lane]
        // LDS layout: lds_b[n_in_tile * b_dim + k_idx] = U_submatrix[k_idx, n_in_tile]
        // So b = lds_b[n_lane * b_dim + (kc + strip)]
        int k_b = kc + strip;
        double b = 0.0;
        if (k_b < b_dim) {
            b = lds_b[n_lane * b_dim + k_b];
        }

        c_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c_acc, 0, 0, 0);
    }

    // ─── Output write-back: subtract-accumulate into W_residual ───
    //
    // lane l holds c_acc[r] = D[m = m_offset + (l/16)*4 + r, n = n_offset_in_tile + (l%16)]
    //                       = c_acc per Φ above
    //
    // gm = m_offset + strip * 4 + r
    // gn_remaining = n_offset_in_tile + n_lane
    // output_col_orig = d_perm[block_end + gn_remaining]
    //
    // W_residual[gm, output_col_orig] -= c_acc[r]
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
            d_w_residual[(long long)gm * (long long)k_dim + (long long)output_col_orig] -= c_acc[r];
        }
    }
}
