// GPTQ Phase 2B — within-block rank-1 update kernel.
//
// See docs/plans/gptq-hip-phase2-obs-kernel.md §3 Kernel 2.
//
// Applies the OBS rank-1 update for ONE column step to the remaining
// (block_end - step - 1) columns inside the current block. This is the
// inner loop at gptq.rs:907-914 restricted to next_step in (step+1)..block_end.
//
// Per-row formula (scalar, FP64):
//   for next_step in (step+1)..block_end:
//       kk_orig = perm[next_step]
//       u_sn    = U[step, next_step]
//       err     = err_block[row, step - block_start]   // err_col_slot
//       if err != 0 and u_sn != 0:
//           W_residual[row, kk_orig] -= err * u_sn
//
// Launch geometry:
//   grid  = ((M + 63) / 64, block_end - step - 1, 1)
//   block = (64, 1, 1)
//
// dim-y iterates over the remaining columns inside the block.
// dim-x iterates over rows. At B=128, max dim-y = 127, max dim-x blocks
// = ceil(M/64). For M=4096: 64 × 127 × 64 = ~520K threads — fine.
//
// U memory layout: column-major in the device buffer (per Phase D's
// `dgeam` transpose at gptq_hip.rs:246-251 — `dgeam(transA=TRANSPOSE,
// transB=NONE, ldc=K)` writes the upper-tri factor into d_u with leading
// dimension K, so `d_u[col*K + row]` is the column-major access pattern).
// To read U[step, next_step] (which has row=step, col=next_step,
// upper-tri so row<=col), the address is d_u[next_step * K + step].
//
// W_residual update is in-place: each (row, next_step) cell is written
// by exactly ONE thread (one row × one dim-y) so there are NO aliasing
// hazards.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64)
void gptq_obs_within_block_rank1_f64(
    double*       __restrict__ d_w_residual,    // [M*K] row-major
    const double* __restrict__ d_err_block,     // [M*B] row-major
    const double* __restrict__ d_u,             // [K*K] COLUMN-major (post-dgeam)
    const int*    __restrict__ d_perm,          // [K] permutation (i32)
    int step,                                    // current column step
    int block_start,                             // start of current block (= floor(step / B) * B)
    int block_end,                               // end of current block (exclusive)
    int k_dim,                                   // K
    int m_dim,                                   // M
    int b_dim                                    // B
) {
    int row  = blockIdx.x * 64 + threadIdx.x;
    int slot = blockIdx.y;                      // 0-based remaining-column index within block
    int next_step = step + 1 + slot;
    if (next_step >= block_end) return;
    if (row >= m_dim) return;

    // U[step, next_step] — column-major: d_u[next_step * K + step].
    double u_sn = d_u[(long long)step * (long long)k_dim + (long long)next_step];
    if (u_sn == 0.0) return;

    int err_slot = step - block_start;
    double err = d_err_block[(long long)row * (long long)b_dim + (long long)err_slot];
    if (err == 0.0) return;

    int kk_orig = d_perm[next_step];
    d_w_residual[(long long)row * (long long)k_dim + (long long)kk_orig] -= err * u_sn;
}
