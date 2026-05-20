// GPTQ Phase 2A — quantize one column of W_residual to MQ4 and write the
// per-row error into a column slot of the d_err_block accumulator.
//
// See docs/plans/gptq-hip-phase2-obs-kernel.md §3 Kernel 1.
//
// M-parallel single-column quantize-and-error kernel. Mirrors the per-row
// work at gptq.rs:869-890 (Phase A of the column-sequential GPTQ loop).
//
// One thread per M-row. Trivial parallelism. Launch geometry:
//   grid = ((M + 255) / 256, 1, 1)
//   block = (256, 1, 1)
//
// Layout contract:
//   d_w_residual : double[M*K] row-major (current residual weights)
//   d_w_quant    : double[M*K] row-major (output quantized weight buffer)
//   d_grids      : packed_grid_t[(M*K)/256] — each 16 bytes: {f32 scale,
//                  f32 min_val, char _pad[8]}. Matches Rust BlockGrid layout
//                  AS PADDED IN THE H2D BUFFER (see gptq_hip.rs).
//
// Per-element formula (mirrors gptq.rs:92-101 quantize_mq4_element_with_clamp,
// FP64 throughout):
//   q_raw    = floor((w - min_val) / scale + 0.5)
//   q_clamp  = clamp(q_raw, 0, 15)
//   q        = q_clamp * scale + min_val
//   err      = (w - q) / u_ss
//
// Note: clamp-diagnostic counters are omitted in this kernel. The CPU
// path's [gptq-clamp] line is generated post-D2H from a separate scan
// in the Rust orchestrator (see Phase 2E plan).

#include <hip/hip_runtime.h>

// Match Rust H2D layout: each grid entry is 16 bytes, scale + min_val + pad.
struct packed_grid_t {
    float scale;
    float min_val;
    char  pad[8];
};

extern "C" __global__ __launch_bounds__(256)
void gptq_quantize_mq4_column_f64(
    const double* __restrict__ d_w_residual,  // [M*K] row-major
    double*       __restrict__ d_w_quant,     // [M*K] row-major (output)
    const packed_grid_t* __restrict__ d_grids,// [(M*K)/256]
    double*       __restrict__ d_err_block,   // [M*B] err accumulator
    int col_orig,                              // un-permuted column index j_orig
    double u_ss,                               // U[step, step] divisor
    int k_dim,                                 // K
    int m_dim,                                 // M
    int err_col_slot,                          // = step - block_start
    int b_dim                                  // = B (128 in production)
) {
    int row = blockIdx.x * 256 + threadIdx.x;
    if (row >= m_dim) return;

    // Block index for the frozen grid: mirrors block_idx_for() in gptq.rs:725.
    // Row-major flat index → block: (row * K + col_orig) / 256.
    long long flat   = (long long)row * (long long)k_dim + (long long)col_orig;
    long long blk_id = flat / 256LL;

    double scale   = (double)d_grids[blk_id].scale;
    double min_val = (double)d_grids[blk_id].min_val;

    double w = d_w_residual[flat];

    // Mirror gptq.rs:92-101 quantize_mq4_element_with_clamp (FP64).
    double q;
    if (scale == 0.0) {
        q = min_val;
    } else {
        double inv_scale = 1.0 / scale;
        double q_raw     = floor((w - min_val) * inv_scale + 0.5);
        // clamp(q_raw, 0.0, 15.0)
        double q_clamp = q_raw < 0.0  ? 0.0  : q_raw;
        if (q_clamp > 15.0) q_clamp = 15.0;
        q = q_clamp * scale + min_val;
    }

    double err;
    if (u_ss == 0.0) {
        err = 0.0;
    } else {
        err = (w - q) / u_ss;
    }

    d_w_quant[flat]                          = q;
    d_err_block[(long long)row * (long long)b_dim + (long long)err_col_slot] = err;
}
