#include <hip/hip_runtime.h>

// Index-aware fused MoE down GEMV (Q8_0 weights) with scaled residual.
// Q8_0 block layout (34 bytes / block): f16 scale + 32 int8 quants.
//
// One launch replaces the 8 per-expert scaled gemv_q8_0 + scaled_add launches
// from the per-expert serialized MoE down loop. Grid is [M, k_top]; each
// workgroup handles one (row, krank) pair, reads the expert id from
// `topk_indices[krank]`, the weight base from `expert_ptrs[expert_id]`,
// and the per-expert hidden from `hidden_batch[krank × K..]`.
//
// Accumulates `topk_weights[krank] * (W_down_expert[row] · hidden_batch[krank])`
// into `x_residual[row]` via atomicAdd. Caller must fold any per-expert
// scaling (`per_expert_scale_host[e]`) into `topk_weights[krank]` before
// upload; mirrors the HFQ4 indexed-down convention.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_q8_0_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_exp] device pointers
    const int*   __restrict__ topk_indices,               // [k_top]
    const float* __restrict__ topk_weights,               // [k_top] (already folded with per_expert_scale)
    const float* __restrict__ hidden_batch,               // [k_top × K]
    float*       __restrict__ x_residual,                 // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const unsigned char* __restrict__ A =
        reinterpret_cast<const unsigned char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = hidden_batch + (long long)krank * K;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (long long)row * blocks_per_row * 34;

    float sum = 0.0f;

    // 8 Q8_0 blocks (256 elements) per outer iter — same unroll as gemv_q8_0
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Tail blocks (when blocks_per_row % 8 != 0)
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * sum);
    }
}
