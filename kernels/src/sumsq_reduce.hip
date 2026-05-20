#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Per-channel sum-of-squares accumulator for AWQ scale derivation.
//
// Computes `acc[k] += Σ_t x[t,k]²` over a tensor of BF16 activations.
// Designed to be called many times (one per activation tile / forward
// step) — `acc` accumulates across calls via `atomicAdd`, and a single
// thread per WG bumps `n_tokens` by `batch` so the caller can divide
// by the running token count at the end.
//
// Foundation of the Tier 1 hipfire-native AWQ calibration path that
// replaces the `scripts/collect_hessian.py` + `llama-imatrix` pair.
// Pairs with `gemm_bf16_mfma.gfx942.hip` (the foundation forward GEMM)
// and `hessian_outer_product.gfx942.hip` (the rank-1 H update on the
// same activations).
//
// Layout:
//   x:          [batch, K]  BF16 row-major   (activations of a layer's input)
//   acc:        [K]         FP32             (per-channel Σx² accumulator)
//   n_tokens:   [1]         FP32 scalar      (running token count)
//
// Geometry:
//   block = [256, 1, 1]                 256 threads cover the batch dim
//   grid  = [K, 1, 1]                   one WG owns one output channel
//   LDS   = 256 × 4 B = 1024 B          single-pass tree reduce
//
// Why one WG per channel (not one per batch row + reduce-across-WGs):
// - Each WG owns a unique `acc[k]` cell, so the inter-WG accumulator
//   write is a single `atomicAdd(&acc[k], partial)` — no contention
//   across WGs on the same address.
// - Column-major access per WG (stride K through HBM) is the load
//   pattern. K is typically ≥ 4096 on Qwen-class models so the strided
//   gather still saturates HBM (no need to bottleneck on row-major
//   reads with cross-WG reduction trees).
// - Atomic semantics let the caller invoke this kernel repeatedly
//   (once per token tile / forward step) without re-zeroing between
//   calls — the caller zeros `acc` + `n_tokens` once at the start of
//   a calibration pass, then dispatches `sumsq_reduce_bf16` N times,
//   then divides at the end.
//
// Numeric note: BF16 has 7 mantissa bits, so each `x² → f32` upconvert
// is well within representable range (BF16 squared maxes at ~3e38 which
// stays inside FP32). Accumulation is in FP32, not BF16, so the
// `K_BATCH × N_CALLS`-element sum doesn't lose precision (the FP32
// accumulator has 24-bit mantissa). Tested for relative error < 1e-3
// vs CPU double-precision reference in `test_sumsq_reduce.rs`.

typedef __bf16 bf16_t;

extern "C" __global__ __launch_bounds__(256)
void sumsq_reduce_bf16(
    const bf16_t* __restrict__ x,        // [batch, K] BF16 row-major
    float*        __restrict__ acc,      // [K] FP32 (atomicAdd-accumulated)
    float*        __restrict__ n_tokens, // [1] FP32 scalar (atomicAdd-accumulated)
    int batch,
    int K
) {
    __shared__ float lds_sum[256];

    const int k   = blockIdx.x;      // 0..K-1: this WG's output channel
    const int tid = threadIdx.x;     // 0..255

    if (k >= K) return;

    // ─── Each thread sums Σ x[t, k]² over its slice of the batch dim ───
    // Stride-256 stripe across `batch` (handles batch > 256 cleanly).
    float local = 0.0f;
    for (int t = tid; t < batch; t += 256) {
        // x[t, k] = x[(long long)t * K + k]
        // BF16 → FP32 upconvert is implicit via the `__bf16 → float` cast
        // (LLVM emits a 16-bit shift + bias).
        float v = (float) x[(long long)t * K + k];
        local += v * v;
    }
    lds_sum[tid] = local;
    __syncthreads();

    // ─── LDS tree reduce: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 ───
    // Pure butterfly; no warp-shuffle (kernel is cross-arch — gfx10/11/12
    // wave32 and gfx9/942 wave64 take the same path).
    #pragma unroll
    for (int s = 128; s >= 1; s >>= 1) {
        if (tid < s) {
            lds_sum[tid] += lds_sum[tid + s];
        }
        __syncthreads();
    }

    // ─── Thread 0 publishes the channel's partial + bumps n_tokens once ───
    if (tid == 0) {
        atomicAdd(&acc[k], lds_sum[0]);
        // Only the first WG (channel 0) updates the token counter — every
        // WG sees the same `batch`, so adding once is sufficient.
        if (k == 0) {
            atomicAdd(&n_tokens[0], (float) batch);
        }
    }
}
