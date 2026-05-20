#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// BF16 × BF16 → FP32 Hessian outer-product accumulator for gfx942 (MI300X
// CDNA3).
//
// Computes `H += xᵀ x` over a tile of BF16 activations:
//
//   H[i, j] += Σ_t x[t, i] · x[t, j]                                (FP32)
//
// Designed to be called repeatedly across a calibration corpus —
// `H` is FP32 in HBM and accumulated via direct write (each WG owns
// a unique 32×32 sub-tile of `H` per call, so no atomics across WGs;
// the accumulation across calls is layered on by reading `H` at start
// and writing back at end).
//
// This is the foundation of the Tier 1 hipfire-native **GPTQ** Hessian
// collection path. Pairs with `sumsq_reduce.hip` (the AWQ side, per-channel
// Σx²) and `gemm_bf16_mfma.gfx942.hip` (the forward GEMM whose activations
// are the input here).
//
// Layout:
//   x:  [batch, K] BF16 row-major  (activations of a layer's input)
//   H:  [K, K]     FP32 row-major  (Hessian accumulator, += per call)
//
// MFMA mapping (`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`):
//
//   A operand  ≡ x viewed as [K_m, batch]    (16 m-rows × 16 contraction k)
//   B operand  ≡ x viewed as [batch, K_n]    (16 contraction k × 16 n-cols)
//   D output   ≡ H[K_m, K_n]                 (16 × 16 FP32 sub-tile)
//
// Both operands are loaded from the SAME `x` tensor but with different
// strides:
//   A:  for lane l, k_lane = (l/16)*4+r, m_lane = l%16
//       → x[batch_offset + k_lane, m_global = m_offset + m_lane]
//   B:  for lane l, k_lane = (l/16)*4+r, n_lane = l%16
//       → x[batch_offset + k_lane, n_global = n_offset + n_lane]
//
// Lane layout (mirrors `gemm_bf16_mfma.gfx942.hip` exactly):
//
//   A: lane l holds a[r] = A[m = l%16,        k = (l/16)*4 + r]
//   B: lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]
//   D: lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]   ← strip-major
//
// The strip-major output is the same trap as the GEMM (output lane l
// holds D[m=(l/16)*4+r] but input A lane l holds A[m=l%16]).
//
// Geometry (mirror of v3 sweet-spot):
//   block = [256, 1, 1]                 4 wave64/WG, 32×32 output tile per WG
//   grid  = [(K+31)/32, (K+31)/32, 1]   one WG per (m_tile, n_tile)
//   LDS   = 32 × K_CHUNK × 2 B × 2 ops  A-tile + B-tile cooperative load
//   K_CHUNK = 128 (along the batch contraction dim)
//
// Wave-tile arrangement within a WG (4 wave64 → 2×2 wave grid over the
// 32×32 output tile):
//
//   wave_id 0 → output sub-tile (m_off=0,  n_off=0)
//   wave_id 1 → output sub-tile (m_off=0,  n_off=16)
//   wave_id 2 → output sub-tile (m_off=16, n_off=0)
//   wave_id 3 → output sub-tile (m_off=16, n_off=16)
//
// All 4 waves cooperatively load:
//   - A-tile: [32 m-rows × K_CHUNK batch-cols] from x[batch_chunk, m_tile]
//   - B-tile: [32 n-rows × K_CHUNK batch-cols] from x[batch_chunk, n_tile]
//
// (Same shape as v3 — A is now also LDS-staged because the row stride in
// HBM is K, identical to B's, and the per-WG K_CHUNK-sized slab amortizes
// HBM traffic across the 8 MFMA iterations per chunk. v3 only LDS-staged
// B because A there was the WEIGHT matrix with row-stride K and the row
// was contiguous in each WG's m-direction; here our `A operand` is built
// FROM x's K-dim, not its batch-dim, so we need LDS to get the gather
// right.)
//
// Memory cost: 2 × 32 × 128 × 2 B = 16 KB LDS per WG. gfx942 has 64 KB
// LDS per CU and supports multiple resident WGs per CU, so 16 KB leaves
// headroom (one CU could fit ~4 WGs, but at 256 threads/WG you're already
// limited to 4 WGs/CU by warps-per-CU on CDNA3).
//
// Diagonal-tile optimization (deliberately NOT taken):
// `H` is symmetric (`H[i,j] = H[j,i]`), so half the tiles are redundant.
// We still compute the full K×K because (a) MFMA cycles are cheap relative
// to the HBM-bound A/B loads (which would happen either way for the lower
// triangle's transpose), (b) the symmetric merge would need atomic adds
// or a separate pass, breaking the "each WG owns a unique sub-tile" rule
// that lets us write directly without contention. Cost: 2× FLOPs, same
// HBM bandwidth. Quantized accumulator readers (GPTQ Cholesky) can read
// either upper or lower half; we provide both.

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

#define K_CHUNK 128
#define TILE    32

extern "C" __global__ __launch_bounds__(256)
void hessian_outer_product_bf16(
    const bf16_t* __restrict__ x,    // [batch, K] BF16 row-major
    float*        __restrict__ H,    // [K, K] FP32 row-major (accumulator, += semantics)
    int batch,
    int K
) {
    // LDS layout:
    //   lds_a[m_in_tile, k_in_chunk]  — A-tile [32 m-rows × K_CHUNK batch-cols]
    //   lds_b[n_in_tile, k_in_chunk]  — B-tile [32 n-rows × K_CHUNK batch-cols]
    //
    // Both indexed [outer_tile_idx * K_CHUNK + k_in_chunk]. Layout chosen
    // so each wave's MFMA load reads K_CHUNK-contiguous from LDS — minimal
    // bank conflicts on gfx942 (LDS has 32 banks × 4 B; K_CHUNK = 128
    // strides cleanly).
    __shared__ bf16_t lds_a[TILE * K_CHUNK];   // 4 KB
    __shared__ bf16_t lds_b[TILE * K_CHUNK];   // 4 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;       // 0..15
    const int strip     = lane_in_w >> 4;       // 0..3

    const int wave_m_off = (wave_id & 2) ? 16 : 0;
    const int wave_n_off = (wave_id & 1) ? 16 : 0;
    const int m_offset_global = blockIdx.x * TILE;             // start of this WG's m-tile in K
    const int n_offset_global = blockIdx.y * TILE;             // start of this WG's n-tile in K
    const int m_offset = m_offset_global + wave_m_off;         // per-wave m-half
    const int n_offset = n_offset_global + wave_n_off;         // per-wave n-half

    // Note on lane geometry (for the MFMA inner loop below):
    //   A operand lane l holds A[m = l%16,        k = (l/16)*4 + r]
    //                          = x[batch_offset + (l/16)*4+r, K_off + l%16]
    //   B operand lane l holds B[k = (l/16)*4 + r, n = l%16]
    //                          = x[batch_offset + (l/16)*4+r, K_off + l%16]
    //   D output lane l holds D[m = (l/16)*4 + r, n = l%16]  (strip-major)
    // (Both operands gather from the same x but indexed against different
    //  K-tiles m_offset_global vs n_offset_global.)

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Batch contraction loop, in chunks of K_CHUNK
    for (int kc = 0; kc < batch; kc += K_CHUNK) {
        // ─── Cooperative LDS load: A-tile and B-tile ───
        // A-tile = x[batch_chunk, m_tile]  laid out as [m_in_tile, k_in_chunk]
        //        = x[(kc + k_in_chunk), (m_offset_global + m_in_tile)]
        // B-tile = x[batch_chunk, n_tile]  laid out as [n_in_tile, k_in_chunk]
        //        = x[(kc + k_in_chunk), (n_offset_global + n_in_tile)]
        //
        // 256 threads load 32 × 128 = 4096 bf16 each tile = 16 per thread.
        __syncthreads();
        #pragma unroll
        for (int li = 0; li < 16; li++) {
            int idx       = tid + li * 256;
            int tile_idx  = idx / K_CHUNK;             // 0..31  (m-row in tile)
            int k_in_chnk = idx % K_CHUNK;             // 0..K_CHUNK-1 (batch offset)
            int gb        = kc + k_in_chnk;             // global batch row in x

            // A-load: x[gb, m_offset_global + tile_idx]
            int g_m = m_offset_global + tile_idx;
            bf16_t va = (gb < batch && g_m < K)
                ? x[(long long)gb * K + g_m]
                : (bf16_t)0;
            lds_a[tile_idx * K_CHUNK + k_in_chnk] = va;

            // B-load: x[gb, n_offset_global + tile_idx]
            int g_n = n_offset_global + tile_idx;
            bf16_t vb = (gb < batch && g_n < K)
                ? x[(long long)gb * K + g_n]
                : (bf16_t)0;
            lds_b[tile_idx * K_CHUNK + k_in_chnk] = vb;
        }
        __syncthreads();

        // ─── MFMA loop over chunk in 16-element contraction steps ───
        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            // A operand: 4 contiguous K-elements (here K = batch contraction)
            //   For lane l: a[r] = A[m = n_lane, k = strip*4 + r] within the tile
            //   In LDS terms: lds_a[(wave_m_off + n_lane), (kk + strip*4 + r)]
            bf16x4 a_reg;
            int a_row_in_tile = wave_m_off + n_lane;
            int a_k_base      = kk + strip * 4;
            a_reg[0] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 0];
            a_reg[1] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 1];
            a_reg[2] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 2];
            a_reg[3] = lds_a[a_row_in_tile * K_CHUNK + a_k_base + 3];

            // B operand: 4 contiguous K-elements (here K = batch contraction)
            //   For lane l: b[r] = B[k = strip*4 + r, n = n_lane] within the tile
            //   In LDS terms: lds_b[(wave_n_off + n_lane), (kk + strip*4 + r)]
            bf16x4 b_reg;
            int b_row_in_tile = wave_n_off + n_lane;
            int b_k_base      = kk + strip * 4;
            b_reg[0] = lds_b[b_row_in_tile * K_CHUNK + b_k_base + 0];
            b_reg[1] = lds_b[b_row_in_tile * K_CHUNK + b_k_base + 1];
            b_reg[2] = lds_b[b_row_in_tile * K_CHUNK + b_k_base + 2];
            b_reg[3] = lds_b[b_row_in_tile * K_CHUNK + b_k_base + 3];

            c_acc = mfma_bf16(a_reg, b_reg, c_acc);
        }
    }

    // ─── Output write-back with += accumulation ───
    // Lane (strip, n_lane) holds c_acc[r] = D[m = m_offset + strip*4 + r,
    //                                          n = n_offset + n_lane]
    // H is [K, K] row-major: H[i, j] = H[i * K + j]
    //
    // Accumulate (read-modify-write) so multiple calls compose. Each WG
    // owns a unique (m_tile, n_tile), so there are no inter-WG races on
    // H within a single call. Across calls, the kernel runs to completion
    // before the next launch — host serializes via stream, no race.
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gi = m_offset + strip * 4 + r;
        const int gj = n_offset + n_lane;
        if (gi < K && gj < K) {
            H[(long long)gi * K + gj] += c_acc[r];
        }
    }
}
