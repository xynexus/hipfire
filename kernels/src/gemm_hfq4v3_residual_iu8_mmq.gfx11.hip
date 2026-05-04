#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// HFQ4v3 — gfx11/RDNA3-native K=64 4-bit weight format with FP16 (d, m)
// per-group metadata. Sister kernel of `gemm_hfq4g256_residual_mmq.hip`.
//
// Per-block layout (K=64 weights):
//   bytes  0.. 1: half d         (FP16 scale)
//   bytes  2.. 3: half m         (FP16 zero/min)
//   bytes  4..35: 32 bytes nibbles (64 weights packed 2/byte, lo nibble first)
// Total: 36 bytes per K=64 group → 4.5 b/w (vs 4.25 b/w for HFQ4-G256).
//
// Why K=64:
//   - K=16 iu8 wmma → 4 wmma calls per K=64 group → enough reuse to
//     amortize the metadata read.
//   - K=256 over-amortizes; one (d, m) per 16 wmma calls drives metadata
//     so cold the BW gain is theoretical only and VGPRs balloon.
//   - K=32 wastes a meta read every 2 wmma calls.
//
// Why FP16 (d, m):
//   - gfx11 wmma accumulates FP32; the half→float cast per tile is free.
//   - Halves metadata BW vs HFQ4-G256's FP32 (d, m).
//   - llama.cpp Q4_K-style quants ship FP16 scales without quality loss.
//
// Activation pairing: same `block_q8_1_mmq` Q8_1 layout as HFQ4-G256 MMQ.
// The Q8_1 quantizer kernel from the v1 MMQ kernel is reused as-is.
//
// Tile + outer-loop structure: identical to v1's MMQ kernel — 128 output
// rows × 128 batch columns per workgroup, 8 warps × 32 threads, i8 WMMA
// over K=16 sub-tiles, outer step K=256. Each outer step covers 4 v3
// K=64 groups (32 nibble bytes × 4 = 128 bytes total nibble run, matching
// v1's HFQ4-G256 group size for the lane→source mapping). Per-K=32 Q8_1
// sub-block one v3 group's (d, m) is replicated into 4 LDS slots so the
// inner dot kernel reads x_dm[i * MMQ_TILE_X_K + 4*ksc + l] with ksc
// running 0..7 across the 8 sub-blocks per outer step.

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define HIPFIRE_RDNA3 1
#else
#define HIPFIRE_RDNA3 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32
#define QK8_1 32
#define QI8_1 8
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K 36
#define MMQ_TILE_X_K 76

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

#if !HIPFIRE_RDNA3

extern "C" __global__ void gemm_hfq4v3_residual_iu8_mmq(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4v3_residual_iu8_mmq_full_add(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4v3_residual_iu8_mmq_full_set(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

#else

template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};

template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
#if HIPFIRE_RDNA3
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
#else
        return ne * (threadIdx.x / 16) + l;
#endif
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return l;
    }
};

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride) {
#if HIPFIRE_RDNA3
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
#else
    static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad layout");
    static_assert(sizeof(t.x) == 16, "bad tile size");
    memcpy_aligned<16>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
#endif
}

template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}

// Load one K=256 outer step of HFQ4v3 weights into LDS for `MMQ_Y` rows.
//
// One outer step = 4 v3 K=64 groups × 36 B/group = 144 B per row, of which
//   - 4 × 4 B = 16 B is metadata (4 (d, m) FP16 pairs)
//   - 4 × 32 B = 128 B is nibbles
// The 128 B nibble run matches v1's HFQ4-G256 group exactly, so the
// per-lane unpack pattern (lane txi reads u32 at offset txi*4 within
// the nibble run, splits into 8 nibbles, writes 2 output ints) is
// reused unchanged.
//
// Because v3 places metadata at the START of each 36-byte group and
// nibbles at offset 4..35, we cannot simply read a contiguous 128-byte
// nibble run from a single base pointer — the nibbles are interleaved
// with metadata on disk. The lane txi belongs to v3 group `txi / 8`
// (0..3 covers all 32 lanes) and reads nibble u32 at index `txi % 8`
// within that group's 32-byte nibble run.
//
// `kbig` is the K=256 outer-step index (0..K/256). v3 group indices for
// this step are `4*kbig + 0..3`.
template <bool FULL>
static __device__ __forceinline__ void load_hfq4v3_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kbig,
    int M,
    int groups64_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;
    const int g_local = txi / 8;     // 0..3 → which v3 group within this outer step
    const int g_lane  = txi % 8;     // 0..7 → which u32 within group's 32-B nibble run

    // ── nibble unpack (4 v3 groups × 32 B = 128 B per row, every lane active)
    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const int kb_g = 4 * kbig + g_local;
        if (kb_g < groups64_per_row) {
            const char* gp = A + ((long long)row * groups64_per_row + kb_g) * 36;
            // gp[0..1]=half d, gp[2..3]=half m, gp[4..35]=nibbles
            const unsigned int qs0 = *(const unsigned int*)(gp + 4 + g_lane * 4);
            const unsigned int q0 = (qs0 >>  0) & 0xFu;
            const unsigned int q1 = (qs0 >>  4) & 0xFu;
            const unsigned int q2 = (qs0 >>  8) & 0xFu;
            const unsigned int q3 = (qs0 >> 12) & 0xFu;
            const unsigned int q4 = (qs0 >> 16) & 0xFu;
            const unsigned int q5 = (qs0 >> 20) & 0xFu;
            const unsigned int q6 = (qs0 >> 24) & 0xFu;
            const unsigned int q7 = (qs0 >> 28) & 0xFu;
            x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
            x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24);
        } else {
            x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = 0;
            x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = 0;
        }
    }

    // ── metadata load: x_dm has 8 active slots per row indexed by k0/8
    //    (k0 ∈ [0..63] → indices 0..7). The inner dot reads
    //    `x_dm[i * MMQ_TILE_X_K + k0/QI8_1]`. Each v3 group spans K=64
    //    = 2 Q8_1 sub-blocks, so v3 group `gidx` covers x_dm slots
    //    [2*gidx, 2*gidx+1] (two slots, both holding the same (d, m)).
    //
    //    Layout per outer K=256 step:
    //       x_dm[0..1] = v3 group 0 (d, m) (K=0..63)
    //       x_dm[2..3] = v3 group 1 (d, m) (K=64..127)
    //       x_dm[4..5] = v3 group 2 (d, m) (K=128..191)
    //       x_dm[6..7] = v3 group 3 (d, m) (K=192..255)
    //
    //    Lane assignment: 16 rows per warp via threadIdx.x/2 (row id),
    //    threadIdx.x & 1 selects the within-group slot parity (0 or 1).
    //    Per (i, ksc_parity), iterate gidx 0..3 and write one slot each.
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const int ksc_parity = threadIdx.x & 1;  // 0 or 1: slot within v3 group
            #pragma unroll
            for (int gidx = 0; gidx < 4; ++gidx) {
                const int kb_g = 4 * kbig + gidx;
                half d_h, m_h;
                if (kb_g < groups64_per_row) {
                    const char* gp = A + ((long long)row * groups64_per_row + kb_g) * 36;
                    d_h = *(const half*)(gp + 0);
                    m_h = *(const half*)(gp + 2);
                } else {
                    d_h = (half)0.0f;
                    m_h = (half)0.0f;
                }
                const half2 dm = make_half2(d_h, m_h);
                // Write one of the two slots this v3 group occupies.
                x_dm[i * MMQ_TILE_X_K + 2 * gidx + ksc_parity] = dm;
            }
        }
    }
}

static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma_v3(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        tile_A A_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);

                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4v3_residual_iu8_mmq(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups64_per_row = K / 64;
    const int outer_steps = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kbig = 0; kbig < outer_steps; ++kbig) {
        load_hfq4v3_tile<false>(A, tile_x, row0, kbig, M, groups64_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kbig) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_v3(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kbig + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kbig + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_v3(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_hfq4v3_residual_iu8_mmq_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;

    const int groups64_per_row = K / 64;
    const int outer_steps = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kbig = 0; kbig < outer_steps; ++kbig) {
        load_hfq4v3_tile<true>(A, tile_x, row0, kbig, M, groups64_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kbig) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma_v3(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kbig + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kbig + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma_v3(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                if constexpr (ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4v3_residual_iu8_mmq_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4v3_residual_iu8_mmq_full_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4v3_residual_iu8_mmq_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4v3_residual_iu8_mmq_full_body<false>(A, Xq, Y, M, K, N);
}

#endif
