#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ4-G256 batched single-output GEMM with fused residual add for
// gfx1030 (RDNA2). Sister of gemm_hfq4g256_residual_wmma.hip; same
// contract + residual semantics (Y += A·x, caller initializes Y with
// residual stream). 136 B/group stride, 4-bit nibble unpack. See
// gemm_gate_up_hfq4g256.gfx1030.hip for the template comments.
// fdot2 inner product for 2× FP16 FMA throughput on gfx1030+.

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g256_residual_gfx1030(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int my_lds_row = tid & 15;
    int safe_row = row_start + my_lds_row;
    if (safe_row >= M) safe_row = M - 1;
    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < batch_size) ? my_col_global : 0;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_col * K;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;

            if ((tid >> 4) == 0) {
                unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
                unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);
                #define DQ(i, pk, sh) lds_a[my_lds_row][i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
                DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
                DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
                DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
                DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
                #undef DQ
            }
            __syncthreads();

            half2_t x_pairs[8];
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                x_pairs[p] = *(const half2_t*)(xg + k_off + p * 2);
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row_in_tile = 2 * j + (tid >> 4);
                const _Float16* a_row = &lds_a[out_row_in_tile][0];
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half2_t a_pair = *(const half2_t*)(a_row + p * 2);
                    acc[j] = __builtin_amdgcn_fdot2(a_pair, x_pairs[p], acc[j], false);
                }
            }
            __syncthreads();
        }
    }

    const int out_col = batch_start + my_col_in_tile;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}
