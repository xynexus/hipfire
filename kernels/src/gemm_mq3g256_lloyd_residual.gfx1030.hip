#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ3-Lloyd-G256 batched single-output GEMM with fused residual add for
// gfx1030 (RDNA2). Sister of gemm_mq3g256_lloyd_residual_wmma.hip; same
// contract, byte-identical kernarg layout, same per-lane output layout
// (acc[j] = C[2*j + (tid>>4)][tid & 15]), and same residual semantics
// (Y += acc; caller initializes Y with residual stream). No WMMA —
// scalar FP32 FMA via LDS-shared dequantized A.
//
// See gemm_gate_up_mq3g256_lloyd.gfx1030.hip for the full template
// commentary. Differences from gate_up: single output projection (no row
// routing), writeback uses += instead of = (residual fusion).

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq3g256_lloyd_residual_wmma(
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
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_col * K;

    // Cooperative-load mapping (32 threads × 4 fp16 = 128 entries per group).
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 4;    // 0 or 4
    int load_a_row = row_start + load_tile_row;
    if (load_a_row >= M) load_a_row = M - 1;
    const char* load_row_base = A + (long long)load_a_row * row_stride;

    __shared__ _Float16 lds_a[16][16];
    __shared__ _Float16 cb_lds[16 * 8];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 112);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cb_lds[load_tile_row * 8 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 112;
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 16);
        const int cb_base = my_lds_row * 8;

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;

            if ((tid >> 4) == 0) {
                const unsigned char* dpkt = dp + kt * 6;
                unsigned int a0 = dpkt[0], a1 = dpkt[1], a2 = dpkt[2];
                unsigned int a3 = dpkt[3], a4 = dpkt[4], a5 = dpkt[5];
                unsigned int q0  = a0 & 7u;
                unsigned int q1  = (a0 >> 3) & 7u;
                unsigned int q2  = ((a0 >> 6) | (a1 << 2)) & 7u;
                unsigned int q3  = (a1 >> 1) & 7u;
                unsigned int q4  = (a1 >> 4) & 7u;
                unsigned int q5  = ((a1 >> 7) | (a2 << 1)) & 7u;
                unsigned int q6  = (a2 >> 2) & 7u;
                unsigned int q7  = (a2 >> 5) & 7u;
                unsigned int q8  = a3 & 7u;
                unsigned int q9  = (a3 >> 3) & 7u;
                unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
                unsigned int q11 = (a4 >> 1) & 7u;
                unsigned int q12 = (a4 >> 4) & 7u;
                unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
                unsigned int q14 = (a5 >> 2) & 7u;
                unsigned int q15 = (a5 >> 5) & 7u;
                #define DQ(i, q) lds_a[my_lds_row][i] = cb_lds[cb_base + (q)]
                DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
                DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
                DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
                DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);
                #undef DQ
            }
            __syncthreads();

            _Float16 x_local[16];
            #pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                x_local[kk] = xg[k_off + kk];
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row_in_tile = 2 * j + (tid >> 4);
                #pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    acc[j] += (float)(lds_a[out_row_in_tile][kk])
                            * (float)(x_local[kk]);
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
                // Residual fusion: caller initialized Y with residual stream.
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}
