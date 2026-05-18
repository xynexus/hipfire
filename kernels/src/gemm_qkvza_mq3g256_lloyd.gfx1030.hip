#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ3-Lloyd-G256 batched 4-way fused QKV+Z+Beta+Alpha GEMM for gfx1030
// (RDNA2). Sister of gemm_qkvza_mq3g256_lloyd_wmma.hip; same contract,
// byte-identical kernarg layout, same per-lane output layout
// (acc[j] = C[2*j + (tid>>4)][tid & 15]), without WMMA.
//
// See gemm_gate_up_mq3g256_lloyd.gfx1030.hip for the full template
// commentary. Differences: 4 output projections (qkv / z / beta / alpha),
// row routing across A_qkv / A_z / A_beta / A_alpha by global row range.

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_qkvza_mq3g256_lloyd_wmma(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_lds_row = tid & 15;
    int my_a_row_global = row_start + my_lds_row;
    if (my_a_row_global >= total_m) my_a_row_global = total_m - 1;

    const char* A;
    int local_row;
    if (my_a_row_global < qkv_m) {
        A = A_qkv;   local_row = my_a_row_global;
    } else if (my_a_row_global < qkv_m + z_m) {
        A = A_z;     local_row = my_a_row_global - qkv_m;
    } else if (my_a_row_global < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = my_a_row_global - qkv_m - z_m;
    } else {
        A = A_alpha; local_row = my_a_row_global - qkv_m - z_m - beta_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + local_row * row_stride;

    // Cooperative-load row mapping (each tile-row may live in a different
    // A — load_tile_row spans 0..15 over tid 0..31).
    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 4;
    int load_a_row_global = row_start + load_tile_row;
    if (load_a_row_global >= total_m) load_a_row_global = total_m - 1;
    const char* load_A;
    int load_local_row;
    if (load_a_row_global < qkv_m) {
        load_A = A_qkv;   load_local_row = load_a_row_global;
    } else if (load_a_row_global < qkv_m + z_m) {
        load_A = A_z;     load_local_row = load_a_row_global - qkv_m;
    } else if (load_a_row_global < qkv_m + z_m + beta_m) {
        load_A = A_beta;  load_local_row = load_a_row_global - qkv_m - z_m;
    } else {
        load_A = A_alpha; load_local_row = load_a_row_global - qkv_m - z_m - beta_m;
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

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
        const _Float16* x_base = X + (long long)safe_col * K + g * 256;
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
                x_local[kk] = x_base[k_off + kk];
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
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < qkv_m) {
                    Y = Y_qkv;   proj_row = out_row;                                  out_stride = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                          out_stride = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - qkv_m - z_m;                    out_stride = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - qkv_m - z_m - beta_m;           out_stride = alpha_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
