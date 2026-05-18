#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ6-G256 batched 3-way fused Q+K+V GEMM for gfx1030 (RDNA2).
// Sister of gemm_qkv_hfq6g256_wmma.hip; 200 B/group stride, 6-bit unpack
// (4 weights per uint32, 4 uint32s per K-tile). See
// gemm_gate_up_hfq6g256.gfx1030.hip for template comments.

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_qkv_hfq6g256_gfx1030(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_q,
    float*         __restrict__ Y_k,
    float*         __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
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
    if (my_a_row_global < q_m) {
        A = A_q;  local_row = my_a_row_global;
    } else if (my_a_row_global < q_m + k_m) {
        A = A_k;  local_row = my_a_row_global - q_m;
    } else {
        A = A_v;  local_row = my_a_row_global - q_m - k_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_col * K + g * 256;

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;
            const int byte_off = kt * 12;

            if ((tid >> 4) == 0) {
                const unsigned char* dp = (const unsigned char*)(gp + 8 + byte_off);
                unsigned int d0 = *(const unsigned int*)(dp);
                unsigned int d1 = *(const unsigned int*)(dp + 3);
                unsigned int d2 = *(const unsigned int*)(dp + 6);
                unsigned int d3 = *(const unsigned int*)(dp + 9);
                #define DQ6(i, dw, sh) lds_a[my_lds_row][i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
                DQ6(0,  d0,  0); DQ6(1,  d0,  6); DQ6(2,  d0, 12); DQ6(3,  d0, 18);
                DQ6(4,  d1,  0); DQ6(5,  d1,  6); DQ6(6,  d1, 12); DQ6(7,  d1, 18);
                DQ6(8,  d2,  0); DQ6(9,  d2,  6); DQ6(10, d2, 12); DQ6(11, d2, 18);
                DQ6(12, d3,  0); DQ6(13, d3,  6); DQ6(14, d3, 12); DQ6(15, d3, 18);
                #undef DQ6
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
                if (out_row < q_m) {
                    Y = Y_q;  proj_row = out_row;               out_stride = q_m;
                } else if (out_row < q_m + k_m) {
                    Y = Y_k;  proj_row = out_row - q_m;          out_stride = k_m;
                } else {
                    Y = Y_v;  proj_row = out_row - q_m - k_m;    out_stride = v_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
