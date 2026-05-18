#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ4-G256 batched gate+up GEMM for gfx1030 (RDNA2, Navi 21+).
// Sister of gemm_gate_up_hfq4g256_wmma.hip; same contract, same per-lane
// output layout (acc[j] = C[2*j + (tid>>4)][tid & 15]) — without WMMA.
//
// MQ4 differs from MQ3 only in per-group stride (136 vs 104 bytes) and
// inner-tile unpack (4-bit nibble vs 3-bit cross-byte). Same scalar-FMA
// strategy via LDS-shared dequantized A weights.

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_gate_up_hfq4g256_gfx1030(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
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
    if (my_a_row_global < gate_m) {
        A = A_gate;  local_row = my_a_row_global;
    } else {
        A = A_up;    local_row = my_a_row_global - gate_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_col * K + g * 256;

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
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
