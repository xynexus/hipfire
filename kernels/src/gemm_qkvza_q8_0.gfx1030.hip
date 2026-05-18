#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Q8_0 batched 4-way fused QKV+Z+Beta+Alpha GEMM for gfx1030 (RDNA2).
// Sister of gemm_qkvza_q8_0_wmma.hip; see gemm_gate_up_q8_0.gfx1030.hip
// for storage-layout / unpack comments.

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_qkvza_q8_0_gfx1030(
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
    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const _Float16* x_row_base = X + (long long)safe_col * K;
    const int n_ktiles = K / 16;
    for (int kt = 0; kt < n_ktiles; kt++) {
        const int group_id = kt >> 1;
        const int half_id = kt & 1;
        const char* gp = row_base + group_id * 34;
        const int k_off = kt * 16;

        if ((tid >> 4) == 0) {
            _Float16 sc;
            __builtin_memcpy(&sc, gp, 2);
            const signed char* w = (const signed char*)(gp + 2 + half_id * 16);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                lds_a[my_lds_row][i] = sc * (_Float16)(float)(int)w[i];
            }
        }
        __syncthreads();

        _Float16 x_local[16];
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            x_local[kk] = x_row_base[k_off + kk];
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
                    Y = Y_qkv;   proj_row = out_row;                                out_stride = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                        out_stride = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - qkv_m - z_m;                  out_stride = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - qkv_m - z_m - beta_m;         out_stride = alpha_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
