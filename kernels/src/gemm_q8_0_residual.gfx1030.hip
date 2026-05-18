#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Q8_0 batched single-output GEMM with fused residual add for gfx1030
// (RDNA2). Sister of gemm_q8_0_residual_wmma.hip; same residual
// semantics (Y += A·x, caller seeds Y with residual stream). See
// gemm_gate_up_q8_0.gfx1030.hip for the template comments.

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_q8_0_residual_gfx1030(
    const char*     __restrict__ A_q8,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= N) return;

    const int my_lds_row = tid & 15;
    int safe_row = row_start + my_lds_row;
    if (safe_row >= M) safe_row = M - 1;
    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + safe_row * row_stride;
    const _Float16* x_row_base = X + (long long)safe_col * K;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}
