#include <hip/hip_runtime.h>

// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] . x.
//
// Mirror of gemv_hfq3g256.hip, with the only difference being a += into y
// instead of an = write. Used for `wo` and `w_down` in the dense MQ3
// forward path to eliminate the add_inplace_f32 follow-up launch (one
// fewer kernel dispatch per layer per token).
//
// Same wave32 layout: 32 threads per block, one block per row, 8
// weights per thread per group, 24 bits packed into 3 bytes per thread
// per group. K must be a multiple of 256.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq3g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 104;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 104;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 3;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];

        int q0 = b0 & 7;
        int q1 = (b0 >> 3) & 7;
        int q2 = ((b0 >> 6) | (b1 << 2)) & 7;
        int q3 = (b1 >> 1) & 7;
        int q4 = (b1 >> 4) & 7;
        int q5 = ((b1 >> 7) | (b2 << 1)) & 7;
        int q6 = (b2 >> 2) & 7;
        int q7 = (b2 >> 5) & 7;

        acc += (scale * (float)q0 + zero) * x[base_idx]
             + (scale * (float)q1 + zero) * x[base_idx + 1]
             + (scale * (float)q2 + zero) * x[base_idx + 2]
             + (scale * (float)q3 + zero) * x[base_idx + 3]
             + (scale * (float)q4 + zero) * x[base_idx + 4]
             + (scale * (float)q5 + zero) * x[base_idx + 5]
             + (scale * (float)q6 + zero) * x[base_idx + 6]
             + (scale * (float)q7 + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;
}
