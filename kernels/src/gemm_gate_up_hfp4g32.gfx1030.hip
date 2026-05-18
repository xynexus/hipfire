#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFP4-G32 batched gate+up GEMM for gfx1030 (RDNA2, Navi 21+).
// Sister of gemm_gate_up_hfp4g32_wmma.hip; same contract, same per-lane
// output layout (acc[j] = C[2*j + (tid>>4)][tid & 15]) — without WMMA.
//
// HFP4-G32 storage (matches gemm_gate_up_hfp4g32_wmma.hip exactly):
//   Per row: 16-B header (FP16 row_scale_a at offset 0 + 14 B pad)
//            + (K/32) blocks × 17 B (1 B UE8M0 exponent + 16 B nibbles
//            = 32 FP4-E2M1 elements).
//   row_stride = 16 + (K/32) * 17 bytes.
//
// Dequant: value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]
//
// K-tile size is 16, so each 32-element block contains TWO K-tiles:
//   half_id 0 → nibbles[0..16]  (bytes bp+1 .. bp+9)
//   half_id 1 → nibbles[16..32] (bytes bp+9 .. bp+17)
//
// Strategy: lanes 0..15 dequantize their LDS row 16 elements at a time;
// inner product is scalar FP32 FMA against 16-elem FP16 X slices.
// Mirrors the Q8_0 gfx1030 template — only the per-K-tile unpack
// differs (FP4 + UE8M0 + row scale vs FP16 scale × int8).
//
// Block: 32 threads (1 wave32). Grid: [ceil((gate_m+up_m)/16), ceil(N/16)].
// LDS: 512 B (16×16 FP16) + 32 B (E2M1 LUT).

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_gate_up_hfp4g32_gfx1030(
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

    // 16-entry signed E2M1 magnitude LUT in LDS (32 B). Lanes 0..15 init.
    // OCP E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Same as WMMA sister.
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

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
    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)local_row * row_stride;
    const char* row_ptr = hdr + 16;

    // Hoist FP16 row_scale_a once per row.
    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    __shared__ _Float16 lds_a[16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const _Float16* x_row_base = X + (long long)safe_col * K;

    const int n_ktiles = K / 16;
    for (int kt = 0; kt < n_ktiles; kt++) {
        const int block_idx = kt >> 1;
        const int half_id = kt & 1;
        const char* bp = row_ptr + block_idx * 17;
        const int k_off = kt * 16;

        if ((tid >> 4) == 0) {
            // UE8M0 block scale → FP32 bit-construct → FP16. Combine
            // with the hoisted row_scale to get this K-tile's scalar.
            const unsigned int e_byte = *(const unsigned char*)bp;
            const float sc_block_f = __builtin_bit_cast(float, e_byte << 23);
            const _Float16 sc_h = row_scale_h * (_Float16)sc_block_f;

            // 16 packed nibbles for this K-tile = 8 bytes = 2 uints.
            // half_id 0 → bytes bp+1, bp+5; half_id 1 → bytes bp+9, bp+13.
            const char* hp = bp + 1 + half_id * 8;
            unsigned int pk0 = *(const unsigned int*)(hp + 0);
            unsigned int pk1 = *(const unsigned int*)(hp + 4);

            #define DQ(i, pk, sh) lds_a[my_lds_row][i] = sc_h * lut[((pk) >> (sh)) & 0xFu]
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
