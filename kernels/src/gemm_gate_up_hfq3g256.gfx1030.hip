#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ3-G256 batched gate+up GEMM for gfx1030 (RDNA2, Navi 21+).
// Sister of gemm_gate_up_hfq3g256_wmma.hip; same contract, same per-lane
// output layout (acc[j] = C[2*j + (tid>>4)][tid & 15]), without WMMA —
// RDNA2 lacks the gfx11 wave32 WMMA builtin.
//
// Strategy: each block holds the 16×16 dequantized FP16 A tile in LDS
// so lanes can share weights across the wave (WMMA does this implicitly
// via the matrix unit). Inner product is scalar FP32 FMA. Throughput is
// ~4-8× below WMMA, but ~10× above per-token forward_scratch fallback at
// typical spec-decode batch sizes (B ≥ 8). Compiles on every gfx10xx
// arch, but only dispatched on gfx1030/1031 (RDNA1 gfx1010 routes
// through per-token forward_scratch — its 8 GiB VRAM can't run 27B
// anyway).
//
// Block: 32 threads (1 wave32). Grid: [ceil((gate_m+up_m)/16), ceil(N/16)].
// LDS: 512 bytes (16×16 FP16).

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_gate_up_hfq3g256_gfx1030(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,        // [N x K], FP16
    float*         __restrict__ Y_gate,    // [N x gate_m]
    float*         __restrict__ Y_up,      // [N x up_m]
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // LDS row index this lane is responsible for dequantizing (lanes 0..15
    // own one row each; lanes 16..31 idle during dequant, contribute during
    // dot product).
    const int my_lds_row = tid & 15;
    int my_a_row_global = row_start + my_lds_row;
    if (my_a_row_global >= total_m) my_a_row_global = total_m - 1;

    const char* A;
    int local_row;
    if (my_a_row_global < gate_m) {
        A = A_gate;
        local_row = my_a_row_global;
    } else {
        A = A_up;
        local_row = my_a_row_global - gate_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 104;
    const char* row_base = A + local_row * row_stride;

    // Per-thread batch column (clamped to in-range so OOB lanes still
    // do useful work without polluting outputs; writeback gates on real
    // bound).
    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    // 16 rows × 16 K-elements of dequantized FP16 A, shared across the wave.
    __shared__ _Float16 lds_a[16][16];

    // Per-thread accumulators for 8 output rows × 1 output col.
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_col * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;

            // Lanes 0..15: dequantize 16 quants from 6 packed bytes into
            // lds_a[my_lds_row][0..16]. Cross-byte 3-bit unpack matches
            // gemv_hfq3g256.hip / gemm_*_hfq3g256_wmma.hip.
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
                #define DQ(i, q) lds_a[my_lds_row][i] = sc_h * (_Float16)(float)(q) + zp_h
                DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
                DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
                DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
                DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);
                #undef DQ
            }
            __syncthreads();

            // All 32 lanes: load 16-K-element FP16 X vector for our col,
            // accumulate 8 dot products in FP32 against the 8 LDS rows.
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

    // Writeback (mirrors WMMA kernel's per-lane row layout).
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
