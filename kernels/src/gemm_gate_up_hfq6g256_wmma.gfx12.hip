#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused HFQ6-G256 GEMM for the Qwen3.5
// FFN preamble (gate + up projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_gate_up_hfq6g256_wmma.hip
// (gfx11 / RDNA3). Combines the hfq6 dequant inner loop (validated in
// gemm_qkv_hfq6g256_wmma.gfx12.hip) with the 2-output gate/up routing
// (validated in gemm_gate_up_hfq4g256_wmma.gfx12.hip).
//
// HFQ6-G256 group layout: 200 bytes / 256 elements
//   [0..4)   f32 scale
//   [4..8)   f32 zero
//   [8..200) 192B packed 6-bit values
//
// gfx12 differences from the gfx11 source:
//   1. _w32 → _w32_gfx12 builtin
//   2. half16_t → half8_t A/B operands; float8_t accumulator unchanged
//   3. K-split via tid >> 4 (each lane carries 8 of 16 K values)
//   4. C-output mapping: contiguous rows per lane-group:
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq6g256_wmma_gfx12(
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

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;  local_row = safe_row;
    } else {
        A = A_up;    local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K2 unroll: 2 K-tiles per loop body.
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            // half-tile base = (kt * 12) + (k_grp * 6); 8 6-bit values = 6 bytes,
            // read as two unaligned uints at byte offsets {0, 3} of the half-tile.
            const int byte_base_a = kt * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_base_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);

            // === Tile B (kt+1) — start loading early ===
            const int byte_base_b = (kt + 1) * 12 + k_grp * 6;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_base_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);

            half8_t b_a = *(const half8_t*)(x_base + kt * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + (kt + 1) * 16 + k_grp * 8);

            // Dequant tile A (8 fp16 per lane).
            half8_t a_reg;
            #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0, d0a, 0);  DQ6(1, d0a, 6);  DQ6(2, d0a, 12); DQ6(3, d0a, 18);
            DQ6(4, d1a, 0);  DQ6(5, d1a, 6);  DQ6(6, d1a, 12); DQ6(7, d1a, 18);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);

            DQ6(0, d0b, 0);  DQ6(1, d0b, 6);  DQ6(2, d0b, 12); DQ6(3, d0b, 18);
            DQ6(4, d1b, 0);  DQ6(5, d1b, 6);  DQ6(6, d1b, 12); DQ6(7, d1b, 18);
            #undef DQ6

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
        }
    }

    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group.
    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;              out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m;     out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
