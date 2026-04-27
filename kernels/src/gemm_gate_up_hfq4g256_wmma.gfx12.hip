#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused HFQ4-G256 GEMM for the Qwen3.5
// FFN preamble (gate + up projections).
//
// **gfx12 (RDNA4) variant** — sister of gemm_gate_up_hfq4g256_wmma.hip
// (gfx11 / RDNA3). Compile with `hipcc --offload-arch=gfx1200` (or 1201).
//
// Differences from the gfx11 kernel — same set as the QKV scaffold
// (gemm_qkv_hfq4g256_wmma.gfx12.hip), validated on R9700 / gfx1201:
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (gfx11 uses _w32 without the _gfx12 suffix).
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t.
//      Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (kABKLane=2):
//        lane (tid >> 4) = 0  -> carries K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> carries K = [8..15] of the 16-K tile
//      Each lane loads HALF as much per WMMA tile as gfx11.
//   4. C-output mapping (VALIDATED 2026-04-27 in the QKV scaffold's
//      channel-test on R9700):
//        gfx11: acc[j] = C[2*j + (tid>>4)][tid & 15]    (rows interleaved)
//        gfx12: acc[j] = C[8*(tid>>4) + j][tid & 15]    (rows contiguous —
//                                                        group 0: rows 0..7,
//                                                        group 1: rows 8..15)
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.
//
// Each 16-row tile may straddle the gate/up boundary. Per-thread output
// routing resolves which Y pointer (Y_gate or Y_up) each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_gfx12(
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

    // Lane decomposition (gfx12 wave32 WMMA):
    //   m_lane = tid & 15   (0..15)  -> row index within the 16-row tile
    //   k_grp  = tid >> 4   (0 or 1) -> which K half-tile this lane carries
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;  local_row = safe_row;
    } else {
        A = A_up;    local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    // B operand indexes BATCH (N) dimension by m_lane (CK kBNLane=16).
    // Each lane holds the 8-K slice of its batch row corresponding to k_grp.
    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // K2 unroll: 2 K-tiles per loop body (8 iterations to cover K=256).
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            // Each lane reads ONE 4-byte (8-nibble) chunk of weights covering
            // its half of K=[k_off..k_off+7] (k_grp=0)
            // or K=[k_off+8..k_off+15] (k_grp=1).
            unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);

            // Load both X tiles early (one half8_t per lane, K-offset by k_grp*8)
            half8_t b_a = *(const half8_t*)(x_base + k_off_a + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + k_off_b + k_grp * 8);

            // Dequant tile A (8 fp16 per lane).
            half8_t a_reg;
            #define DQ8(pk) \
                a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
                a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
                a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
                a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
                a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
                a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
                a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
                a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h
            DQ8(pka);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);

            // Dequant tile B (reuse a_reg register).
            DQ8(pkb);
            #undef DQ8

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
        }
    }

    // --- Output ---
    // gfx12 wave32 WMMA C-mapping: lane group 0 holds rows 0..7,
    // group 1 holds rows 8..15 (validated on R9700 via the QKV scaffold).
    const int out_col = batch_start + m_lane;  // batch index
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
