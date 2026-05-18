#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ3-Lloyd-G256 batched gate+up GEMM for gfx1030 (RDNA2, Navi 21+).
// Sister of gemm_gate_up_mq3g256_lloyd_wmma.hip; same contract, same
// kernarg layout (so dispatch can swap kernel names transparently), same
// per-lane output layout (acc[j] = C[2*j + (tid>>4)][tid & 15]), without
// WMMA — RDNA2 lacks the gfx11 wave32 WMMA builtin.
//
// MQ3-Lloyd group layout (112 B/group): 16 B fp16 codebook (8 entries
// per row) + 96 B 3-bit indices (16 K-tiles × 6 bytes, identical
// cross-byte pack to HFQ3). Decode is per-row codebook lookup
// `cb_lds[my_lds_row * 8 + q]` instead of HFQ3's `sc * q + zp`.
//
// Strategy mirrors gemm_gate_up_hfq3g256.gfx1030.hip:
//   - 32-thread wave32 block, 16×16 output tile per WG.
//   - LDS-shared dequantized A: lds_a[16][16] (512 B FP16).
//   - Additional codebook stage: cb_lds[16 * 8] (256 B FP16),
//     cooperatively loaded once per group.
//   - Inner product is scalar FP32 FMA. Throughput ~4-8× below WMMA,
//     but compiles + runs on every gfx10xx arch.
//
// Block: 32 threads (1 wave32). Grid: [ceil((gate_m+up_m)/16), ceil(N/16)].
// LDS: 768 B (lds_a 512 + cb_lds 256).

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_gate_up_mq3g256_lloyd_wmma(
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

    // LDS row this lane owns for dequant (lanes 0..15 each own one row;
    // lanes 16..31 idle during dequant, contribute during dot product).
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
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + local_row * row_stride;

    // Cooperative-load row mapping for codebook staging (independent of
    // my_lds_row: load_tile_row spans 0..15 over tid 0..31, each tile-row
    // may live in a different A matrix when straddling gate/up boundary).
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 4;   // 0 or 4
    int load_a_row_global = row_start + load_tile_row;
    if (load_a_row_global >= total_m) load_a_row_global = total_m - 1;
    const char* load_A;
    int load_local_row;
    if (load_a_row_global < gate_m) {
        load_A = A_gate; load_local_row = load_a_row_global;
    } else {
        load_A = A_up;   load_local_row = load_a_row_global - gate_m;
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    // 16 rows × 16 K-elements of dequantized FP16 A, shared across the wave.
    __shared__ _Float16 lds_a[16][16];
    // 16 rows × 8-entry codebook, restaged once per group.
    __shared__ _Float16 cb_lds[16 * 8];

    // Per-thread accumulators for 8 output rows × 1 output col.
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load (32 threads × 4 fp16 = 128 entries).
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

            // Lanes 0..15: unpack 16 3-bit indices, look up in this row's
            // codebook, store into lds_a[my_lds_row][0..16]. Cross-byte
            // 3-bit pack identical to HFQ3 / MQ3 uniform.
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

            // All 32 lanes: load 16-K-element FP16 X vector, accumulate
            // 8 dot products in FP32 against 8 LDS rows.
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
