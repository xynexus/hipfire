#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped routed-expert OQ4 W4A16 GEMM. The routing/scatter contract matches
// gemm_hfq4g256_moe_grouped_wmma_k2; only the weight block differs:
//   OQ8 indexed block = [f32 scale | 256 int8] (260 B), dequant w = scale * int8.
// Two producers of that block stream exist and `weight_byte_offset` selects
// between them, since the expert pointer table always stores the buffer base:
//   - hipfire-runtime::oq_moe repack (indexed-decode opt-in) — offset 0.
//   - hipfire-runtime::oq4_arch combined layout (the DEFAULT for resident
//     qt=34/37 experts) — the interleaved region trails the split int8 and
//     split f32 scales, so offset = M*K + M*(K/256)*4.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` maps a sorted routed slot back to its source activation row:
// gate_up uses K_TOP; down uses 1.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_oq8g256_moe_grouped_wmma(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total,
    unsigned long long weight_byte_offset
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id])
        + weight_byte_offset;
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base =
        (x_row >= 0) ? (X_src + (long long)x_row * K)
                     : (const _Float16*)nullptr;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 260;
    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 260;
        const _Float16 scale =
            (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            // OQ8: 16 consecutive int8 per WMMA k-tile, so the k offset is not
            // halved as it is for packed nibbles.
            const signed char* wa = (const signed char*)(gp + 4 + k_off_a);
            const signed char* wb = (const signed char*)(gp + 4 + k_off_b);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                a[i] = scale * (_Float16)(float)(int)wa[i];
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_a, acc);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                a[i] = scale * (_Float16)(float)(int)wb[i];
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_b, acc);
        }
    }

    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
