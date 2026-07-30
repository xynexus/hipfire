#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped routed-expert OQ4 W4A16 GEMM. The routing/scatter contract matches
// gemm_hfq4g256_moe_grouped_wmma_k2; only the weight block differs:
//   OQ4 indexed block = [f32 scale | 128 signed-int4 nibbles] (132 B).
// Two producers of that block stream exist and `weight_byte_offset` selects
// between them, since the expert pointer table always stores the buffer base:
//   - hipfire-runtime::oq_moe repack (indexed-decode opt-in) — offset 0.
//   - hipfire-runtime::oq4_arch combined layout (the DEFAULT for resident
//     qt=34/37 experts) — the interleaved region trails the split nibbles and
//     split f32 scales, so offset = M*(K/2) + M*(K/256)*4.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ int oq4_sext(unsigned int nibble) {
    int value = (int)(nibble & 0xFu);
    return value > 7 ? value - 16 : value;
}

// `x_row_div` maps a sorted routed slot back to its source activation row:
// gate_up uses K_TOP; down uses 1.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_oq4g256_moe_grouped_wmma(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 132;
    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 132;
        const _Float16 scale =
            (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            const unsigned int pk0a = *(const unsigned int*)(gp + 4 + k_off_a / 2);
            const unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            const unsigned int pk0b = *(const unsigned int*)(gp + 4 + k_off_b / 2);
            const unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a;
            #define OQ4_DQ(reg, i, packed, shift) \
                reg[i] = scale * (_Float16)(float)oq4_sext((packed) >> (shift))
            OQ4_DQ(a, 0,  pk0a,  0); OQ4_DQ(a, 1,  pk0a,  4);
            OQ4_DQ(a, 2,  pk0a,  8); OQ4_DQ(a, 3,  pk0a, 12);
            OQ4_DQ(a, 4,  pk0a, 16); OQ4_DQ(a, 5,  pk0a, 20);
            OQ4_DQ(a, 6,  pk0a, 24); OQ4_DQ(a, 7,  pk0a, 28);
            OQ4_DQ(a, 8,  pk1a,  0); OQ4_DQ(a, 9,  pk1a,  4);
            OQ4_DQ(a, 10, pk1a,  8); OQ4_DQ(a, 11, pk1a, 12);
            OQ4_DQ(a, 12, pk1a, 16); OQ4_DQ(a, 13, pk1a, 20);
            OQ4_DQ(a, 14, pk1a, 24); OQ4_DQ(a, 15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_a, acc);

            OQ4_DQ(a, 0,  pk0b,  0); OQ4_DQ(a, 1,  pk0b,  4);
            OQ4_DQ(a, 2,  pk0b,  8); OQ4_DQ(a, 3,  pk0b, 12);
            OQ4_DQ(a, 4,  pk0b, 16); OQ4_DQ(a, 5,  pk0b, 20);
            OQ4_DQ(a, 6,  pk0b, 24); OQ4_DQ(a, 7,  pk0b, 28);
            OQ4_DQ(a, 8,  pk1b,  0); OQ4_DQ(a, 9,  pk1b,  4);
            OQ4_DQ(a, 10, pk1b,  8); OQ4_DQ(a, 11, pk1b, 12);
            OQ4_DQ(a, 12, pk1b, 16); OQ4_DQ(a, 13, pk1b, 20);
            OQ4_DQ(a, 14, pk1b, 24); OQ4_DQ(a, 15, pk1b, 28);
            #undef OQ4_DQ
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
