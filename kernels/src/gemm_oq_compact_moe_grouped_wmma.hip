// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped routed-expert OqPlusCompact W4A16 GEMM. The routing/scatter contract
// matches gemm_oq4g256_moe_grouped_wmma; the WEIGHT LAYOUT is what differs, and
// it is not a per-block interleave like Oq4's:
//
//   SPLIT-PLANE. Nibbles first, then side records.
//     nibble plane : A + row*(K/2) + g*128          (128 B per 256-group)
//     side region  : A + M*(K/2) + row*n_groups*side_stride + g*side_stride
//     side record  : [f16 scale][n_ov x (u8 idx, i8 val)]
//     side_stride  = block_stride - 128,  n_ov = (side_stride - 2) / 2
//
// so the SCALE lives in the side record, not ahead of the nibbles, and
// block_stride (136 at the shipped N_out=3) is a parameter rather than a
// constant.
//
// WHY COMPACT HERE: routed MoE prefill is weight-bandwidth-bound, and compact is
// 4.25 bits/weight against Oq8's 8.06 -- 1.90x less weight traffic on the axis
// that actually binds, on top of what grouping itself removes.
//
// THE OVERLAY IS APPLIED INLINE. The loader zeroes the bulk nibble under each
// side record, so a nibbles-only pass would drop those weights entirely and a
// separate correction pass would have to be expert-indexed. Instead each decoded
// value checks the group's n_ov records and substitutes `scale * val` for the
// zeroed nibble. oq8_arch.rs measures that inline side-record read at ~1%
// (226.4 vs 229.3 GB/s).
//
// The substitution is written as a compile-time unrolled scan over the 16 lanes
// of the fragment rather than an indexed write `a[idx - k_off] = ...`, because a
// runtime index into a vector register drops to M0-relative v_movrels — the
// trap levers.md 16 documents, which cost 20.6 instructions per MAC in the
// sparse overlay before it was fixed.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ int oqc_sext(unsigned int nibble) {
    int value = (int)(nibble & 0xFu);
    return value > 7 ? value - 16 : value;
}

// `x_row_div` maps a sorted routed slot back to its source activation row:
// gate_up uses K_TOP; down uses 1.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_oq_compact_moe_grouped_wmma(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total,
    int block_stride
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
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
    const int side_stride = block_stride - 128;
    const int n_ov = (side_stride - 2) / 2;

    const char* nib_row = A + (long long)safe_row * (K / 2);
    const char* side_row = A + (long long)M * (K / 2)
                             + (long long)safe_row * groups_per_row * side_stride;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int g = 0; g < groups_per_row; g++) {
        const char* np = nib_row + g * 128;
        const char* sp = side_row + g * side_stride;
        const _Float16 scale = *(const _Float16*)(sp);
        const unsigned char* rec = (const unsigned char*)(sp + 2);
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            const unsigned int pk0a = *(const unsigned int*)(np + k_off_a / 2);
            const unsigned int pk1a = *(const unsigned int*)(np + k_off_a / 2 + 4);
            const unsigned int pk0b = *(const unsigned int*)(np + k_off_b / 2);
            const unsigned int pk1b = *(const unsigned int*)(np + k_off_b / 2 + 4);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            // Decode 16 nibbles, then let any side record for this position win.
            // `i` is compile-time so every write lands in a real register.
            half16_t a;
            #define OQC_DQ(reg, i, packed, shift, koff)                        \
                do {                                                           \
                    float w = (float)oqc_sext((packed) >> (shift));            \
                    _Pragma("unroll") for (int e = 0; e < 4; e++) {            \
                        if (e >= n_ov) break;                                  \
                        if ((int)rec[2 * e] == ((koff) + (i))) {               \
                            w = (float)(signed char)rec[2 * e + 1];            \
                        }                                                      \
                    }                                                          \
                    reg[i] = scale * (_Float16)w;                              \
                } while (0)

            OQC_DQ(a, 0,  pk0a,  0, k_off_a); OQC_DQ(a, 1,  pk0a,  4, k_off_a);
            OQC_DQ(a, 2,  pk0a,  8, k_off_a); OQC_DQ(a, 3,  pk0a, 12, k_off_a);
            OQC_DQ(a, 4,  pk0a, 16, k_off_a); OQC_DQ(a, 5,  pk0a, 20, k_off_a);
            OQC_DQ(a, 6,  pk0a, 24, k_off_a); OQC_DQ(a, 7,  pk0a, 28, k_off_a);
            OQC_DQ(a, 8,  pk1a,  0, k_off_a); OQC_DQ(a, 9,  pk1a,  4, k_off_a);
            OQC_DQ(a, 10, pk1a,  8, k_off_a); OQC_DQ(a, 11, pk1a, 12, k_off_a);
            OQC_DQ(a, 12, pk1a, 16, k_off_a); OQC_DQ(a, 13, pk1a, 20, k_off_a);
            OQC_DQ(a, 14, pk1a, 24, k_off_a); OQC_DQ(a, 15, pk1a, 28, k_off_a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_a, acc);

            OQC_DQ(a, 0,  pk0b,  0, k_off_b); OQC_DQ(a, 1,  pk0b,  4, k_off_b);
            OQC_DQ(a, 2,  pk0b,  8, k_off_b); OQC_DQ(a, 3,  pk0b, 12, k_off_b);
            OQC_DQ(a, 4,  pk0b, 16, k_off_b); OQC_DQ(a, 5,  pk0b, 20, k_off_b);
            OQC_DQ(a, 6,  pk0b, 24, k_off_b); OQC_DQ(a, 7,  pk0b, 28, k_off_b);
            OQC_DQ(a, 8,  pk1b,  0, k_off_b); OQC_DQ(a, 9,  pk1b,  4, k_off_b);
            OQC_DQ(a, 10, pk1b,  8, k_off_b); OQC_DQ(a, 11, pk1b, 12, k_off_b);
            OQC_DQ(a, 12, pk1b, 16, k_off_b); OQC_DQ(a, 13, pk1b, 20, k_off_b);
            OQC_DQ(a, 14, pk1b, 24, k_off_b); OQC_DQ(a, 15, pk1b, 28, k_off_b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_b, acc);
            #undef OQC_DQ
        }
    }

    if (slot_idx < m_total && x_row >= 0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)slot_idx * M + out_row] = acc[j];
            }
        }
    }
}
