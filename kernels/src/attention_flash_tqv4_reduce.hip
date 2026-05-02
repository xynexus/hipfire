#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_tqv4_reduce(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles
) {
    const int h = blockIdx.x;
    if (h >= n_heads || (head_dim != 128 && head_dim != 256)) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    const float* my_partials = partials + h * max_tiles * stride;

    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f) global_max = fmaxf(global_max, p[0]);
    }

    float global_sum = 0.0f;
    const int n_halves = head_dim / 128;
    const int d0 = (head_dim == 256) ? tid * 8 : tid * 4;
    float g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        float corr = expf(p[0] - global_max);
        g0 += p[2 + d0 + 0] * corr; g1 += p[2 + d0 + 1] * corr;
        g2 += p[2 + d0 + 2] * corr; g3 += p[2 + d0 + 3] * corr;
        if (head_dim == 256) {
            g4 += p[2 + d0 + 4] * corr; g5 += p[2 + d0 + 5] * corr;
            g6 += p[2 + d0 + 6] * corr; g7 += p[2 + d0 + 7] * corr;
        }
        global_sum += tile_sum * corr;
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    g0 *= inv_sum; g1 *= inv_sum; g2 *= inv_sum; g3 *= inv_sum;
    if (head_dim == 256) {
        g4 *= inv_sum; g5 *= inv_sum; g6 *= inv_sum; g7 *= inv_sum;
        fwht_shfl_inverse_256(g0, g1, g2, g3, g4, g5, g6, g7, signs1, signs2, tid);
    } else {
        fwht_shfl_inverse(g0, g1, g2, g3, signs1, signs2, tid);
    }

    float* my_out = out + h * head_dim;
    my_out[d0 + 0] = g0; my_out[d0 + 1] = g1; my_out[d0 + 2] = g2; my_out[d0 + 3] = g3;
    if (n_halves == 2) {
        my_out[d0 + 4] = g4; my_out[d0 + 5] = g5; my_out[d0 + 6] = g6; my_out[d0 + 7] = g7;
    }
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_tqv4_reduce_batched(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads || (head_dim != 128 && head_dim != 256)) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;

    const int global_bid = batch_offset + local_bid;
    const int seq_len = (block_cols > 0) ? (block_start + block_cols)
                                          : (positions[global_bid] + 1);
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f) global_max = fmaxf(global_max, p[0]);
    }

    float global_sum = 0.0f;
    const int n_halves = head_dim / 128;
    const int d0 = (head_dim == 256) ? tid * 8 : tid * 4;
    float g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        float corr = expf(p[0] - global_max);
        g0 += p[2 + d0 + 0] * corr; g1 += p[2 + d0 + 1] * corr;
        g2 += p[2 + d0 + 2] * corr; g3 += p[2 + d0 + 3] * corr;
        if (head_dim == 256) {
            g4 += p[2 + d0 + 4] * corr; g5 += p[2 + d0 + 5] * corr;
            g6 += p[2 + d0 + 6] * corr; g7 += p[2 + d0 + 7] * corr;
        }
        global_sum += tile_sum * corr;
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    g0 *= inv_sum; g1 *= inv_sum; g2 *= inv_sum; g3 *= inv_sum;
    if (head_dim == 256) {
        g4 *= inv_sum; g5 *= inv_sum; g6 *= inv_sum; g7 *= inv_sum;
        fwht_shfl_inverse_256(g0, g1, g2, g3, g4, g5, g6, g7, signs1, signs2, tid);
    } else {
        fwht_shfl_inverse(g0, g1, g2, g3, signs1, signs2, tid);
    }

    const int q_dim = n_heads * head_dim;
    float* my_out = out + local_bid * q_dim + h * head_dim;
    my_out[d0 + 0] = g0; my_out[d0 + 1] = g1; my_out[d0 + 2] = g2; my_out[d0 + 3] = g3;
    if (n_halves == 2) {
        my_out[d0 + 4] = g4; my_out[d0 + 5] = g5; my_out[d0 + 6] = g6; my_out[d0 + 7] = g7;
    }
}
