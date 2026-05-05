#include <hip/hip_runtime.h>
#include "turbo_common.h"

__device__ float q8_tqv_value_table(int idx, int value_bits, int head_dim, const float* __restrict__ centroids) {
    if (centroids != nullptr) return centroids[idx];
    if (value_bits == 1 && head_dim == 256) return TURBO_C1_256[idx];
    if (value_bits == 1) return TURBO_C1[idx];
    if (value_bits == 4 && head_dim == 256) return TURBO_C4_256[idx];
    if (value_bits == 4) return TURBO_C4[idx];
    if (value_bits == 3 && head_dim == 256) return TURBO_C3_256[idx];
    if (value_bits == 3) return TURBO_C3[idx];
    if (head_dim == 256) return TURBO_C2_256[idx];
    return TURBO_C2[idx];
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_tqv4_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int value_bits,
    const float* __restrict__ tqv_centroids,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    if (h >= n_heads || (head_dim != 128 && head_dim != 256) || (value_bits != 1 && value_bits != 2 && value_bits != 3 && value_bits != 4)) return;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int n_halves = head_dim / 128;

    const int q8_blocks_per_head = head_dim / 32;
    const int q8_total_blocks_per_pos = n_kv_heads * q8_blocks_per_head;
    const int q8_head_block_start = kv_h * q8_blocks_per_head;
    const int q8_row_stride = q8_total_blocks_per_pos * 34;

    const int storage_bits = (value_bits == 1) ? 2 : value_bits;
    const int v_bytes_per_head = 4 + (head_dim * storage_bits) / 8;
    const int v_bytes_per_pos = n_kv_heads * v_bytes_per_head;
    const int v_head_off = kv_h * v_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;
    const float* q_head = q + h * head_dim;

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int k_bi = half * 4 + tid / 8;
            const int k_off = (tid % 8) * 4;
            const unsigned char* kb = k_cache + (size_t)t * q8_row_stride + (q8_head_block_start + k_bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            const float* qb = q_head + k_bi * 32 + k_off;
            partial += qb[0] * (ks * (float)((signed char)kb[2 + k_off + 0]))
                     + qb[1] * (ks * (float)((signed char)kb[2 + k_off + 1]))
                     + qb[2] * (ks * (float)((signed char)kb[2 + k_off + 2]))
                     + qb[3] * (ks * (float)((signed char)kb[2 + k_off + 3]));
        }
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32) local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    float out_vec[8] = {0,0,0,0,0,0,0,0};
    for (int t_local = 0; t_local < tile_len; t_local++) {
        const float w = scores[t_local];
        const int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
        const float cnorm = *(const float*)vb;
        if (value_bits == 4 && head_dim == 256) {
            const unsigned char* packed = vb + 4 + tid * 4;
            unsigned char b0 = packed[0], b1 = packed[1], b2 = packed[2], b3 = packed[3];
            out_vec[0] += w * (cnorm * q8_tqv_value_table((b0)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((b0 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((b1)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((b1 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[4] += w * (cnorm * q8_tqv_value_table((b2)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[5] += w * (cnorm * q8_tqv_value_table((b2 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[6] += w * (cnorm * q8_tqv_value_table((b3)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[7] += w * (cnorm * q8_tqv_value_table((b3 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
        } else if (value_bits == 4) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * q8_tqv_value_table((b0)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((b0 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((b1)      & 0xF, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((b1 >> 4) & 0xF, value_bits, head_dim, tqv_centroids));
        } else if (value_bits == 3 && head_dim == 256) {
            const unsigned char* packed = vb + 4 + tid * 3;
            unsigned int pack = (unsigned int)packed[0] | ((unsigned int)packed[1] << 8) | ((unsigned int)packed[2] << 16);
            out_vec[0] += w * (cnorm * q8_tqv_value_table((pack)       & 7, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((pack >> 3)  & 7, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((pack >> 6)  & 7, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((pack >> 9)  & 7, value_bits, head_dim, tqv_centroids));
            out_vec[4] += w * (cnorm * q8_tqv_value_table((pack >> 12) & 7, value_bits, head_dim, tqv_centroids));
            out_vec[5] += w * (cnorm * q8_tqv_value_table((pack >> 15) & 7, value_bits, head_dim, tqv_centroids));
            out_vec[6] += w * (cnorm * q8_tqv_value_table((pack >> 18) & 7, value_bits, head_dim, tqv_centroids));
            out_vec[7] += w * (cnorm * q8_tqv_value_table((pack >> 21) & 7, value_bits, head_dim, tqv_centroids));
        } else if (value_bits == 3) {
            const unsigned char* packed = vb + 4 + (tid >> 1) * 3;
            unsigned int pack = (unsigned int)packed[0] | ((unsigned int)packed[1] << 8) | ((unsigned int)packed[2] << 16);
            int shift = (tid & 1) ? 12 : 0;
            out_vec[0] += w * (cnorm * q8_tqv_value_table((pack >> shift)       & 7, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((pack >> (shift + 3)) & 7, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((pack >> (shift + 6)) & 7, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((pack >> (shift + 9)) & 7, value_bits, head_dim, tqv_centroids));
        } else if (head_dim == 256) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * q8_tqv_value_table((b0)      & 3, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((b0 >> 2) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((b0 >> 4) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((b0 >> 6) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[4] += w * (cnorm * q8_tqv_value_table((b1)      & 3, value_bits, head_dim, tqv_centroids));
            out_vec[5] += w * (cnorm * q8_tqv_value_table((b1 >> 2) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[6] += w * (cnorm * q8_tqv_value_table((b1 >> 4) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[7] += w * (cnorm * q8_tqv_value_table((b1 >> 6) & 3, value_bits, head_dim, tqv_centroids));
        } else {
            unsigned char b = *(vb + 4 + tid);
            out_vec[0] += w * (cnorm * q8_tqv_value_table((b)      & 3, value_bits, head_dim, tqv_centroids));
            out_vec[1] += w * (cnorm * q8_tqv_value_table((b >> 2) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[2] += w * (cnorm * q8_tqv_value_table((b >> 4) & 3, value_bits, head_dim, tqv_centroids));
            out_vec[3] += w * (cnorm * q8_tqv_value_table((b >> 6) & 3, value_bits, head_dim, tqv_centroids));
        }
    }

    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) { p[0] = tile_max; p[1] = tile_sum; }
    if (head_dim == 256) {
        const int d0 = tid * 8;
        #pragma unroll
        for (int i = 0; i < 8; i++) p[2 + d0 + i] = out_vec[i];
    } else {
        const int d0 = tid * 4;
        p[2 + d0 + 0] = out_vec[0];
        p[2 + d0 + 1] = out_vec[1];
        p[2 + d0 + 2] = out_vec[2];
        p[2 + d0 + 3] = out_vec[3];
    }
}
