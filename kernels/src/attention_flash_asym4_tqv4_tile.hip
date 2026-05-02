#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym4_tqv4_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int value_bits,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    if (h >= n_heads || (head_dim != 128 && head_dim != 256) || (value_bits != 2 && value_bits != 4)) return;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos = n_kv_heads * k_bytes_per_head;
    const int k_head_off = kv_h * k_bytes_per_head;
    const int v_bytes_per_head = 4 + (head_dim * value_bits) / 8;
    const int v_bytes_per_pos = n_kv_heads * v_bytes_per_head;
    const int v_head_off = kv_h * v_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + h * head_dim;
    float mq[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0], b = q_head[d0 + 1], c = q_head[d0 + 2], d = q_head[d0 + 3];
        givens_forward(a, b, c, d, cos_theta, sin_theta, half * 32 + tid);
        mq[half * 4 + 0] = a; mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c; mq[half * 4 + 3] = d;
    }

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        const float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned short pk = *(const unsigned short*)(kb + 4 + half * 64 + tid * 2);
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C4[(pk)       & 0xF])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C4[(pk >> 4)  & 0xF])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C4[(pk >> 8)  & 0xF])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C4[(pk >> 12) & 0xF]);
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
            out_vec[0] += w * (cnorm * TURBO_C4_256[(b0)      & 0xF]);
            out_vec[1] += w * (cnorm * TURBO_C4_256[(b0 >> 4) & 0xF]);
            out_vec[2] += w * (cnorm * TURBO_C4_256[(b1)      & 0xF]);
            out_vec[3] += w * (cnorm * TURBO_C4_256[(b1 >> 4) & 0xF]);
            out_vec[4] += w * (cnorm * TURBO_C4_256[(b2)      & 0xF]);
            out_vec[5] += w * (cnorm * TURBO_C4_256[(b2 >> 4) & 0xF]);
            out_vec[6] += w * (cnorm * TURBO_C4_256[(b3)      & 0xF]);
            out_vec[7] += w * (cnorm * TURBO_C4_256[(b3 >> 4) & 0xF]);
        } else if (value_bits == 4) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * TURBO_C4[(b0)      & 0xF]);
            out_vec[1] += w * (cnorm * TURBO_C4[(b0 >> 4) & 0xF]);
            out_vec[2] += w * (cnorm * TURBO_C4[(b1)      & 0xF]);
            out_vec[3] += w * (cnorm * TURBO_C4[(b1 >> 4) & 0xF]);
        } else if (head_dim == 256) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * TURBO_C2_256[(b0)      & 3]);
            out_vec[1] += w * (cnorm * TURBO_C2_256[(b0 >> 2) & 3]);
            out_vec[2] += w * (cnorm * TURBO_C2_256[(b0 >> 4) & 3]);
            out_vec[3] += w * (cnorm * TURBO_C2_256[(b0 >> 6) & 3]);
            out_vec[4] += w * (cnorm * TURBO_C2_256[(b1)      & 3]);
            out_vec[5] += w * (cnorm * TURBO_C2_256[(b1 >> 2) & 3]);
            out_vec[6] += w * (cnorm * TURBO_C2_256[(b1 >> 4) & 3]);
            out_vec[7] += w * (cnorm * TURBO_C2_256[(b1 >> 6) & 3]);
        } else {
            unsigned char b = *(vb + 4 + tid);
            out_vec[0] += w * (cnorm * TURBO_C2[(b)      & 3]);
            out_vec[1] += w * (cnorm * TURBO_C2[(b >> 2) & 3]);
            out_vec[2] += w * (cnorm * TURBO_C2[(b >> 4) & 3]);
            out_vec[3] += w * (cnorm * TURBO_C2[(b >> 6) & 3]);
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

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym4_tqv4_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int value_bits,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads || (head_dim != 128 && head_dim != 256) || (value_bits != 2 && value_bits != 4)) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int n_halves = head_dim / 128;
    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos = n_kv_heads * k_bytes_per_head;
    const int k_head_off = kv_h * k_bytes_per_head;
    const int v_bytes_per_head = 4 + (head_dim * value_bits) / 8;
    const int v_bytes_per_pos = n_kv_heads * v_bytes_per_head;
    const int v_head_off = kv_h * v_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + local_bid * q_dim + h * head_dim;
    float mq[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0], b = q_head[d0 + 1], c = q_head[d0 + 2], d = q_head[d0 + 3];
        givens_forward(a, b, c, d, cos_theta, sin_theta, half * 32 + tid);
        mq[half * 4 + 0] = a; mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c; mq[half * 4 + 3] = d;
    }

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        const float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned short pk = *(const unsigned short*)(kb + 4 + half * 64 + tid * 2);
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C4[(pk)       & 0xF])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C4[(pk >> 4)  & 0xF])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C4[(pk >> 8)  & 0xF])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C4[(pk >> 12) & 0xF]);
        }
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= block_start) s += tree_bias[global_bid * block_cols + (t - block_start)];
            scores[t_local] = s;
        }
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
            out_vec[0] += w * (cnorm * TURBO_C4_256[(b0)      & 0xF]);
            out_vec[1] += w * (cnorm * TURBO_C4_256[(b0 >> 4) & 0xF]);
            out_vec[2] += w * (cnorm * TURBO_C4_256[(b1)      & 0xF]);
            out_vec[3] += w * (cnorm * TURBO_C4_256[(b1 >> 4) & 0xF]);
            out_vec[4] += w * (cnorm * TURBO_C4_256[(b2)      & 0xF]);
            out_vec[5] += w * (cnorm * TURBO_C4_256[(b2 >> 4) & 0xF]);
            out_vec[6] += w * (cnorm * TURBO_C4_256[(b3)      & 0xF]);
            out_vec[7] += w * (cnorm * TURBO_C4_256[(b3 >> 4) & 0xF]);
        } else if (value_bits == 4) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * TURBO_C4[(b0)      & 0xF]);
            out_vec[1] += w * (cnorm * TURBO_C4[(b0 >> 4) & 0xF]);
            out_vec[2] += w * (cnorm * TURBO_C4[(b1)      & 0xF]);
            out_vec[3] += w * (cnorm * TURBO_C4[(b1 >> 4) & 0xF]);
        } else if (head_dim == 256) {
            const unsigned char* packed = vb + 4 + tid * 2;
            unsigned char b0 = packed[0], b1 = packed[1];
            out_vec[0] += w * (cnorm * TURBO_C2_256[(b0)      & 3]);
            out_vec[1] += w * (cnorm * TURBO_C2_256[(b0 >> 2) & 3]);
            out_vec[2] += w * (cnorm * TURBO_C2_256[(b0 >> 4) & 3]);
            out_vec[3] += w * (cnorm * TURBO_C2_256[(b0 >> 6) & 3]);
            out_vec[4] += w * (cnorm * TURBO_C2_256[(b1)      & 3]);
            out_vec[5] += w * (cnorm * TURBO_C2_256[(b1 >> 2) & 3]);
            out_vec[6] += w * (cnorm * TURBO_C2_256[(b1 >> 4) & 3]);
            out_vec[7] += w * (cnorm * TURBO_C2_256[(b1 >> 6) & 3]);
        } else {
            unsigned char b = *(vb + 4 + tid);
            out_vec[0] += w * (cnorm * TURBO_C2[(b)      & 3]);
            out_vec[1] += w * (cnorm * TURBO_C2[(b >> 2) & 3]);
            out_vec[2] += w * (cnorm * TURBO_C2[(b >> 4) & 3]);
            out_vec[3] += w * (cnorm * TURBO_C2[(b >> 6) & 3]);
        }
    }

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
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
