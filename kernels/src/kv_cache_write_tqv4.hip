#include <hip/hip_runtime.h>
#include "turbo_common.h"

__device__ int tqv_quantize_table(float x, const float* __restrict__ thresholds, int value_bits) {
    int idx = 0;
    const int n = (value_bits == 1) ? 2 : ((1 << value_bits) - 1);
    for (int i = 0; i < n; i++) idx += (x > thresholds[i]);
    return idx;
}

__device__ float tqv_centroid_table(int idx, int value_bits, int head_dim, const float* __restrict__ centroids) {
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

__device__ int turbo_quantize_ternary(float x, int head_dim) {
    const float* t = (head_dim == 256) ? TURBO_T1_256 : TURBO_T1;
    return (x > t[1]) ? 2 : ((x > t[0]) ? 1 : 0);
}

extern "C" __global__ void kv_cache_write_tqv4(
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim,
    int value_bits,
    const float* __restrict__ tqv_centroids,
    const float* __restrict__ tqv_thresholds
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads || (head_dim != 128 && head_dim != 256) || (value_bits != 1 && value_bits != 2 && value_bits != 3 && value_bits != 4)) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const int storage_bits = (value_bits == 1) ? 2 : value_bits;
    const int bytes_per_head = 4 + (head_dim * storage_bits) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const float* src = v_src + h * head_dim;
    unsigned char* out = v_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    const int d0 = (head_dim == 256) ? tid * 8 : tid * 4;
    float v0 = src[d0 + 0], v1 = src[d0 + 1], v2 = src[d0 + 2], v3 = src[d0 + 3];
    float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
    if (head_dim == 256) {
        v4 = src[d0 + 4]; v5 = src[d0 + 5]; v6 = src[d0 + 6]; v7 = src[d0 + 7];
    }

    float lsq = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7;
    for (int off = 16; off > 0; off >>= 1) lsq += __shfl_xor(lsq, off);
    const float orig_norm = sqrtf(lsq);
    const float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    v0 *= inv_norm; v1 *= inv_norm; v2 *= inv_norm; v3 *= inv_norm;
    v4 *= inv_norm; v5 *= inv_norm; v6 *= inv_norm; v7 *= inv_norm;
    if (head_dim == 256) {
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);
    } else {
        fwht_shfl_forward(v0, v1, v2, v3, signs1, signs2, tid);
    }

    int i0, i1, i2, i3, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
    if (tqv_thresholds != nullptr) {
        i0 = tqv_quantize_table(v0, tqv_thresholds, value_bits);
        i1 = tqv_quantize_table(v1, tqv_thresholds, value_bits);
        i2 = tqv_quantize_table(v2, tqv_thresholds, value_bits);
        i3 = tqv_quantize_table(v3, tqv_thresholds, value_bits);
        if (head_dim == 256) {
            i4 = tqv_quantize_table(v4, tqv_thresholds, value_bits);
            i5 = tqv_quantize_table(v5, tqv_thresholds, value_bits);
            i6 = tqv_quantize_table(v6, tqv_thresholds, value_bits);
            i7 = tqv_quantize_table(v7, tqv_thresholds, value_bits);
        }
    } else if (value_bits == 1) {
        i0 = turbo_quantize_ternary(v0, head_dim);
        i1 = turbo_quantize_ternary(v1, head_dim);
        i2 = turbo_quantize_ternary(v2, head_dim);
        i3 = turbo_quantize_ternary(v3, head_dim);
        if (head_dim == 256) {
            i4 = turbo_quantize_ternary(v4, head_dim);
            i5 = turbo_quantize_ternary(v5, head_dim);
            i6 = turbo_quantize_ternary(v6, head_dim);
            i7 = turbo_quantize_ternary(v7, head_dim);
        }
    } else if (value_bits == 4 && head_dim == 256) {
        i0 = turbo_quantize_4bit_256(v0);
        i1 = turbo_quantize_4bit_256(v1);
        i2 = turbo_quantize_4bit_256(v2);
        i3 = turbo_quantize_4bit_256(v3);
        i4 = turbo_quantize_4bit_256(v4);
        i5 = turbo_quantize_4bit_256(v5);
        i6 = turbo_quantize_4bit_256(v6);
        i7 = turbo_quantize_4bit_256(v7);
    } else if (value_bits == 4) {
        i0 = turbo_quantize_4bit(v0);
        i1 = turbo_quantize_4bit(v1);
        i2 = turbo_quantize_4bit(v2);
        i3 = turbo_quantize_4bit(v3);
    } else if (value_bits == 3 && head_dim == 256) {
        i0 = turbo_quantize_3bit_256(v0);
        i1 = turbo_quantize_3bit_256(v1);
        i2 = turbo_quantize_3bit_256(v2);
        i3 = turbo_quantize_3bit_256(v3);
        i4 = turbo_quantize_3bit_256(v4);
        i5 = turbo_quantize_3bit_256(v5);
        i6 = turbo_quantize_3bit_256(v6);
        i7 = turbo_quantize_3bit_256(v7);
    } else if (value_bits == 3) {
        i0 = turbo_quantize_3bit(v0);
        i1 = turbo_quantize_3bit(v1);
        i2 = turbo_quantize_3bit(v2);
        i3 = turbo_quantize_3bit(v3);
    } else if (head_dim == 256) {
        i0 = turbo_quantize_2bit_256(v0);
        i1 = turbo_quantize_2bit_256(v1);
        i2 = turbo_quantize_2bit_256(v2);
        i3 = turbo_quantize_2bit_256(v3);
        i4 = turbo_quantize_2bit_256(v4);
        i5 = turbo_quantize_2bit_256(v5);
        i6 = turbo_quantize_2bit_256(v6);
        i7 = turbo_quantize_2bit_256(v7);
    } else {
        i0 = turbo_quantize_2bit(v0);
        i1 = turbo_quantize_2bit(v1);
        i2 = turbo_quantize_2bit(v2);
        i3 = turbo_quantize_2bit(v3);
    }

    float rsq;
    if (tqv_centroids != nullptr) {
        rsq = tqv_centroid_table(i0, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i0, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i1, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i1, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i2, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i2, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i3, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i3, value_bits, head_dim, tqv_centroids);
        if (head_dim == 256) {
            rsq += tqv_centroid_table(i4, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i4, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i5, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i5, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i6, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i6, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i7, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i7, value_bits, head_dim, tqv_centroids);
        }
    } else if (value_bits == 1 && head_dim == 256) {
        rsq = TURBO_C1_256[i0]*TURBO_C1_256[i0] + TURBO_C1_256[i1]*TURBO_C1_256[i1]
             + TURBO_C1_256[i2]*TURBO_C1_256[i2] + TURBO_C1_256[i3]*TURBO_C1_256[i3]
             + TURBO_C1_256[i4]*TURBO_C1_256[i4] + TURBO_C1_256[i5]*TURBO_C1_256[i5]
             + TURBO_C1_256[i6]*TURBO_C1_256[i6] + TURBO_C1_256[i7]*TURBO_C1_256[i7];
    } else if (value_bits == 1) {
        rsq = TURBO_C1[i0]*TURBO_C1[i0] + TURBO_C1[i1]*TURBO_C1[i1]
             + TURBO_C1[i2]*TURBO_C1[i2] + TURBO_C1[i3]*TURBO_C1[i3];
    } else if (value_bits == 4 && head_dim == 256) {
        rsq = TURBO_C4_256[i0]*TURBO_C4_256[i0] + TURBO_C4_256[i1]*TURBO_C4_256[i1]
             + TURBO_C4_256[i2]*TURBO_C4_256[i2] + TURBO_C4_256[i3]*TURBO_C4_256[i3]
             + TURBO_C4_256[i4]*TURBO_C4_256[i4] + TURBO_C4_256[i5]*TURBO_C4_256[i5]
             + TURBO_C4_256[i6]*TURBO_C4_256[i6] + TURBO_C4_256[i7]*TURBO_C4_256[i7];
    } else if (value_bits == 4) {
        rsq = TURBO_C4[i0]*TURBO_C4[i0] + TURBO_C4[i1]*TURBO_C4[i1]
             + TURBO_C4[i2]*TURBO_C4[i2] + TURBO_C4[i3]*TURBO_C4[i3];
    } else if (value_bits == 3 && head_dim == 256) {
        rsq = TURBO_C3_256[i0]*TURBO_C3_256[i0] + TURBO_C3_256[i1]*TURBO_C3_256[i1]
             + TURBO_C3_256[i2]*TURBO_C3_256[i2] + TURBO_C3_256[i3]*TURBO_C3_256[i3]
             + TURBO_C3_256[i4]*TURBO_C3_256[i4] + TURBO_C3_256[i5]*TURBO_C3_256[i5]
             + TURBO_C3_256[i6]*TURBO_C3_256[i6] + TURBO_C3_256[i7]*TURBO_C3_256[i7];
    } else if (value_bits == 3) {
        rsq = TURBO_C3[i0]*TURBO_C3[i0] + TURBO_C3[i1]*TURBO_C3[i1]
             + TURBO_C3[i2]*TURBO_C3[i2] + TURBO_C3[i3]*TURBO_C3[i3];
    } else if (head_dim == 256) {
        rsq = TURBO_C2_256[i0]*TURBO_C2_256[i0] + TURBO_C2_256[i1]*TURBO_C2_256[i1]
             + TURBO_C2_256[i2]*TURBO_C2_256[i2] + TURBO_C2_256[i3]*TURBO_C2_256[i3]
             + TURBO_C2_256[i4]*TURBO_C2_256[i4] + TURBO_C2_256[i5]*TURBO_C2_256[i5]
             + TURBO_C2_256[i6]*TURBO_C2_256[i6] + TURBO_C2_256[i7]*TURBO_C2_256[i7];
    } else {
        rsq = TURBO_C2[i0]*TURBO_C2[i0] + TURBO_C2[i1]*TURBO_C2[i1]
             + TURBO_C2[i2]*TURBO_C2[i2] + TURBO_C2[i3]*TURBO_C2[i3];
    }
    for (int off = 16; off > 0; off >>= 1) rsq += __shfl_xor(rsq, off);
    const float recon_norm = sqrtf(rsq);
    const float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    if (value_bits == 4) {
        unsigned char* base = out + 4 + ((head_dim == 256) ? tid * 4 : tid * 2);
        base[0] = (unsigned char)((i1 << 4) | (i0 & 0xF));
        base[1] = (unsigned char)((i3 << 4) | (i2 & 0xF));
        if (head_dim == 256) {
            base[2] = (unsigned char)((i5 << 4) | (i4 & 0xF));
            base[3] = (unsigned char)((i7 << 4) | (i6 & 0xF));
        }
    } else if (value_bits == 3 && head_dim == 256) {
        unsigned char* base = out + 4 + tid * 3;
        unsigned int pack = (unsigned int)(i0 & 7)
                          | ((unsigned int)(i1 & 7) << 3)
                          | ((unsigned int)(i2 & 7) << 6)
                          | ((unsigned int)(i3 & 7) << 9)
                          | ((unsigned int)(i4 & 7) << 12)
                          | ((unsigned int)(i5 & 7) << 15)
                          | ((unsigned int)(i6 & 7) << 18)
                          | ((unsigned int)(i7 & 7) << 21);
        base[0] = (unsigned char)(pack & 0xFF);
        base[1] = (unsigned char)((pack >> 8) & 0xFF);
        base[2] = (unsigned char)((pack >> 16) & 0xFF);
    } else if (value_bits == 3) {
        int j0 = __shfl_xor(i0, 1);
        int j1 = __shfl_xor(i1, 1);
        int j2 = __shfl_xor(i2, 1);
        int j3 = __shfl_xor(i3, 1);
        if ((tid & 1) == 0) {
            unsigned char* base = out + 4 + (tid >> 1) * 3;
            unsigned int pack = (unsigned int)(i0 & 7)
                              | ((unsigned int)(i1 & 7) << 3)
                              | ((unsigned int)(i2 & 7) << 6)
                              | ((unsigned int)(i3 & 7) << 9)
                              | ((unsigned int)(j0 & 7) << 12)
                              | ((unsigned int)(j1 & 7) << 15)
                              | ((unsigned int)(j2 & 7) << 18)
                              | ((unsigned int)(j3 & 7) << 21);
            base[0] = (unsigned char)(pack & 0xFF);
            base[1] = (unsigned char)((pack >> 8) & 0xFF);
            base[2] = (unsigned char)((pack >> 16) & 0xFF);
        }
    } else {
        unsigned char* base = out + 4 + ((head_dim == 256) ? tid * 2 : tid);
        base[0] = (unsigned char)((i0 & 3) | ((i1 & 3) << 2) | ((i2 & 3) << 4) | ((i3 & 3) << 6));
        if (head_dim == 256) {
            base[1] = (unsigned char)((i4 & 3) | ((i5 & 3) << 2) | ((i6 & 3) << 4) | ((i7 & 3) << 6));
        }
    }
}

extern "C" __global__ void kv_cache_write_tqv4_batched(
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ v_src,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim,
    int value_bits,
    const float* __restrict__ tqv_centroids,
    const float* __restrict__ tqv_thresholds,
    int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size || (head_dim != 128 && head_dim != 256) || (value_bits != 1 && value_bits != 2 && value_bits != 3 && value_bits != 4)) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int kv_dim = n_kv_heads * head_dim;
    const int storage_bits = (value_bits == 1) ? 2 : value_bits;
    const int bytes_per_head = 4 + (head_dim * storage_bits) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const float* src = v_src + bid * kv_dim + h * head_dim;
    unsigned char* out = v_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    const int d0 = (head_dim == 256) ? tid * 8 : tid * 4;
    float v0 = src[d0 + 0], v1 = src[d0 + 1], v2 = src[d0 + 2], v3 = src[d0 + 3];
    float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
    if (head_dim == 256) {
        v4 = src[d0 + 4]; v5 = src[d0 + 5]; v6 = src[d0 + 6]; v7 = src[d0 + 7];
    }

    float lsq = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7;
    for (int off = 16; off > 0; off >>= 1) lsq += __shfl_xor(lsq, off);
    const float orig_norm = sqrtf(lsq);
    const float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    v0 *= inv_norm; v1 *= inv_norm; v2 *= inv_norm; v3 *= inv_norm;
    v4 *= inv_norm; v5 *= inv_norm; v6 *= inv_norm; v7 *= inv_norm;
    if (head_dim == 256) {
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);
    } else {
        fwht_shfl_forward(v0, v1, v2, v3, signs1, signs2, tid);
    }

    int i0, i1, i2, i3, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
    if (tqv_thresholds != nullptr) {
        i0 = tqv_quantize_table(v0, tqv_thresholds, value_bits);
        i1 = tqv_quantize_table(v1, tqv_thresholds, value_bits);
        i2 = tqv_quantize_table(v2, tqv_thresholds, value_bits);
        i3 = tqv_quantize_table(v3, tqv_thresholds, value_bits);
        if (head_dim == 256) {
            i4 = tqv_quantize_table(v4, tqv_thresholds, value_bits);
            i5 = tqv_quantize_table(v5, tqv_thresholds, value_bits);
            i6 = tqv_quantize_table(v6, tqv_thresholds, value_bits);
            i7 = tqv_quantize_table(v7, tqv_thresholds, value_bits);
        }
    } else if (value_bits == 1) {
        i0 = turbo_quantize_ternary(v0, head_dim);
        i1 = turbo_quantize_ternary(v1, head_dim);
        i2 = turbo_quantize_ternary(v2, head_dim);
        i3 = turbo_quantize_ternary(v3, head_dim);
        if (head_dim == 256) {
            i4 = turbo_quantize_ternary(v4, head_dim);
            i5 = turbo_quantize_ternary(v5, head_dim);
            i6 = turbo_quantize_ternary(v6, head_dim);
            i7 = turbo_quantize_ternary(v7, head_dim);
        }
    } else if (value_bits == 4 && head_dim == 256) {
        i0 = turbo_quantize_4bit_256(v0);
        i1 = turbo_quantize_4bit_256(v1);
        i2 = turbo_quantize_4bit_256(v2);
        i3 = turbo_quantize_4bit_256(v3);
        i4 = turbo_quantize_4bit_256(v4);
        i5 = turbo_quantize_4bit_256(v5);
        i6 = turbo_quantize_4bit_256(v6);
        i7 = turbo_quantize_4bit_256(v7);
    } else if (value_bits == 4) {
        i0 = turbo_quantize_4bit(v0);
        i1 = turbo_quantize_4bit(v1);
        i2 = turbo_quantize_4bit(v2);
        i3 = turbo_quantize_4bit(v3);
    } else if (value_bits == 3 && head_dim == 256) {
        i0 = turbo_quantize_3bit_256(v0);
        i1 = turbo_quantize_3bit_256(v1);
        i2 = turbo_quantize_3bit_256(v2);
        i3 = turbo_quantize_3bit_256(v3);
        i4 = turbo_quantize_3bit_256(v4);
        i5 = turbo_quantize_3bit_256(v5);
        i6 = turbo_quantize_3bit_256(v6);
        i7 = turbo_quantize_3bit_256(v7);
    } else if (value_bits == 3) {
        i0 = turbo_quantize_3bit(v0);
        i1 = turbo_quantize_3bit(v1);
        i2 = turbo_quantize_3bit(v2);
        i3 = turbo_quantize_3bit(v3);
    } else if (head_dim == 256) {
        i0 = turbo_quantize_2bit_256(v0);
        i1 = turbo_quantize_2bit_256(v1);
        i2 = turbo_quantize_2bit_256(v2);
        i3 = turbo_quantize_2bit_256(v3);
        i4 = turbo_quantize_2bit_256(v4);
        i5 = turbo_quantize_2bit_256(v5);
        i6 = turbo_quantize_2bit_256(v6);
        i7 = turbo_quantize_2bit_256(v7);
    } else {
        i0 = turbo_quantize_2bit(v0);
        i1 = turbo_quantize_2bit(v1);
        i2 = turbo_quantize_2bit(v2);
        i3 = turbo_quantize_2bit(v3);
    }

    float rsq;
    if (tqv_centroids != nullptr) {
        rsq = tqv_centroid_table(i0, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i0, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i1, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i1, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i2, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i2, value_bits, head_dim, tqv_centroids)
            + tqv_centroid_table(i3, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i3, value_bits, head_dim, tqv_centroids);
        if (head_dim == 256) {
            rsq += tqv_centroid_table(i4, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i4, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i5, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i5, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i6, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i6, value_bits, head_dim, tqv_centroids)
                +  tqv_centroid_table(i7, value_bits, head_dim, tqv_centroids) * tqv_centroid_table(i7, value_bits, head_dim, tqv_centroids);
        }
    } else if (value_bits == 1 && head_dim == 256) {
        rsq = TURBO_C1_256[i0]*TURBO_C1_256[i0] + TURBO_C1_256[i1]*TURBO_C1_256[i1]
             + TURBO_C1_256[i2]*TURBO_C1_256[i2] + TURBO_C1_256[i3]*TURBO_C1_256[i3]
             + TURBO_C1_256[i4]*TURBO_C1_256[i4] + TURBO_C1_256[i5]*TURBO_C1_256[i5]
             + TURBO_C1_256[i6]*TURBO_C1_256[i6] + TURBO_C1_256[i7]*TURBO_C1_256[i7];
    } else if (value_bits == 1) {
        rsq = TURBO_C1[i0]*TURBO_C1[i0] + TURBO_C1[i1]*TURBO_C1[i1]
             + TURBO_C1[i2]*TURBO_C1[i2] + TURBO_C1[i3]*TURBO_C1[i3];
    } else if (value_bits == 4 && head_dim == 256) {
        rsq = TURBO_C4_256[i0]*TURBO_C4_256[i0] + TURBO_C4_256[i1]*TURBO_C4_256[i1]
             + TURBO_C4_256[i2]*TURBO_C4_256[i2] + TURBO_C4_256[i3]*TURBO_C4_256[i3]
             + TURBO_C4_256[i4]*TURBO_C4_256[i4] + TURBO_C4_256[i5]*TURBO_C4_256[i5]
             + TURBO_C4_256[i6]*TURBO_C4_256[i6] + TURBO_C4_256[i7]*TURBO_C4_256[i7];
    } else if (value_bits == 4) {
        rsq = TURBO_C4[i0]*TURBO_C4[i0] + TURBO_C4[i1]*TURBO_C4[i1]
             + TURBO_C4[i2]*TURBO_C4[i2] + TURBO_C4[i3]*TURBO_C4[i3];
    } else if (value_bits == 3 && head_dim == 256) {
        rsq = TURBO_C3_256[i0]*TURBO_C3_256[i0] + TURBO_C3_256[i1]*TURBO_C3_256[i1]
             + TURBO_C3_256[i2]*TURBO_C3_256[i2] + TURBO_C3_256[i3]*TURBO_C3_256[i3]
             + TURBO_C3_256[i4]*TURBO_C3_256[i4] + TURBO_C3_256[i5]*TURBO_C3_256[i5]
             + TURBO_C3_256[i6]*TURBO_C3_256[i6] + TURBO_C3_256[i7]*TURBO_C3_256[i7];
    } else if (value_bits == 3) {
        rsq = TURBO_C3[i0]*TURBO_C3[i0] + TURBO_C3[i1]*TURBO_C3[i1]
             + TURBO_C3[i2]*TURBO_C3[i2] + TURBO_C3[i3]*TURBO_C3[i3];
    } else if (head_dim == 256) {
        rsq = TURBO_C2_256[i0]*TURBO_C2_256[i0] + TURBO_C2_256[i1]*TURBO_C2_256[i1]
             + TURBO_C2_256[i2]*TURBO_C2_256[i2] + TURBO_C2_256[i3]*TURBO_C2_256[i3]
             + TURBO_C2_256[i4]*TURBO_C2_256[i4] + TURBO_C2_256[i5]*TURBO_C2_256[i5]
             + TURBO_C2_256[i6]*TURBO_C2_256[i6] + TURBO_C2_256[i7]*TURBO_C2_256[i7];
    } else {
        rsq = TURBO_C2[i0]*TURBO_C2[i0] + TURBO_C2[i1]*TURBO_C2[i1]
             + TURBO_C2[i2]*TURBO_C2[i2] + TURBO_C2[i3]*TURBO_C2[i3];
    }
    for (int off = 16; off > 0; off >>= 1) rsq += __shfl_xor(rsq, off);
    const float recon_norm = sqrtf(rsq);
    const float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    if (value_bits == 4) {
        unsigned char* base = out + 4 + ((head_dim == 256) ? tid * 4 : tid * 2);
        base[0] = (unsigned char)((i1 << 4) | (i0 & 0xF));
        base[1] = (unsigned char)((i3 << 4) | (i2 & 0xF));
        if (head_dim == 256) {
            base[2] = (unsigned char)((i5 << 4) | (i4 & 0xF));
            base[3] = (unsigned char)((i7 << 4) | (i6 & 0xF));
        }
    } else if (value_bits == 3 && head_dim == 256) {
        unsigned char* base = out + 4 + tid * 3;
        unsigned int pack = (unsigned int)(i0 & 7)
                          | ((unsigned int)(i1 & 7) << 3)
                          | ((unsigned int)(i2 & 7) << 6)
                          | ((unsigned int)(i3 & 7) << 9)
                          | ((unsigned int)(i4 & 7) << 12)
                          | ((unsigned int)(i5 & 7) << 15)
                          | ((unsigned int)(i6 & 7) << 18)
                          | ((unsigned int)(i7 & 7) << 21);
        base[0] = (unsigned char)(pack & 0xFF);
        base[1] = (unsigned char)((pack >> 8) & 0xFF);
        base[2] = (unsigned char)((pack >> 16) & 0xFF);
    } else if (value_bits == 3) {
        int j0 = __shfl_xor(i0, 1);
        int j1 = __shfl_xor(i1, 1);
        int j2 = __shfl_xor(i2, 1);
        int j3 = __shfl_xor(i3, 1);
        if ((tid & 1) == 0) {
            unsigned char* base = out + 4 + (tid >> 1) * 3;
            unsigned int pack = (unsigned int)(i0 & 7)
                              | ((unsigned int)(i1 & 7) << 3)
                              | ((unsigned int)(i2 & 7) << 6)
                              | ((unsigned int)(i3 & 7) << 9)
                              | ((unsigned int)(j0 & 7) << 12)
                              | ((unsigned int)(j1 & 7) << 15)
                              | ((unsigned int)(j2 & 7) << 18)
                              | ((unsigned int)(j3 & 7) << 21);
            base[0] = (unsigned char)(pack & 0xFF);
            base[1] = (unsigned char)((pack >> 8) & 0xFF);
            base[2] = (unsigned char)((pack >> 16) & 0xFF);
        }
    } else {
        unsigned char* base = out + 4 + ((head_dim == 256) ? tid * 2 : tid);
        base[0] = (unsigned char)((i0 & 3) | ((i1 & 3) << 2) | ((i2 & 3) << 4) | ((i3 & 3) << 6));
        if (head_dim == 256) {
            base[1] = (unsigned char)((i4 & 3) | ((i5 & 3) << 2) | ((i6 & 3) << 4) | ((i7 & 3) << 6));
        }
    }
}
