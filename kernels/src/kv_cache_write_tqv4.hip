#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __global__ void kv_cache_write_tqv4(
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim,
    int value_bits
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads || (head_dim != 128 && head_dim != 256) || (value_bits != 2 && value_bits != 3 && value_bits != 4)) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + (head_dim * value_bits) / 8;
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
    if (value_bits == 4 && head_dim == 256) {
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
    if (value_bits == 4 && head_dim == 256) {
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
    int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size || (head_dim != 128 && head_dim != 256) || (value_bits != 2 && value_bits != 3 && value_bits != 4)) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int kv_dim = n_kv_heads * head_dim;
    const int bytes_per_head = 4 + (head_dim * value_bits) / 8;
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
    if (value_bits == 4 && head_dim == 256) {
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
    if (value_bits == 4 && head_dim == 256) {
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
