#include <hip/hip_runtime.h>

// Experimental RDNA3.5 MQ4 x I4 decode path.
//
// Weights stay in the existing HFQ4-G256/MQ4 layout:
//   [f32 scale][f32 zero][128B u4 nibbles]
//
// Activations are pre-rotated by the normal MQ4 rotation path, then packed per
// 256-element group by mq4_pack_i4_x:
//   dx      = clip(abs(x_rot)) / qmax   (default qmax=8 from dispatch)
//   a_i     = clamp(round(x_rot_i / dx), -8, 7)
//   sum_a_g = sum(a_i)
//
// The dot kernel uses signed i4 dot8. Unsigned q in [0, 15] is converted to
// signed q-8 by toggling the i4 sign bit (q ^ 8), then the +8*sum(a)
// correction is folded into the per-group scalar term:
//
//   sum((sc*q + zp) * dx*a)
//     = sc*dx*dot_i4(q - 8, a) + (zp + 8*sc)*dx*sum(a)

__device__ __forceinline__ int clamp_i4(int v) {
    return v < -8 ? -8 : (v > 7 ? 7 : v);
}

__device__ __forceinline__ int signed_i4_dot8(unsigned int q_shifted, unsigned int a_pack) {
    return __ockl_sdot8((int)q_shifted, (int)a_pack, 0, false);
}

extern "C" __launch_bounds__(32, 16)
__global__ void mq4_pack_i4_x(
    const float* __restrict__ x,
    unsigned int* __restrict__ x_i4,
    float* __restrict__ x_dx,
    int* __restrict__ x_sum_a,
    int K,
    float qmax,
    float clip_rms
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups = K / 256;
    if (group >= groups) return;

    const int base = group * 256 + tid * 8;

    float v0 = x[base];
    float v1 = x[base + 1];
    float v2 = x[base + 2];
    float v3 = x[base + 3];
    float v4 = x[base + 4];
    float v5 = x[base + 5];
    float v6 = x[base + 6];
    float v7 = x[base + 7];

    float amax = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                              fmaxf(fabsf(v2), fabsf(v3))),
                       fmaxf(fmaxf(fabsf(v4), fabsf(v5)),
                              fmaxf(fabsf(v6), fabsf(v7))));

    __shared__ float s_amax[32];
    __shared__ float s_sumsq[32];
    __shared__ int s_sum[32];
    s_amax[tid] = amax;
    s_sumsq[tid] = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    float group_amax = s_amax[0];
    if (clip_rms > 0.0f) {
        float rms = sqrtf(s_sumsq[0] * (1.0f / 256.0f));
        group_amax = fminf(group_amax, clip_rms * rms);
    }
    qmax = fminf(fmaxf(qmax, 1.0f), 8.0f);
    const float dx = group_amax > 0.0f ? group_amax / qmax : 0.0f;
    const float inv_dx = group_amax > 0.0f ? qmax / group_amax : 0.0f;

    int a0 = clamp_i4(__float2int_rn(v0 * inv_dx));
    int a1 = clamp_i4(__float2int_rn(v1 * inv_dx));
    int a2 = clamp_i4(__float2int_rn(v2 * inv_dx));
    int a3 = clamp_i4(__float2int_rn(v3 * inv_dx));
    int a4 = clamp_i4(__float2int_rn(v4 * inv_dx));
    int a5 = clamp_i4(__float2int_rn(v5 * inv_dx));
    int a6 = clamp_i4(__float2int_rn(v6 * inv_dx));
    int a7 = clamp_i4(__float2int_rn(v7 * inv_dx));

    unsigned int pack = ((unsigned int)(a0 & 0xF))
        | ((unsigned int)(a1 & 0xF) << 4)
        | ((unsigned int)(a2 & 0xF) << 8)
        | ((unsigned int)(a3 & 0xF) << 12)
        | ((unsigned int)(a4 & 0xF) << 16)
        | ((unsigned int)(a5 & 0xF) << 20)
        | ((unsigned int)(a6 & 0xF) << 24)
        | ((unsigned int)(a7 & 0xF) << 28);

    x_i4[group * 32 + tid] = pack;

    int sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    s_sum[tid] = sum;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        x_dx[group] = dx;
        x_sum_a[group] = s_sum[0];
    }
}

__device__ __forceinline__ float load_f32(const char* p) {
    return __builtin_bit_cast(float, *(const unsigned int*)p);
}

#define MQ4_DOT8_GROUP(gidx, acc_var) do {                                      \
    const char* gp = row_ptr + (gidx) * 136;                                     \
    float sc = load_f32(gp);                                                     \
    float zp = load_f32(gp + 4);                                                 \
    unsigned int q = *(const unsigned int*)(gp + 8 + boff);                      \
    unsigned int a = x_i4[(gidx) * 32 + tid];                                    \
    int dot = signed_i4_dot8(q ^ 0x88888888u, a);                                \
    float dx = x_dx[(gidx)];                                                     \
    (acc_var) += sc * dx * (float)dot;                                           \
    if (tid == 0) {                                                              \
        (acc_var) += (zp + 8.0f * sc) * dx * (float)x_sum_a[(gidx)];              \
    }                                                                            \
} while (0)

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_i4_dot8(
    const char* __restrict__ A,
    const unsigned int* __restrict__ x_i4,
    const float* __restrict__ x_dx,
    const int* __restrict__ x_sum_a,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        MQ4_DOT8_GROUP(g, acc0);
        MQ4_DOT8_GROUP(g + 1, acc1);
        MQ4_DOT8_GROUP(g + 2, acc2);
        MQ4_DOT8_GROUP(g + 3, acc3);
    }
    if (tail >= 1) MQ4_DOT8_GROUP(quads << 2, acc0);
    if (tail >= 2) MQ4_DOT8_GROUP((quads << 2) + 1, acc1);
    if (tail >= 3) MQ4_DOT8_GROUP((quads << 2) + 2, acc2);

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_i4_dot8_residual(
    const char* __restrict__ A,
    const unsigned int* __restrict__ x_i4,
    const float* __restrict__ x_dx,
    const int* __restrict__ x_sum_a,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        MQ4_DOT8_GROUP(g, acc0);
        MQ4_DOT8_GROUP(g + 1, acc1);
        MQ4_DOT8_GROUP(g + 2, acc2);
        MQ4_DOT8_GROUP(g + 3, acc3);
    }
    if (tail >= 1) MQ4_DOT8_GROUP(quads << 2, acc0);
    if (tail >= 2) MQ4_DOT8_GROUP((quads << 2) + 1, acc1);
    if (tail >= 3) MQ4_DOT8_GROUP((quads << 2) + 2, acc2);

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] += acc;
}

#undef MQ4_DOT8_GROUP
