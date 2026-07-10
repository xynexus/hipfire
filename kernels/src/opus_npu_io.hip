// GPU producer/consumer for the AIE2P whole-array scaled Opus dma-buf layout.
// The math and FWHT sign convention match mq8_rotate_quantize_x and the CPU
// Opus oracle; only the destination/source addressing is AIE physical layout.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void opus_npu_pack_activations(
    const float* __restrict__ input,
    unsigned char* __restrict__ packed,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K,
    int rows,
    int cache_rows,
    int groups,
    int n_macros,
    int outblocks,
    int mode_w8
) {
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    if (group >= groups || row >= cache_rows) return;

    const int d0 = tid * 8;
    const int group_base = group * 256;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
    if (row < rows) {
        const long long row_base = (long long)row * K;
        #define LOAD_VALUE(dst, lane) do { \
            const int col = group_base + d0 + (lane); \
            if (col < K) { \
                float value = input[row_base + col]; \
                if (awq_scale) value /= awq_scale[col]; \
                (dst) = value * signs1[d0 + (lane)]; \
            } \
        } while (0)
        LOAD_VALUE(v0, 0); LOAD_VALUE(v1, 1); LOAD_VALUE(v2, 2); LOAD_VALUE(v3, 3);
        LOAD_VALUE(v4, 4); LOAD_VALUE(v5, 5); LOAD_VALUE(v6, 6); LOAD_VALUE(v7, 7);
        #undef LOAD_VALUE
    }

    float t;
    t = v0; v0 += v1; v1 = t - v1;
    t = v2; v2 += v3; v3 = t - v3;
    t = v4; v4 += v5; v5 = t - v5;
    t = v6; v6 += v7; v7 = t - v7;
    t = v0; v0 += v2; v2 = t - v2;
    t = v1; v1 += v3; v3 = t - v3;
    t = v4; v4 += v6; v6 = t - v6;
    t = v5; v5 += v7; v7 = t - v7;
    t = v0; v0 += v4; v4 = t - v4;
    t = v1; v1 += v5; v5 = t - v5;
    t = v2; v2 += v6; v6 = t - v6;
    t = v3; v3 += v7; v7 = t - v7;

    #define HBFLY(v, pat, str) do { \
        float p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        (v) = (tid & (str)) ? p - (v) : (v) + p; \
    } while (0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)
    HBFLY8(0x041F, 1); HBFLY8(0x081F, 2); HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8); HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float norm = 0.0625f;
    v0 *= norm * signs2[d0];     v1 *= norm * signs2[d0 + 1];
    v2 *= norm * signs2[d0 + 2]; v3 *= norm * signs2[d0 + 3];
    v4 *= norm * signs2[d0 + 4]; v5 *= norm * signs2[d0 + 5];
    v6 *= norm * signs2[d0 + 6]; v7 *= norm * signs2[d0 + 7];

    float max_abs = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
                          fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    for (int offset = 16; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, offset));
    const float scale = (row < rows && max_abs > 0.0f) ? max_abs / 127.0f : 0.0f;
    signed char q[8];
    #define QUANTIZE(v) (scale > 0.0f \
        ? (signed char)fminf(fmaxf(roundf((v) / scale), -127.0f), 127.0f) \
        : (signed char)0)
    q[0] = QUANTIZE(v0); q[1] = QUANTIZE(v1);
    q[2] = QUANTIZE(v2); q[3] = QUANTIZE(v3);
    q[4] = QUANTIZE(v4); q[5] = QUANTIZE(v5);
    q[6] = QUANTIZE(v6); q[7] = QUANTIZE(v7);
    #undef QUANTIZE

    const int inner_k = mode_w8 ? 8 : 16;
    const int mr = mode_w8 ? 8 : 4;
    const int m_macro = row / 96;
    const int within_macro = row - m_macro * 96;
    const int stripe = within_macro / 24;
    const int within_stripe = within_macro - stripe * 24;
    const int lm = within_stripe / mr;
    const int local_row = within_stripe - lm * mr;
    const int inblocks = outblocks * groups;
    for (int n_macro = 0; n_macro < n_macros; ++n_macro) {
        const int outblock = m_macro * n_macros + n_macro;
        const int block = outblock * groups + group;
        const long long base = ((long long)stripe * inblocks + block) * 8192;
        #pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int inner = d0 + lane;
            const int kt = inner / inner_k;
            const int kk = inner - kt * inner_k;
            const int target = (lm * (256 / inner_k) + kt) * 64 + local_row * inner_k + kk;
            ((signed char*)packed)[base + target] = q[lane];
        }
        if (tid == 0) {
            *(float*)(packed + base + 6144 + within_stripe * 4) = scale;
        }
    }
}

extern "C" __global__ void opus_npu_unpack_output(
    const float* __restrict__ packed,
    float* __restrict__ out0,
    float* __restrict__ out1,
    float* __restrict__ out2,
    int rows,
    int N,
    int width0,
    int width1,
    int width2,
    int cols,
    int n_macros,
    int outblocks,
    int mode_w8
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * N) return;
    const int row = index / N;
    const int col = index - row * N;
    const int lm_count = mode_w8 ? 3 : 6;
    const int ln_count = mode_w8 ? 3 : 6;
    const int mr = mode_w8 ? 8 : 4;
    const int cols_stripe = mode_w8 ? 48 : 96;
    const int c_core = mode_w8 ? 1152 : 2304;
    const int macro_n = cols * cols_stripe;

    const int m_macro = row / 96;
    const int within_macro = row - m_macro * 96;
    const int row_stripe = within_macro / 24;
    const int within_stripe = within_macro - row_stripe * 24;
    const int lm = within_stripe / mr;
    const int local_row = within_stripe - lm * mr;
    const int n_macro = col / macro_n;
    const int within_n_macro = col - n_macro * macro_n;
    const int col_stripe = within_n_macro / cols_stripe;
    const int within_col_stripe = within_n_macro - col_stripe * cols_stripe;
    const int ln = within_col_stripe / 16;
    const int local_col = within_col_stripe - ln * 16;
    const int outblock = m_macro * n_macros + n_macro;
    const int c_join = 4 * c_core;
    const long long source = ((long long)col_stripe * outblocks + outblock) * c_join
        + row_stripe * c_core
        + (lm * ln_count + ln) * mr * 16
        + local_row * 16 + local_col;
    const float value = packed[source];

    if (col < width0) {
        out0[(long long)row * width0 + col] = value;
    } else if (col < width0 + width1) {
        const int role_col = col - width0;
        out1[(long long)row * width1 + role_col] = value;
    } else {
        const int role_col = col - width0 - width1;
        out2[(long long)row * width2 + role_col] = value;
    }
}
