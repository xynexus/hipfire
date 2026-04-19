#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// __restrict__ stripped: hipcc/clang (ROCm 7.2) miscompile on gfx1100 —
// leaving restrict on pointer params of this kernel produces a DN state
// that is deterministically wrong at layer 0 and nondeterministic at
// layers ≥ 1 in batched prefill (diagnosed via prefill_state_diff harness).
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    signed char* s_q8,
    float* s_scales,
    float* output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // 4KB

    // Dequant Q8 tile → FP32 in LDS
    signed char* sq_base = s_q8 + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        int r = i / HD;
        float scale = s_scales[h * HD + row_start + r];
        S_tile[i] = scale * (float)sq_base[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        float q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Requant FP32 → Q8 with stochastic rounding.
    // Deterministic roundf systematically crushes small values to 0, causing
    // cumulative state corruption over hundreds of tokens.
    // Stochastic rounding: floor(x + uniform) has E[result] = x (unbiased).
    // PRNG: position-derived LCG — deterministic, no shared state, no extra registers.
    for (int r = 0; r < TILE_ROWS; r++) {
        float* sr = S_tile + r * HD + col;
        float my_max = fmaxf(fmaxf(fabsf(sr[0]), fabsf(sr[1])), fmaxf(fabsf(sr[2]), fabsf(sr[3])));
        for (int o = 16; o > 0; o >>= 1)
            my_max = fmaxf(my_max, __shfl_xor(my_max, o));

        float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
        signed char* dst = sq_base + r * HD + col;

        // Stochastic rounding: generate 4 uniform random values in [0, 1)
        // Seed from position + row max (changes every token, independent of individual values)
        unsigned int rng = (unsigned int)(h * 1664525u + (row_start + r) * 1013904223u
                         + col * 214013u + __float_as_uint(my_max) * 2531011u);
        #define STOCH_ROUND(val) ({ \
            rng = rng * 1664525u + 1013904223u; \
            float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
            float scaled = (val) * inv_s; \
            (signed char)fminf(fmaxf(floorf(scaled + noise), -128.0f), 127.0f); \
        })
        dst[0] = STOCH_ROUND(sr[0]);
        dst[1] = STOCH_ROUND(sr[1]);
        dst[2] = STOCH_ROUND(sr[2]);
        dst[3] = STOCH_ROUND(sr[3]);
        #undef STOCH_ROUND
        if (tid == 0) s_scales[h * HD + row_start + r] = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
    }
}
