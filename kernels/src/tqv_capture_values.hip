#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Capture the exact scalar distribution seen by TQV value quantization:
// normalize each V head, apply the same FWHT/sign transform as
// kv_cache_write_tqv4, then write transformed f32 scalars. One block per
// KV head, 32 lanes per block. Output layout is [n_kv_heads, head_dim].
extern "C" __global__ void tqv_capture_values(
    float* __restrict__ dst,
    const float* __restrict__ v_src,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads || (head_dim != 128 && head_dim != 256)) return;
    const int tid = threadIdx.x;

    const float* src = v_src + h * head_dim;
    float* out = dst + h * head_dim;

    const int d0 = (head_dim == 256) ? tid * 8 : tid * 4;
    float v0 = src[d0 + 0], v1 = src[d0 + 1], v2 = src[d0 + 2], v3 = src[d0 + 3];
    float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
    if (head_dim == 256) {
        v4 = src[d0 + 4]; v5 = src[d0 + 5]; v6 = src[d0 + 6]; v7 = src[d0 + 7];
    }

    float lsq = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7;
    for (int off = 16; off > 0; off >>= 1) lsq += __shfl_xor(lsq, off);
    const float inv_norm = (lsq > 1e-20f) ? rsqrtf(lsq) : 0.0f;

    v0 *= inv_norm; v1 *= inv_norm; v2 *= inv_norm; v3 *= inv_norm;
    v4 *= inv_norm; v5 *= inv_norm; v6 *= inv_norm; v7 *= inv_norm;
    if (head_dim == 256) {
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);
        out[d0 + 0] = v0; out[d0 + 1] = v1; out[d0 + 2] = v2; out[d0 + 3] = v3;
        out[d0 + 4] = v4; out[d0 + 5] = v5; out[d0 + 6] = v6; out[d0 + 7] = v7;
    } else {
        fwht_shfl_forward(v0, v1, v2, v3, signs1, signs2, tid);
        out[d0 + 0] = v0; out[d0 + 1] = v1; out[d0 + 2] = v2; out[d0 + 3] = v3;
    }
}
