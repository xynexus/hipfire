// Exact (erf-based) GELU. PyTorch `nn.GELU()` default — used by Gemma 4
// E-series per-layer-embedding block (per `nn.GELU()` in modeling code).
// Distinct from gelu_tanh.hip which is the tanh approximation used by
// Gemma's FFN SwiGLU activation (`gelu_pytorch_tanh`).
//
//   gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// The two approximations agree to ~0.03 max absolute error per element,
// but the per-layer-embed signal compounds across 35-42 layers under the
// learned `layer_scalar` gain — small per-element divergence accumulates.
#include <hip/hip_runtime.h>

extern "C" __global__ void gelu_erf_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    // 1 / sqrt(2) = 0.70710678118f
    out[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118f));
}
