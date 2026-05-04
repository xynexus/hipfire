#include <hip/hip_runtime.h>

// Batched GELU-tanh + element-wise multiply: out[b][i] = gelu_tanh(gate[b][i]) * up[b][i].
// Used by Gemma 4 MoE Phase 2.5 to fuse the 8 per-expert (gelu + mul)
// launches in apply_moe_branch into a single dispatch (16 → 1 per layer).
//
// Grid: [ceil(K/256), batch]; blockDim 256; each thread handles one element.
// gate, up, out are [batch × K] row-major (per-expert hidden lives at
// offset b*K). In-place capable when out aliases gate or up.
//
// gelu_tanh(v) = 0.5 * v * (1 + tanh(0.7978845608 * (v + 0.044715 * v^3)))
// (matches the standalone gelu_tanh_f32 kernel; same constants).

extern "C" __global__ void gelu_tanh_mul_batched_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int K
) {
    const int chunk = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int i = chunk * 256 + tid;
    if (i >= K) return;
    const long long off = (long long)b * K + i;
    float g = gate[off];
    float u = up[off];
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    float gv = 0.5f * g * (1.0f + tanhf(inner));
    out[off] = gv * u;
}
