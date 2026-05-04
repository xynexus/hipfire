#include <hip/hip_runtime.h>

// Compose Gemma 4 MoE per-expert composite weights on GPU:
//   fused_weights[ki] = topk_weights[ki] * per_expert_scale[topk_indices[ki]]
//
// Replaces the CPU-side fold (which required two D2H syncs of
// topk_indices and topk_weights, plus a host-to-device upload of the
// fused result, per layer per token). With this kernel, the entire
// MoE branch becomes hipGraph-capture-safe end-to-end.
//
// Single workgroup, k_top threads (8 for Gemma 4 26B-A4B); each thread
// handles one (ki) slot. Bounds-clamping per_expert_scale[expert_id]
// is deferred to the GPU top-K kernel which already produces values in
// [0, n_exp).

extern "C" __global__ void fold_topk_with_per_expert_scale_k8(
    const int*   __restrict__ topk_indices,        // [k_top]
    const float* __restrict__ topk_weights,        // [k_top]
    const float* __restrict__ per_expert_scale,    // [n_exp]
    float*       __restrict__ fused_weights,       // [k_top]
    int k_top
) {
    const int ki = threadIdx.x;
    if (ki >= k_top) return;
    const int e = topk_indices[ki];
    fused_weights[ki] = topk_weights[ki] * per_expert_scale[e];
}
