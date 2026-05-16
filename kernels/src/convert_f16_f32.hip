#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP16→FP32 widening conversion kernel.
// Dual of the convert_f32_to_f16 kernel embedded in
// gemm_hfq4g256_residual_fp16.hip. Used by the DFlash draft pipeline post-
// Phase 2 (2026-05-16): target_hidden is stored as F16 to halve VRAM, and
// the FC GEMM input expects F32, so this routine widens the per-cycle delta
// slice on the fly. Bandwidth: read N*2 + write N*4 = 6 bytes/elem.
extern "C" __launch_bounds__(256)
__global__ void convert_f16_to_f32(
    const _Float16* __restrict__ in,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}
