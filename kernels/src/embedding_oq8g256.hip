// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Gather one row of an Oq8G256 (Opus W8A8) embedding table, PLANAR layout.
//
// Q8F16 has been the only 8-bit gather format, and every Opus body routes its
// embed to it (cli.rs `is_embed` arm) because Oq4/Oq8 were treated as GEMV-only.
// Measured on Qwen3.5-0.8B with the body held at oq4.25++, round-tripping the
// embed through each candidate codec:
//
//     q8f16   8.50 bpw   KLD 0.044742      <- previous default
//     oq8     8.06 bpw   KLD 0.044724      <- this kernel: same quality, narrower
//
// The layout is the one `oq8_combined` already produces for body weights:
//
//     [ m*k signed int8, row-major ][ m*ng f32 group scales ]      ng = k/256
//
// NOT the 258-byte interleaved on-disk block. That choice is the point: on a
// tied model the embedding table IS the lm_head, and the head is served by
// `gemv_oq8_grouped_v2`, which reads planar. Keeping the gather on the same
// buffer means one encoding serves both — no second copy, and no f32 expansion
// of a [vocab, dim] head. Reading the interleaved form here instead faults the
// head GEMV, which is how this was found.
//
// Symbols are stored FWHT-rotated (seeds 42/1042), so the gather rotates back;
// the inverse is the same signed transform with the sign vectors SWAPPED — s2
// applied first, s1 last — matching `dequant_oq8g256` on the host.
//
// One block per gathered row. The butterfly is a plain LDS reduction rather than
// the wave-swizzle form in the megakernel: a gather decodes one row per token, so
// this is not the hot path, and it must compile identically on RDNA2/3/4.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_oq8g256(
    const signed char* __restrict__ codes,   // m*k int8, row-major
    const float* __restrict__ scales,        // m*ng f32
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ output,
    int token_id, int dim
) {
    __shared__ float buf[256];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int ng = dim / 256;
    const signed char* row = codes + (size_t)token_id * (size_t)dim;
    const float* row_scales = scales + (size_t)token_id * (size_t)ng;

    for (int g = 0; g < ng; ++g) {
        const float scale = row_scales[g];
        const signed char* q = row + (size_t)g * 256;

        // Dequantize into LDS and apply the FIRST sign vector of the inverse
        // rotation (s2, because the forward transform ended with it).
        for (int i = tid; i < 256; i += nthreads) {
            buf[i] = (float)q[i] * scale * signs2[i];
        }
        __syncthreads();

        // Radix-2 Walsh-Hadamard butterfly, 8 stages over 128 pairs.
        for (int stride = 1; stride < 256; stride <<= 1) {
            for (int p = tid; p < 128; p += nthreads) {
                const int blk = p / stride;
                const int j = p - blk * stride;
                const int a = blk * (stride << 1) + j;
                const int b = a + stride;
                const float x = buf[a];
                const float y = buf[b];
                buf[a] = x + y;
                buf[b] = x - y;
            }
            __syncthreads();
        }

        // 1/sqrt(256), then the second sign vector (s1).
        for (int i = tid; i < 256; i += nthreads) {
            output[g * 256 + i] = buf[i] * 0.0625f * signs1[i];
        }
        __syncthreads();
    }
}
