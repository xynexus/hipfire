// MagnumQuant 4-bit GEMV: g256 block, 4 × g64 sub-blocks, 16-entry fp16 Lloyd codebook.
//
// === Format (256 bytes per g256 block) ===
//
// Block layout, repeated for each g256 group (256 consecutive weights):
//
//   Codebook section (128 bytes, sub-block major):
//     Sub-block s ∈ {0,1,2,3}: 16 × fp16 Lloyd centroids = 32 bytes each
//     Byte offset within block: s * 32
//     Total: 4 × 32 = 128 bytes
//
//   Index section (128 bytes, sub-block major):
//     Sub-block s ∈ {0,1,2,3}: 64 × 4-bit indices packed 2-per-byte = 32 bytes each
//     Byte offset within block: 128 + s * 32
//     Within each sub-block: byte b holds weight[s*64 + b*2 + 0] in bits [3:0]
//                                   and weight[s*64 + b*2 + 1] in bits [7:4]
//     Total: 4 × 32 = 128 bytes
//
//   Grand total: 256 bytes per g256 block.
//
// === Thread assignment (wave32, 32 threads) ===
//
// Thread tid handles 8 weights: 2 per sub-block, consecutive within that sub-block.
//   Sub-block s: weights s*64 + tid*2 and s*64 + tid*2 + 1.
//   Codebook for sub-block s: gptr + s*32 → 16 fp16 entries.
//   Index byte: gptr + 128 + s*32 + tid (one byte = 2 nibbles for 2 weights).
//
// Codebook broadcast: all 32 threads in the wave share the same sub-block
// codebook. We load it into __shared__ LDS once per sub-block per group so
// the 16-entry lookup is register-to-LDS (no global-memory per-weight read).
//
//   LDS budget: 4 sub-blocks × 16 fp32 entries = 256 B per workgroup.
//   Comfortably within gfx1100's 64 KB LDS.
//
// === Python reference (ground truth) ===
//
// learn_butterfly_mq.py:
//   _lloyd_max_dequant_per_group(w_fwht, quant_bits=4, subscale_size=64)
//   block_size = 64, n_centroids = 16, n_subblocks = 4
//   dequant[row, g*256 + s*64 + i] = centroids_s[assignment_s[i]]
//   (no scale/zero multiply — centroids ARE the dequantized values)
//
// The Python trainer writes centroids as f32, serialized here as f16 to
// match the MQ2/MQ3-Lloyd convention (centroids are in rotated weight-space,
// magnitudes ≤ 1.0, f16 precision loss < 1e-3 per entry).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// LDS layout: 4 sub-blocks × 16 f32 centroids = 64 floats = 256 bytes.
// Shared across all 32 threads in the workgroup.
#define N_SUBS 4
#define CB_ENTRIES 16

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g64sub_lloyd(
    const char* __restrict__ A,    // packed weight matrix [M × groups_per_row × 256 bytes]
    const float* __restrict__ x,   // activation vector [K]
    float* __restrict__ y,         // output vector [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    // 256 bytes per block: 128 bytes codebook + 128 bytes indices
    const int block_stride = 256;
    const char* row_ptr = A + (long long)row * groups_per_row * block_stride;

    // LDS: 4 sub-blocks × 16 f32 centroids.
    __shared__ float sh_cb[N_SUBS][CB_ENTRIES];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * block_stride;

        // Load all 4 × 16 = 64 fp16 codebook entries into LDS.
        // 32 threads × 2 loads = 64 entries. Thread tid loads entries
        // at positions (tid*2) and (tid*2 + 1) across the flat 64-entry span.
        // Flat offset: sub*16 + entry_within_sub.
        {
            const __half* cb_global = reinterpret_cast<const __half*>(gptr);
            // Two loads per thread (tid handles flat indices tid*2 and tid*2+1).
            {
                int flat0 = tid * 2;
                int flat1 = flat0 + 1;
                int sub0 = flat0 / CB_ENTRIES;
                int ent0 = flat0 % CB_ENTRIES;
                int sub1 = flat1 / CB_ENTRIES;
                int ent1 = flat1 % CB_ENTRIES;
                sh_cb[sub0][ent0] = __half2float(cb_global[flat0]);
                sh_cb[sub1][ent1] = __half2float(cb_global[flat1]);
            }
        }
        __syncthreads();

        // Load 8 index nibbles (2 per sub-block), one byte per sub-block.
        // Byte for thread tid in sub-block s: gptr + 128 + s*32 + tid.
        const unsigned char* idx_base = (const unsigned char*)(gptr + 128);

        int base_idx = g * 256;

        // Process 4 sub-blocks. Each thread handles 2 weights per sub-block.
        for (int s = 0; s < N_SUBS; s++) {
            unsigned char byte = idx_base[s * 32 + tid];
            unsigned int q0 = byte & 0xFu;
            unsigned int q1 = byte >> 4;

            float w0 = sh_cb[s][q0];
            float w1 = sh_cb[s][q1];

            int wi = base_idx + s * 64 + tid * 2;
            acc += w0 * x[wi] + w1 * x[wi + 1];
        }

        __syncthreads();  // Guard before next group's LDS fill.
    }

    // Wave-level reduction: sum across 32 lanes.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
