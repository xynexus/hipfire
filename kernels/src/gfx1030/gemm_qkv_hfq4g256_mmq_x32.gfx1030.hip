// HFQ4 qkv MMQ — mmq_x=32, MMQ_Y=128. Long-prefill tile.

#define MMQ_X 32
#define KERNEL_NAME gemm_qkv_hfq4g256_mmq_x32
// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include "gemm_qkv_hfq4g256_mmq_body.cuh"
