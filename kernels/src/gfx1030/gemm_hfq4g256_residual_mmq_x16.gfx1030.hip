// MMQ residual variant — mmq_x=16, MMQ_Y=128. Mid-batch tile for HFQ4.
//
// Mirrors gemm_hfq3g256_residual_mmq_x16. Chosen by the auto-selector
// for batch sizes in (12, 64) where mmq_x=16 balances tile granularity
// against per-WG efficiency.

#define MMQ_X 16
#define KERNEL_NAME gemm_hfq4g256_residual_mmq_x16
// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include "gemm_hfq4g256_residual_mmq_body.cuh"
