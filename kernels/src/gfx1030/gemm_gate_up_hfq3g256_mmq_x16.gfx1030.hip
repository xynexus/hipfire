// HFQ3 gate_up MMQ variant — mmq_x = 16.
#define MMQ_X 16
#define KERNEL_NAME gemm_gate_up_hfq3g256_mmq_x16
// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include "gemm_gate_up_hfq3g256_mmq_body.cuh"
