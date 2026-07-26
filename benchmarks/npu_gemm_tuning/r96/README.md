# R96: compact resident-W4 fragment ring

R96 adds one runtime owner/source loop for R25's existing three-row down
activation fragment exchange. It replaces five statically duplicated eight-owner
driver sequences while preserving the direct-stream order, packed bytes,
weights, and arithmetic. Together with R95's unified GEMM body, this recovers
program space for the first canonical-BF16 gate/up preparation stage.

The existing kernel parameter remains the platform correctness workaround.
This changes no LDS policy or immutable tensor-block order.
