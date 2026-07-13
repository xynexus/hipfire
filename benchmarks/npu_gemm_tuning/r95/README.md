# R95: unified resident-W4 init/accumulate body

R95 is a capacity rung for fusing R94 with the first native gate/up stage. It
replaces R25's separately instantiated W4 init and accumulate GEMM bodies with
one runtime-flagged body, while preserving the weight, activation, local nibble
decode/swizzle, GeGLU, down-pack, and DMA contracts.

The existing kernel parameter remains the platform correctness workaround.
This experiment changes no LDS policy or immutable tensor-block order.
