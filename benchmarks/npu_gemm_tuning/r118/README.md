# R118: stage compact activations once, stream two N32 blocks

R118 is the scalable successor to R117. Each core copies its three 2,080-byte
R113 chunks into one 6,336-byte local stage with a 2,112-byte aligned stride,
releases activation FIFOs, and then streams two full-K N32 weight/output blocks.
Activation DMA remains one
589,824-byte pass while useful output grows to N64.

The output DMA scatters the two N32 objects directly into canonical `[256,64]`
f32 rows. Immutable weight records remain loader/offline `.rdna2.hfp` data and
no tensor-block reorder occurs in the kernel.

The added kernel parameter is still the workaround that stops the platform
issue. It is not LDS avoidance. R118 deliberately uses local memory to preserve
activation bandwidth; admission depends on bank allocation, parity, and timing.

The first 6,240-byte concatenation placed group 1 at offset 2,080, which is only
32-byte aligned for a 64-byte MMUL activation load. The 2,112-byte stride keeps
every group 64-byte aligned. This is an explicit load-alignment correction, not
the platform workaround and not a rule against local memory.

With aligned staging, N32 block 0 passes but a single repeated output descriptor
publishes only that first block. Two explicit output tasks (queue depth two)
restore both blocks. Final N64 parity is zero mismatches with `5e-9` maximum
error at 3,736 bytes maximum core text.

Nine passing fresh 1,000-dispatch processes average 0.106058 ms; one other
fresh process returns all zeros. N64 is only 22.0% slower than R117 N32 while
doing twice the useful work and retaining one activation pass. Admit the
activation-once N64 topology with the existing context-stability caveat.

The next rung tests the compiler/runtime task-repeat attribute together with
the outer DMA tiling dimension. That must work before scaling to many N32 blocks
without creating one live output task per block.

Durable rows: `../results/r118-staged-fullk-n64-20260713.csv`.
