# R79: offline paired whole-scaled HFP layout

R79 implements the immutable-layout prerequisite for paired compact projection.
`NpuOpusExecutor::prepack_paired_whole_scaled_cached` converts a validated
whole-scaled `.rdna2.hfp` from `(column, block)` to
`(adjacent-column-pair, block, lane)` order. Every encoded block remains
byte-identical; only complete records move, once, in the loader/offline path.

The derivative uses `OpusHfpLayout::PairedWholeScaledV1`, retains the source
encoding and geometry, records the source payload size, and is cache-keyed by
the complete source artifact. Local nibble decode and lane swizzle remain
kernel work. No tensor block is reordered during dispatch.

The deterministic unit oracle covers every source block, verifies each block's
bytes are unchanged, checks pair/block/lane order, validates the derivative
descriptor, and confirms cache reuse. R79 is an offline-layout checkpoint, not
a kernel correctness or performance result. The next rung must consume this
layout in a paired compact projection graph and match the R65/R70 stage oracle.
