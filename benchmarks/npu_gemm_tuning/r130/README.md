# R130: fused R108 QKV weight reuse

R130 is the bounded transplant of R129's acquire-once/apply-two-documents
topology into R108. It keeps a second QKV accumulator pair on each odd compute
core, transposes the existing document-contiguous activation BO in the shim DMA,
and consumes each paired QKV weight FIFO object once for both documents. The
later Q/K/V packing, block-diagonal attention, output projection, direct-X
handoff, and inverse-state layout remain the admitted per-document R108/R128
schedule.

The image is deliberately B2-only. B1 continues to use byte-identical R108,
and larger batches remain closed.

## Result: rejected

The direct form lowered its four-dimensional activation transpose but produced
16,876-byte odd-core ELFs, exceeding AIE2P's 16 KiB program store (the admitted
R108 odd core is already 16,396 bytes including ELF text). Moving Q packing and
inverse relay wholesale to even cores merely moved the overflow there. Splitting
Q packing across each even/odd pair and emitting inverse metadata from the even
core balanced the image sufficiently for `aiecc` to produce an xclbin.

The compiled form was then tested against two independent R108 M256 hardware
oracles with deliberately distinct documents. It returned an all-zero completed
state: document 0 reported `x_cosine=NaN`, `x_max=3.5781250`,
`x_mean=0.64745794`, `inverse_cosine=NaN`, and `inverse_max=1.2935559`.
Reordering the inverse DMA behind the norm-output drains and replacing the shim
transpose with an explicitly record-interleaved input ABI did not change that
result. R130 is therefore preserved as a negative topology and is not selected
by the runtime.

The concrete boundary is the fused image's program-memory pressure: the
straightforward acquire-once topology cannot be represented without exceeding
the odd-core store, while code motion sufficient to fit disturbs the already
tightly coupled Q/direct-X FIFO schedule. R129 remains the admitted weight-reuse
primitive; R128 remains the admitted fused segmented resident layer.
