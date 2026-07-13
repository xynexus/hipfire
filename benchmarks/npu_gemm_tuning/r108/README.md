# R108: consume completed residual directly in attention

R108 removes R48's standalone residual-copy context. The post-FFN tail already
stores the correctly rounded completed scalar as the high BF16 plane of each
compensated row. The attention graph DMAs that plane directly from the shared
completed-state BO into its existing 16 KiB residual FIFO. Each physical DMA
row transfers 1,536 useful high-plane bytes plus 512 ignored low-plane bytes;
the kernel uses a 2,048-byte row stride.

The first form used a sixth DPU argument and is rejected because the amdxdna DPU
register map holds at most five. The admitted form places completed BF16x2 in
the first 884,736 bytes of the existing attention input argument and R34
activation records after it. R109 writes that suffix in place, retaining the
five-argument ABI.

Layer 0 reaches FFN cosine `0.99991644`, tail cosine `0.99999886`, and completed
layer cosine `0.99996836`. Alternating full-model trials measure the R48 control
at 813.690/816.898 ms and R108/R109 at 801.536/807.908 ms. The paired means are
815.294 and 804.722 ms, a 1.30% full-model latency reduction. Preparation falls
from roughly 9-12 ms to 7-9 ms per layer, while direct strided residual DMA
offsets part of that local gain in the attention phase.
