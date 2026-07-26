# R107: fuse next-layer activation and residual preparation

R107 adds the residual-record copy to R47's existing RMS input pass. Both
outputs consume the same compensated BF16x2 completed state and write disjoint
regions of the next attention input BO. The fused graph therefore removes the
separate R48 context and its fifth full input read without changing the R34
activation prefix, residual-record ABI, immutable weight order, or tensor
layout.

The graph is rejected before hardware. Adding four residual-output object FIFOs
per memory tile exceeds the AIE2P memory-tile input DMA-channel budget. R108
therefore moves the residual handoff into the existing attention residual FIFO
instead of adding output channels to R47.
