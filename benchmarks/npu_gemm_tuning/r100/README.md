# R100: interleaved compensated post-FFN tail

R100 changes only the split-X post-FFN tail reader. It deinterleaves R99's
per-scalar `(BF16-high, BF16-low)` pairs locally, then executes the unchanged
post-FFN RMSNorm, residual addition, and compensated output path. Its DMA reads
the existing 3,072-byte prefix of each 4,608-byte combined row.

No immutable weight layout or tensor-block order changes in this rung. The
required R15 kernel controls and R97 fragment-buffer fix remain independent.

The standalone hardware oracle passes at `0.99999861` cosine and `0.0039062`
maximum error. An initial verifier falsely exposed stale host-written split-X
pages because its explicit host-sync helper flushed only the combined FFN
buffer. The helper now flushes both combined and residual buffers; production
NPU-to-NPU handoffs continue to skip host synchronization.
