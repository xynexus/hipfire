# R99: combined-row DMA scatter

R99 changes only R98's output DMA destination stride. Each token's 3,072-byte
interleaved compensated-BF16 FFN result is placed at the start of the existing
4,608-byte post-FFN combined row, leaving its third plane reserved. The NPU
still transfers 884,736 useful output bytes; the larger 1,327,104-byte BO is a
sparse destination layout shared with the resident split-X tail.

The compute kernel keeps canonical token/column order. This is mutable output
placement by DMA, not an immutable tensor-block conversion and not an LDS rule.

The full hardware oracle is unchanged from R98. Three 100-command runs with
recycle-every-7 measure 6.4729, 6.6113, and 6.6000 ms (median 6.6000 ms,
38,788 M256 rows/s). Admit the combined-row scatter; it has no material cost
relative to R98.
