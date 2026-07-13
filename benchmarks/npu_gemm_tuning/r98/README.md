# R98: compensated BF16 output boundary

R98 retains R97's canonical-BF16 input, native resident W4 FFN, dedicated gate
fragment buffers, and required R15 kernel controls. It changes only the final
representation: each F32 output is replaced in place by its compensated
`BF16-high + BF16-low` pair, interleaved per scalar. The physical output remains
884,736 bytes and token/column order is unchanged.

This is the smallest output-side rung toward the resident post-FFN tail. It
must pass the full R97 oracle and fit the 16-KiB program store before the next
rung changes the DMA scatter or tail input ABI.

Hardware admission passes. Maximum core text is 16,032 bytes. The full oracle
retains `0.99998228` cosine, `0.2596474` maximum error, and `0.03750556` mean
error. Three 100-command runs with recycle-every-7 measure 6.5832, 6.5873, and
6.5774 ms (median 6.5832 ms, 38,886 M256 rows/s). The compensated conversion
costs about 1.5% versus R97's sustained median.
