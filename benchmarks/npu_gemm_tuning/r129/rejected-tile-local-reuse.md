# Rejected R129 tile-local reuse variants

The final R129 graph acquires each 8,320-byte N32 weight FIFO record once and
calls the established R118 projection for every staged document before
releasing it. Two deeper reuse variants were tested and rejected rather than
hidden behind the final wrapper.

## Eight-accumulator B2

The first variant kept both N16 halves for both documents live across the full
K loop. Each of the four weight vectors for a K tile was loaded once and fed to
eight MMUL accumulators. It compiled, but the hardware result failed absolute
parity: doc0 cosine `0.993069742`, maximum absolute error `0.2605845`, and
81,851 mismatches. The live accumulator/vector set exceeds the reliable
scheduling envelope for this kernel shape.

## Two N16 passes

The second variant reduced the live set to four accumulators. It computed the
first N16 half for both documents, then the second N16 half. This is bit-exact,
including fresh and reused contexts, but it reloads both activation vectors on
the second pass. B2 measured `1.682993 ms` versus `1.059922 ms` for R121, only
`1.2596x` row throughput and no improvement over the simpler FIFO-reuse
wrapper. It is rejected on performance.

These results show that the single-copy external/FIFO weight traffic is useful,
but tile-local dense compute remains row-linear. The final R129 source retains
the simpler correct wrapper; exact rows are in
`../results/r129-staged-fullk-batched-weight-reuse-20260716.csv`.
