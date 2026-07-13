# R105: direct-X inverse application boundary

R105 retains R44's proven physical output unchanged. Each of eight columns
reads 32 contiguous canonical BF16 X rows. Its four cores compute RMS from the
tile-local X object, then traverse that object a second time to emit canonical
unit-RMS BF16, padded to 288 rows. This avoids a third DMA channel for R44's
separate physical inverse records.

The kernel deliberately does not apply immutable pre-FFN norm. The following
native-W4 loader folds that vector into the existing activation divisor, so no
third input FIFO or online tensor reorder is required. Physical mutable traffic
is 393,216 bytes read plus 393,216 bytes written, compared with the current two
5.1-MiB host synchronizations.

R15's `rounding=floor` and `saturation=none` numerical controls remain enabled.
The separately added kernel parameter remains the platform-issue workaround;
R105 changes neither mechanism and imposes no LDS-avoidance rule.

## Result

The hardware oracle passes at cosine `0.99999122` and maximum absolute error
`0.0078125`. BF16 bit equality is intentionally not the gate: AIE2P reduction
and inverse-square-root ordering produce 98,380 one-rounding-boundary changes
relative to the scalar CPU reference while retaining the numerical contract.

Three fresh 100-command processes measure 0.1331, 0.1250, and 0.1295 ms
(median 0.1295 ms), or 1,977,526 M256 rows/s and 5.658 GiB/s of physical
source-plus-output traffic. No four-second timeout occurs. The low fraction of
the 56 GB/s feed ceiling shows that this small stage is dispatch/scheduling
limited. It is negligible only as a standalone context: when chained between
the resident attention and FFN contexts, explicit cache maintenance raises the
observed phase to 2.35-4.14 ms per layer. Durable standalone rows are in
`../results/r105-direct-x-unit-rms-20260713.csv`.
