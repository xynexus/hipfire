# R115: direct R113 compact-chunk consumer, group 0 x N16

R115 is the first consumer-side rung after rejecting R114 assembly. It reads
R113's admitted per-core diagnostic slots directly and computes one int8
K256-by-N16 scaled projection. All 32 cores own eight tokens. The same offline
packed N16 weight record is broadcast to each token owner, and outputs scatter
to canonical `[256,16]` row-major order.

This rung deliberately does not assemble 24-token R34 records and does not
materialize five N-macro activation replicas. It reads 196,608 physical bytes
from the group-0 diagnostic plane, containing 66,560 bytes of unique chunks.
The initial graph retains R113's 6,144-byte diagnostic slot padding so the first
gate isolates direct-consumer addressing and matrix math. A later DMA-gather
rung may remove that padding only if byte parity and throughput are preserved.

Immutable weights use an offline/loader-provided record. No tensor block is
reordered in the kernel. The `.rdna2.hfp` suffix remains the required durable
layout tag.

The added kernel parameter is the platform workaround that stops the platform
issue. It is not LDS avoidance; tile-local ObjectFIFO placement in this graph is
an independent capacity and performance decision.

The image builds with 1,692 bytes maximum core text. Locked hardware parity is
exact within floating-point scaling noise: zero mismatches and `2e-9` maximum
absolute error. Six fresh 1,000-dispatch processes pass and average 0.092506 ms.
One earlier fresh process returned mostly zero output; the immediate six-process
series passed. Keep that as context-transition evidence, separate from the
platform workaround and from local-memory choices.

R115 admits the direct-consumer mapping and one-group matrix stage. It is not a
full-K or full-N throughput claim. The next rung adds groups 1 and 2 with local
float accumulation while preserving the same per-core chunk ABI.

Durable rows: `../results/r115-direct-compact-group-n16-20260713.csv`.
