# R111: one-pass completed-state preparation

R111 keeps R109's in-place prefix/suffix ABI and exact R47 math, but changes the
core schedule from group-major to row-major. Each core copies one completed
BF16x2 row into a 3,072-byte tile-local buffer and immediately releases the
input FIFO. It then computes the RMS inverse and packs all three K256 groups
into separate local chunks from that explicit local copy.

The completed-state allocation is 884,736 bytes because it contains 32 padded
rows, but each physical sweep reads only the 256 active 3,072-byte rows:
786,432 bytes. R111 reduces that active input traffic from four sweeps
(3,145,728 bytes) to one (786,432 bytes), saving 2,359,296 bytes, and cuts its
shim DMA tasks from 32 to 8. Parameter preload, R34 output block order, offline
`.rdna2.hfp` ordering, and the existing platform-workaround kernel parameter
are unchanged. LDS/tile-memory placement remains a separate capacity and
performance decision. Final core text ranges from 9,072 to 10,592 bytes.

The initial form held the input FIFO across RMS plus all three pack calls and
reproduced R54's producer/consumer schedule failure; it is rejected. Only the
copy-then-release form is a candidate for admission.

The first copy-then-release build preserved every scale but corrupted only the
packer-owned group-1/group-2 Q bytes. Those chunks were 32-byte aligned while
`copy_chunk_to_block` used a 64-byte vector load. R111 therefore uses 32-byte
Q copies, matching the allocator's guaranteed alignment without changing bytes.

The corrected hardware gate passes with five one-code Q differences, maximum
Q delta 1, and maximum scale error `7e-9`. Three paired fresh-process timings
measure R109 at 5.1355, 5.1636, and 5.1281 ms (mean 5.1424 ms), and R111 at
5.0971, 5.1357, and 5.0519 ms (mean 5.0949 ms). The 0.9% standalone reduction
is intentionally treated as small until the full resident layer is compared.
