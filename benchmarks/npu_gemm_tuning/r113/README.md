# R113: tail-local next-layer RMS and three-group pack

R113 keeps R112's joined canonical row input and contiguous eight-token core
ownership. Each core preloads one 9,216-byte record containing the three
next-layer K256 parameter groups, then computes row RMS and all three
AWQ/FWHT/int8 chunks from each still-local two-row BF16x2 output using R111's
admitted math. It does not retain a separate eight-row completed-state buffer.

The first rung exposes each core's three 2,080-byte chunks in padded diagnostic
slots through R112's existing completed-output route. It deliberately does not
yet assemble or replicate R34 blocks. This proves local math and capacity before
adding core chains. No completed-state input DMA is needed for the pack, no new
memory-tile channel is added, and no immutable tensor block is reordered.

The first implementation retained all eight completed rows in a separate
24,576-byte buffer and failed bank allocation. R113 instead preloads the next
parameters and packs each two-row output while it is still local. Every core
then links at 9,984 text bytes with 9,216 bytes of next parameters, a 6,240-byte
three-group pack blob, 1,024 bytes of scratch, and 32 bytes of RMS state.

The shim S2MM queue cannot retain four completed-output tasks plus three
diagnostic tasks. With all seven live, group 2 remains zero. R113 therefore
launches every row/half stripe, retires the oldest completed-output task for
each stripe, and only then publishes the three diagnostic tasks. Retiring a
task inside the stripe-construction loop was correct but serialized the eight
stripes and measured 13.3578 ms, so that schedule is rejected.

The final locked oracle reports tail cosine `1.00000000`, maximum error
`0.0000310`, three one-code int8 differences, maximum Q delta 1, and `7e-9`
maximum scale error. Four live 50-command samples average 5.056325 ms. Current
live controls average 0.236051 ms for R112 plus 5.024850 ms for R111, or
5.260901 ms combined. Fusion therefore saves 0.204576 ms (3.8886%) while
removing R111's 786,432-byte completed-state input pass. This admits the fused
math/topology rung, not the still-unimplemented R34 assembly or a full-model
throughput claim.

One fresh process returned an all-zero tail during the first attempted
order-balanced series; four immediate fresh-context reproductions and the
complete second series passed. Keep that transition as context-lifetime
diagnostic evidence rather than attributing it to local memory.

The added kernel parameter remains the platform-issue workaround. It is not
LDS avoidance; local buffers here are an independently measured capacity and
performance decision.

Durable rows: `../results/r113-tail-next-pack-fusion-20260713.csv`.
