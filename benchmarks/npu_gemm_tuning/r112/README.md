# R112: fusion-ready compensated post-FFN tail topology

R112 changes only R100's mutable DMA ownership and architectural-X placement.
Each core now owns eight contiguous tokens across the four two-token phases.
Strided DMA gathers the existing canonical rows into that execution order and
scatters the completed rows back to the unchanged canonical output. Architectural
X occupies the already-reserved third plane of R99's 4,608-byte mutable row, so
the same memory-tile broadcast carries interleaved FFN high/low plus X and the
horizontal core stream used by R100's split-X relay becomes free.

This rung does not reorder immutable tensor blocks, alter `.rdna2.hfp` weight
order, or add layout conversion to the kernel. X remains canonical token-major
BF16; only the preceding mutable producer's destination placement changes. The route is an enabling seam
for chaining three adjacent eight-token core owners into one 24-row R111-style
next-layer preparation group. It does not yet fuse the preparation math.

The separately added kernel parameter remains the platform-issue workaround
that stops the failure. It is preserved independently of this rung. LDS/tile
memory remains a capacity and performance choice; it is not the workaround.
R15 rounding/saturation controls, R97 fragment-buffer lifetime, and context
recycling also remain separate concerns.

The first attempted route sent split X through a second memory-tile broadcast.
It is rejected because the tile already uses its output DMA channels for the
four FFN consumers and the completed-state join. Joining X into R99's reserved
row suffix fits without adding a channel. Maximum core text falls from R100's
4,208 bytes to 3,696 bytes and all 24 horizontal core flows disappear. Active
input DMA is unchanged at 1,179,648 bytes.

The locked hardware oracle is identical to R100: cosine `0.99999861` and
maximum absolute error `0.0039062`. Four counterbalanced 100-command pairs all
favor R112. Mean dispatch falls from `0.324965 ms` to `0.218271 ms`, a 32.84%
reduction. This admits the fusion-ready tail topology, not the still-unbuilt
R111 pack fusion or full-model throughput.
