# R121: activation-once full-K N1280

R121 scales the admitted R120 topology to all 40 N32 blocks of an
EmbeddingGemma 768x1280 projection. Each core stages the three R113 activation
chunks once and retains the same 6,336-byte local footprint while the weight
stream grows to the complete projection. Activation DMA remains one
589,824-byte pass containing 199,680 unique bytes, with no N-macro replicas.

One output task per stream uses an outer tiling dimension of 40 and
`repeat_count=39`. Output row strides scale to N1280 while each produced object
remains one N32 block. The 7,987,200-byte W8 diagnostic weight payload is laid
out offline as `.rdna2.hfp` records; tensor blocks are not reordered in the
kernel.

The added kernel parameter remains the platform workaround that stops the
platform issue. LDS placement, activation staging, output repetition, and
full-width payload size are separate performance and capacity concerns.

The complete 256x768 by 768x1280 byte oracle passes with zero mismatches and
`6e-9` maximum absolute error. All ten fresh 1,000-dispatch contexts pass at
0.319049-0.325542 ms, mean 0.320640 ms. This is about 798,402 M256 projection
rows/s and 30.84 GB/s over the W8 input, compact activation, and f32 output
bytes. It is a single-projection schedule measurement, not end-to-end encoder
tokens/s. Maximum core text is 3,848 bytes. Durable rows:
`../results/r121-staged-fullk-n1280-20260713.csv`.
