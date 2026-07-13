# R116: direct R113 compact full-K consumer, N16

R116 extends R115 from one K256 group to all three groups. Every token-owning
core consumes its three R113 chunks in group order and accumulates the scaled
partials locally in f32 before emitting canonical `[256,16]` output.

The graph reads the complete 589,824-byte padded R113 diagnostic ABI containing
199,680 unique chunk bytes. It still materializes zero N-macro activation
replicas; the canonical R34 activation input would occupy 2,949,120 bytes.
This rung keeps padding to isolate full-K accumulation. Padding removal and N
extension remain later, separately gated steps.

Weights are prepacked by the loader/offline tool and tagged `.rdna2.hfp`; the
kernel does not reorder immutable tensor blocks. The added kernel parameter is
the platform workaround that stops the platform issue. It is not LDS avoidance;
local f32 accumulation is an independent compute/memory design choice.

The first full-K build corrupts only group 1's low eight output columns; group 2
and columns 8-15 remain exact. Padding the 4,160-byte immutable weight records
to 4,224 bytes for 128-byte starts does not change the error and is rejected.
A subsequent isolation removed the dynamic three-group core loop to test
whether the scheduler overlapped the low MMUL accumulator between iterations.
Neither experiment changes the platform workaround or establishes an LDS rule.

Unrolling the group loop and splitting each group into a separate DMA task both
produce all-zero output and are rejected. Keeping the loop but copying the
previous 8x16 f32 output tile into a 512-byte local staging array before the
next MMUL fixes the dependency. The K512 unit-scale oracle is bit-exact. Full
K768 has zero mismatches and `4e-9` maximum error. Maximum core text is 2,220
bytes.

Eight passing fresh 1,000-dispatch processes average 0.096384 ms. Two additional
fresh processes returned all-zero output before later fresh processes passed.
This admits full-K compact-consumer math, but not context-stable production
selection. The context symptom remains separate from the kernel-parameter
platform workaround and from LDS decisions.

Durable rows: `../results/r116-direct-compact-fullk-n16-20260713.csv`.
