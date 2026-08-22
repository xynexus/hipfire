# Alternatives to the compact overlay — three measured dead, and a QAT scope

The sparse overlay forces a per-(row, group) gather, which is what made the
correction cost 30-100% of the GEMM. Before optimising the gather further, this
asks whether the overlay should exist at all. Three replacements were measured at
**matched bit budget**; all three lose badly.

Setup: Qwen3.8-27B bf16, 256 rows x 4 tensor types, weight SSE in the rotated
basis, identical scales across arms. The overlay costs `3 x 2 B / 256 w` =
**0.1875 bits/weight** and the format totals 4.25 (4 nibble + 0.0625 f16 scale).

| arm | bits/w | SSE (down_proj) | vs plain int4 |
|---|---|---|---|
| plain int4, G=256 | 4.0625 | 7.6590 | — |
| **overlay, n_out=3 (shipped)** | **4.25** | **4.4536** | **−41.8%** |
| shared positions | 4.19 | 7.4852 | −2.3% |
| low-rank residual, matched rank | 4.25 | ~7.54 | −1.5% |
| G=64, no overlay | 4.25 | 7.0611 | −7.8% |

**The overlay is remarkably bit-efficient and nothing tested comes close.**

## Why each alternative fails

**Shared outlier positions** (details in the sibling study): positions are chosen
AFTER the FWHT rotation, which exists to destroy the channel structure that would
make positions shareable. Captures 5-6% of the overlay's benefit. 1.68x SSE.

**Low-rank residual.** `W ≈ Q4(W) + A·Bᵀ` is attractive because the correction is
two dense GEMMs — no gather at all — and it is already in-tree
(`HIPFIRE_LOWRANK_R`). Matched bits on a 5120x17408 tensor is rank 46, i.e. 0.9%
of the rank. Measured on the dumped residual, ~1% of the rank buys ~1.5%. The
reason is structural: the int4 residual is essentially **white quantisation
noise**, and low-rank captures correlated structure, of which it has almost none.
(The prior "−13% @ 2 bits" is not a contradiction — at 2 bits the residual is
much larger and more structured.)

**Finer groups, G=64 with no overlay.** Exactly matches 4.25 bits: scale cost goes
0.0625 -> 0.25 b/w, and the side table disappears, so there is no gather. But
−7.8% against the overlay's −41.8%. A smaller group means a SHORTER FWHT and so
less mixing, which partly cancels the finer scale. **58% worse than the overlay at
identical bits.**

## Does the rotation choice change any of this? Measured: no.

The verdict above ("shared positions capture 5%") is really a claim about the
FWHT basis, so it is fair to ask whether a different rotation rescues it. The
quantizer does offer alternatives: `RotationVariant::{Plain, PlainG128, Givens,
WithRmsnorm, WithSwiGLU}`, and `--rotate` is a **SpinQuant R1 deploy merge** which
builds `R1 = FᵀM` so the codec's own FWHT composes away (`F·Fᵀ·M = M`) and the
LEARNED rotation replaces it outright.

Rather than train an M, measure the bound. The unrotated basis is where the AWQ
channel structure that would make positions shareable is MAXIMALLY present, so it
upper-bounds what any rotation can do for sharing:

| basis | int4 | per-row overlay | shared | capture |
|---|---|---|---|---|
| FWHT (current) | 7.6590 | 4.4536 | 7.4852 | **5.4%** |
| **unrotated** | 8.3732 | 4.5102 | 7.8527 | **13.5%** |

Sharing does work better without the rotation — 2.5x better, so the rotation is
genuinely part of the obstacle. But 13.5% is still nowhere near viable, and a
learned rotation can only land BETWEEN these two. **Weight outliers are inherently
per-row**: the AWQ insight is about ACTIVATION channels being outliers, not weight
positions, and a row's extreme weights are a property of that row's learned
function rather than of the channel.

The second thing a better rotation could do is reduce the outlier NEED. The data
bounds that too: FWHT beats no-rotation on plain int4 by **8.5%** (8.3732 ->
7.6590), while the overlay is worth **41.8%**. Rotation is a ~10% lever and the
overlay is a ~40% one — they are not substitutes, and a learned rotation would
have to be several times better than the FWHT is over identity to make the
overlay redundant.

## What this leaves

Two options survive, and neither is a format change:

1. **Fuse the correction into the GEMM epilogue.** Each overlay's `idx` lands in
   exactly one BK-strip (`idx / BK` selects it), so it can be applied while that
   strip's activation tile is STILL IN LDS — the reuse the standalone pass can
   never get, because it re-reads an activation the GEMM already had. This keeps
   the format, keeps the quality, and removes the separate pass rather than
   optimising it. **This is the recommendation.**
2. **The layout work already landed** (K-major + hoisted table + dword gather,
   1.5-3.0x, bit-identical) as the fallback if fusion proves awkward.

## QAT scope

Measured ladder, tiny fixture, qwen3_5_moe_indexed:

| variant | mean KLD |
|---|---|
| oq4 | 0.0601 |
| oq4+ (AWQ calib) | 0.0572 |
| oq4++ (LDLQ) | 0.0476 |
| **oq4.25++ (overlay)** | **0.0382** |
| oq8 | 0.0033 |

**The overlay buys −19.7% KLD over oq4++.** So QAT's bar is: recover 19.7% on top
of oq4++ without the overlay. Precedent says that is plausible — light-QAT +
LoRA recovery-FT recovered ~52% of the W3 loss, and this is a strictly easier
target at W4.

**But there is a constraint that changes the conclusion.** Dropping the overlay
gives 4.0625 bits/weight, which is BELOW the project's hard 4.25 floor. So the
bits have to be respent — and the three respend candidates above are all far
worse than the overlay. Concretely:

- QAT + oq4++ at 4.06 bits: violates the floor even if quality matches.
- QAT + G=64 at 4.25: measured 58% worse in SSE before QAT even starts.
- QAT + oq4.25++ (keep everything): strictly better QUALITY, no speed change.

So **QAT is a quality lever here, not a speed one** — unless the 4.25 floor is
relaxed, which is a product decision rather than a measurement. The speed comes
from fusing the correction into the GEMM, which needs no requantisation, no
training run, and no format change.

Recommended order: fuse the correction (kernel work, unblocks the 2.1-3.2x iu4
GEMM), and treat QAT as an independent quality track whose payoff is a better
oq4.25++ rather than a cheaper one.
