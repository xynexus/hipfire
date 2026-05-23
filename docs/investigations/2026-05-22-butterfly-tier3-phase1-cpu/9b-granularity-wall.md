# 9B is granularity-bound at MQ4, not rotation-bound (2026-05-23)

**Headline:** Learnable-FWHT sign tables give −20.7% KLD on Qwen3.5-0.8B but
**nothing** on Qwen3.5-9B. After ruling out config causes (alpha sweep,
imatrix quality), the granularity-ceiling diagnostic proves the 9B MQ4 error
is **quantizer-granularity-bound, not rotation-bound** — so a rotation lever
(sign flips, and by extension continuous Givens) is the wrong tool for 9B.

## The decisive diagnostic

theta=0 baseline KLD (current MQ4+AWQ pipeline, FWHT seeds 42/1042, alpha=0.55,
32-seq Python eval) at varying quantizer bit-width on the SAME rotation:

| Model | 4-bit (MQ4) | 6-bit | 8-bit |
|---|---:|---:|---:|
| **Qwen3.5-9B** | 0.4055 | **0.0884** | 0.0369 |
| Qwen3.5-0.8B | 0.1469 | — | (nan, edge case) |

The 9B 4→6-bit jump recovers **78%** of the KLD (0.405 → 0.088) on the
**same FWHT rotation**. This is the signature of a granularity-bound error:
the rotation is fine, but 16 uniform levels (4-bit min-max RTN) are too coarse
for 9B's rotated per-group distribution. 6-bit (64 levels) captures it.

## Why sign-learning helps 0.8B but not 9B

- **0.8B**: 4-bit baseline 0.109 (64-seq) → sign-learning 0.086 (−20.7%). 0.8B's
  rotated distribution has outlier structure that sign flips can flatten,
  tightening the per-group range so 16 uniform levels capture it better.
- **9B**: 4-bit baseline 0.353 (64-seq) → sign-learning ~flat (+0.4% best).
  9B's rotated distribution is ALREADY well-flattened by the seed FWHT
  (6-bit on it = 0.088). Sign flips have no rotation-addressable headroom left;
  the residual 4-bit error is pure quantizer coarseness, which no orthogonal
  rotation can fix.

## What was ruled out first (so this conclusion is sound)

1. **AWQ over-scaling (hypothesis b)** — REFUTED. Alpha sweep: 9B baseline at
   alpha 0/0.25/0.55/0.80 = 0.696/0.452/0.405/0.421. AWQ HELPS 9B (0→0.55
   improves), alpha=0.55 is near-optimal. Not over-scaled.
2. **Imatrix quality (hypothesis a)** — REFUTED. 907K-token unsloth imatrix
   (0.401) ≈ 65K-token tier1.mix (0.405) at alpha=0.55. 8× more calibration
   tokens barely moved it. (Codex's HF↔blk.N name-alias fix enabled this test.)
3. **Eval slice sensitivity (hypothesis e)** — partially real (32-seq 0.405 vs
   64-seq 0.353, ~15% swing) but does not explain the 0.8B-vs-9B gap (both
   measured identically).

## Implications for the goal (match PARO on 0.8B/9B/27B/A3B)

The learnable-FWHT-signs approach is a **0.8B-class lever**. It cannot, alone,
improve granularity-bound models (9B, and almost certainly the larger 27B /
A3B which will be at least as granularity-bound).

To improve 9B/27B/A3B at 4-bit, the lever must attack the **quantizer**, not
the rotation:
- **Non-uniform codebook (Lloyd-Max)**: fit 16 levels per group where the mass
  concentrates, instead of uniform min-max. hipfire already ships MQ3-Lloyd /
  MQ2-Lloyd; an MQ4-Lloyd (or learnable codebook) is the natural lever. Pays a
  runtime cost (codebook lookup vs uniform dequant).
- **FWHT-signs + Lloyd compose**: signs flatten where there's outlier structure
  (0.8B-style gains), Lloyd recovers granularity (9B-style gains). Together they
  may match PARO across the trunk.

**Open question — PARO's actual 9B number.** PARO uses continuous rotation +
scaling, not a better quantizer. If 9B is granularity-bound, PARO's 4-bit
rotation also can't beat the granularity floor — so PARO's 9B 4-bit KLD may
ALSO be ~0.3-0.4, meaning the current MQ4 baseline already ~matches it. We do
not yet have PARO's 9B KLD. There IS a native PARO A3B model on mi300
(shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed) that can anchor the
PARO-vs-MQ4 comparison on A3B.

## Next experiment

Test whether a non-uniform 4-bit codebook (Lloyd-Max, simulated in Python
per-group) recovers the 9B 4→6-bit granularity gap. If Lloyd-4-bit on 9B
approaches the 6-bit ceiling (0.088), the codebook is the lever for the
granularity-bound models, and FWHT-signs + Lloyd is the path to match PARO.

## Methodology note

The `--quant-bits` diagnostic (commit 6819cf63) and the alpha sweep
(`/workspace/sweep9b/`) + granularity sweep (`/workspace/gdiag/`) artifacts
are on mi300. All measurements are Python in-memory per-token KLD (oracle BF16
vs pseudo-quant student), which has ~15% slice variance and differs in absolute
scale from the production kldref pipeline — relative comparisons within a fixed
seq-count are reliable; absolute numbers should be anchored to production eval
before any ship decision.

Credit: the rotation-addressability fork + alpha-sweep + name-fix came from a
Codex rescue collaboration (see [[feedback_codex_collaboration]]).
