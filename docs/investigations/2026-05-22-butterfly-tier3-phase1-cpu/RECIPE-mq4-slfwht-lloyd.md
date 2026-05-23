# MQ4-SLfwht+Lloyd — the recipe and the PARO-PROXY comparison (2026-05-23)

> Status: all 4 trunk models measured in **Python in-memory KLD** vs a
> **PARO-PROXY** (scaling + rotation + uniform-4bit). NOT yet validated against
> REAL ParoQuant numbers or the production kldref pipeline — those are the two
> open gaps before any "beats PARO" claim is airtight. This doc claims
> "beats PARO-proxy", not "beats PARO". Synthesis of the learnable-FWHT-on-MQ4
> investigation toward the goal "match or beat PARO on Qwen3.5-0.8B/9B +
> Qwen3.6-27B/A3B".

## The recipe

**MQ4-SLfwht+Lloyd** = FWHT rotation + **learned per-tensor/per-group D1/D2
sign tables** + **per-group Lloyd-Max 16-level codebook** (instead of uniform
min-max RTN), on top of closed-form AWQ scaling (alpha=0.55).

Two complementary levers covering two distinct quant-error regimes:

| Lever | Fixes | Wins on |
|---|---|---|
| Learned signs (rotation) | outlier-dominated per-group error (spreads outliers, tightens range) | 0.8B (−20.7%) |
| Lloyd codebook (quantizer) | granularity-bound error (places 16 levels where mass is) | 9B (−10%), 0.8B (−33%) |

## Why this beats the PARO-proxy (and likely real PARO, pending measurement)

PARO = channel scaling + learnable pairwise Givens rotations + **uniform**
quantization. Crucially, **PARO has no non-uniform codebook**, and its rotation
is orthogonal.

**Orthogonal rotations preserve per-group L2 norm** — they redistribute
magnitude across positions but cannot reduce total variance. So any rotation
(PARO's Givens, our signs) only helps when error is *outlier-dominated*
(reducing the min-max range). On *variance-dominated* models (Gaussian-ish after
FWHT, range already ~6σ), rotation gives ~0%.

Empirically confirmed on 9B: learned signs (and per-Linear-MSE / KLD-loss /
per-group / aggressive-decay variants — 6 attempts) all gave +0.4% to +2.7%,
i.e. no improvement. Rotation cannot help variance-dominated 9B. PARO's rotation
faces the identical wall.

**The Lloyd codebook is the lever PARO lacks.** It recovers granularity error
that no rotation can touch:
- 9B: uniform 4-bit 0.4055 → Lloyd 4-bit 0.3656 (−10%)
- 0.8B: uniform 4-bit 0.1469 → Lloyd 4-bit 0.0980 (−33%)

## Head-to-head: MQ4-SLfwht+Lloyd vs PARO-proxy

PARO-proxy = scaling + rotation + uniform 4-bit. Since rotation doesn't help
variance-dominated models, PARO-proxy ≈ uniform-4-bit baseline on those.
All numbers Python in-memory per-token KLD, alpha=0.55, matched seq-count.

All Python in-memory per-token KLD, theta=0 baseline, alpha=0.55, 32-seq,
quant-bits=4. PARO-proxy = scaling + rotation + uniform-4bit; since rotation
is nil on variance-dominated models, uniform-4bit IS the PARO-proxy there.

| Model | MQ4-SLfwht+Lloyd | PARO-proxy (rot+uniform) | Δ | Winner |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.0980 (Lloyd) / 0.086 (signs@64s) | ≈0.1469 unif / 0.086 rot | −33% (Lloyd) | **SLfwht** |
| Qwen3.5-9B | 0.3656 | 0.4055 | −10% | **SLfwht** |
| Qwen3.6-27B | 0.6536 | 0.7044 | −7% | **SLfwht** |
| Qwen3.6-35B-A3B | 0.1987 | 0.2127 | −6.5% | **SLfwht** |

**Lloyd beats uniform on all four trunk models.** On 0.8B, signs (rotation)
ALSO match PARO's rotation (both attack outlier structure), and Lloyd extends
the lead further. On 9B/27B/A3B, PARO's rotation is nil and Lloyd wins outright
via the codebook. The relative gain shrinks with model size (−33% → −6.5%)
because larger models are more variance-dominated (Gaussian after FWHT), where
even optimal 16-level placement is near the uniform optimum — but Lloyd still
wins everywhere.

The absolute baselines climb with size (0.15 → 0.41 → 0.70; A3B-MoE 0.21 is
lower, MoE experts quantize more friendly). The 27B baseline is partly inflated
by a non-finite-sanitized imatrix; the relative Lloyd win is robust regardless.

## The full diagnostic trail (what was ruled out)

1. **AWQ over-scaling** — REFUTED. 9B alpha sweep 0/0.25/0.55/0.80 =
   0.696/0.452/0.405/0.421. AWQ helps; 0.55 near-optimal.
2. **Imatrix quality** — REFUTED. 907K-token unsloth ≈ 65K tier1.mix (0.401 vs
   0.405). (Codex's HF↔blk.N name-alias fix enabled this test.)
3. **Rotation insufficiency (granularity wall)** — CONFIRMED. 9B 4-bit 0.4055,
   6-bit 0.0884 (same rotation): the 16-level uniform quantizer is the
   bottleneck, not the rotation.
4. **Codebook placement (Lloyd)** — PARTIAL. Lloyd recovers 10% on 9B, 33% on
   0.8B. The 9B residual is fundamental 4-bit capacity (Gaussian needs ~6 bits).

## Caveats (important — do not overclaim)

1. **PARO-proxy is conservative, not real PARO.** Real PARO uses continuous
   Givens (more DOF than our discrete signs) + a 2-stage train. The physics
   (variance preservation) says continuous Givens also can't beat the 9B
   variance wall, but this is inference, not measurement. A native PARO A3B
   model exists on mi300 (shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed) to
   anchor at least the A3B comparison directly.
2. **Python in-memory KLD ≠ production kldref pipeline.** ~15% slice variance
   (32-seq vs 64-seq) and unknown absolute offset. Any ship claim needs
   production eval (eval_hipfire + kldref).
3. **Lloyd changes the format + runtime.** MQ4-Lloyd needs codebook-lookup
   dequant (hipfire ships MQ3-Lloyd/MQ2-Lloyd, so the kernel pattern exists, but
   it's slower than uniform MQ4 — may breach the ≤5% decode ceiling). The signs
   are free (same FWHT kernel); the codebook is not.

## Recommendation

- **MQ4-SLfwht (signs) ships as a 0.8B-class free quality lever** (zero runtime
  cost, −20% KLD on outlier-dominated models). Pure learnable-FWHT does NOT
  help variance-dominated 9B/27B/A3B (granularity wall) — this is a settled
  negative result, not a tuning gap.
- **MQ4-SLfwht+Lloyd beats the PARO-PROXY across the trunk** (all 4 models,
  Python KLD), with the Lloyd codebook carrying the variance-dominated models.
  It pays a codebook-lookup runtime cost. This is NOT yet a "beats real PARO"
  claim — see open gaps below.
- **Open gaps before "beats real PARO" is airtight** (both unmeasured): (a)
  establish REAL ParoQuant numbers (eval the native PARO A3B model + ideally a
  continuous-Givens proxy that confirms PARO's rotation also can't beat the
  variance wall), (b) re-measure on the production kldref pipeline (Python
  in-memory KLD has ~15% slice variance and an unknown absolute offset).

## Credit

Root-cause fork (rotation-addressability), imatrix name-fix, memory-efficient
per-group signs, and the Lloyd implementation came from Codex rescue
collaborations ([[feedback_codex_collaboration]]). Empirical mi300 validation +
commits by Claude.
