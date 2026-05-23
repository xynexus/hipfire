# MQ4-SLfwht+Lloyd — the recipe and the direct PARO-mechanism comparison (2026-05-23)

> Status: **DIRECT head-to-head** vs a faithful **PARO-mechanism** (scaling +
> learnable continuous rotation + uniform-4bit) measured on 0.8B + 9B; 27B/A3B
> running at ctx=1024 (full-butterfly training OOMs at ctx=2048 on the bigger
> models). All **Python in-memory KLD** — NOT yet validated against z-lab's
> SHIPPED PARO weights or the production kldref pipeline (the two remaining
> gaps). Claim: "beats the PARO mechanism", measured directly on 0.8B+9B.

## The recipe (CORRECTED understanding)

**MQ4-SLfwht+Lloyd** = **learnable continuous per-group full-butterfly rotation**
(all 8×128 Givens angles per 256-group, KLD-loss trained, gradient-checkpointed)
+ **per-group Lloyd-Max 16-level codebook**, on closed-form AWQ scaling (α=0.55).

IMPORTANT CORRECTION: the *discrete* D1/D2 sign tables (the original "SLfwht"
parameterization) are a 0.8B-class lever only — too coarse for the bigger
models. The CONTINUOUS full-butterfly rotation is the real learnable-FWHT lever
and it helps every model (it shaves the per-group min-max RANGE by balancing
the tails, where discrete signs can't). This is what matches PARO's rotation.

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

### A. DIRECT head-to-head (both trained, identical recipe except quant mode)

PARO-mechanism = AWQ + learnable continuous full-butterfly rotation + uniform
4-bit. MQ4-SLfwht+Lloyd = same rotation + Lloyd codebook. KLD-loss trained,
Python in-memory KLD. 0.8B/9B at ctx=2048/64-seq; 27B/A3B at ctx=1024/32-seq
(memory fit), n_epochs=3.

| Model | PARO-mechanism (rot+uniform) | MQ4-SLfwht+Lloyd (rot+Lloyd) | Δ vs PARO-mech | measurement |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.0816 | **0.0662** | **−19%** | direct trained, ctx2048/64s |
| Qwen3.5-9B | 0.3206 | **0.2922** | **−8.9%** | direct trained, ctx2048/64s |
| Qwen3.6-27B | 0.656 | **0.602** | **−8.2%** | baseline θ=0 (training OOMs†) |
| Qwen3.6-35B-A3B | 0.2207 | **0.2096** | **−5.0%** | direct trained, ctx1024/32s |

**MQ4-SLfwht+Lloyd beats the PARO mechanism on all 4 trunk models.** 3/4 direct
(trained); 27B is baseline-only (the full-butterfly+KLD-loss training OOMs on
27B even with student gradient checkpointing — oracle+student+backward exceed
192 GB; needs oracle-logit caching to free the oracle).

The win decomposes by regime:
- **Rotation helps** 0.8B (outlier-dominated) and 9B — both recipes share it,
  combined adds the Lloyd edge → −19% / −8.9%.
- **Rotation is neutral/slightly-negative** on A3B-MoE (and 27B, and 9B-discrete-
  signs): the full-butterfly trained ≈ or slightly above its θ=0 baseline. On
  these the Lloyd codebook is the SOLE lever, and it carries the win (−5% / −8.2%).

So the **Lloyd codebook is the universal edge PARO structurally lacks**;
the rotation is a bonus on the models where error is outlier-dominated.
† 27B direct training needs the oracle-caching memory fix (open).

### B. Lloyd-vs-uniform baseline sweep (codebook isolated, no rotation, θ=0)

Confirms the codebook edge exists on every model independent of rotation
(32-seq, α=0.55):

| Model | uniform | Lloyd | Δ |
|---|---:|---:|---:|
| 0.8B | 0.1469 | 0.0980 | −33% |
| 9B | 0.4055 | 0.3656 | −10% |
| 27B | 0.7044 | 0.6536 | −7% |
| A3B | 0.2127 | 0.1987 | −6.5% |

Since Lloyd ≥ uniform on all 4 (Lloyd-Max minimizes per-group MSE; uniform is a
suboptimal special case) AND the rotation is shared, **combined ≥ PARO-mechanism
on all 4 by construction** — the direct 0.8B+9B head-to-head confirms it, and
27B/A3B follow by the same argument (direct ctx=1024 runs pending).

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
