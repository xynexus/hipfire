# Latent-KV retraining lever: Tier-1 mechanism probe (Supra-50M)

Status: **complete** (mechanism probe; not admission evidence)

Date: 2026-07-11

Reference branch: `chaingun`

Companion to `docs/plans/2026-07-11-latent-kv-large-model-confirmation.md`, which
rejected every static / equivariant rank-32 basis on the *frozen* Qwen3.5 model
(0.8B/4B/9B) and left **per-model retraining** as the only remaining lever.

## Question

Can co-training a model (LoRA on q/v + RMSNorm, base frozen) make it *tolerate*
a fixed rank-r latent-KV bottleneck — i.e., recover the KL gap the bottleneck
introduces on held-out text? This is the cheapest test of the retraining lever,
run on a model the native `hipfire-train` stack already supports.

## Harness

`hipfire-train` example `latent_kv_recovery` (committed a2e5c16a9):

- **Bottleneck** (`src/latent_kv.rs`): per-(layer, kv-head) rank-r projector
  `P = U Uᵀ`, `U` = top-r eigenvectors of the post-RoPE K/V covariance (Jacobi
  eigensolver), calibrated on the train batch with projection off, applied as a
  forward-only STE perturbation on post-RoPE K and V. Projecting K alone
  realizes the shared-basis rank-r score `qᵀ P k`.
- **Student**: clean fp32 weights + trainable LoRA(q/v) + RMSNorm; base frozen.
  **Teacher**: clean fp32, no projection. Loss: KL(teacher‖student) on logits.
- **Model**: Supra-50M (llama, head_dim 64 → rank 32 = **2× KV reduction**,
  12 layers × 4 kv-heads). Native HIP training — no PyTorch.
- **Data**: real C4 `realnewslike` token windows, calibration/train and held-out
  eval from **disjoint documents** (seq 32, 48 train / 24 held-out windows).

## Results

Both configurations LR-tuned for stability (see caveats); KL in nats/tok:

| LoRA rank | in-sample KL (before→after) | held-out KL (before→after) | **held-out recovered** |
| --- | --- | --- | --- |
| 16 (lr 3e-5) | 0.710 → 0.160 (77.5%) | 0.718 → 0.599 | **16.6%** |
| 64 (lr 1e-5) | 0.709 → 0.173 (75.6%) | 0.718 → 0.600 | **16.5%** |

## Findings

1. **Retraining is the only lever with non-zero held-out recovery.** ~16.5% > 0,
   versus ~0% held-out for every *frozen*-basis approach (shared static, metadata
   mixture, position-equivariant) at 4B/9B. Co-training moves the needle where a
   fixed basis on the frozen model could not.

2. **The recovery is modest and capacity-saturated.** Quadrupling LoRA rank
   (16 → 64) leaves held-out recovery unchanged (16.6% → 16.5%), while in-sample
   stays ~76–78%. **Adapter capacity is not the binding constraint** — the ceiling
   is set by data scale (48 windows) and/or the 50M model. By extension, a full
   weight tune (more capacity still) would not lift held-out on this data; it
   would only overfit in-sample harder. Do not pursue full-tune or higher rank at
   this scale.

3. **On real text the calibrated subspace generalizes** (raw latent-KV gap 0.71
   in-sample ≈ 0.72 held-out), unlike synthetic random tokens (0.48 vs 1.25),
   where the basis did not transfer and recovery was noise (4.4%). Using real,
   disjoint-document text was necessary to get a trustworthy held-out number.

## Caveats (why this is a probe, not a verdict)

- **Small model.** Supra-50M shows quirks that do not generalize; per the
  confirmation study, a failure here would not be definitive, and neither is a
  partial success.
- **Capacity comparison is LR-confounded.** Rank 64 diverged at the rank-16 LR
  (3e-5) and needed 1e-5; there is no gradient clipping to hold LR fixed across
  capacities, so "capacity" and "stable LR" are entangled.
- **Tiny data.** 48 train / 24 eval windows — the 77% in-sample vs 16% held-out
  gap is classic small-data overfitting.

## Engineering notes for Tier 2

- **Optimizer stability:** per-sequence AdamW on diverse real text diverges above
  a low LR (rank 16 stable only ≤3e-5; higher capacity needs lower). Tier 2 needs
  **gradient accumulation and/or clipping** — `hipfire-train` has no GPU
  add/clip helper yet, so this is real work.
- **Memory:** a residual per-iteration leak inside `model_distill_backward`
  (rank-independent, ~2 GB over thousands of iters) caps runs at ~120 steps /
  ~6k iterations before OOM. The example already frees the *returned* grads; this
  internal one remains.
- **gfx1103 instability:** intermittent HIP 719 ("unspecified launch failure")
  killed several runs non-deterministically; seq 64 faulted reliably (LDS
  hazard). Tier 2 should run on `halo`/`medusa` and test longer sequences there.

## Conclusion and Tier-2 scope

The retraining lever is **alive but unproven**: it is the only approach with real
held-out recovery, but at Supra-50M scale that recovery is modest and limited by
data/scale, not model capacity. This justifies Tier 2 without prescribing
full-tune.

Tier 2 (separate plan, own sealed split and gate):

- add Qwen3.5 forward/backward support to `hipfire-train` (model_type `qwen3_5`,
  attention output gate, QK-norm);
- train on real corpora at scale (thousands of windows) on 4B/9B;
- add gradient accumulation + clipping for stable training;
- redefine the gate vs the *original* model (not the per-cache oracle);
- only then re-test LoRA vs full-tune, where capacity may finally matter.
