# Phase 3 — Butterfly Residual Smoke Gate (2026-05-22)

**Result: PASS** — trained KLD (0.012440) < baseline KLD (0.012759), Δ = −2.50%.

The proxy-doesn't-transfer failure mode that killed three prior session
levers ([[project_mq4_falsified_levers_2026_05_22]],
[[project_grad_scale_learn_falsified_2026_05_22]],
[[project_paro_k0_falsified_2026_05_22]]) is NOT manifesting at the smoke
scale. Direction is correct. Phase 4 (full 0.8B, 138 tensors × 128 seqs ×
~5 epochs) is justified.

## Setup

- Model: Qwen3.5-0.8B BF16 HF safetensors
- Wrapped tensors (6, layer-0 only):
  - `model.layers.0.linear_attn.in_proj_a`
  - `model.layers.0.linear_attn.in_proj_b`
  - `model.layers.0.linear_attn.in_proj_qkv`
  - `model.layers.0.linear_attn.in_proj_z`
  - `model.layers.0.mlp.gate_proj`
  - `model.layers.0.mlp.up_proj`
- IMPLEMENTATION_PLAN.md said "5 tensors (q_proj, k_proj, v_proj, gate_proj,
  up_proj)". Qwen3.5-0.8B layer 0 is DeltaNet (linear_attn), not standard
  self_attn, so the actual AWQ-F1 layer-0 superset is the 6 tensors above.
  Structurally identical to the plan's intent (small layer-0 subset).
- Imatrix: `/workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf`
- AWQ scales: closed-form paper formula at α = 0.55 (FROZEN)
- Calibration corpus: `/workspace/calibration-mix-v1.txt`
  (md5 `68a1d2e62117e692e0e04c2811349aaf`)
- Train: 8 sequences × 2048 ctx × 2 epochs = 16 SGD steps
- Optimizer: SGD lr=1e-3 momentum=0.9 cosine_floor=0.05 grad_clip=1.0
- Eval (smoke gate): logit KLD between BF16 oracle and student over the
  same 8 sequences (in-distribution; OK for direction check)
- Hardware: mi300 (gfx942 / CDNA3), single GPU
- Wall-clock: 15.7s training + ~5s eval pre/post = ~30s total
- Commit: `03986c59` on `feat/learnable-fwht`

## Results

| Metric | Value |
|---|---:|
| Baseline KLD (theta = 0) | 0.012759 |
| Trained KLD (theta after 16 steps) | 0.012440 |
| Δ KLD | −0.000319 |
| Rel. Δ | −2.50% |

### Final theta norms (post-training)

| Tensor | ||theta|| |
|---|---:|
| linear_attn.in_proj_a | 4.62 × 10⁻⁵ |
| linear_attn.in_proj_b | 1.41 × 10⁻⁵ |
| linear_attn.in_proj_z | 1.64 × 10⁻⁶ |
| linear_attn.in_proj_qkv | 1.08 × 10⁻⁶ |
| mlp.gate_proj | 1.02 × 10⁻⁷ |
| mlp.up_proj | 4.11 × 10⁻⁸ |

All well below the π warning threshold (rotations < half-turn). The
movement is conservative — 16 steps at lr=1e-3 with smooth cosine decay.

## Why this matters

Three prior session levers were FALSIFIED *despite* their proxy objectives
moving in the right direction. The pattern:

1. Proxy loss (per-Linear MSE) decreased during training.
2. Production KLD (BF16 vs final quantized) increased.
3. Root cause: per-256-group min-max RTN breaks the `(x/s)·(W·s) = x·W`
   invariance once `s` varies non-uniformly within a group.

The butterfly residual avoids this failure mode by construction: it does
not perturb the channel scales OR the weight, only the rotation applied
to the same `W*s` value. The per-group dynamic range of the rotated
weights changes, but in a direction that the optimizer can target via
gradient (since the rotation is differentiable).

Phase 3 measures KLD directly (not the proxy), confirming the direction.
Phase 4 will scale this to the full model and answer whether the
production-grade ≤ 0.10 KLD threshold is reachable.

## Caveats

1. **Smoke-eval is in-distribution.** The 8 sequences used for KLD eval
   are the SAME 8 used for training. The 2.5% improvement is over a tiny
   sample; full validation requires a larger held-out slice.
2. **Only 6 of 138 tensors wrapped.** The full-model baseline KLD is
   roughly 0.13 (per prior session). The layer-0-only KLD of 0.012 is much
   smaller — this is a per-tensor smoke, not a model-level measurement.
3. **16 training steps is severely undertrained.** The paper hparams call
   for ~500-700 steps. Phase 4 will run the full schedule.

These caveats are fine for the smoke gate's purpose: verify direction
before committing to the expensive full run.

## Next step

Phase 4 — full Python validation on 0.8B with all 138 AWQ-F1 tensors,
128 seqs × 2048 ctx, ~5 epochs (~640 steps total). HARD gate:
0.8B butterfly KLD ≤ 0.10 to continue. Estimated wall-clock 30-60 min on
mi300, ~$10 compute.

## Artifacts

- Mi300: `/workspace/butterfly-phase3-smoke/`
  - `butterfly_residuals.npz` (29 KB)
  - `butterfly_residuals.hfbf` (25 KB)
  - `paper_awq_scales.hfsc` (15 KB)
  - `butterfly_meta.json`
  - `smoke.log`
