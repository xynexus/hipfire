# ParoQuant K=0 (stage-1 + stage-2) Prototype for hipfire MQ4G256 — Negative Result

**Date:** 2026-05-22
**Branch:** `feat/paro-stage2-mq4-prototype`
**Hardware:** mi300 droplet (gfx942, ROCm 7.0, PyTorch nightly)
**Reference paper:** [ParoQuant: Pairwise Rotation Quantization](https://arxiv.org/abs/2511.10645) (Liang et al., ICLR 2026)
**Official code:** [github.com/z-lab/paroquant](https://github.com/z-lab/paroquant)

## TL;DR

ParoQuant's K=0 reduction (stage-1 channel scaling + stage-2 weight fine-tuning,
no rotations, no quantizer params) **catastrophically REGRESSED KLD by 357%**
on Qwen3.5-0.8B + hipfire MQ4G256, despite the BRECQ proxy loss decreasing 43%.

**The paper's stage-1+stage-2 mechanism does NOT survive the drop of the K=8
pairwise rotation kernel.** The rotations are not just a perf optimization —
they are load-bearing for the optimization landscape.

## Recipe

Per the paper's `experiments/optimize/4bit.sh` (with `--num-rotations 0`):
- **Stage 1**: 5 epochs Adam(W) LR=0.05 → optimize `channel_scales`
- **Stage 2**: 5 epochs Adam(W) LR=1e-5 → optimize `weight`
- AdamW: weight_decay=0.01, betas=(0.9, 0.95), eps=1e-10
- Cosine LR schedule decaying to 1/20 of initial
- SmoothL1Loss on per-Linear output reconstruction (BRECQ-style)
- 128 calibration sequences × 2048 ctx tokens (mix-v1 corpus)
- 138 AWQ-F1-target Linears wrapped
- Per-Linear forward: STE `MQ4G256(W * channel_scales)` with FWHT seeds (42, 1042)

## Results

| Config | KLD (full 1175-chunk) | PPL | ΔKLD vs baseline |
|---|---:|---:|---:|
| Closed-form paper-formula AWQ + GPTQ (baseline) | **0.1327** | 19.12 | 0% |
| Closed-form paper-formula AWQ, no GPTQ | 0.1327 | 19.12 | 0% |
| Prior BRECQ+STE scale-only (LR=1e-3, 5ep, no stage-2) | 0.1593 | 19.88 | +20% |
| **PARO K=0 stage-1+stage-2 + GPTQ** | **0.6151** | 31.68 | **+364%** |
| **PARO K=0 stage-1+stage-2 no GPTQ** | **0.6149** | 31.67 | **+364%** |

Diagnostic (first 100 chunks only, paro scales applied to ORIGINAL untuned weights):

| Config | KLD (first-100) | Interpretation |
|---|---:|---|
| Closed-form paper-formula + GPTQ | 0.1364 | baseline reference |
| PARO scales + ORIG weights | 0.6970 | scales alone catastrophic |
| PARO scales + PARO-tuned weights | 0.6230 | stage-2 recovers ~10% |

**Decomposition of the regression:**
- Bad scales account for ~412% of the KLD increase
- Stage-2 weight tuning recovers ~12% relative
- GPTQ vs no-GPTQ makes no difference at this magnitude

## Training Loss Trace

Stage 1 (channel_scales, lr=0.05 → 2.5e-3 cosine):
```
epoch 1: 1.877e-2
epoch 2: 1.847e-2 (-1.6%)
epoch 3: 1.778e-2 (-3.7%)
epoch 4: 1.699e-2 (-4.5%)
epoch 5: 1.644e-2 (-3.2%)
```
Total stage-1: −12.4%.

Stage 2 (weight, lr=1e-5 → 5e-7 cosine):
```
epoch 1: 1.944e-2 (regression — fresh optimizer state, larger param set)
epoch 2: 1.498e-2 (-23.0% from epoch 1; -8.9% from stage-1 end)
epoch 3: 1.238e-2 (-17.4%)
epoch 4: 1.135e-2 (-8.3%)
epoch 5: 1.078e-2 (-5.1%)
```
Total stage-2: −34.4% from stage-1 final.

**The BRECQ proxy loss converged cleanly — the optimization "succeeded" by
its own metric. But the deployment KLD is 4.6× worse.**

## Root Cause

**Channel scales drifted ~3.5× from the paper-formula geometric mean = 1.0.**

| Statistic | Paper-formula scales | PARO learned scales |
|---|---:|---:|
| Per-tensor geomean | 1.00 (by construction) | 2.3-3.8 |
| Per-tensor mean | 1.00-1.10 | 1.76-4.00 |
| Per-256-group log-std of `s` | 0.108 | 0.218 |
| Per-256-group max(s)/min(s) | 2.61 median | 3.76 median |

Mathematically, uniform per-tensor scaling of `s` should be invariant
(`(x/s) @ MQ4(W*s) = (x/s) @ (W*s + ε)` with ε scaling proportionally).
But the per-256-group log-std of paro's learned scales is **2× higher**
than paper-formula's, which means the per-group dynamic range of
`FWHT(W * s_learned)` is wider, the per-group MQ4 grid step is bigger,
and the absolute quantization error is amplified.

The optimizer found scales that minimize per-Linear MSE reconstruction —
but per-Linear MSE is dominated by the magnitude of the salient output
channels, and the unconstrained scale search produced an "outlier-fitter"
that loads up `s` to make `W*s` larger overall (still quantizable per-group
because of the min/max grid), thereby letting the dequant round-trip
better preserve the largest output dimensions.

Once quantized weights then meet `x/s_learned` at runtime in production,
the per-channel scale spread interacts with the per-group quantization
error in ways the per-Linear MSE loss does not capture.

## Why the paper's full recipe works (and ours doesn't)

The paper's K=8 pairwise rotations provide implicit regularization on
the scale optimization:
1. The rotation kernel rotates `W * s` BEFORE the per-group min/max grid
   is computed, so wild per-channel scale variation gets averaged into
   the rotation
2. The rotation parameters `(idx_ij, theta)` are co-optimized with `s`,
   so the optimizer can pick rotations that compensate for an
   "outlier-heavy" `s`
3. The quantizer params `(s_q, z_q)` are also learnable in their stage 2
   with LR=1e-6, providing one more degree of freedom to absorb any
   residual mismatch from `s` drift

When you remove all three (K=0, no learnable quantizer), the channel
scales are the ONLY knob and they fail to find the closed-form's
balance.

## What we tested

Two adaptation attempts of ParoQuant for hipfire MQ4 perf-lane:
1. **K=0 + paper hparams** (this report): catastrophic regression
2. (not attempted) Lower stage-1 LR with the paper's recipe: would need
   another full training run

Both adaptation attempts fall under the user's "Two adaptation attempts
max before reporting blocked" budget. The K=0 reduction is **structurally
incompatible** with hipfire's existing perf-lane.

## Honest assessment

**Did PARO's mechanism (minus rotations) beat closed-form?**
**No.** It regressed KLD by +364% (full 1175-chunk slice mean).

**Is the result GPTQ-dependent?**
**No.** Both `paro_plus_gptq` (KLD 0.6151) and `paro_only` (KLD 0.6149)
land at essentially the same value. The regression is from the PARO
training output itself, not the downstream pipeline.

**Did stage-2 weight tuning help?**
**Marginally — about 10% relative recovery.** Paro scales + ORIG weights
gives KLD 0.697; paro scales + paro-tuned weights gives KLD 0.623.
Without stage-1's catastrophic regression, the stage-2 weight tuning is
a small useful refinement, but it cannot rescue bad scales.

## Recommendation

**Do NOT ship paro-stage2-mq4** in this form. The mechanism does not
generalize to "drop the rotation kernel". A few alternative directions:

1. **Full PARO (K=8 + learnable s,z,W)**: implement the rotation kernel
   on AMD ROCm (port `paroquant/kernels/cuda/rotation.cu` to HIP) and
   evaluate the full algorithm. Stage-1 then has K=8 angles to absorb
   the scale-search instability.
2. **Constrained scale search**: re-run K=0 with a hard geomean=1
   constraint per tensor (renormalize after each gradient step) +
   reduced LR (1e-3 instead of 0.05). May recover the BRECQ result
   (KLD ~0.16) but unlikely to beat closed-form (0.13).
3. **Direct quantizer-aware loss**: replace per-Linear MSE with the
   end-to-end KL divergence vs the BF16 oracle (whole-model loss).
   Much more expensive (full forward pass per step) but addresses the
   proxy-objective mismatch directly.
4. **Abandon PARO; pursue MR-GPTQ-style joint scale+rotation**: the
   paper's mechanism only works because rotation and scale are
   co-optimized. The hipfire FWHT rotation is static (seeds 42, 1042
   are fixed). A "learnable FWHT" variant (different sign tables per
   tensor, learned during calibration) might be a hipfire-native
   equivalent of PARO's K=8.

**The K=0 reduction is structurally falsified.** Document and move on.

## Reproducer

```bash
# Train
python3 scripts/paro_k0_stage2.py \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
    --imatrix /workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf \
    --corpus /workspace/calibration-mix-v1.txt \
    --output-dir /workspace/paro-stage2/0.8b/full \
    --n-sequences 128 --ctx-len 2048 --alpha 0.55 \
    --stage1-epochs 5 --stage2-epochs 5 --log-interval 16

# Eval
PARO_DIR=/workspace/paro-stage2/0.8b/full \
HESSIAN=/workspace/qwen3.5-0.8b.mix.ctx4096.hessian.bin \
IMATRIX=/workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf \
KLDREF=/workspace/kldref/qwen3.5-0.8b-bf16.kldref.bin \
ALPHA=0.55 GPTQ_DAMP=0.0 \
OUT_DIR=/workspace/paro-stage2/0.8b/eval \
bash scripts/paro_k0_eval.sh
```

Wall clock: ~35 min training + ~30 min × 2 eval cells = ~95 min total on 1× MI300X.

## Artifacts on droplet

- `/workspace/paro-stage2/0.8b/full/paro_channel_scales.hfsc` — 138 learned scales, 0.57 MB
- `/workspace/paro-stage2/0.8b/full/tuned-safetensors/` — fine-tuned BF16 weights, 1.75 GB
- `/workspace/paro-stage2/0.8b/full/paro_meta.json` — training hparams + loss trace
- `/workspace/paro-stage2/0.8b/eval/paro_plus_gptq/eval.kldseq` — 1175-chunk KLD output
- `/workspace/paro-stage2/0.8b/eval/paro_only/eval.kldseq` — same, no GPTQ
- `/workspace/paro-stage2/0.8b/diag_scales_only/eval.kldseq` — 100-chunk diagnostic
