# PTQ tuning-step survey (cited) — for AWQ'd, mixed-precision MQ4 residual

Scope: step 3, tune an already-AWQ'd, kmap-mixed 4-bit model. Bars: 0.8B 0.027 / 9B 0.036.
Constraint: calib-only/hours, prod-kldref, RDNA-HIP+Rust runtime (CUDA-only = blocker).

| method | mech | bpw | cost | code | CUDA-lock | adapt |
|---|---|---|---|---|---|---|
| AdaRound (2004.10568) | learn round up/down per layer | 4 | hrs | y | no | rounding only; pairs w/ our Lloyd, cheap first shot |
| BRECQ (2102.05426) | block recon, Fisher | 4 | hrs | y | no | block loss, no full bp — tier1 |
| OmniQuant (2308.13137) | learnable clip+scale, block | 4 | hrs | y | no | weights frozen, learn clip — fits FWHT |
| EfficientQAT (2407.11062) | block train all params + e2e | 4 | hrs-1d | y | no | **best fit: calib QAT, weight-only, runs ROCm** |
| OmniQuant/AffineQuant (2403.12544) | affine pre-transform | 4 | hrs | y | no | overlaps our rotation; marginal |
| GPTQ/AWQ (2210.17323/2306.00978) | done (baseline) | 4 | min | y | no | floor, already in pipeline |
| QuIP#/QTIP (2402.04396/2406.11235) | E8 codebook+rotate | 2-4 | days | y | **yes** | CUDA codebook kernel — RDNA blocker |
| AQLM (2401.06118) | additive VQ + e2e tune | 2-3 | days | y | **yes** | best <3bpw quality; CUDA kernels |
| QLoRA/QA-LoRA/LoftQ (2305.14314/2309.14717/2310.08659) | frozen-quant + LoRA adapters | 4+lora | hrs | y | no | recover quality w/o touching weights; merge adapters |
| logit-KLD distill (`--loss-kld`) | match bf16 logits | any | 1d | ours | no | heaviest, last resort |

## KEY: flat-MQ4 stays. EfficientQAT/AdaRound/OmniQuant/LoftQ tune weights to the
4.25bpw uniform grid — no helical kernel, no kmap, ships on existing GEMV. Retraining
may delete helical, not just cheapen it. (bar = prod-kldref, unproven til run.)

## Top-3 for us
1. **EfficientQAT** — weight-only block QAT, calib-only, hours, ROCm-runnable, reports near-FP16 at 4bpw on Llama; tune the flat layers, keep helical-Q8 on hot. Closest to bar without CUDA lock.
2. **AdaRound/BRECQ first** — cheap block recon to confirm retraining helps before QAT.
3. **LoftQ adapters** — if weight tune underperforms, low-rank merge recovers quality, no kernel change.
## Avoid: QuIP#/AQLM (CUDA codebook kernels, RDNA blocker); proxy-only claims (butterfly residual −8% MSE → +0.3% prod).
