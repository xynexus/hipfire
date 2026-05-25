# Assignment: Survey post-quantization training methods we can adapt (cited)

Owner: hermes research agent  ·  Branch: `feat/learnable-fwht`  ·  Date: 2026-05-25
Type: **research / literature review** — return cited methodologies, not code.

## Why

We quantize Qwen3.5/3.6 to ~4-bit (MQ4) for VRAM fit. Quantizer geometry is exhausted —
FWHT rotation + Lloyd codebooks + per-g64 subscales + Q8 codebook all measured; range
was −0.06% to +78% KLD, noise next to the bottom-2-bit cliff. MQ6 (~6.5bpw) is near-F16;
MQ4 (~4.5bpw) holds ~0.05–0.13 KLD. Helical MQ4 Lloyd pays MQ6-like decode tax for MQ4
quality. Conclusion: the lever is **retraining to the quantized forward**, not codebooks.
Before spending GPU on it, we want a cited map so we don't take shots in the dark.

## Deliverable

A `docs/investigations/2026-05-25-ptq-retraining-survey.md` report: for each method —
1-paragraph mechanism, paper/repo citation (title, authors, year, arXiv/GitHub URL),
bpw + reported quality, compute cost, code availability, and **how it adapts to our stack**
(4-bit, FWHT-rotated weights, Lloyd/codebook, RDNA/HIP, Rust runtime, no CUDA). End with a
ranked top-3 for us with rationale, and a flagged "avoid/known-falsified" list.

## Cover at minimum (find the rest)
- **PTQ rounding:** AdaRound, BRECQ, OBQ, GPTQ, AWQ, SmoothQuant.
- **Rotation/incoherence:** QuIP#, QTIP, SpinQuant, QuaRot (vs our FWHT — what's new).
- **Block reconstruction:** OmniQuant, AffineQuant, OS+/learnable clip.
- **QAT-lite:** EfficientQAT, LSQ, PEQA, low-rank adapters (QLoRA/QA-LoRA, LoftQ).
- **Vector/codebook + retrain:** AQLM, QuIP# codebooks, GPTVQ (cheap codebook recovery?).
- **Distillation:** logit-KLD distill of a quantized student (cost vs gain on ≤9B).

## Constraints to score against
- Must reach ~4.5bpw, recover toward MQ6 KLD; decode tax ≤ MQ6, ideally Q8-codebook.
- Inference is Rust/HIP gfx10-12 — needs a buildable dequant+codebook GEMV, no CUDA-only.
- Calib-only / hours-not-days preferred; flag method's CUDA-only-pipeline blockers.
- Required test = production kldref (proxy↔prod ~2×, direction unreliable — butterfly
  residual: −8% proxy MSE → +0.3% prod KLD). Cite methods that report real eval not proxy.

## Do not return code or run GPU. Cited methods + adaptation notes + ranked picks only.
