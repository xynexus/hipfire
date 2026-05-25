# Assignment: Post-Quantization Retraining to recover MQ6-class quality at MQ4 size

Owner: hermes research agent  ·  Branch: `feat/learnable-fwht`  ·  Date: 2026-05-25

## TL;DR of the decision that triggered this

Quantizer cleverness is exhausted. We tried codebook geometry every way:

| recipe | bpw | 0.8B KLD (mix.ctx4096, a0.55, ne0) | notes |
|---|---|---|---|
| per-sub g64 Lloyd (helical) | ~5.5 | 0.054 | best quality, 4 cb/256 |
| shared 1 cb/256 fp16 | 4.9 | 0.081 | |
| shared 1 cb/256 **Q8_0** | ~4.5 | 0.081 | codebook q8 is ~free (−0.06%) |
| global 1 cb/tensor | 4.25 | 0.144 | falsified, +78% |

MQ6 (~6.5bpw) is already near-F16. The whole point of MQ4 is the ~2bpw / ~30% VRAM
saving. **But helical MQ4 Lloyd pays MQ6-like decode tax** (4 subscales + codebook
lookup + FWHT per group) — so it spends MQ6's perf without MQ6's quality. Worst trade.
Geometry won't close the cliff; the bottom two bits are an information problem.
**The only lever left is retraining the model to its own quantized forward.**

Strong prior (do not relitigate): "Tier 3 butterfly residual FALSIFIED" — a per-Linear
proxy win of −8% MSE translated to +0.3% prod KLD. Any approach that optimizes a Python
proxy metric and does NOT eval through the production kernel will mislead. Treat
proxy↔prod as ~2× and unreliable in direction.

## Goal

Recover MQ6-class KLD at ~4.5bpw on Qwen3.5 0.8B/9B (and check A3B), measured in
**production kldref**, by post-quantization retraining — not by more codebook design.
Target: prod-kldref MQ4 from current ~0.179 (9B) / ~0.13 floor toward MQ6 (near-0).
A win = sub-PARO prod KLD AND coherence-gate clean AND decode tax ≤ shared-Q8.

## Two tiers — do them in this order

1. **Block-wise reconstruction (cheap, do first).** AdaRound / OmniQuant-style:
   per layer, learn weight-rounding + clip (and the Q8 codebook subscales) to minimise
   ||W_q·x − W·x|| on calibration activations. No full backprop, hours on mi300. Cheap
   falsification of "retraining helps at all." Stop if block-recon ≤ shared-Q8.
2. **End-to-end KLD distillation (heavy, gate on tier-1).** Existing `--loss-kld` path in
   `scripts/learn_butterfly_mq.py`. Fine-tune weights (or low-rank adapters) of the real
   MQ4-helical-Q8 model against bf16 logits. Watch 27B OOM (cache oracle logits, free
   oracle: `--cache-oracle-logits`). Distill 0.8B+9B; A3B only if 9B clears.

## Hard rule: prod kldref, not proxy

Cheap pre-step before any claim: quantize 9B plain-MQ4 via existing hipfire-quantize,
eval `examples/eval_hipfire` vs `/workspace/kldref/qwen3.5-9b-bf16.kldref.bin`, compare to
Python baseline 0.353. If they don't map, the proxy is lying and proxy deltas are void.
Every retrain result reports BOTH proxy KLD and prod kldref + coherence-gate output.

## Assets (mi300, ssh `mi300`)
- 0.8B HF: `/workspace/qwen3.5-0.8b-hf` · 9B: `/workspace/9b-hf` · A3B: `/workspace/hf-models/qwen3.6-35b-a3b`
- imatrix: `/workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf`, `/workspace/qwen3.5-9b.tier1.mix.imatrix.gguf`
- corpus: `/workspace/calibration-mix-v1.txt` · kldref: `/workspace/kldref/*.bin`
- sim: `scripts/learn_butterfly_mq.py` (g64 subscale, `--shared-codebook`, `HIPFIRE_CB_Q8=1`, `--loss-kld`, `--cache-oracle-logits`)

## Deliverables
1. block-recon ladder vs blessed bars (0.8B 0.027 / 9B 0.036 / A3B 0.0149) + prod kldref.
2. go/no-go on distillation with prod numbers.
3. if win: prod kernel path for MQ4-helical-Q8 + decode tax vs MQ6 + coherence-gate report.
4. write findings (incl negatives) to docs/investigations/, commit per CLAUDE.md.

## Guardrails: do NOT chase tensor-global cb, more sub-PARO codebook geometry, or proxy-only wins; do NOT push master; reproduce blessed imatrix for the absolute bar before claiming "beats PARO."
