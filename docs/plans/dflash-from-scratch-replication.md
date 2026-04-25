# DFlash From-Scratch Replication — Post-Mortem

**Date:** 2026-04-19
**Goal:** Reproduce z-lab's DFlash training recipe locally so we can train domain-specialized drafts for agentic/tool-calling workloads. Stretch: port the same recipe to non-Qwen target architectures (Gemma, Llama, etc.) to prove cross-arch generality.
**Outcome:** **Did not match z-lab baseline.** Our best 9B from-scratch draft reached τ = 0.15 vs z-lab's τ = 2.65-4.39 (**~20× worse**) on the same target + prompts. Training infrastructure is correct; the gap is in training **dynamics / hyperparameters**, which we cannot close on a single MI300X with batch=1. **Pivoting to sidecar-only domain-specialization** (documented separately).

## What we tried (7 runs)

Chronological summary of the seven 9B/4B training attempts:

| # | target | arch | LR | steps | resume? | outcome (Rust τ vs z-lab) |
|---|---|---|---|---|---|---|
| 1 | Qwen3.5-4B | 16/256 + partial rotary (inherited) | 3e-4 | 5000 | no | τ=0.09 — truly broken (wrong arch) |
| 2 | Qwen3.5-4B | zlab-match (32/128 full rotary) | 3e-4 | killed at 1300 | no | (killed on false τ-probe signal) |
| 3 | Qwen3.5-4B | zlab-match | 5e-5 | killed at 500 | no | (killed on false τ-probe signal) |
| 4 | Qwen3.5-4B | zlab-match | 1e-5 | 2000 | warm-start z-lab | τ byte-identical to z-lab (LR too small) |
| 5 | Qwen3.5-4B | zlab-match | 1e-4 | killed at 1500 | warm-start z-lab | τ REGRESSED on all eval prompts |
| 6 | Qwen3.5-9B | zlab-match | 5e-5 | 25000 | no | τ=0.12 @ hermes_test, 0.20 @ hermes_agent |
| 7 | Qwen3.5-9B | zlab-match | 5e-5 | 14000 (killed) | warm-start run 6 | τ=0.15 / 0.19 — marginal improvement |

## What went right — infrastructure

Ten weeks of rigorous verification confirmed our training pipeline is **correct**:

1. **Forward / mask / loss** verified via `scripts/dflash_diag_zlab_loss.py`: loading z-lab's weights through our forward gives `weighted_loss = 1.81` on our corpus — within ±0.1 of z-lab's paper baseline. Means the sparse multi-anchor attention mask (paper Figure 4), target-layer hidden-state injection (T1), multi-block concat (T2), and position-weighted CE loss (T4) are all implemented correctly.

2. **Training mask is paper-faithful** (proven in `scripts/dflash_diag_mask.py`): `j < a_k` context visibility + bidirectional within-block noise attention matches the inference spec_generate semantics (which uses `attention_mask=None, is_causal=False`).

3. **Tokenization parity** verified in `scripts/dflash_diag_tokenization.py`: training-style and inference-style ChatML produce identical token IDs on Qwen3.5 tokenizer. ChatML mismatch was NOT a bug source.

4. **Config instantiation is correct**: rewrote `build_draft_config` to construct a flat `Qwen3Config` (not fragile composite `Qwen3_5Config`), force all layers `"full_attention"`, route `rope_parameters` through transformers-5.x schema, assert `num_hidden_layers == len(layer_types)`. End-to-end `save_pretrained` → `dflash_convert --mq4` → Rust engine load is validated.

5. **Architecture matches z-lab exactly** via `--match-zlab-arch`: 32 heads × 128 head_dim, full rotary, rope_theta=1e7, tied embeddings, intermediate=9728. z-lab's weights load into our `DFlashDraftModel(cfg)` with missing=0, unexpected=0. Shape-bit-identical.

## What went wrong — training dynamics

The EMA loss trajectory reveals the core problem. From run #6 (9B scratch 25k):

```
step 0:      EMA = 12.17 (random init)
step 5000:   EMA = 6.63
step 10000:  EMA = 4.22
step 15000:  EMA = 3.48
step 20000:  EMA = 3.04
step 25000:  EMA = 2.95
```

Asymptote: ~2.9-3.0. z-lab's baseline on our training forward: **1.81**. We plateaued 60% higher than z-lab's converged loss.

Resume run #7 (25k → 50k) at LR=5e-5 warmup=100:

```
resume step 2500  (cumulative 27500): EMA 2.33
resume step 7500  (cumulative 32500): EMA 2.16  ← brief minimum
resume step 10000 (cumulative 35000): EMA 2.22
resume step 14000 (cumulative 39000): EMA 2.49  ← rising again
```

EMA oscillated around 2.2 — got closer to z-lab's 1.81 but never converged there, and began drifting back up. This is the classic "batch=1 gradient noise causes optimizer thrashing" regime.

**Why we can't match z-lab on 1× MI300X:**

1. **Batch=1 gradient variance.** Paper's "6 epochs AdamW" recipe almost certainly used a much larger effective batch (likely 32-128 via data-parallel + grad accum). Noise in batch=1 updates is `~sqrt(batch_ratio)` larger — for us ~8× the noise of batch=8. Optimizer trajectory is noisier, converges to a worse local minimum.

2. **Loss ≠ τ proxy problem.** Run #5 demonstrated cleanly: even when loss dropped from 2.99 → 1.0 EMA on agentic corpus (warm-started from known-good z-lab weights), **τ regressed on ALL eval prompts** including the domain-matched Hermes template. The model was over-indexing on abundant structural tokens (`<|im_end|>`, `<|im_start|>`, `\n`, JSON boilerplate) at the expense of the novel-content tokens that matter for deployment speculation. Per-position CE loss is a systematically biased proxy for τ when the training corpus has heavy template/structural token density.

3. **Single-sample anchors are too sparse.** With K=4 masked anchors per 4096-token sequence, each step only provides gradient signal on 4 × 15 = 60 predicted positions. Over 25k steps that's 1.5M supervised predictions — far less than z-lab's equivalent given their larger effective batch.

## What we verified is NOT the problem

- **Architecture match**: z-lab arch verified shape-identical (no regression from architecture choice).
- **Rotary convention**: full rotary (not partial) confirmed correct via `--match-zlab-arch`.
- **Tokenization**: training/inference parity confirmed.
- **Loss forward/backward**: z-lab weights produce reference loss through our pipeline.
- **KV-cache inference path**: the `τ=1.0` probes during training were a **false negative** from a transformers-5.5.4 `DynamicCache(config=...)` bug on Qwen3.5 hybrid-attention targets (proof in `scripts/dflash_cache_test.py`). Rust engine uses its own cache correctly.

## Could clustering fix it?

Potentially. Three approaches we could try on 8× MI300X:

1. **Pure data parallelism (DDP)** — effective batch 8 across 8 GPUs. Cost: ~10hr × $48/hr = $480 for 50k steps. Would quadruple gradient quality. Unknown whether that's enough to close the 20× τ gap.

2. **DDP + grad accum** — effective batch 64. ~20hr × $48 = ~$960 for 50k steps. Approaches z-lab's probable effective batch. Higher probability of success.

3. **Wait for z-lab to release training code** (their README promises it "soon"). Would skip all recipe-reverse-engineering risk.

None of these are guaranteed. A single failed run at $480-960 buys much more sidecar progress.

## What worked for comparison: sidecar domain-specialization

An earlier test showed a Hermes-corpus-calibrated sidecar paired with `carnice-9b` produced clean `<tool_call>` output on a kimi Hermes trace, while a wikitext-calibrated sidecar on the same model degenerated into `<function=write>` loops. That's an **orthogonal** signal from draft quality, pairs with any working draft (z-lab's included), and is roughly 2 hours of MI300X time per target.

**4 sidecars × 2hr × $2/hr = $16** to produce agentic-specialized sidecars for our full Qwen3.5 + 3.6-A3B matrix. vs $480-960 for a speculative draft retrain.

## Decision

Stop pursuing from-scratch draft training with current infrastructure. Keep z-lab's 4B / 9B / 27B drafts as the dense baselines. Invest the compute budget in:

1. **Agentic sidecar calibration for all 4 targets** (primary value lever, proven effective).
2. **Qwen3.6-35B-A3B from-scratch draft** (ONLY target where z-lab has no baseline — our draft is the baseline; even a modest quality draft is a research contribution).
3. **Optional future**: if DDP is added + a dedicated 8× rental window opens, retry 9B scratch as a methodology experiment. Low priority.

## Scripts & artifacts preserved

- `scripts/dflash_train_poc.py` — training script with `--match-zlab-arch`, `--resume`, `--grad-ckpt-target` flags. Production-ready; just waits for a better recipe.
- `scripts/dflash_diag_zlab_loss.py` — loads z-lab weights through our forward, validates loss-parity to paper.
- `scripts/dflash_diag_mask.py` — verifies sparse attention mask matches paper Figure 4.
- `scripts/dflash_diag_tokenization.py` — verifies training/inference tokenization parity.
- `scripts/dflash_diag_config.py` — shape-diffs safetensors vs fresh model.
- `scripts/dflash_cache_test.py` — documents the transformers-5.5.4 hybrid-cache bug.
- `scripts/dflash_diag_iter1_tau.py` — tests a draft at max_cycles=1.
- `scripts/mi300_chain_runner.sh` — sequential training-job runner.
- `scripts/fetch_calibration_corpus.sh` — corpus builder with `agentic_xl` recipe (Nemotron-Agentic-v1 + hermes + hermes-filtered + tool-calls-multiturn + xlam).

## Open research questions for later

1. **Can we write a τ-direct loss?** CE is only a proxy. A direct τ-maximizing objective (expectation over acceptance-length) would be non-differentiable but could be surrogate-trained via policy gradient. Significant R&D.

2. **Is there a sample-efficient draft training recipe?** Batch=1 with K=4 anchors gives only 60 supervised predictions per step. Curriculum, active learning, or GAN-style adversarial sampling might close the gap with fewer tokens.

3. **DDTree for inference-time τ recovery.** DDTree (Ringel & Romano, arXiv:2604.12989) is a tree-verify extension that can multiplex several drafted paths per cycle. We have a partial implementation but per-cycle overhead dominates; our 2026-04-14 test showed +130% τ on hard creative prompts but wall-clock SLOWER due to 30ms/cycle overhead. Worth revisiting after kernel-level tree-verify optimization.

4. **Reach out to z-lab.** Their paper claims code release "soon". Their recipe would shortcut every open question above.
