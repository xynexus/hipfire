<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Generate-based distillation for DFlash/DSpark drafters (fine-tune phase)

**Status: TODO.** Follow-on to the medgemma-27b drafter build (2026-07-09). Applies
to any DFlash/DSpark drafter, not just medgemma.

## The gap it closes

A spec-decode drafter is distilled from the target (teacher): data-gen runs the
target over a corpus, captures its hidden states at the extract layers + its
output distribution, and the drafter is trained to match that distribution
(`0.9·TV + 0.1·CE + conf`). Today the corpus is **teacher-forced human text**
(BioASQ + tulu-3): the teacher *reads* an existing sequence and we capture its
per-position predictions.

But at **deployment the drafter operates on the target's OWN generation stream** —
it drafts continuations of what the target is actively producing. So the ideal
training distribution is the target's own outputs, not human-written text.
Teacher-forcing on a human corpus is a cheap, effective proxy, but it leaves a
train/inference distribution gap that caps acceptance (τ). Generate-based
distillation removes that gap.

## What it is

Same trainer, same loss, same data-gen capture — **only the corpus source
changes**: instead of human text, the corpus is the **target's own generations**.

1. **Regenerate.** Take a prompt set (medical prompts for medgemma — questions
   *without* answers; general instructions) and have the target **generate**
   responses (AR decode). Use the fast quantized target (e.g.
   `medgemma-27b-text-it-q8f16.hfq`) with batched generation.
2. **Capture.** Run the existing `dspark_labels` data-gen over the
   prompt++generation sequences → a generate-based DSLB (hidden at the extract
   layers + target next-token labels, exactly as now).
3. **Fine-tune.** Continue training the drafter from the teacher-forced base
   checkpoint: **warm-start, lower LR, fewer epochs** — a distribution-shift
   fine-tune, not a from-scratch run.

## Why a fine-tune step (not from scratch)

- Generation is much more expensive than teacher-forcing (AR decode vs. one
  forward pass) — the dominant cost, so you want to spend the cheap teacher-forced
  pass first and only fine-tune on the expensive generate-based data.
- It's a distribution *shift*, not more of the same — warm-starting from the base
  drafter and nudging toward the deployment distribution is the right shape
  (cf. staged EAGLE/DFlash recipes).
- The base (teacher-forced) drafter already learns the mechanics (block drafting,
  hidden-conditioning); the fine-tune only re-centers its distribution.

## Sequencing (fits the existing pipeline)

- Phase 1 — teacher-forced pretrain (**current build**): human corpus → DSLB →
  train the drafter (`init_dspark_model`, the f16s/f16sc2 backward).
- Phase 2 — generate-based fine-tune (**this TODO**): target generations → DSLB →
  resume-train the base drafter at lower LR for a few epochs → convert → validate
  τ (expect a measurable acceptance bump vs. the teacher-forced-only drafter).

## What hipfire needs to add

- A **regenerate driver**: prompts → batched AR generation on the target →
  prompt++response token sequences in the `{"tokens":[...]}` corpus format the
  data-gen already consumes. (hipfire has the fast quantized target + generation;
  this is mostly a batched-decode + dump loop, no new kernels.)
- The trainer already supports resume/warm-start + LR schedule — reuse for the
  fine-tune. No new loss/arch.

## Open questions

- **Prompt source** for generation — medical prompt bank (BioASQ questions
  stripped of answers; USMLE-style; clinical instructions) + general prompts.
- **Sampling** — temperature/top-p for the teacher's generation (match the
  intended deployment sampling; greedy vs. sampled changes the target
  distribution the drafter must match).
- **Data volume + mix** — how much generate-based data, and whether to blend a
  little teacher-forced text to retain coverage.
- **Compute** — generation over a large prompt set on one gfx1151 is a multi-day
  job (AR decode on 27B); batch aggressively, or run on stronger hardware.
- **Validation** — measure τ (acceptance length) teacher-forced-only vs.
  +generate-based to quantify the gap this closes.

## References

- `docs/plans/dflash-trainer.md` — the "regenerate" phase (§ The 3-phase pipeline,
  step 1) names this as hipfire's real data-gen advantage.
- DSpark / DFlash papers (arXiv 2607.05147, 2602.06036) — target-distribution
  distillation objective.
- `crates/hipfire-arch-gemma3/examples/dspark_labels.rs` — the data-gen capture to
  reuse; `~/.hipfire/datasets/medgemma-27b-dspark/build_med_corpus.py` — the
  teacher-forced corpus builder this fine-tune phase replaces the corpus of.
