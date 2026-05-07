# 04. Phase 2 Results — SE Hypothesis REFUTED

**Status:** Phase 2 ablation complete on hiptrx 2026-05-07 01:08 UTC.
30 runs (5 variants × 3 trials × 2 A3B models) on 4× R9700 gfx1201,
total walltime ~50 min after the layer-0 falsy-int fix at d7866f6.

**TL;DR:** Pre-registered SE hypothesis (per `03-super-expert-confirmation.md`)
**REFUTED on both 3.5-A3B and 3.6-A3B**. Three different proxy candidate
sets — D2 (weight ratio_p99 at layer 0), D3 (activation absmax_max at
layers 38-39), and D2∪D3 union — all close <3% of the All-MQ4 → All-Q8
PPL gap. The cliff is broadly distributed across all 10,240 routed
experts × 40 layers, not concentrated in any small candidate subset.

## Final results

### Qwen3.5-35B-A3B

| variant | mean PPL | gap closure | criterion verdict |
|---|---:|---:|---|
| V1 (All-MQ4 baseline) | 7.4496 | 0.0% | — |
| V2 (All-Q8 ceiling) | 7.0785 | 100.0% | — |
| **V3a** (17 D2 layer-0 experts pinned) | 7.4410 | **2.33%** | REFUTED |
| **V3b** (19 D3 layer-38/39 experts pinned) | 7.4499 | **−0.09%** | REFUTED |
| **V3c** (36 D2∪D3 union pinned) | 7.4410 | **2.31%** | REFUTED |

Absolute gap to close: 0.371 PPL (~5% relative).

### Qwen3.6-35B-A3B

| variant | mean PPL | gap closure | criterion verdict |
|---|---:|---:|---|
| V1 (All-MQ4 baseline) | 7.4607 | 0.0% | — |
| V2 (All-Q8 ceiling) | 7.1653 | 100.0% | — |
| **V3a** (17 D2 layer-0 experts pinned) | 7.4621 | **−0.48%** | REFUTED |
| **V3b** (19 D3 layer-38/39 experts pinned) | 7.4603 | **+0.11%** | REFUTED |
| **V3c** (36 D2∪D3 union pinned) | 7.4615 | **−0.29%** | REFUTED |

Absolute gap to close: 0.295 PPL (~4% relative).

### Cross-model concordance

Both models REFUTE the SE hypothesis at every candidate set. The signs
of the V3 deltas are different but all are in the noise (|closure| < 3%
in every cell). The cliff shape is structurally identical: ~5% PPL gap
between MQ4 and Q8 ceiling, broadly distributed.

3.6-A3B's gap is ~25% smaller than 3.5-A3B's (0.295 vs 0.371). 3.6 has
slightly more headroom under MQ4 in absolute PPL but the relative
difficulty of "find a small pin set that closes the gap" is the same
on both — empirically, no such set exists.

## What this means

### The MQ4 cliff is structural, not localizable

The SE narrative (per arXiv 2507.23279) is that <1% of MoE experts
produce extreme down_proj outputs and dominate quantization damage. We
identified two candidate sets via different proxies:

- **D2 weight-side**: 17 experts at layer 0 with extreme per-row
  `ratio_p99` (5e7-6.9e7 vs dense max ~40). 19/20 cross-model concordant
  in our `02-survey-results.md`.
- **D3 activation-side**: 19 experts at layers 38-39 with extreme
  per-channel `absmax_max` (~35 in fp32). 19/20 cross-model concordant.

Both proxies cleanly identify outliers under their respective criteria,
and both show structural concordance across 3.5/3.6. **Neither
corresponds to "experts whose Q4 quantization noise damages model
output."** The MQ4 quality damage is roughly uniform across all 10,240
experts at the relative-noise level — pinning 17 or 19 of them recovers
nothing because each contributes ~0.01% of the total damage.

This contradicts the strong reading of arXiv 2507.23279 for the
Qwen3.5/3.6 MoE family. We do not refute the paper's claim on
Mixtral-8x7B / DeepSeek-V2 (their reported models); we only refute that
the same SE-pinning structure applies to Qwen3.5/3.6 MoE under MQ4G256.

### Two distinct phenomena, both real but neither is the cliff

The D2 and D3 cliffs are real (per the `02-survey-results.md` data) but
each is its own thing:

- D2 = **MQ4 row-precision-loss artifact** at layer 0 down_proj for 17
  specific experts. The extreme weight-row tail forces the per-256-element
  absmax to a value that collapses small magnitudes into noise. This
  damages output ACCURACY for those rows when the input aligns with the
  outlier columns — but the calibration corpus rarely triggers that
  alignment.
- D3 = **arXiv 2507.23279 SE phenomenon** — extreme down_proj output
  magnitude at layers 38-39 for 19 specific experts. These experts
  produce large outputs because the residual stream is large at the
  model's depth; their relative quantization error is similar to other
  experts'.

Both are documented in `02-survey-results.md`. Neither is the proximate
cause of the MQ4 cliff that #171 reports.

### PPL on wikitext may be the wrong metric

The user-visible cliff that issue #171 reports is **token-attractors on
agentic prompts** — specifically a 200-word agent_prompt that triggers
incoherent loops on 3.6-A3B at MQ4. Wikitext-103 is general-domain prose
where MQ4 noise averages out across tokens; the 5% PPL gap we measured
is real but doesn't capture the catastrophic-on-agentic-prompts failure
mode. A coherence-style eval (running the same V3 variants through
`agent_prompt` and looking for token-attractors qualitatively) is the
direct test. That is a follow-up; this synthesis covers what the
pre-registered criterion measured, on the dataset it specified.

### Architectural implication for the engine

The `outlier_pin_set: HashSet<(LayerIdx, ExpertIdx)>` proposal floated
during 03-doc design (per-expert precision pinning as an Architecture
trait surface) **is not warranted by the Phase 2 evidence**. We
empirically tested whether pinning 17, 19, or 36 specific experts at Q8
recovers PPL — it doesn't. There's no per-expert precision lever to
ship, at least not via the value-ranking proxies we tried.

A coarser "Q8 these named tensors per arch" surface (which is what
fivetide's #171 patch already targets at `mlp.gate.weight`) is the
precision-pinning shape that's been validated empirically. We should
ship at that granularity, not finer.

## Path forward (highest-leverage first)

1. **Land tasks #24/#25 — fivetide's Q8-router patch.** Different
   surface (router gate `mlp.gate.weight`, not expert weights). Already
   validated by the contributor on #171's headline failure. Phase 2
   doesn't address this surface at all. Fast follow-up.

2. **Coherence-style eval on Phase 2 variants.** Run #171's
   `agent_prompt` through V1/V2/V3a/V3b/V3c on 3.6-A3B; check
   qualitatively for token-attractors. Cheap (~10 min). Tells us whether
   PPL was the wrong metric or the SE hypothesis is genuinely broken
   for this family.

3. **GPTQ/AWQ trial.** Calibration-aware quantization that minimizes
   per-layer output error rather than ranking by raw value.
   AutoGPTQ/AutoAWQ are off-the-shelf. If they cure the cliff at
   MQ4-equivalent bit budget, the engine integration question shifts to
   "support GPTQ format in HFQ4."

4. **HFP4 / NVFP4-equivalent format** (per the vision doc). RDNA-native
   block-floating-point. Better encoding density at the format level,
   generalizes across the whole family. Months-long format-design effort.

**What NOT to do:** spend more cycles on per-expert pinning at MQ4 with
new proxies. Three different proxies (D2 weight-tail, D3 activation
magnitude, union of both) all close <3% of the gap. The cliff is
structural.

## Pre-registration audit

Per `03-super-expert-confirmation.md`:

- Criterion fixed BEFORE measurement: gap closure ≥ 80% (CONFIRMED) /
  ≥ 30% (PARTIAL) / < 30% or negative (REFUTED).
- 5-variant design fixed BEFORE measurement (after the D3 amendment
  that added V3b/V3c at the time D3 results landed but before any
  Phase 2 PPL was collected — disclosed in the 23:50 UTC amendment
  block of `03-super-expert-confirmation.md`).
- 30 runs completed end-to-end with no post-hoc criterion changes.
- Trials are bit-identical (no GPU nondeterminism) — 3-trial loop
  confirmed determinism but added no statistical power.

The pre-registration succeeded in producing a clean REFUTED verdict
without ambiguity. Would not have been clean had we picked candidate
sets after seeing PPL data.

## Files

- `/tmp/hiptrx-survey/runs/phase2-qwen3.5-a3b/phase2_results.jsonl`
  — 15 records (5 variants × 3 trials)
- `/tmp/hiptrx-survey/runs/phase2-qwen3.6-a3b/phase2_results.jsonl`
  — 15 records (5 variants × 3 trials)
- pulled to k9lin at `/tmp/hiptrx-survey-pull/phase2/`
- `scripts/quant-survey/phase2_runner.py` (commit b4b6785 + ed8acef + d7866f6)
- `scripts/quant-survey/phase2_queue.sh` (commit 2572782 + 0790a03)
- `scripts/quant-survey/phase2_eval_corpus.jsonl` (md5 877d9ae1...)

## Cross-reference

- `00-recon-synthesis.md` — pre-survey synthesis (deprecated)
- `01-survey-runner-design.md` — methodology spec
- `02-survey-results.md` — Phase 1B+1C synthesis + D3 update
- `03-super-expert-confirmation.md` — pre-registered criterion (this doc verifies the verdict)
- `INVESTIGATION-LOG.md` — append-only timeline
- arXiv 2507.23279 — "Super Experts in MoE" (paper claim does not generalize cleanly to Qwen3.5/3.6 family per our test)
- GitHub issue #171 — fivetide's router-Q8 finding (orthogonal surface, untested by Phase 2)
