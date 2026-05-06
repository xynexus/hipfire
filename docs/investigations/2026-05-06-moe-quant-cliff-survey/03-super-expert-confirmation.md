# 03. Phase 2: Super-Expert Hypothesis Confirmation (PRE-REGISTERED)

**Status:** PRE-REGISTERED. Criterion fixed BEFORE measurement. Drafted
2026-05-06 23:30 UTC, before any Phase 2 perplexity data was collected.

**Amendment 2026-05-06 23:50 UTC (BEFORE Phase 2 PPL collection):** the
D3 forward-pass results landed and showed activation-side SE candidates
at layers 38-39 with completely different expert IDs from the D2
weight-side set. To cleanly resolve which signal drives the actual
quality cliff, Phase 2 now tests **three independent candidate sets**
(D2-only / D3-only / D2∪D3 union) against the same All-MQ4 baseline
and All-Q8 ceiling. Criterion thresholds (gap closure ≥ 80% / size
overhead ≤ 2%) are unchanged.

**Branch:** `survey/moe-quant-cliff-2026-05-06`

**Hypothesis source:** arXiv 2507.23279 ("Super-Expert" identification in MoE LLMs).

**Phase 1 evidence backing this design:**
- `02-survey-results.md` — weight-side D2: 17 SE-candidate expert IDs concordant
  between Qwen3.5-A3B and Qwen3.6-A3B at layer 0 `mlp.experts.X.down_proj`,
  ratio_p99 between **5e7 and 6.9e7** vs dense controls' max ~40 (six orders
  of magnitude separation).
- D3 per-expert activation absmax (in flight at draft time) — corroboration.

## Hypothesis (formal)

In Qwen3.5/3.6 MoE models quantized at MQ4G256+FWHT, a small subset of
routed experts at layer 0 `down_proj` carry weight-row tails so heavy
that 4-bit per-256-element absmax bucketing collapses the row's signal
into floating-point noise. Pinning **only those experts** at higher
precision (Q8) closes most of the perplexity gap to a fully-Q8 model,
at negligible model-size cost.

## Pre-Registered Criterion

The hypothesis is **CONFIRMED for this model family** if and only if BOTH
hold simultaneously on BOTH 3.5-A3B and 3.6-A3B:

1. **Perplexity gap closure ≥ 80%:**
   `(PPL_All-MQ4 - PPL_Outlier-Q8) / (PPL_All-MQ4 - PPL_All-Q8) ≥ 0.80`

2. **Size cost ≤ 2%:**
   `(SIZE_Outlier-Q8 - SIZE_All-MQ4) / SIZE_All-MQ4 ≤ 0.02`

The hypothesis is **REFUTED for this model family** if EITHER:

- Gap closure is below 30% on either model (the SE candidate set is wrong
  or the cliff is mostly an activation-side issue uncaptured by weight
  ratios), OR
- Outlier-Q8 perplexity is **higher** than All-MQ4 (the SE candidates are
  miscalibrated and pinning them harms the model).

The intermediate range (30%-79% gap closure) is **PARTIAL CONFIRMATION**:
the hypothesis applies in spirit but the exact candidate set needs
refinement (Phase 2.5 — extend the SE set with D3 per-expert activation
ranking, retest).

### Rationale for thresholds

- **80% / 2%** is the same shape arXiv 2507.23279 uses for its
  Mixtral-8x7B / DeepSeek-V2 ablations (their Outlier-Q8 set was 0.6%
  of experts at ~80% gap closure).
- **30%** is a generous floor — a chance result on 17/256 randomly chosen
  experts on layer 0 alone would be far below this.
- **2% size** is roughly 17 / 256 / 64 layers expanded to 50% size cost
  per pinned expert ≈ 0.05% — the 2% threshold is slack for
  implementation overhead (per-expert metadata, alignment).

## Variants

Three variants per model (3.5-A3B, 3.6-A3B), three independent trials each.

### V1: All-MQ4 (baseline)

Production quantization at MQ4G256+FWHT. Same as `~/.hipfire/models/qwen3.5-35b-a3b.mq4`.
FWHT seeds 42 / 1042. No precision pins.

### V2: All-Q8 (ceiling)

Q8G256 on every weight tensor. No FWHT (Q8 has enough headroom that
FWHT rotation isn't required for correctness). ~2× model size. Used
ONLY as the perplexity ceiling reference.

### V3a: Outlier-Q8 (D2-derived)

MQ4G256+FWHT on every weight tensor EXCEPT:
- Layer 0 `mlp.experts.X.down_proj` for X ∈ {3, 8, 42, 49, 70, 115, 119,
  132, 164, 167, 190, 195, 203, 225, 237, 239, 253}: pinned at Q8G256, no
  FWHT.

17 experts × 1 layer × 1 projection = 17 weight tensors elevated.
Source: D2 weight-side ratio_p99 cliff at layer 0 down_proj
(`02-survey-results.md` original synthesis, before D3 update).

### V3b: Outlier-Q8 (D3-derived)

MQ4G256+FWHT on every weight tensor EXCEPT:
- `mlp.experts.X.down_proj` for the (layer, expert) pairs in:
  ```
  layer 38: {48, 103, 209}
  layer 39: {5, 21, 27, 37, 101, 108, 113, 149, 155, 170, 200,
             209, 229, 238, 251, 255}
  ```
  pinned at Q8G256, no FWHT.

19 weight tensors elevated. Source: D3 forward-pass per-expert absmax_max
output side, top-20 union concordant across 3.5-A3B and 3.6-A3B (19/20
concordance, see `02-survey-results.md` D3 update section).

### V3c: Outlier-Q8 (D2 ∪ D3 union)

Both V3a and V3b's pinning sets simultaneously: 17 + 19 = 36 weight
tensors elevated (no overlap; D2 is layer 0, D3 is layers 38-39).

## Methodology

### Quantization implementation

Python harness (no hipfire engine modification required for this
ablation):

1. Load model via `transformers.AutoModelForCausalLM.from_pretrained(..., dtype=bfloat16)`.
2. For each weight tensor, apply the appropriate round-trip from
   `scripts/quant-survey/quant_ops.py`:
   - V1: `quantize_then_dequantize_mq4g256_fwht_vectorized()` everywhere.
   - V2: `quantize_then_dequantize_q8g256()` everywhere (need to add this).
   - V3a/b/c: V1 everywhere except the pinned tensors which use V2.
3. Replace each weight in-place via `tensor.copy_(round_tripped)`.

The bf16 reference path (V_ref) is the unquantized model. PPL_All-Q8 (V2)
should land within ε of V_ref by construction.

### Test set

`benchmarks/calib/calib-1m.txt` was used for the D3 calibration corpus
sampling. Phase 2 perplexity uses a **disjoint slice** to avoid eval
contamination: take `calib-5m.txt` (md5 5dc7dc29..., 5M chars) and
sample chunks at offsets that DO NOT overlap any used in
`calibration_corpus.jsonl`. Build `phase2_eval_corpus.jsonl` with
~100 chunks × 1024 tokens ≈ 100K tokens evaluated per variant.

Cross-entropy is summed over each chunk's input_ids (model in eval mode,
no caching, batch_size=1, max_seq_len 1024). Perplexity is
`exp(total_ce / total_tokens)`.

### Trials and noise

Three independent trials per variant. Trial-to-trial variation should be
near-zero (deterministic given identical weights + tokenizer + corpus),
but trials catch numerical-noise issues (e.g., bf16 GEMM nondeterminism
from non-deterministic reduce ordering on multi-GPU).

If trial stddev exceeds 0.5% of mean PPL on any variant, the result is
discarded and the run is restarted with `torch.use_deterministic_algorithms(True)`
explicitly. The criterion math uses **mean** of 3 trials.

### Hardware

hiptrx, all 4 R9700s via `device_map="auto"` for the A3B inference.

### Wall time estimate

Per variant per trial: ~5 min (model load + 100 chunks of 1024 tokens
forward at ~300 tok/s). With the 3-set design:
5 variants (V1, V2, V3a, V3b, V3c) × 3 trials × 2 models = 30 runs × 5 min
= **~150 min total**. Sequenced on hiptrx (each A3B uses all 4 R9700s).

## Outputs

Per model:

- `phase2_results.jsonl` — one line per (variant, trial) with: ppl_total,
  ce_sum, n_tokens, weights_modified_count, model_size_bytes.
- `phase2_summary.json` — mean/stddev per variant, gap closure %,
  CONFIRMED / REFUTED / PARTIAL classification per the criterion.

Cross-model: extension of this doc with results table + verdict.

## Decision tree

For each model in {3.5-A3B, 3.6-A3B} and each variant in {V3a, V3b, V3c}:

```
gap_closure(variant) = (PPL_V1 - PPL_variant) / (PPL_V1 - PPL_V2)
size_overhead(variant) = (SIZE_variant - SIZE_V1) / SIZE_V1

if gap_closure >= 0.80 and size_overhead <= 0.02:
  -> CONFIRMED for this variant on this model
elif gap_closure >= 0.30:
  -> PARTIAL — set is partially correct
elif PPL_variant > PPL_V1:
  -> REFUTED — pinning that set HURT
else:
  -> REFUTED — gap closure too low
```

Family-level verdict for the SE hypothesis:

- If V3b (D3-derived) CONFIRMED on both models: arXiv 2507.23279 SE
  applies cleanly to this family; the activation-side absmax_max signal
  identifies the correct pinning set.
- If V3a (D2-derived) CONFIRMED on both models: weight-tail ratio at
  layer 0 is the dominant signal; the SE label per arXiv 2507.23279
  is misnamed for this family but the pinning works.
- If V3c (D2 ∪ D3) is the only CONFIRMED variant: both signals are
  necessary but neither is sufficient — additive cure required.
- If V3a is CONFIRMED but V3b is REFUTED (or vice versa): the cliffs
  are independent failure modes. Document each.
- All three REFUTED: neither survey-derived candidate set is correct;
  the cliff lives in a different proxy (router-precision, attention
  outliers, or per-tensor sensitivity ranking via Hessian-trace methods).

## Halt point

After this phase ships and `phase2_summary.json` is written,
**HALT for human review** before any Phase 3 work (engine integration of
per-expert precision pinning, downstream sidecar plans, etc.).

## Cross-references

- `00-recon-synthesis.md` — pre-survey synthesis (deprecated; superseded by 02)
- `01-survey-runner-design.md` — methodology spec
- `02-survey-results.md` — Phase 1B+1C synthesis (the SE candidate set lives there)
- `INVESTIGATION-LOG.md` — append-only timeline
- arXiv 2507.23279 — "Super Experts in MoE" reference paper
- GitHub issue #171 — fivetide's Q8-router fix (compatible/additive surface)
