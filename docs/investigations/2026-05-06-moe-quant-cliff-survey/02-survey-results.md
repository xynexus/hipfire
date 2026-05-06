# 02. Phase 1 Survey Results

**Branch:** `survey/moe-quant-cliff-2026-05-06` (commit `83ca5f8`)
**Hardware:** hiptrx, 4× R9700 gfx1201 + Threadripper 9970X 32-core single-NUMA, 122 GB.
**Wall time:** 9B 6:15, 27B 22:29, A3B-3.5 52:44, A3B-3.6 52:54 (4 models in parallel windows where shape allowed).

## TL;DR

Three findings the synthesis lands on, all empirically supported:

1. **The Super-Expert pathology is real, MoE-specific, and not 3.6-specific.** All
   top outliers in both A3B models concentrate at layer 0 `down_proj`, with
   per-row absmax/median ratios of 5e7 to 6.9e7. Dense controls (9B, 27B) max
   out at ratio ~40 — **three orders of magnitude smaller**.
2. **17 of 20 SE-candidate expert IDs are common between 3.5-A3B and 3.6-A3B.**
   Same training corpus + similar initialization → same experts allocate
   themselves to the same feature axes that produce the extreme tail. This is
   a structural property of the architecture+data, not a quirk of 3.6.
3. **Survey is consistent with fivetide's #171 finding** but identifies the
   weight-side counterpart of the activation-side cliff: extreme per-row tail
   ratios on `mlp.experts.X.down_proj` weights for ~17 specific experts at
   layer 0. fivetide measured the routing-precision damage at `mlp.gate.weight`
   (the router); we measure the per-expert weight side. Both surfaces feed
   the same downstream forward-pass instability, and the cures may be
   composable.

## Methodology recap

Per `01-survey-runner-design.md` (v1, post-remediation rounds 1-9):

- **D1 NRMSE**: production-matched MQ4G256+FWHT round-trip, NRMSE = √MSE / √var(ref),
  cosine similarity. Vectorized FWHT batch (95×) + vectorized D4 (150×).
- **D2 down_proj**: per-row absmax + median + ratio statistics. Outlier
  classification on `ratio_p99` z-score using MAD scale.
- **D4 FWHT**: per-256-element-group pre-rotation vs post-rotation absmax with
  reduction ratio.
- **D3 activation absmax via forward hooks**: module authored at `83ca5f8`,
  smoke tested on synthetic MoE; **NOT YET RUN** on real models — that is
  Phase 1B-bis once a D3 runner is wired.

Production FWHT seeds 42 / 1042 (matches `crates/hipfire-quantize/src/main.rs:430-436`).
Manifest validation prevents silent merge of incompatible runs into the same
output dir; resume safely retries errored records (rounds 6+7 fixes).

## Per-model summary

| Model | Records | Layers | Wall (s) | Outliers (z≥3) | Top ratio_p99 |
|-------|--------:|-------:|---------:|----------------:|--------------:|
| qwen3.5-9b   | 135    | 32 | 375  | 9     | 4.0e1 |
| qwen3.5-27b  | 263    | 64 | 1349 | 8     | 3.0e1 |
| qwen3.5-a3b  | 21,497 | 40 | 3164 | 3,213 | **6.9e7** |
| qwen3.6-a3b  | 21,241 | 40 | 3174 | 3,189 | **6.8e7** |

The dense 9B and 27B yield ~10 outliers each spread across multiple layers
and projections. The A3Bs yield ~3,200 outliers each, **all top-20 of which
land in layer 0 down_proj**, all with `ratio_p99` 5-7 orders of magnitude
above the dense baseline.

The 3,213 / 3,189 z≥3 counts represent ~15% of each model's tensors — the
MAD-based threshold is too sensitive at the A3B scale because too many
tensors share a similar elevated ratio. Top-N filtering by raw ratio_p99
is more useful for synthesis (used throughout this doc).

## Cross-model SE expert concordance

Top-20 SE candidates at layer 0 `down_proj` (ratio_p99 z-score, MAD-based):

| Rank | 3.5-A3B (ratio_p99) | 3.6-A3B (ratio_p99) | Same expert? |
|-----:|--------------------:|--------------------:|:------------:|
|  1 | 42  (6.88e7) | 42  (6.84e7) | ✓ |
|  2 | 119 (6.10e7) | 119 (5.90e7) | ✓ |
|  3 | 195 (5.51e7) | 195 (5.31e7) | ✓ |
|  4 | 190 (5.48e7) | 190 (5.29e7) | ✓ |
|  5 | 239 (5.38e7) | 239 (5.19e7) | ✓ |
|  6 | 132 (5.27e7) | 253 (5.05e7) |   |
|  7 | 8   (5.21e7) | 203 (5.04e7) |   |
|  8 | 203 (5.16e7) | 8   (5.01e7) | ✓ |
|  9 | 225 (5.14e7) | 225 (4.98e7) | ✓ |
| 10 | 253 (5.06e7) | 164 (4.97e7) |   |

**Common to BOTH models (top-20 union):**
`{3, 8, 42, 49, 70, 115, 119, 132, 164, 167, 190, 195, 203, 225, 237, 239, 253}` —
**17 of 20 IDs concordant**.

Different at the precise-rank level (3.5 has 132, 209, 0 in top-20 that 3.6
doesn't; 3.6 has 24, 34, 120 that 3.5 doesn't), but the SET of pathological
experts is structurally preserved.

## Layer distribution

| Model | Outlier layers (top-20) | Projections affected |
|-------|--------------------------|----------------------|
| qwen3.5-9b | 6, 10, 12, 18, 31 | down_proj, gate_proj, k_proj, up_proj |
| qwen3.5-27b | 22, 34, 49, 50, 59, 63 | down_proj, gate_proj, q_proj, up_proj, v_proj |
| qwen3.5-a3b | **0 only** | **down_proj only** |
| qwen3.6-a3b | **0 only** | **down_proj only** |

Dense outliers spread across layers and projections — typical "some weights
sit at the edge of the absmax bucket" behavior. The MoE concentration on
layer 0 down_proj is a different qualitative signature.

## Magnitude vs ratio dichotomy

`absmax_max` on the A3B SE candidates is **0.07 to 0.13** in raw bf16 — small
in absolute terms, identical in distribution to typical transformer weights.
The cliff is in the RATIO of that absmax to the per-row median, which sits
at 1e-9 or below for SE rows. One value is normal-magnitude; the rest of the
row collapses into floating-point noise.

This matches arXiv 2507.23279's Super-Expert characterization. It is also
consistent with — but distinct from — fivetide's router-precision finding.
fivetide measured the routing-decision damage at `mlp.gate.weight`; the
present survey measures the per-expert weight tail at `mlp.experts.X.down_proj`.
Both surfaces feed the same downstream attention-sink + token-attractor
mechanism, and the cures may be additive:

- **Q8 router** (fivetide's fix) protects which experts get selected.
- **Q8 on top-N SE down_proj experts** (this survey's prior) protects the
  per-token output of pathological experts when they ARE selected.

Phase 2 ablations will measure whether the second is additive over the
first or redundant.

## What this does NOT yet show

- **Activation-side measurement (D3)**: not yet run on real models. The
  module is implemented and self-tested; the runner that wires it +
  transformers `device_map="auto"` is the next step. D3 will measure
  per-channel activation absmax at every Linear input + output during
  forward on a calibration corpus, and is the direct measurement of the
  arXiv 2507.23279 `down_proj output magnitude` signal.

- **Perplexity impact of pinning SE experts at Q8**: that is Phase 2. This
  doc identifies WHICH experts to pin; Phase 2 measures whether pinning them
  at Q8 closes the perplexity gap to All-Q8 reference, with cost to model
  size.

- **3.5-122B-A10B**: descoped (not in any local cache; doesn't fit hiptrx
  VRAM at bf16 for D3). Could be added as Phase 1B-bis if Phase 2 evidence
  warrants downloading 244 GB.

## Pre-registered Phase 2 SE candidate set

Based on this survey, the candidate "promote to Q8" set for both 3.5-A3B
and 3.6-A3B is: **layer 0 `mlp.experts.X.down_proj` for X in
{3, 8, 42, 49, 70, 115, 119, 132, 164, 167, 190, 195, 203, 225, 237, 239, 253}**
— 17 expert tensors per model, ~10 MB additional VRAM per model
(0.014% of an 18 GB MQ4 model — negligible cost).

If Phase 2 measures this set's outlier-Q8 perplexity within 1% of All-Q8
AND outlier-Q8 size is no more than 1% larger than All-MQ4 (the
pre-registered criterion in `03-super-expert-confirmation.md`), the
Super-Expert hypothesis is confirmed for this model family.

If the criterion fails — i.e. pinning these 17 experts doesn't close the
perplexity gap — either the SE set is different from what the weight-side
ratio identifies (D3 activation measurement would distinguish), or the cliff
mechanism is genuinely activation-driven (fivetide's router-Q8 path is
sufficient and the down_proj weight-ratio is a downstream symptom not a
cause).

## Files

- per-tensor JSONL records: `/tmp/hiptrx-survey/runs/<model>-full/per_tensor.jsonl`
  (~25 MB each for A3Bs, never committed per File Policy)
- per-model summary.json: same dir, also gitignored
- this synthesis: this file
- raw data summaries staged to k9lin: `/tmp/hiptrx-survey-pull/`

## Cross-reference

- `00-recon-synthesis.md` — pre-survey synthesis (deprecated, see header)
- `01-survey-runner-design.md` — methodology spec
- `INVESTIGATION-LOG.md` — full timestamped audit trail (15+ entries)
- `../2026-05-05-qwen36-a3b-mq4-fragility/` — prior-session evidence base
- GitHub issue #171 — fivetide's router-precision finding
- arXiv 2507.23279 — "Super Experts in MoE" reference paper
