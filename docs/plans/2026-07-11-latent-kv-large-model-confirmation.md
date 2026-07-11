# Latent-KV static-basis rejection: large-model confirmation

Status: **active** (supersedes the rejection verdict of
`docs/plans/2026-07-10-hierarchical-calibrated-latent-kv.md`)

Date: 2026-07-11

Reference branch: `chaingun`

## Why this plan exists

The parent plan
(`docs/plans/2026-07-10-hierarchical-calibrated-latent-kv.md`) records three
successive rejections of the shared static / page-local-mixture rank-32 basis
family, and concludes that Phase 1 is blocked and the family is "not admitted."

**Every one of those runs used only `Qwen3.5-0.8B`.** All four
`benchmarks/results/latent-kv-*/plan.json` artifacts pin
`/srv/huggingface/models--Qwen--Qwen3.5-0.8B` and share one
`combined_sha256`. No 4B/9B confirmation run exists, and neither the parent plan
nor any result artifact records one.

That violates the standing bring-up protocol: **0.8B is a prototype rung only.**
Qwen3.5-0.8B has size-driven quirks — sharp, low-redundancy attention and little
head-to-head KV redundancy — that bias specifically *against* a shared low-rank
basis. Any result that *fails* on 0.8B must be reconfirmed on `Qwen3.5-4B` and
`Qwen3.5-9B` before it is treated as a real rejection. Larger models generally
carry more low-rank structure in their KV subspaces, so both failure modes seen
at 0.8B may shrink or vanish with scale:

- the **representational gap** — the contaminated in-sample ceiling sat only
  marginally over the bar (four-expert ceiling `0.0749` KLD / `1.0625` PPL vs the
  frozen `0.05` / `1.05`); and
- the **generalization gap** — held-out basis transfer was the actual wall
  (`0.31`–`0.39` held-out KLD, 4–6x the in-sample ceiling).

This plan does **not** reinterpret, promote, or re-run the rejected 0.8B
artifacts. It runs a fresh, correctly-sealed confirmation on the larger models
the protocol required, with the corrected evaluator and untouched validation
splits, under the same frozen thresholds.

## What is inherited unchanged

The full mathematical contract, storage geometry, page/scheduler contract,
runtime kernel shape, and Phase 1–4 definitions are inherited verbatim from the
parent plan and are not restated here. This plan changes only the **model
coverage** and the **status of the rejection verdict**. It changes nothing about
the frozen admission gate.

Frozen, unchanged:

- Admission gate: `max_static_vs_oracle_kld_delta = 0.05` and
  `max_static_vs_oracle_ppl_ratio = 1.05`, measured against the same-rank
  per-cache SVD oracle. **Not loosened.** Loosening the thresholds or silently
  raising the Phase-0 rank remains an unacceptable resolution.
- Base rank: 32 (the mandatory first-milestone comparison).
- Corrected evaluator only. The causal-mask defect recorded in
  `benchmarks/results/latent-kv-evaluator-invalidation-20260711.json` must be
  fixed in the evaluator fingerprint of every run here; the
  `test_latent_qwen35_forward_identity_preserves_gate_and_wo` identity test
  must pass first.

## Confirmation experiment

Run the identical Phase-0 pipeline that rejected 0.8B, on the models it should
have been confirmed against.

### Models

1. `Qwen3.5-4B` — primary confirmation rung.
2. `Qwen3.5-9B` — second rung, to establish a trend rather than a single point.

Both are resident on the local mount
(`/srv/huggingface/models--Qwen--Qwen3.5-4B`,
`/srv/huggingface/models--Qwen--Qwen3.5-9B`); no download is required. 4B runs on
the local gfx1103 box; 9B runs here if it fits UMA or on `halo` (gfx1151, HIP
device 1) otherwise.

### Basis families

Confirm the two families the parent plan actually rejected, in order:

1. `shared_static_v0`, one global basis — the canonical scheduler-clean bet.
2. `page_local_mixture_v1`, eight experts — the smallest still-authorized
   research exception from the parent plan.

The mixture family is only run for a model if the single global basis has not
already cleared the gate for that model.

### Corpus

A **new, untouched** validation split per the parent contract — the 0.8B splits
and offsets may not be reused. Keep the in-domain C4 `realnewslike`
train-calibration / held-out-validation structure that the balanced 0.8B run
used (its only defensible design choice), but at fresh, disjoint offsets, and
seal a new manifest under
`benchmarks/corpora/latent-kv-large-confirm-4b-20260711/` (and the 9B analogue).
Calibration and validation offsets must not overlap each other or any offset
used by a 0.8B corpus manifest already in the tree.

### Length / position stratification

Preserve the parent plan's requirement to evaluate held-out sequence lengths and
relative offsets at least 4x beyond the calibration regime. Calibrate at the same
length/position strata used for 0.8B; evaluate the extrapolation lengths.

## Decision gate

For each model, the pipeline emits: component reference (static vs same-cache
oracle), full-model KLD/PPL (`latent-kv-model-eval`), and — only if the honest
held-out arm is rejected — a contaminated validation-fit ceiling
(`latent-kv-feasibility`).

- **Confirm the rejection.** If both 4B and 9B fail the frozen gate on the honest
  held-out arm *and* their contaminated ceilings also miss `0.05` / `1.05`, then
  the shared static / page-local-mixture rank-32 family is confirmed dead across
  scale. The parent plan's verdict stands, now with the model coverage the
  protocol required. Phase 1 remains unstarted for this family. The only
  remaining door is a materially different mathematical contract (see below).

- **Overturn the rejection.** If either 4B or 9B clears the frozen gate on the
  honest held-out arm, the 0.8B rejection was a small-model artifact. Reopen
  Phase 1 (BF16/F16 latent runtime oracle) for that model class, unchanged from
  the parent plan's Phase 1 definition.

- **Partial / trend.** If the contaminated ceiling passes but held-out does not,
  and the gap narrows monotonically 0.8B → 4B → 9B, the failure is a
  generalization/transfer problem that scales away rather than a representational
  wall. Record the trend and treat the retraining lever below as the next
  Phase-0 successor rather than declaring the family dead.

## Experiment results (2026-07-11)

Run on `Qwen3.5-4B` and `Qwen3.5-9B`, `shared_static_v0` single global basis,
rank 32, corrected evaluator, fresh sealed C4 `realnewslike` split
(`latent-kv-large-confirm-4b-20260711`, calibration offset 60000 / validation
offset 12000). Held-out full-model KLD/PPL vs baseline:

| Model | oracle KLD / PPL | static KLD / PPL | static-vs-oracle kld_delta / ppl_ratio | admitted |
| --- | --- | --- | --- | --- |
| 0.8B (parent) | 0.110 / 28.41 | 0.422 / 41.59 | 0.312 / 1.464 | no |
| 4B | 0.050 / 15.62 | 0.334 / 22.36 | 0.284 / 1.431 | no |
| 9B | 0.036 / 13.20 | 0.311 / 18.81 | 0.275 / 1.425 | no |

The single static basis is **rejected at every size** on the honest held-out
arm. The 0.8B rejection was therefore not a small-model artifact for
`shared_static_v0`. The per-cache oracle grows near-lossless with scale
(KLD 0.110 -> 0.050 -> 0.036), confirming larger models carry more low-rank KV
structure.

Contaminated in-sample ceiling (admission-ineligible; basis fit to the eval
caches), vs-oracle kld_delta / ppl_ratio against the frozen 0.05 / 1.05 limits:

| Selector (experts) | 0.8B | 4B | 9B |
| --- | --- | --- | --- |
| global (1) | fail | 0.145 / 1.184 | 0.115 / 1.143 |
| length x position (4) | 0.075 / 1.063 (fail) | **0.044 / 1.038 (pass)** | **0.031 / 1.025 (pass)** |
| per-stratum oracle (8) | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |

**This overturns the parent plan's core argument.** At 0.8B, even the
contaminated 4-expert ceiling missed both limits, which is why the parent plan
declared the family representationally dead and demanded a "materially different
mathematical contract." At 4B and 9B the contaminated 4-expert length x position
ceiling **clears both frozen limits**. The representational wall is gone at
scale; the only remaining obstacle for that selector is the generalization gap
(does a calibration-fit basis transfer to held-out?). The parent plan's disproof
of two- and four-expert metadata selectors is 0.8B-specific and does not carry
to 4B/9B.

### Justified next experiment (admission-eligible)

Before invoking retraining, run the cheaper metadata-selector experiment the
larger-model ceiling now re-opens:

- basis family `page_local_mixture_v1`, four experts, selector = length x
  position strata (deterministic, known at page seal);
- experts fit from **calibration captures only** (not the validation caches);
- evaluated on the same held-out split, same frozen 0.05 / 1.05 limits, on 4B and
  9B.

If the calibration-fit four-expert mixture clears held-out, `shared_static`-style
latent KV is viable at scale with no retraining, and Phase 1 reopens for the
mixture family. If it fails held-out despite the passing ceiling, the obstacle is
a pure generalization gap and the retraining lever below becomes the next step.

### Result: the metadata-selector experiment is rejected (2026-07-11)

Ran `page_local_mixture_v1` with the new admission-eligible `length_x_position_v1`
selector (four experts = calibration lengths x offsets, fit on calibration only,
routed to each held-out page's nearest calibrated regime at seal time). Same
frozen split and limits. Held-out static-vs-oracle:

| Model | length x position (calib-fit) | single static (shared_static_v0) | contaminated 4-expert ceiling |
| --- | --- | --- | --- |
| 4B | 0.304 / 1.456 | 0.284 / 1.431 | 0.044 / 1.038 |
| 9B | 0.300 / 1.463 | 0.275 / 1.425 | 0.031 / 1.025 |

Both **rejected**, and the calibration-fit mixture is marginally *worse* than a
single shared basis. Held-out routing is correct (offset-0 pages -> the offset-0
expert, offset-256 -> the offset-64 expert; both held-out lengths route to the
length-64 experts as nearest regime), so this is the generalization gap itself,
not a routing defect. Splitting calibration into narrow length regimes and then
extrapolating a length-64-fit expert to lengths 128/256 is strictly harder than
using one broadly-fit basis; length specialization hurts extrapolation.

**Conclusion.** The contaminated ceiling overstated reachable capacity: it fit
each expert to the validation stratum it was scored on. No calibration-fit
metadata selector can recover that capacity because the held-out length/position
regimes are disjoint from calibration by the 4x extrapolation contract. The
metadata-selector door is closed by admission-eligible evidence. The remaining
lever is per-model retraining (co-train the model into the rank-32 subspace) or a
position-equivariant (e.g. RoPE-compatible) basis; both are separate successor
plans with their own sealed splits and gates.

## If the metadata-selector experiment still fails at scale

If the static basis fails honestly on 4B and 9B, the remaining relaxation is
**per-model adaptation**, not more experts or a looser threshold (both disproven
/ prohibited by the parent plan). Co-train the model (LoRA or QAT) to live in the
rank-32 subspace instead of approximating a frozen model post-hoc. This attacks
the generalization gap at its root and preserves the shared-basis scheduler
contract, because training is offline and one-time per model package. It changes
the admission oracle from "within 5% of the frozen model's per-cache SVD" to
"downstream quality vs the original model," and requires its own new sealed split
and gate. That is a separate successor plan, opened only if this confirmation
fails.

## Provenance

- Supersedes the *verdict* of `2026-07-10-hierarchical-calibrated-latent-kv.md`;
  inherits its contract.
- Rejected 0.8B artifacts under `benchmarks/results/latent-kv-*-2026071{0,1}*`
  remain sealed and admission-ineligible; they are not inputs to this plan.
- Evaluator defect and corrected fingerprint:
  `benchmarks/results/latent-kv-evaluator-invalidation-20260711.json`.
