# Qwen3.5-0.8B broad-corpus latent-KV Phase 0

Status: original full-model result invalidated by
`../latent-kv-evaluator-invalidation-20260711.json`. No loader, runtime,
allocator, or kernel implementation follows from this experiment. Component
evidence remains valid diagnostic evidence; corrected full-model reruns on the
consumed split are explicitly post-hoc and admission-ineligible.

## Corpus

Calibration used 64 documents from the revision-pinned FineWeb `sample-10BT`
viewer slice. Validation used 16 articles from the revision-pinned
WikiText-103 raw test split. Raw downloads, normalized JSONL, licenses, source
revisions, and SHA-256 fingerprints are recorded in
`../../corpora/latent-kv-20260711/manifest.json`.

The sealed capture contract used 16 calibration samples per length/position
cell (64 windows and 384 layer records total) and two validation samples per
cell (eight windows and 48 layer records total). Validation lengths and
position offsets remain at least four times the calibration regime.

## Frozen gate

- maximum static-vs-same-cache-oracle KLD delta: 0.05
- maximum static-vs-same-cache-oracle PPL ratio: 1.05
- gated order: latent attention, `R_v`, sigmoid gate, existing `W_o`

## Held-out result

The following historical full-model numbers are invalid and must not be used:

- KLD delta: 0.2799866805096883 (failed)
- PPL ratio: 6.7415601323653584 (failed)
- baseline, static, and oracle logits were all finite

No comparison or admission conclusion may be drawn from those numbers.

The predeclared component rank curve also remained far from parity:

| Rank | Static-vs-oracle attention KLD delta | Static gated-output relative error |
| ---: | ---: | ---: |
| 32 | 0.6109231946816327 | 0.8002832091335869 |
| 64 | 0.34044334520603564 | 0.6405281920757729 |
| 96 | 0.18481440894045506 | 0.5024678683148919 |

Full-model rank 64/96 evaluation was not run: the sealed evaluator correctly
fails closed because the Phase-0 admission contract is fixed to rank 32.

All 12 ReCalKV-style value-refinement candidates were rejected by the frozen
stacked-`W_o` output-error rule. The exact evaluator sources hashed by
`plan.json` are retained under `evaluator-snapshot/`.

## Decision

Do not admit the shared static basis or implement downstream runtime phases from
this experiment. A successor must use the corrected evaluator and a newly
sealed untouched validation split. The post-hoc ceiling below further indicates
that reopening the same single-basis hypothesis is not technically justified.

## Post-hoc feasibility ceiling

After rejection, Astrea fitted one global rank-32 KQ-SVD basis directly on all
already-consumed validation caches. This deliberately contaminated diagnostic
is not admission evidence; it is an optimistic ceiling for the current
single-shared-basis family.

With explicit causality restored, an in-sample basis fitted on the validation
caches still fails both numeric limits. The corrected artifact at
`feasibility/feasibility.json` is explicitly marked
`heldout_contaminated_by_fit=true` and `admission_eligible=false`:

- one global basis: `0.2574588821698663` KLD delta, `1.3444232507327558` PPL ratio
- two position experts selected at page seal: `0.19276862962815072` KLD delta,
  `1.2448213361494171` PPL ratio
- four length-by-position experts: `0.10863553410571519` KLD delta,
  `1.1105987097435825` PPL ratio; this is not stable for already-sealed pages
- eight per-stratum experts: `0.0` KLD delta, `1.0` PPL ratio by construction;
  this is the per-cache oracle control, not a deployable selector
- component one-basis attention KLD delta: `0.25657979368019074`
- frozen limits: `0.05` KLD delta and `1.05` PPL ratio

No tested metadata selector with at most four experts meets the gate. The only
passing control assigns a separately fitted basis to every consumed cache,
which collapses back to the explicitly non-production per-cache oracle.

The subsequent paper audit also found and fixed a ReCal-style implementation
defect: the alternating right-factor solve omitted the captured activation
covariance and the returned down/up factors were swapped. The feasibility
calculation disables ReCal refinement, so this correction does not alter the
ceiling above. Any new experiment must use the corrected evaluator fingerprint.

The paper audit then found a separate causal-evaluator defect: the replacement
attention path relied on an explicit Transformers mask and failed to synthesize
the causal mask when the real model delegated causality to its backend. This
allowed future-token leakage and invalidates both sealed full-model evaluation
artifacts. Capture, calibration, and component reference evidence are
unaffected because the component evaluator always applies an explicit causal
mask. Admission now requires a new untouched validation split.
