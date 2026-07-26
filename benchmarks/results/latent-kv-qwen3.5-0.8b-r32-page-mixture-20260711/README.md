# Qwen3.5-0.8B page-local latent-KV Phase 0

Status: **rejected**. This experiment is not admission evidence for Phase 1.

## Frozen contract

- basis family: eight-expert `page_local_mixture_v1`
- selector: calibration-fit standardized K/V mean-absolute-value and RMS moments
- selection time: page seal, with the selected basis ID stored per page
- calibration: revision-pinned FineWeb-Edu `sample-10BT`
- validation: revision-pinned C4 `realnewslike` validation
- rank: 32
- maximum candidate-vs-same-cache-oracle KLD delta: `0.05`
- maximum candidate-vs-same-cache-oracle PPL ratio: `1.05`

The exact corpus and raw-download hashes are in
`../../corpora/latent-kv-page-mixture-20260711/manifest.json`. The immutable plan,
evaluator fingerprints, calibration artifact, captures, and results are retained
under this directory.

## Held-out result

All logits were finite, but the page-local candidate failed both frozen limits:

- KLD delta: `0.39190371365759386`
- PPL ratio: `1.6734282322965865`
- component attention-KLD delta: `0.794678364364919`

Every held-out page selected expert 1. This was not merely a selector-routing
failure: a post-hoc ideal choice among the eight calibration experts still left a
mean component KLD delta of `0.6318382827740812`.

## Post-hoc ceiling

The consumed validation capture was then used only for an explicitly
admission-ineligible feasibility ceiling:

- one validation-fit global basis: `0.18758506406859266` KLD delta,
  `1.2831165674963918` PPL ratio
- two validation-fit position experts: `0.1282161993037968`,
  `1.1775975595597528`
- four validation-fit length-by-position experts: `0.068818063816514`,
  `1.074942971121751`
- eight per-cache controls: `0.0`, `1.0` by construction

The failure is a cross-domain expert-coverage failure, not evidence that a
calibration-trained page-local selector passes. Phase 1 remains unstarted. A
single in-domain successor with balanced expert fitting is technically justified;
it must use a new untouched validation slice and the same numeric limits.
