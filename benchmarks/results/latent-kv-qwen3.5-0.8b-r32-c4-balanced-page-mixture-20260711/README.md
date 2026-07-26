# Qwen3.5-0.8B balanced page-local latent-KV Phase 0

Status: **rejected**. Phase 1 remains unstarted.

## Frozen contract

- basis family: eight-expert `page_local_mixture_v1`
- selector: standardized K/V mean-absolute-value and RMS moments
- fitting: deterministic balanced farthest-point k-means, eight calibration pages
  per expert
- selection: nearest packaged expert at page seal
- source accounting: 786,432 bytes of BF16 K/V per maximum 64-token open page;
  zero full-K/V bytes retained after seal
- corpus: revision-pinned C4 `realnewslike`, with disjoint train calibration and
  previously untouched validation offsets
- rank: 32
- frozen candidate-vs-same-cache-oracle limits: `0.05` KLD delta and `1.05`
  PPL ratio

The exact raw and normalized corpus hashes are in
`../../corpora/latent-kv-c4-balanced-page-mixture-20260711/manifest.json`.

## Held-out result

The full-model evaluator produced finite logits but rejected the candidate:

- KLD delta: `0.31225071167719615`
- PPL ratio: `1.4640812293141279`
- component attention-KLD delta: `0.6857957214556386`

The selector assigned the eight held-out pages to experts 3 and 5. Matching the
calibration and validation domain and balancing expert membership improved the
prior cross-domain full-model result (`0.3919`, `1.6734`) but did not approach
the frozen limits.

## Post-hoc ceiling

The consumed validation capture was fitted only after rejection and is explicitly
admission-ineligible:

- one validation-fit global basis: `0.18370473145931493` KLD delta,
  `1.2243690897838049` PPL ratio
- two validation-fit position experts: `0.13240185005502553`,
  `1.149408647786211`
- four validation-fit length-by-position experts: `0.07486126898100394`,
  `1.0625047187039358`
- eight per-cache controls: `0.0`, `1.0` by construction

Even the four-expert contaminated ceiling misses both limits. The tested
calibration-trained eight-expert page-local family is rejected at rank 32; no
runtime, allocator, packing, or kernel phase may be inferred from this result.
