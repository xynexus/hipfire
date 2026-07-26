# TODO: cold-merge sidecar (similarity grouping)

Status: idea. Owner: KV. Relates to the hier-KV plan
(`docs/plans/2026-07-12-hot-cold-hierarchical-kv-implementation.md`) and the
KV-compression adoption plan (`docs/plans/2026-07-13-kv-compression-adoption-plan.md`).

## Why

The only real quality cost of the two-tier hierarchical cache is the **cold
merge**, and the loss is **content, not phase** (ceiling analysis): the current
merge groups the non-core cold tail by **position-adjacency**, so it averages
tokens that differ in content. The CASK reframe from this session concluded the
importance *scorer* is a spent lever (`vnorm ≥ TriAttn`), but CASK's genuinely
useful contribution is the **similarity-based merge grouping** — merging
near-duplicate keys is ~lossless.

`HIPFIRE_KV_MERGE=similarity` already exists (greedy K-cosine clustering). What it
lacks is a *calibrated* grouping distance.

## What a merge sidecar would carry

An offline-calibrated grouping distance for the cold merge, so it groups by
content-similarity instead of position:

- **κ-weighted distance** `d_κ = Σ_f |κ(ω_f)|·‖k_{i,f}−k_{j,f}‖` using the
  future-relevance kernel from the TriAttention centers — reuse the TRIA sidecar
  we already load for `ImportanceMode::TriAttn`. This weights the near-duplicate
  detection by which key channels actually matter for future attention.
- Optionally a per-(layer,head) merge-safety mask: which channels tolerate
  averaging vs which must stay exact.

Sidecar generation is offline (calibration set), no hot-path training — same
machinery as the TRIA/hessian sidecars.

## Gate (now measurable)

Does κ-weighted similarity-merge beat position-merge (and plain cosine
similarity-merge) on **KLD in the retrieval regime**? This was un-measurable
before; it is now testable via the long-context eval suites
(`hipfire eval … --suite niah,needle_chain,… --kv-mode kvarn --kv-hierarchical`)
and the long-context KLD bridge (`--battery perplexity --corpus <fixture> --ctx N
--kldref <bf16>`). If κ-grouping doesn't beat position-adjacency on KLD, the
merge-sidecar thesis is closed and hierarchical stays a memory-only play.

## Sketch

1. Reuse `TriAttnCenters` load path; derive the κ grouping distance.
2. Extend `similarity_groups` in `kv_compact.rs` to accept the κ distance (behind
   `HIPFIRE_KV_MERGE=similarity_kappa` or a sidecar-present auto-select).
3. Measure vs position/cosine on the retrieval suites + long-ctx KLD.
