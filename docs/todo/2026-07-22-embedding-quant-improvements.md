# TODO: future improvements to embedding quantization

Status: open / research
Owner: unassigned
Related:
- `crates/hipfire-quantize/src/main.rs` — `--embed-precision`, `embed_precision_override`
- `docs/todo/2026-07-21-embed-per-row-lloyd-codebook.md` (REJECTED per-row Lloyd)
- commits e45c30965 (bf16/f16 gather kernels + format) and 92831c295 (source default)

## Background (what we know)

The embedding table is **disproportionately downstream-sensitive**: it seeds
the residual stream and rides it **unnormalized** across every layer (RMSNorm
only normalizes the per-layer *input* copy, not the residual). Measured on
Qwen3.5-35B-A3B, 8-bit embed (Q8 or a 256-level Lloyd codebook) costs ~0.06 KLD
vs bf16 — about **40% of the entire oq4 model's KLD budget (0.139)** for one
tensor of ~500 MB.

Current state (as of 2026-07-22):
- `--embed-precision {source,q8,bf16,f16}`, **default `source`** — keep the
  gather table at model source precision (bf16/f16). This un-quantizes the
  largest per-tensor KLD contributor by default.
- `q8` opts back down to the ~500 MB-smaller table.
- Gather kernels convert bf16/f16 → f32 in-kernel (portable RDNA2/3/4); no
  F32-widening on disk.

Two hard constraints for any future embed codec:
1. **Evaluate downstream (KLD), never by reconstruction MSE/per-row cosine.**
   Both Q8 and the Lloyd codebook reconstruct at per-row cos 0.99999 yet cost
   ~0.06 KLD — whole-vector residual fidelity is what matters, not per-row error.
2. Embed is a **gather** (row lookup), so the format must be gather-friendly.
   Trellis/QTIP-style codecs suit `lm_head` (a matmul) but not the embed gather.

## Directions worth investigating

1. **Quantify the real win.** Measure the full-model KLD delta of source-vs-q8
   embed on real OQ models (oq4/oq8) across a few families. The ~40% figure was
   isolated on a bf16 base; confirm it carries through end-to-end and size the
   quality/MB trade so we can recommend `source` vs `q8` per model.

2. **The ~6-bit sweet spot.** Between q8 (~1 B/elem, ~40% KLD tax) and bf16
   (2 B/elem, ~0 tax) there may be a gather-friendly group-scaled 6-bit format
   (oq6-style) that recovers most of bf16 quality at ~0.75× the bf16 size. Sweep
   downstream KLD vs bytes for {q8, oq6-gather, f16, bf16}.

3. **Outlier-row mixed precision.** Keep only high-energy embed rows (common
   tokens / high-norm rows) at bf16 and the tail at a lower width. The residual
   seed for a given prompt only touches the rows actually gathered, so protecting
   the heavy-hitters may capture most of the quality at a fraction of the bytes.

4. **Tied embeddings split.** When embed doubles as `lm_head`, the gather
   objective (residual seed fidelity) and the matmul objective (logit fidelity)
   differ. Investigate storing/serving the two roles separately (bf16 gather +
   a trellis/QTIP lm_head) instead of one shared table.

5. **Router-aware policy (MoE).** Re-baselining showed qwen3_5_moe KLD drift UP
   in-tolerance when embed went to source precision: a more-accurate embed shifts
   the residual, so the *quantized* router flips experts vs the unquantized-router
   reference. Investigate whether embed precision should be chosen jointly with
   router precision for MoE, or whether the router should be protected (cf.
   `HIPFIRE_OQ8_ROUTER`) whenever embed is high-precision.

## Explicitly rejected (do not re-open without new evidence)

- **Per-row / per-group Lloyd codebook for embed** — see
  `docs/todo/2026-07-21-embed-per-row-lloyd-codebook.md`. oq8l beat Q8 only
  marginally downstream (~27% on the p99 tail) — not worth a new codec + kernel
  for a marginal gain on a bad trade. Keeping the table at source precision is
  the better lever.
