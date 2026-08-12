# Gemma 4 native calibration and CASK

Status: native calibration/CASK producer and heterogeneous F32 runtime implemented; full admission open

Owner: Gemma 4 / calibration / CASK

Canonical contract: `docs/plans/2026-07-15-gemma4-support.md`

Related: [bundled model induction program](2026-07-22-bundled-model-induction-program.md)

## Relationship to the canonical plan

This document fills two implementation gaps; it does not replace the canonical
Gemma 4 plan or reset its Phase 5 evidence. Gemma 4 arch 24, registry-driven
quantization, boxed serving, prompt/tool behavior, and existing OQ8-family
rejections remain authoritative. A newly calibrated or CASK-enabled candidate
must pass the frozen gates; previous failures are not erased by a new producer.

Initial target snapshots:

- `Gemma-4-31B-it` (dense); and
- `Gemma-4-26B-A4B-it` (MoE/PLE family).

## Current gap

Gemma 4 has config, weight, forward, serving, quantizer, and tiny-eval support.
The former missing arch-24 calibration registration and Qwen-specific CASK
producer have been replaced. The remaining gap is full-scale evidence. Gemma 4 can vary attention kind,
local/global head dimension, value projection, KV producer, PLE, and MoE plan by
layer, so copying the Qwen adapter or forcing a uniform TRIA shape is incorrect.

## Implementation evidence: 2026-07-22

Arch 24 now registers `gemma4-stream-v1` through the family-owned
`calibration_stream` module. The adapter uses production Gemma 4 layer math,
preserves the BF16 embedding scale/round boundary, applies final-logit
soft-capping, emits alias-aware tensor plans, captures attention/dense/router/
selected-expert inputs, and splits each stacked expert payload after one source
read. Variants with PLE or shared KV still fail explicitly; neither requested
snapshot uses those features.

Dry plans pass against the immutable local snapshots:

- 31B dense: arch 24, 60 layers, 833 logical tensors, 61,394,690,680 unique
  source bytes; and
- 26B-A4B: arch 24, 30 layers, 568 logical tensors, 50,465,776,700 unique
  source bytes.

Production-path gfx1103 checkpoint smokes committed layer 0 for both. The dense
run consumed 15 logical tensors with no duplicate reads. The MoE run consumed
20, read the stacked expert tensors once, routed 2 tokens into 16 slots, and
reconciled 16 router hits with 16 gate/down observations (15 admitted per role
after the one-row quota). Both persisted resumable boundary and full-ledger
state. Focused arch tests pass 10/10, including logical-to-physical CASK layer
mapping.

The dense 31B adapter has also completed an uninterrupted 60-layer production
pass over the full immutable snapshot. The final artifact contains 410
Hessians and 410 imatrices, consumed all 833 logical reads (832 canonical)
without missing or duplicate entries, read 61,394,690,680 source bytes, and
reported maximum Hessian/imatrix consistency error zero. Structural audit is
valid with no errors or warnings (`fnv64:05df11caa5127e0d`). This was a bounded
one-sequence/two-token, no-KLD mechanism run, not a quant admission corpus.

That run also exposed two recovery defects which are now explicit contracts.
The part combiner uses index-only `pread` plus page-cache release rather than a
whole-file mmap; the mapped implementation had slowed to a multi-day projection
after all 60 durable parts were already committed. `--resume
--finalize-completed` now validates the original run/engine identities and all
parts before publishing exactly that completed spool, without re-executing the
model. Separately, `--cask-only` is a fresh, RAM-boundary, no-KLD mode that keeps
the full architecture capture roster for identity validation while skipping
Hessian/imatrix accumulation. It exists to regenerate CASK without writing
another model-sized calibration artifact; it cannot resume, pause, emit
residual probes, or masquerade as calibration admission evidence.

The first full CASK-only attempt exposed another layer-stream lifecycle bug:
finished weights and scratch were returned to the GPU free-list but the offline
engine retained every differently sized bucket until process exit, producing a
real `hipMalloc` OOM at layer 38. The engine now invalidates weight-pointer
caches and drains pooled allocations after embedding, every completed layer,
and finalization. The unchanged retry passed the former failure point with
roughly 60 GiB host memory available and completed all 60 layers. Its 3,465,216-
byte CASK is arch 24, `hipfire.triattn.v2`, has exactly physical layers 0..59,
two samples per layer, half-split RoPE, full and sliding policies, and artifact
fingerprint `fnv64:5fff1a8dc3246d07`. CASK-only failures also clean their
non-resumable scratch spools automatically.

The full 26B-A4B snapshot now has the same bounded mechanism evidence. Its
30-layer run emitted 235 Hessians and 1,483 imatrices, consumed all 568 logical
reads (567 canonical) over 50,465,776,700 bytes with no missing/duplicate
entries, and reported consistency error zero. The 2,294,994,816-byte artifact
passes structural audit without errors or warnings
(`fnv64:ea47e239e14f7322`; run `fnv64:695168e2a0a03eea`). The intentionally
one-row expert quota left 6,848/7,680 expert capture points deficient and
preserved 3,424 experts at high precision, so this remains mechanism evidence,
not a calibrated product candidate. Its 872,448-byte CASK is arch 24, has all
30 physical layers, two samples per layer, half-split RoPE, full/sliding
policies, and fingerprint `fnv64:e63fbb4c6c896798`.

The common `calibrate --cask-output` path now writes canonical
`hipfire.triattn.v2` HFQM directly from the family adapter. Gemma records its HF
`rotate_half` convention as half-split (not adjacent/interleaved), and the
accumulator pairs `(f, f + head_dim/2)` accordingly. Serving retains the
heterogeneous package instead of coercing it to uniform TRIA. The registered
Gemma backend validates each layer against `LayeredKvPlan`, keeps sliding layers
in bounded rings, and scores/compacts each owned full-context F32 cache using
that layer's own Q/KV geometry, rotary dimension, theta, convention, and center
bank. Shared-KV and packed-KVarN combinations fail explicitly until compatible
aggregation/readers exist.

Live gfx1103 evidence covers two heterogeneous eviction cycles across a sliding
layer plus differently shaped half-split and interleaved full layers, including
post-compaction writes. Packed Q8 half-split scoring also matches the CPU oracle
with Pearson 1.000000 and maximum relative error `5.78e-7`; all F32/Q8/asym2/
asym3/asym4 scoring kernels compile for gfx1030, gfx1100, gfx1103, gfx1151, and
gfx1201. Product-scale calibration/center statistics, long-context recall, and
frozen quant admission remain open, so this scope is not complete.

## Scope A: arch-24 streamed calibration adapter

Add `crates/hipfire-arch-gemma4/src/calibration_stream.rs` and register arch 24.
Reuse the family-neutral calibration contracts and the production Gemma 4
forward primitives.

The adapter owns:

- parsing all supported Gemma 4 config variants;
- a deterministic alias-aware tensor plan;
- exact source-precision handling;
- resource estimates for dense and MoE/PLE layers;
- embedding/finalizer loads and logit soft-capping;
- per-layer loads respecting `AttentionKind`, `ValueProjection`, `KvProducer`,
  and `FfnPlan`;
- capture descriptors for attention, dense FFN, router, shared/routed experts,
  and any PLE projections; and
- routed-expert capture quotas and high-precision fallback records.

Shared K/V must have one logical owner in the read ledger even when multiple
consumer layers reference it. `attention_k_eq_v` and pre-norm/shared-value
semantics must not cause duplicate reads or missing calibration captures.

Start with a dry tensor plan for both 31B and 26B-A4B. A dense-only adapter is
not considered complete for the requested model set.

## Scope B: family-neutral CASK calibration seam

Move production CASK generation out of the Qwen-specific runtime example into a
registered offline Rust workflow. The common producer owns corpus sampling,
GPU/CPU accumulation, validation, serialization, fingerprints, and reporting.
The family adapter owns model loading, eligible layer descriptions, pre-RoPE Q
taps, and state reset/prefill.

Define per-layer center geometry rather than one global
`n_heads/head_dim/rope_theta` tuple. Each layer record binds:

- physical layer index and attention kind;
- Q heads, KV heads, head dimension, and rotary dimension;
- RoPE basis/theta/convention;
- sliding/full context policy;
- KV producer identity; and
- center payload range and sample count.

Legacy TRIA v1 remains readable for uniform Qwen-style models. New heterogeneous
Gemma 4 output should be a canonical TriAttention HFQM component, while compose
must also accept and losslessly carry legacy raw TRIA.

Add the pre-RoPE tap at the architecture-owned point before Gemma 4 applies its
layer-specific RoPE. GPU accumulation must validate head dimension support;
unsupported geometry uses the CPU oracle or fails explicitly, never truncates.

## Scope C: induction integration

Once both adapters pass their gates, extend `scripts/induct_model.py` without a
Gemma-specific command fork. Tool discovery already scans registered calibration
adapter crates and should discover Gemma 4 through the common mechanism.

Requested primary candidates:

```text
Gemma-4-26B-A4B-it.triattn.oq4.25++.hfq
Gemma-4-31B-it.triattn.oq4.25++.hfq
```

When a compatible DFLASH component is supplied, packaging may add `.dflash.`.
That only proves carriage and parsing. Gemma 4 target-side DFLASH verification
must be implemented and gated before the bundle may advertise executable
DFLASH support.

## Calibration gates

1. No-GPU tensor-plan fixtures for dense 31B and MoE/PLE 26B-A4B.
2. Every logical source tensor consumed once; aliases and shared K/V have one
   owner; no missing or duplicate ledger entries.
3. One-layer and mixed-layer streamed-versus-resident activation comparisons.
4. Segmented resume at boundaries around attention-kind, KV-sharing, and
   dense/MoE transitions.
5. Full artifact structural audit with exact source, corpus, token, geometry,
   adapter, engine, and KLDREF fingerprints.
6. Resident-oracle calibration comparison on a model that fits the validation
   host, then full 31B/26B evidence.
7. Any `oq4.25++` candidate receives finite-logit and frozen Phase 5 KLD/PPL
   verdicts; thresholds are not revised after results.

## CASK gates

1. Serialization round trip for heterogeneous per-layer geometry.
2. CPU versus GPU accumulator parity per attention geometry.
3. Center count/finite/statistical validation for every eligible layer.
4. Same-model loose versus embedded center parity.
5. Long-context retrieval/recall against `asym3` and uncompressed baselines.
6. Combined CASK + target quant evaluation.
7. Atlas AR rows; DFLASH rows only after target-side DFLASH execution exists.

## Verification commands

Workflow-only changes run `./tests/no-gpu-ci.sh`. Runtime/calibration/quant
changes run `./tests/tiny-affected-gate.sh --require-coverage`. GPU examples and
evals use the shared `hipfire lock` unless the invoked gate already owns it.

## Non-goals

- Do not reopen or relabel rejected OQ8-family candidates.
- Do not add a Gemma-specific branch to central induction orchestration.
- Do not make `/srv/huggingface` a runtime dependency.
- Do not infer executable DFLASH support from a successfully composed component.

## Definition of done

Both target snapshots produce audited native calibration artifacts and valid
per-layer CASK data through registered family adapters. Their `oq4.25++`
candidates receive explicit frozen-gate verdicts, and any promoted bundle loads
embedded CASK through the common runtime component path.
