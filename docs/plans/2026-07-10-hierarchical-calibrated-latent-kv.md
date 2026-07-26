# Hierarchical calibrated latent KV

Status: **proposed**

Date: 2026-07-10

Reference branch: `chaingun`

## Experiment status

The rank-32 shared-static-basis prerequisite is currently unresolved for
Qwen3.5-0.8B. The first two sealed full-model artifacts are invalid: the custom
attention evaluator did not synthesize a causal mask when Transformers passed
`None` and delegated causality to its backend, allowing compressed arms to
attend to future tokens. Their numeric KLD/PPL verdicts must not be used. The
capture, calibration, and component artifacts remain valid; the broad-corpus
rank-96 component proxy retained a `0.1848` static-vs-oracle attention KLD gap.

Therefore Phases 1 and later remain unstarted. More samples alone are not the
next step. A successor proposal must change the static-basis hypothesis and
seal a new untouched validation split before it can reopen this plan.

A corrected post-hoc feasibility diagnostic then fitted the shared rank-32 basis
directly on every already-consumed validation cache. Although this is
deliberately contaminated and cannot serve as admission evidence, it is an
optimistic in-sample ceiling for the current basis family. It still failed:
full-model KLD delta was `0.2575` and PPL ratio was `1.344`, versus the frozen
`0.05` and `1.05` limits; a two-expert absolute-position selector improved
these only to `0.1928` and `1.245`. The component attention-KLD gap was
`0.2566` for one global basis.

A broader selector-capacity curve reached `0.1086` KLD and `1.111` PPL ratio
with four length-by-position experts, still outside both limits and not stable
for already-sealed pages. Eight per-stratum experts reached exact oracle parity
only because this assigns one separately fitted basis to every consumed cache;
it is the per-cache oracle under another name, not a deployable static selector.

This creates a contract blocker rather than a corpus-selection problem. The
single shared rank-32 KQ-SVD basis cannot satisfy the current gate on the tested
Qwen3.5 distribution even when fitted to the evaluation caches themselves.
Reopening Phase 0 now requires explicit authorization to change the canonical
basis family. The smallest still-plausible research exception is now an
eight-expert page-local mixture with a calibration-trained, seal-time-stable
selector and explicit source-retention/accounting contract; two- and four-expert
metadata selectors are disproven by the post-hoc ceiling. Any such exception
must then seal a completely new validation split.
Changing the frozen thresholds or silently raising the Phase-0 rank is not an
acceptable resolution.

The page-local exception was explicitly authorized on 2026-07-11. The successor
Phase-0 experiment therefore changes the canonical basis family to an eight-expert
`page_local_mixture_v1` package. Expert factors and a standardized nearest-centroid
selector are fit from calibration captures only. The selector uses compact K/V
mean-absolute-value and RMS moments accumulated while a BF16 open page exists,
chooses one expert at page seal, stores that basis ID with the sealed page, and
does not permit post-seal reselection after the BF16 source is discarded. Query
projection is once per distinct expert represented by the supplied block list.

The new untouched corpus is revision-pinned in
`benchmarks/corpora/latent-kv-page-mixture-20260711/manifest.json`: FineWeb-Edu
`sample-10BT` is calibration-only and C4 `realnewslike` validation is held out.
The original `0.05` KLD-delta and `1.05` PPL-ratio limits remain frozen. This is
a new basis-family experiment, not a retroactive pass for the rejected shared
static basis.

That first authorized exception run is now rejected. On the FineWeb-Edu to C4
held-out transfer, all pages selected one expert; full-model candidate-vs-oracle
KLD delta was `0.3919` and PPL ratio was `1.6734`. An ideal post-hoc choice among
the eight calibration experts still left a `0.6318` component attention-KLD gap,
so routing alone cannot explain the failure. A contaminated validation-fit ceiling
improved four length-by-position experts to `0.0688` and `1.0749`, still outside
both limits. Phase 1 remains blocked. The only justified next Phase-0 experiment
is an in-domain C4 train/validation successor with balanced calibration experts,
the same frozen limits, and a new untouched validation slice.

That bounded successor is also rejected. Balanced `8x8` calibration membership
and in-domain C4 validation improved the full-model result only to `0.3123` KLD
delta and `1.4641` PPL ratio. A post-rejection validation-fit ceiling reached
`0.0749` and `1.0625` with four length-by-position experts, still outside both
limits; only the per-cache control passed. The rank-32 page-local mixture family
tested here is therefore not admitted, and Phase 1 remains unstarted. Reopening
the basis-family search again requires a materially different mathematical
contract rather than another corpus or selector rebalance.

## Decision

Make calibrated low-rank KV the primary long-context cache representation for
models with head dimensions large enough for the rank reduction to amortize its
projection cost:

```text
calibrated Q/K/V projection contract
    -> latent KV pages
    -> KVarN-packed sealed pages
    -> fused low-rank attention
    -> optional token-aligned rank-extension sidecars
    -> optional CASK cold merge / residency migration
```

Paged allocation is part of this design, but PagedAttention is not the design.
The page table owns logical-to-physical placement and residency; each page also
carries its latent rank, codec, residual stripes, merge state, and generation.
The scheduler/orchestrator can eventually manage those transitions without
changing the model's calibrated projection contract.

This supersedes **simple per-cache cold SVD residuals as the lead design**. A
per-cache SVD remains the required quality oracle for the static-basis bet. It is
not the default runtime basis.

Related work already in this tree:

- `docs/plans/2026-06-22-hierarchical-kv-followups.md` documents the current
  default-off hot-F32 plus CASK/KVarN cold-segment implementation and its measured
  quality/performance limitations.
- `docs/plans/2026-06-23-kv-merge-design-levers.md` establishes that CASK merge
  noise is real information loss dominated by RoPE-phase blur, not a bias that
  fine-tuning reliably recovers.
- `Quantization/kv_explore/FINDINGS.md` records the original low-rank and
  CASK/KVarN attention-output fidelity exploration.
- `Quantization/kv_explore/ARCHITECTURE.md` records the deferred-compaction thesis
  and the current two-tier scaffold.

Local paper and reference checkouts live under `~/KV-Compression`:

- KQ-SVD: `2512.05916v1-KQ-SDV/`
- ReCalKV: `2505.24357v3-ReCalKV/`
- QSVD: `2510.16292v1-QSVD/`
- OjaKV: `2509.21623v2-OjaKV/`
- TriAttention and CASK: `2604.04921v1-TriAttention/` and
  `2604.10900v1-CASK/`

## Why this replaces the earlier compromise

The current deferred hierarchy proved several useful facts:

1. KVarN quantization is not the dominant quality loss at the measured operating
   points. Token merging is.
2. CASK merging is valuable for very aggressive cold compression, but it should
   not define the representation used by every token.
3. The existing `HierKvState` is a good parity oracle, but its F32 hot rings,
   CPU download/upload compaction, per-segment dequant/read/merge loop, fixed
   `head_dim=256`, and forced serial prefill/decode are not a production serving
   substrate.
4. A static calibrated basis has a cleaner scheduler contract than per-cache SVD
   or online Oja updates. The same model package and policy can be shared by every
   session, page, batch, and device.

That last point is a hypothesis, not established local evidence. The existing
low-rank measurements use per-cache SVD. Phase 0/1 must measure the penalty of
replacing that oracle with one static basis shared across sessions before the
scheduler-simplicity argument is accepted.

The primary quality/byte lever should therefore be feature-dimension reduction,
with KVarN providing multiplicative storage compression after the reduction.
CASK and residual storage remain available when age, pressure, or observed
underfit justifies them.

## Mathematical contract

For layer `l` and GQA group `g`, let `d` be the original head dimension. Calibrate
key and query factors `A_k[l,g]` and `B_q[l,g]`, with rank `r_k`, using a KQ-SVD
objective over the query heads that share the GQA KV head:

```text
K_lat = RoPE(X W_k) A_k
Q_lat = RoPE(X W_q) B_q
scores = Q_lat K_lat^T / sqrt(d)
```

The denominator remains `sqrt(d)`, not `sqrt(r_k)`. The approximation targets the
original score matrix. For GQA calibration, stack the calibration queries from all
query heads in the group when solving the shared key-side problem.

The projection is intentionally after RoPE. Arbitrary calibrated bases do not
commute with RoPE, so the first implementation must not fold `A_k` or `B_q` into
`W_k`/`W_q`. A future rotary-compatible constrained basis may revisit this, but it
is a separate optimization requiring its own quality evidence.

This placement fits the true post-RoPE score matrix but sacrifices exact
relative-position equivariance: the fixed low-rank product sits between the two
position-dependent rotations. A basis calibrated at one length may therefore fail
at larger relative offsets. Calibration should be position/length-stratified, and
Phase 0/1 must evaluate sequence lengths and relative offsets at least 4x beyond
the calibration regime before the basis is admitted.

For values, calibrate a low-rank factorization:

```text
V ~= V_lat R_v
V_lat = X L_v
Z_lat = softmax(scores) V_lat
output = Z_lat (R_v W_o)  # ungated layers
```

Use KQ-SVD's value-output objective and its closed-form solution as the starting
point, then apply ReCalKV-style offline calibration refinement and matrix fusion.
ReCalKV supplies the valuable calibration/fusion implementation pattern, but its
raw value-reconstruction objective does not include the output-projection
weighting. The output-sensitive objective is:

```text
min || V W_o - V_lat R_v W_o ||
```

rather than relying only on raw `V` or `W_v X` reconstruction. Under GQA, one KV
head serves `m` query heads with distinct output-projection slices. Fit one shared
value basis against the whole group:

```text
min_(L_v,R_v) sum_i || V W_o^(i) - V_lat R_v W_o^(i) ||^2
```

Equivalently, use the stacked output matrix
`[W_o^(1) | ... | W_o^(m)]`. Fold `R_v` into every corresponding query-head slice
offline. The production decode path must not reconstruct full-dimensional cached
values or a full-context V shadow.

### Gated attention-output exception

Qwen3.5 full-attention layers apply a token-dependent, per-dimension sigmoid
gate after attention and before `W_o`:

```text
output = ((Z_lat R_v) .* sigmoid(g_token)) W_o
```

For these gated layers, `R_v` cannot commute through the gate and no static
`R_v W_o` matrix is algebraically equivalent. The gated Qwen3.5 path therefore
reconstructs only the current query-head attention result `Z_lat R_v`, applies
the existing gate, and then executes the existing `W_o`. It must never
reconstruct cached values or materialize a full-context V shadow. This
rank-to-head projection is a measured runtime component, not a hidden fallback.

Ungated models and layers still use the statically fused `R_v W_o` path. Model
policy records which value-output contract applies, and loaders fail closed if
the packaged contract does not match the model's attention-output gating
semantics. Component parity for gated models compares explicit reconstruction
against the authorized current-output reconstruction path; component parity for
ungated models compares explicit reconstruction against static `R_v W_o`
fusion.

## Prefix-stable ranks and residual stripes

Ranks are selected from hardware-friendly buckets, initially `{32, 64, 96}`.
Calibrate one maximum-rank factorization per layer/GQA group and require smaller
ranks to be prefixes of that factorization. Rank-specific recalibration must
preserve this prefix contract.

Represent rank as 32-wide stripes:

```text
stripe 0: dimensions  0..31   (base, present on every page)
stripe 1: dimensions 32..63   (optional extension)
stripe 2: dimensions 64..95   (optional extension)
```

This is both the rank-allocation unit and the cold residual representation. A page
that underfits at rank 32 can store stripe 1 without changing its basis or token
identity. A more sensitive page can store stripes 1 and 2. Query projections use
the matching calibrated stripes.

Promotion provenance is part of the format contract. Once a sealed page retains
only rank-32 coordinates, it cannot synthesize stripes 1/2 later. Promotion must be
decided while the maximum-rank latent is still available: normally at seal time,
with the mutable open tail carrying rank 96, or within a bounded source-retention
window whose memory is explicitly accounted. No implementation may promise
arbitrary late promotion without retaining or recomputing the source.

For a token with extension stripes, attention must compute:

```text
score_t = dot(Q_base, K_base_t)
        + dot(Q_ext1, K_ext1_t) when stripe 1 is present
        + dot(Q_ext2, K_ext2_t) when stripe 2 is present
```

The value accumulator must apply the same token probability to its base and
extension value coordinates. A residual correcting the same token must **not** be
run as an independent softmax tier. Doing so gives it a different normalization
and is mathematically incorrect.

Each 32-wide stripe may carry its own signed FWHT rotation. Applying the same
orthonormal rotation to the matching K and Q stripes preserves their dot product.
This lets ranks 32, 64, and 96 use the same stripe primitive without requiring a
96-wide Hadamard transform. Value rotation is optional and must be folded into the
fused output factor when used.

## CASK interaction

CASK is a cold-page transformation, not a residual implementation. Keep error
sources and remedies separate:

| Error source | Correct response |
| --- | --- |
| Calibrated subspace underfit | Add a nested rank stripe |
| KVarN quantization error | Increase K or V bits, or change stripe quant policy |
| CASK merge error | Reduce `fold_m`, increase the core, or stop merging that page |
| Severe outlier page | Promote to maximum rank or a full-dimensional KVarN override |
| Sustained distribution shift after promotion | Evaluate OjaKV-style adaptation |

Once tokens have been merged, a rank extension can improve the merged slot's
feature approximation but cannot restore the removed token identities. Admission
telemetry must attribute loss before choosing a promotion action.

## Storage geometry

Keep KVarN's 128-token block geometry. It aligns with the existing packed record
and flash tile and is the proposed latent-KV scheduler page size; no generic
128-token scheduler page exists yet.

For a K stripe stored feature-major as `[rank, 128]` at `b` bits:

```text
K bytes/token/head = rank*b/8 + rank/32 + 2
```

For a V stripe stored token-major as `[128, rank]`:

```text
V bytes/token/head = rank*b/8 + rank/64 + 4
```

At KVarN4 with equal K/V ranks and original `head_dim=256`:

| Base rank | K+V bytes/token/KV-head | Compression vs FP16 K+V |
| ---: | ---: | ---: |
| 32 | 39.5 | 25.9x |
| 64 | 73.0 | 14.0x |
| 96 | 106.5 | 9.6x |

These figures include packed-record scale metadata but exclude static bases, page
descriptors, allocator fragmentation, and sparse extension pages. All measured
reports must include those real costs rather than quoting the formula alone.
Static basis/fused-weight deltas are expected to be on the order of tens of MB for
Qwen3.5-class packages, depending on model size, number of full-attention layers,
rank, dtype, and whether replaced weights are counted net or gross. Phase 0 must
report the exact net package and resident-memory delta.

Independent K/V ranks and bit widths are allowed by the policy contract. The
initial runtime should keep the matrix small: K4/V4 and ranks 32/64/96, followed by
K2/V4 or K2/V2 only after the KVarN4 path has quality and performance evidence.

The `{32,64,96}` buckets and headline ratios target `head_dim=256`. For
`head_dim=128`, rank 96 is only a 1.33x low-rank reduction and may not amortize the
post-RoPE projections; prefer `{32,64}` and admit the latent path only after a
measured crossover. KVarN still degrades gracefully, but the plan's "primary
representation" claim is scoped to operating points with a real net win.

## Page and scheduler contract

Separate immutable model policy from mutable session state.

### Model policy

The model package owns:

- calibration/evaluator fingerprint and policy ID;
- per-layer and per-GQA-group base/max K and V ranks;
- KQ-SVD K/Q factors;
- calibrated V down factors plus either fused output projection weights or the
  gated current-output reconstruction contract;
- stripe ordering, rotations, codec parameters, and supported bit widths;
- legal rank/format transitions;
- kernel capability requirements;
- quality and performance evidence references.

The target is embedded HFQ package sections, not mandatory loose sidecars. A
provisional section layout is:

```text
kv.policy
kv.latent.k_basis
kv.latent.q_basis
kv.latent.v_down
kv.latent.v_up
kv.latent.wo_fused
kv.latent.rank_map
kv.latent.calibration
kv.evidence
```

Names may be tightened when the HFQ typed-section writer/loader is implemented,
but producers and consumers must move together. The loader must reject a policy
whose required kernels, ranks, arch capabilities, or weight shapes are unavailable.
Embedded sections are the target. Reserve a dotted artifact feature such as `.lkv`
only if latent KV later ships as an independently loaded loose sidecar.

### Session page state

The runtime owns fine-grained block descriptors similar to:

```text
LatentKvBlockDesc {
    sequence_id
    layer
    logical_start
    valid_tokens
    gqa_group
    base_k_rank / base_v_rank
    extension_mask
    k_bits / v_bits
    merge_state
    residency
    generation
    base_handles
    extension_handles
}
```

The exact Rust type should avoid storing per-block strings and should use typed IDs,
enums, and allocator handles. `LatentKvBlockDesc` is the hot-path/block-list type;
it does not supersede `hipfire_state::SequenceStatePageDescriptor`. The existing
descriptor remains the coarse scheduler/reservation/health view of a session's KV
state. The latent page table must aggregate its blocks into that existing type,
preserving `SequenceStateHandle`, `allocation_epoch`, logical position, placement,
ownership, and resident-byte accounting rather than creating a second session-state
lifecycle.

The page table must support:

- one session with many pages and many concurrent sessions;
- prefix-page sharing without mutable aliasing, only when tokens, policy, RoPE
  convention, and baked absolute positions are identical;
- device, GTT/host, and future offload residency;
- asynchronous migration with generation checks;
- immutable sealed pages plus one mutable open tail page;
- batched prefill/decode block lists;
- deterministic reset, cancellation, and teardown;
- accounting visible to scheduler health/telemetry.

The inference runtime executes a supplied page/block list. The orchestrator chooses
when to seal, promote, merge, spill, prefetch, or evict pages. Keep those policy
decisions out of attention kernels.

## Runtime kernel shape

The production read should be one block-list attention family, not one launch per
cold segment:

1. Project the current post-RoPE queries into the active K rank stripes.
2. Launch over `(batch row, query head, logical KV block)`.
3. Read an open BF16/F16 latent page or inline-dequant a sealed KVarN base record.
4. Add token-aligned extension-stripe dots when the page descriptor enables them.
5. Produce online-softmax `(m, l, output)` partials.
6. Accumulate base and value-extension coordinates with the same probabilities.
7. Reduce block partials and either apply the fused low-rank output projection
   or, for an explicitly packaged gated layer, reconstruct only the current
   query-head result before the existing gate and `W_o`.

Do not materialize a full dequantized latent cache or a full-dimensional K/V shadow.
Per-call scratch follows the `OwnedTensor` rules in the runtime/RDNA `AGENTS.md`.

The recurring read-side projection overhead is the current query's `B_q` projection.
The K-side `A_k` projection runs once when each token is appended, not once per
cached-token read. On ungated layers, the fused `R_v W_o` replaces the baseline
full-width `W_o` and should be cheaper rather than booked as new overhead. On
gated Qwen3.5 layers, separately measure the authorized current-result `R_v`
projection, gate, and unchanged `W_o`; do not report the ungated fusion cost for
that path.

The first kernels should be register/wave tiled and zero-LDS on the local gfx1103
path. Rank stripes are multiples of wave32. WMMA versions are a later measured
optimization and must use the vendored AMD Matrix Instruction Calculator when lane
mapping or accumulator layout is selected.

## Rank allocation

Borrow QSVD's global marginal-importance idea, not its exact VLM joint-QKV runtime.
The allocator chooses a discrete configuration for each layer/GQA group under real
storage and compute budgets.

The scoring data should include:

- KQ score-matrix or attention-output error for each K rank increment;
- output-sensitive V error for each V rank increment;
- KLD/NLL/PPL delta from end-to-end rank ablations;
- expected context length and number of resident sessions;
- KVarN record bytes including metadata;
- fused `W_o` shape and supported weight-format group constraints;
- projection and attention kernel cost on the target arch;
- local/sliding/global attention behavior and hybrid non-KV layers.

Initially allocate ranks per layer or GQA group, but restrict choices so the fused
`W_o` input dimension remains compatible with existing MQ/HFQ group sizes. Do not
accept arbitrary per-head ranks that force padding away the storage win or silently
fall back to a slow weight path. Irregular head ranks can be revisited after grouped
or ragged fused-output kernels exist.

## Implementation phases

### Phase 0: Contract and calibration artifact

Deliver a host-side calibration/reference path before changing production cache
allocation.

- Add an Astrea latent-KV planning/calibration artifact with model/engine/dataset
  fingerprints.
- Capture or consume post-RoPE Q/K and value/output evidence per layer and GQA
  group.
- Implement KQ-SVD key/query factors and the KQ-SVD closed-form value-output
  factors, followed by an optional ReCalKV-style calibration refinement.
- Emit prefix-stable maximum-rank factors plus candidate rank maps.
- Evaluate ranks 32/64/96 in an offline attention reference.
- Add a per-cache SVD oracle arm at the same rank and on the same captured caches.
  Rank 32 is the mandatory first-milestone comparison; higher ranks are diagnostic.
- Before inspecting held-out results, record numeric
  `max_static_vs_oracle_kld_delta` and `max_static_vs_oracle_ppl_ratio` admission
  thresholds in the experiment artifact. Do not loosen them after evaluation.
- Stratify calibration by position and sequence length, then evaluate held-out
  lengths and relative offsets at least 4x beyond the calibration regime.
- Record actual commands, calibration corpus, validation corpus, sequence lengths,
  artifact sizes, and net static basis/fused-weight bytes.

Exit gate: a reproducible policy artifact and held-out evidence showing that the
rank-32 static factorization is useful before KVarN quantization is introduced and
lands within the predeclared KLD/PPL delta of the same-rank per-cache SVD oracle,
including the extrapolation-length evaluation.

### Phase 1: BF16/F16 latent runtime oracle

Start with Qwen3.5 full-attention layers at rank 32 and keep the feature default-off.

- Load the calibrated bases and the packaged value-output contract (fused for
  ungated layers, current-output reconstruction for gated Qwen3.5 layers).
- Project post-RoPE Q/K and generate latent V.
- Store an unquantized rank-32 latent cache first.
- Define the real single-session `LatentKvBlockDesc` and block-list producer-consumer
  interface now. The oracle may use a simple allocator, but its attention kernel
  must consume the same typed block-list shape that later batched/page-table work
  extends.
- Project aggregate latent-block inventory into the existing
  `SequenceStatePageDescriptor` view for memory/placement accounting; do not fork a
  second session-state descriptor hierarchy.
- Add an explicit-reconstruction debug oracle for values and prove parity with
  the packaged value-output contract: fused `R_v W_o` for ungated layers, or
  current-result reconstruction followed by the existing gate and `W_o` for
  gated Qwen3.5 layers.
- Run the same rank-32 prompts/caches through the offline per-cache SVD oracle and
  report the static-basis penalty separately from the total low-rank penalty.
- Preserve full-KV/asym3/KVarN baselines and make unsupported models fail closed.
- Support single-token decode and the existing serial prefill reference first.
- Repeat quality checks at lengths/relative offsets at least 4x beyond calibration.
- Profile append-time `A_k`, decode-time `B_q`, the packaged value-output path
  (including current-result reconstruction, gate, and `W_o` for Qwen3.5), VGPR
  occupancy, and scratch/register spills at rank 32 and at a synthetic
  maximum-rank-96 kernel shape. Zero LDS must not be achieved by silently spilling
  the working set.

Exit gate: end-to-end coherent generation, finite logits, component parity, and
held-out KLD/PPL evidence versus full or accepted high-precision KV, plus a
rank-32 static basis within the Phase-0 predeclared delta of the same-rank per-cache
SVD oracle. The extrapolation-length checks and block-list interface must pass.

### Phase 2: KVarN latent pages

- Generalize KVarN producer/consumer geometry to latent ranks 32/64/96.
- Use feature-major K and token-major V records.
- Add CPU codec, GPU quant/dequant, and host/device byte-layout parity tests.
- Seal 128-token pages; retain a small BF16/F16 open-tail page. The base-only
  Phase-2 path may retain rank 32, but the later promotion-capable path must carry
  maximum-rank source coordinates until its seal decision.
- Implement fused inline-dequant latent attention without a full latent shadow.
- Measure KVarN4 independently from low-rank loss.

Exit gate: KVarN latent results remain within the predeclared quality budget versus
the BF16 latent oracle and show an actual decode-memory/bandwidth benefit.

### Phase 3: Batched allocator and scheduler integration

- Extend the Phase-1 typed block-list contract with page allocator ownership,
  refcounted/shared handles, and multi-session block-list construction outside
  `KvCache`'s quant-mode boolean soup.
- Support multi-session batched prefill and decode without forcing
  `SerialReference`.
- Integrate prefix sharing, cancellation, teardown, and residency accounting.
- Keep `SequenceStatePageDescriptor` as the aggregate scheduler-visible view and
  verify its reservation/health bytes against the fine-grained latent allocator.
- Preserve DFlash/spec-decode cache semantics and tree masks.

Exit gate: batched and serial paths agree numerically, session isolation is proven,
and hierarchical latent mode no longer disables fused serving backends.

### Phase 4: Rank allocation and nested residual promotion

- Add the global discrete rank allocator to Astrea.
- Store base stripe 0 on every page and optional stripes 1/2 on promoted pages.
- Keep the mutable open tail at maximum rank, make promotion a seal-time decision,
  and define any bounded post-seal source-retention window explicitly.
- Add calibration-derived promotion thresholds, provenance, and runtime telemetry.
- Compute extension logits and values inside the same attention tile.
- Add maximum-rank/full-KVarN fallback pages for severe outliers.

Exit gate: sparse promotion recovers a meaningful fraction of base-rank quality at
lower bytes than a uniform higher-rank cache.

### Phase 5: Cold CASK and scheduler residency policy

- Port TriAttention scoring to latent pages or score before the hot source is
  discarded.
- Port CASK separately as the core-preservation plus scratch-consolidation page
  transform; do not describe it as a scorer.
- Redefine CASK's semantic answer-anchoring core/scratch policy for age/pressure
  cold pages, or specify and validate an explicit age/pressure surrogate. Do not
  assume semantic core is equivalent to old pages.
- Make consolidation/core selection a page transformation with explicit
  position/token metadata.
- Defragment cold pages during idle windows rather than accumulating one segment
  launch per turn.
- Add GPU/GTT/host residency transitions, prefetch, and scheduler pressure inputs.
- Keep OjaKV deferred until static bases plus maximum-rank promotion show measured,
  sustained distribution-shift failure.

Exit gate: a scheduler-controlled quality/byte/latency policy that retains coherent
multi-turn and long-context behavior under real memory pressure.

## Evaluation and admission

Every stage must isolate its own damage:

1. full/BF16 KV baseline;
2. same-rank per-cache BF16 SVD oracle;
3. static-basis BF16 latent KV;
4. KVarN-packed latent KV;
5. base plus promoted rank stripes;
6. CASK-merged cold pages;
7. residency migration and batched serving.

Required quality evidence:

- finite-logit checks;
- KLD, NLL, and PPL against the same engine fingerprint and RoPE convention;
- the static-versus-per-cache-oracle KLD/PPL gap at rank 32, checked against
  thresholds fixed before held-out evaluation;
- position-stratified held-out evaluation at sequence lengths and relative offsets
  at least 4x beyond the calibration regime;
- long-context retrieval/needle and committed long-context batteries;
- multi-turn recall and conversation continuation;
- agentic/tool-call and structured-output behavior where relevant;
- DFlash/spec-decode acceptance and coherence;
- at least one small bring-up model and one model/context large enough for KV
  bandwidth and capacity to dominate.

Required systems evidence:

- actual bytes per token/page/session including record metadata, bases, page tables,
  allocator fragmentation, scratch, and residual prevalence;
- exact net package and resident bytes for static bases and replaced/fused weights;
- prefill throughput and decode tokens/s over increasing context lengths;
- separate append-time K projection, decode-time Q projection, packed-attention,
  and fused-output timing;
- VGPR occupancy and spill/scratch evidence for base and maximum-rank stripe shapes;
- first-token latency and migration/prefetch stalls;
- page promotion, merge, spill, and hit-rate telemetry;
- serial versus fused-batch parity and performance;
- Atlas AR and DFlash rows before promotion claims;
- RDNA2, RDNA3, and RDNA4 compile/routing coverage, with live hardware evidence
  where available.

Changes touching runtime, dispatch, kernels, quant formats, or speculative decode
must run the narrow relevant parity tests and `./tests/coherence-gate-dflash.sh`.
Workflow-only slices run `./tests/no-gpu-ci.sh`. Coordinate non-daemon GPU binaries
with `hipfire lock` unless their gate already does so.

## Promotion rules

Do not call the design promoted or ship-ready until:

- the model package embeds or deterministically resolves its calibrated policy;
- unsupported policy/kernel/arch combinations fail closed;
- the rank-32 static basis stays within its predeclared KLD/PPL delta of a
  same-rank per-cache SVD oracle, including 4x length/offset extrapolation;
- static-basis, packed-latent, residual, and CASK losses have been measured
  separately;
- fused batch serving no longer silently bypasses the cache path;
- actual total memory is lower at the admitted operating point;
- decode performance is neutral or better at the contexts where the policy is
  intended to activate;
- DFlash and AR quality gates pass;
- static calibrated bases have been tested on held-out and multi-turn distributions.

OjaKV becomes justified only when those tests show a repeatable distribution-shift
failure that nested calibrated promotion cannot cover within budget.

## Non-goals for the first milestone

- Replacing all existing `KvCache` modes at once.
- Making hierarchical latent KV the default.
- Adding Vulkan, wgpu, or a cross-vendor backend.
- Computing per-session SVD or Oja bases in the decode hot path.
- Shipping arbitrary per-head ranks before fused-output kernel constraints are
  understood.
- Treating a dry-run Astrea policy or output-cosine probe as runtime admission.
- Reconstructing full K/V on every attention read, as in cache schemes that store
  a shared latent but replay an up-projection over the entire context.
- Requiring Phase 1 to beat CASK at equal bytes; BF16 latent and KVarN/CASK are not
  matched storage budgets and are complementary stages in this plan.

## First executable milestone

The first implementation goal is deliberately narrower than the full system:

> Produce a reproducible, default-off Qwen3.5 full-attention calibration and BF16
> latent-runtime oracle at rank 32, with KQ-SVD keys, the KQ-SVD value-output
> closed form plus ReCalKV-style calibration/fusion, an explicit value-
> reconstruction parity oracle, and a same-rank per-cache SVD quality oracle. Fix
> the allowed static-versus-oracle KLD/PPL delta before held-out evaluation and
> require the static basis to meet it, including sequence lengths/relative offsets
> at least 4x beyond calibration. Build the oracle on the real single-session typed
> block-list interface and export aggregate state through the existing
> `SequenceStatePageDescriptor`. Do not add KVarN packing or scheduler migration
> until this factorization contract is proven end to end.

This milestone establishes the producer-consumer contract all later storage,
kernel, residual, and scheduler work depends on.
