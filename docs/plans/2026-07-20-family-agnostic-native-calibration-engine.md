# Family-agnostic native calibration engine and expert microbatching

Status: implementation in progress; native engine and mechanism gates landed,
full-scale production/admission runs pending

Primary validation host: gfx1151, 120 GiB unified aperture

First production adapter: Qwen3.5-397B-A17B

Reference source: `/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B`

## Implementation evidence snapshot — 2026-07-20

The native mechanism is implemented, but the production quality/admission
ladder is not complete. Keep this distinction explicit when reporting status.

Implemented and verified in this checkout:

- family-neutral calibration contracts, sample boundaries, deterministic
  scheduling, F32 RAM/mmap boundary stores, tensor plans, read ledger, resume,
  KLDREF packing, and canonical calibration metadata;
- registered Qwen3.5 and Gemma3-text source adapters selected by architecture
  metadata rather than a family-specific CLI option;
- per-layer/per-expert gate-up and down telemetry, the 2,048-row hard floor,
  strict and preserve-undercovered policies, and routing-aware tile admission
  with seen/admitted/slack/quota-skipped accounting;
- shared grouped-MoE routing/scratch/capture machinery, K=8/K=10 tests, mixed
  OQ4 plus BF16/F16 expert execution, and canonical paged OQ4 expert layout;
- native two-pass orchestration, atomic recipe manifests, artifact fingerprints,
  interrupted-run resume, and quantizer enforcement of high-precision fallback;
- durable per-layer phase timing for source load/upload, teacher execution,
  capture serialization, collector finalization, and part sync/hash, persisted
  in checkpoints and the completed artifact for resume-safe ETA analysis;
- bounded family-neutral source lookahead: the engine selects the next owner's
  canonical physical ranges without consuming the read ledger, warms at most
  16 GiB through one fixed 8 MiB worker buffer during current-layer execution,
  preserves 32 GiB of live host-memory headroom, and persists read/wait/error
  timing while remaining compatible with older checkpoints;
- bounded quantizer storage: completed tensors spill to disk for bounded RSS,
  and Linux final assembly releases each copied-and-hashed spill range with
  hole punching rather than retaining two full artifact-sized payloads;
- canonical induction naming for the target, typed BF16/F16 DFlash sidecars,
  TriAttention sidecars, calibration artifact, and two-pass manifest;
- `./tests/no-gpu-ci.sh`, `./tests/coherence-gate-dflash.sh`, GPU calibration
  reduction/grouped-capture/KLD tests, and mixed/paged expert parity on gfx1151;
- an index-only dry run of the 397B source: 60 layers, 1,038 logical tensors,
  792,692,717,952 unique source bytes across 94 shards, K=10, sequence batch 4,
  time tile 64, and the complete one-read ledger contract;
- the same native CLI/source-plan contract dry-runs against the installed
  Gemma3-text checkpoint (`medgemma-27b-text-it`, architecture 12): 62 layers,
  809 logical tensors, and 54,018,004,480 unique source bytes across 11 shards;
- conservative storage preflight for the production Qwen calibration: about
  12.54 GB capture payload, 135 MB KLDREF, 8.59 GB boundary spool, 25.21 GB
  simultaneous part/assembly payload, and 38.16 GB required free space after
  fixed overhead and safety margin; the current `/home` artifact root passes;
- a real bounded Qwen3.5-397B layer-0 run with K=10: 20 routed slots from two
  tokens at both capture roles, zero dropped indices, 203,066,368 part bytes,
  19 canonical reads totaling 15,184,552,832 bytes, and zero duplicates; and
- a real Gemma3-text layer-0 run followed by a layer-1 resume: the ledger grew
  monotonically from 14 to 27 canonical reads and 3,644,369,408 to
  4,470,166,528 bytes with zero duplicate reads;
- a real Qwen layer-0 sequence-batch sweep over the same 32x8-token sample set:
  fresh-process wall time was 6.06/4.89/4.48/4.31/4.18 seconds for sequence
  batches 1/4/8/16/32 respectively. Every run recorded 2,560 K=10 routes at
  both capture roles, zero dropped or duplicate rows, identical 209,968,128-byte
  parts, and zero Hessian/imatrix diagonal inconsistency;
- an extended Qwen layer-0 batch-32 run over 4,096 tokens: 40,960 K=10 routes
  were executed and observed at both capture roles with no invalid, duplicate,
  or quota-dropped rows. The deliberately small sample was still far below the
  production coverage gate: 31 experts received no route, only one reached the
  2,048-row floor, and preserve-undercovered retained 511 experts. Cold wall
  time was 138.88 seconds with 15.1 GB maximum RSS, so this is throughput and
  coverage-shape evidence rather than an admissible calibration;
- a complete 62-layer Gemma3-text stream after a real pause/resume: the
  38,692,740,100-byte artifact contains 434 Hessians, 434 imatrices, one matched
  KLDREF position, and a complete 809-logical/808-canonical read ledger over
  54,018,004,480 bytes with zero missing or duplicate reads; and
- a bounded second-pass `oq4++` join over Gemma layer 0: AWQ+LDLQ found all seven
  requested Hessians (`success=7`, `missing=0`, `k_mismatch=0`) and wrote a
  249,080,134-byte HFQ whose metadata binds the 38.69 GB calibration artifact;
  and
- a fresh Gemma layer-0 timing check measured 64.6 ms source load/upload,
  118.1 ms execution, 1.226 s capture serialization, 0.5 ms collector finish,
  and 1.437 s part sync/hash (2.846 s before checkpoint commit). This validates
  the persisted timing schema and identifies artifact I/O as the limiter for
  that deliberately tiny two-token smoke; and
- a same-sample Qwen row-geometry sweep over 4,096 tokens produced identical
  normalized descriptors, coverage telemetry, and consistency at 256, 512,
  1,024, 2,048, and 4,096 rows. Layer execution fell from 7.56 s at 256 rows to
  3.76/2.55/2.16/1.89 s respectively. End-to-end pre-checkpoint time bottomed
  at 4.64 s for 2,048 rows because the 4,096-row run paid higher capture-write
  variance. At 2,048 rows, sequence batches 32/64/128 took 2.55/2.15/2.27 s of
  execution on an identical 128x32 sample set. The CLI auto-tuning ceiling is
  now 2,048 rows; live estimates/allocation probes still select lower geometry
  on constrained hosts, and the target production recipe uses batch 64.

The full Gemma stream also exposed file-backed safetensor residency. Ordinary
planned tensors are released after upload, while a canonical tensor with a
declared tied alias remains resident until the alias is consumed. The initial
`posix_fadvise(DONTNEED)` fix allowed Gemma finalization to fall from roughly
57 GB RSS to 15 GB, but production Qwen proved that mapped 8--9 GB shard pages
could still accumulate. Mapping-level `MADV_DONTNEED` now removes the resident
PTEs immediately and `posix_fadvise` retains the backing-cache hint. A bounded
Qwen resume fell from a 44.8 GB two-layer peak to 21.9 GB while preserving
refault correctness and the one-read ledger.

The first six steady production Qwen layers read about 13.15 GB each. Layers
1--5 spent 123/152/229/240/277 seconds in source load plus upload versus
144/142/219/156/162 seconds executing, so the 1 Gbps network source is a
co-limiter or dominant limiter. The bounded lookahead above is intended to
overlap that source read with current-layer execution; its actual reduction is
still an evidence item and is not inferred from the implementation alone.

Still required before declaring the engine complete or promoting a production
397B quant:

- the full Qwen3.5-397B teacher pass and calibration/KLDREF artifact;
- matched prefetch-on versus prefetch-off layer timings on the same network
  source, separating background read duration, foreground wait, and upload;
- the second target-source read and completed quantized artifact;
- full-run confirmation that the bounded-layer 2,048-row, batch-64 winner
  remains optimal, the full Qwen layer stream, and production per-expert
  coverage/fallback results from the real corpus;
- resident-versus-streamed comparison and matched held-out KLD/PPL evidence;
- the controlled minimum-coverage and capture-target sweeps;
- resident-versus-streamed second-family parity (the complete streamed artifact
  and quantizer-consumption half are now proven); and
- admitted-path channel evidence beyond the currently tested gfx1151 host.

`/srv` cannot host the production outputs on the current host snapshot: it has
approximately 4.3 GiB free while the source checkpoint alone is approximately
807 GB. `/home` is the proposed artifact/spool root and had approximately
381 GiB free at the latest check. Treat that number as transient: the native
production preflight reports live availability and must pass immediately before
starting each full pass. A dry run is not throughput, coverage, quality, or
source-payload-read evidence.

## Outcome

Build a Rust-native calibration engine that can read a Hugging Face checkpoint
directly, stream one layer at a time, process many independent corpus sequences
in parallel, and write one canonical `.calib.hfq` containing:

- dense projection Hessians and imatrices;
- per-expert imatrices and routed-token counts;
- model-family-neutral router histograms and co-occurrence records;
- matched-corpus KLDREF top-k logits and log-normalizers;
- exact source, tokenizer, corpus, sample-geometry, precision, and execution
  provenance.

The engine is family-agnostic. A model family supplies tensor-name/layout rules,
embedding/finalization operations, and one-layer execution. Qwen3.5 is the first
adapter and the 397B A17B model is the scale target, but no scheduling, artifact,
capture, KLDREF, or grouped-expert policy should be named after Qwen.

The intended induction contract remains two reads of the target checkpoint:

1. one layer-streamed teacher pass produces calibration + KLDREF;
2. one quantizer pass consumes that artifact and writes the quantized HFQ.

There is no intermediate BF16 HFQ and no Python model forward. Python may remain
temporarily as a parity oracle and workflow wrapper, but the calibration data
plane, model math, expert routing, reductions, and artifact writer are native.

## Decisions locked by this plan

1. **Corpus samples, not one concatenated token stream.** Calibration input is a
   deterministic list of independent token sequences. KV, convolution, and
   recurrent state reset at every sample boundary. The artifact records sample
   IDs, token counts, context length, truncation/padding policy, and token hashes.
2. **KLDREF is collected in the same teacher pass.** It is the matched reference
   used to judge a quantized candidate; Hessian/imatrix records drive the weight
   optimization. Both use the same corpus rows without requiring another model
   load.
3. **F32 boundary activations by default.** Layer-streamed residual boundaries
   are held in host memory or an mmap-backed spool as F32, preserving the native
   BF16/F16 model's F32 residual stream. A BF16 boundary spool is an explicit,
   provenance-stamped memory-saving mode, not the quality default.
4. **Hessians for dense/shared projections; imatrix for routed experts.** Full
   per-expert Hessians remain prohibitively large. Expert coverage and routed
   counts are first-class admission evidence.
5. **Logical capture identities replace pointer identity.** Pointer-to-name
   lookup remains a compatibility path for resident models. Streamed, paged,
   fused, and grouped weights use stable logical capture descriptors so changing
   GPU addresses cannot lose or misattribute expert activations.
6. **The direct-safetensors command is offline tooling.** Expose it as
   `hipfire-coexistence calibrate ...`; keep import/interop orchestration out of
   the daemon/server/runtime binaries. The existing resident-HFQ
   `hipfire collect-artifacts` and daemon `Collect` path remain supported.
7. **Reuse the current grouped MoE pipeline before adding kernels.** First
   extract and measure the existing scatter, grouped GEMM, unscatter, and combine
   path. Add or port kernels only when a profile identifies the limiting stage.
8. **Minimum expert activation coverage is a hard quality gate.** The initial
   default is 2,048 routed activation rows for every `(layer, expert)` in the
   model, measured separately at the gate-up and down capture points. Model-wide
   averages do not satisfy this requirement. The strict default refuses to
   finalize an undercovered artifact; an explicit fallback policy may instead
   preserve undercovered experts at high precision.
9. **Saturated experts stop capture, not execution.** Calibration admission is
   decided per valid `(token, expert)` route and capture role. Once an expert
   reaches its predeclared capture target, later routes still execute and still
   contribute to the residual, downstream routing, dense statistics, and
   KLDREF, but they do not incur more expert activation copies or reductions.
   Never drop a whole token because one of its routed experts is saturated.
10. **Capture filtering must preserve batch economics.** Apply the saturation
    mask after routing to the expert-capture stream; do not compact or shrink the
    teacher's model-execution microbatch. Accumulate admitted rows across
    microbatches until a full reduction tile is available. After the quality
    target, admit only the bounded slack that fills a reduction tile which was
    already going to launch; a saturated expert may never cause another capture
    tile. This avoids paying approximately full-batch cost for a half-full
    expert reduction while keeping every teacher route semantically intact.

## What exists today

| Surface | Current capability | Gap for the target |
|---|---|---|
| `hipfire_runtime::calibration::CalibCollector` | Generic GPU sum-of-squares and Hessian accumulation, compact streaming HFQM writer | Capture is string/pointer keyed; no grouped/segmented expert capture |
| `collect()` | One forward over one resident model | A 397B BF16 model cannot be resident |
| `collect_grouped()` | Bounds accumulator memory by registering a subset of resident layers | Re-runs the entire model for every layer group; it is not layer-streamed weight execution |
| `SafetensorsSource` / `ModelSource` | Mmap-backed shard/tensor lookup and model metadata | Qwen's calibration loader still consumes `HfqFile`; no one-layer source plan or read ledger |
| Arch calibration modules | Qwen3.5, Gemma3, LFM2-MoE, MiniMax, Nemotron, Zaya and others share the same collect/KLD packaging pattern | `CalibOpts`, KLD tensor packing, progress, capture-name maps, and telemetry are duplicated |
| Qwen `forward_prefill_*_session_batch` | Independent-session pointer tables and fused multi-session state routing exist | Full-stack, embedding-owning entry points; no reusable single-layer boundary-in/boundary-out interface |
| Qwen grouped routed MoE | Scatter by expert, grouped gate-up/down GEMM, unscatter/combine; raw F16/BF16 path on gfx1151 | Scratch, dtype admission, and executor live inside Qwen; routed capture is absent; general K-top contract is obscured by `_k8` names |
| KLD GPU reducer | Exact tiled top-256 + logsumexp kernel exists | Calibration collectors still download a full logits row and duplicate host packing |
| Python layer stream | Direct safetensors, one-read guard, boundary spooling, calibration + KLDREF | Duplicates model semantics through Transformers/Torch and is Qwen-specialized |

The important distinction is:

- **grouped capture**: resident full model, capture a few layers, replay the full
  forward for the next group;
- **layer-streamed execution**: load one source layer once, run every sample
  through that layer, write the next residual boundary, then release the layer.

Only the second shape satisfies the 397B memory and two-checkpoint-pass contract.

## Target architecture

```text
hipfire-coexistence calibrate
        |
        v
CalibrationJob + SampleSet + ModelSource
        |
        v
family-neutral LayerStreamEngine
        |---- BoundaryStore (F32 RAM/mmap, double-buffered)
        |---- MicrobatchPlanner (sequences x time tile <= row budget)
        |---- CaptureRegistry / CalibCollector
        |---- ExpertTelemetry
        |---- KldRefBuilder
        |---- ReadLedger + progress/checkpoint manifest
        |
        v
dyn CalibrationFamilyAdapter
        |
        +---- Qwen35CalibrationAdapter (first)
        +---- dense decoder adapter (second proof)
        +---- other family adapters
        |
        v
Gpu + family-neutral GroupedMoeExecutor
        |
        v
canonical <family>-<size>.calib.hfq
```

### Generic engine interfaces

The exact Rust spelling can change during implementation, but the ownership
contract should look like this:

```rust
pub trait CalibrationFamilyAdapter {
    fn family(&self) -> ArchFamily;
    fn inspect(&self, source: &dyn ModelSource) -> Result<ModelPlan, CalibError>;
    fn capture_plan(&self, model: &ModelPlan) -> Result<CapturePlan, CalibError>;

    fn load_embedding(
        &mut self,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
    ) -> Result<Box<dyn CalibrationEmbedding>, CalibError>;

    fn load_layer(
        &mut self,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        layer: usize,
    ) -> Result<Box<dyn CalibrationLayer>, CalibError>;

    fn load_finalizer(
        &mut self,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
    ) -> Result<Box<dyn CalibrationFinalizer>, CalibError>;
}

pub trait CalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &LayerMicrobatch,
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError>;
}
```

The engine owns corpus order, boundary storage, microbatch geometry, retries,
read accounting, artifact construction, and progress. The adapter owns only
family math and source tensor layout. It must not choose corpus samples, write
HFQM metadata directly, or invent a family-specific KLD format.

### Source and read accounting

Keep `hipfire_model::ModelSource` as the common source abstraction and add an
offline wrapper with:

- a parsed per-layer `TensorLoadPlan`;
- tensor role, source shard, stored dtype, shape, byte range, and aliases;
- typed BF16/F16/F32 upload/conversion helpers;
- an atomic read ledger recording first/duplicate/missing reads;
- explicit persistent tensors (embedding, final norm, lm-head) versus
  layer-owned tensors;
- a completion assertion that every planned teacher tensor was read exactly
  once, except declared aliases such as tied embeddings.

Mmap page faults are not a useful definition of a model load. The contract is
logical tensor consumption: a tensor may be viewed/uploaded once in the teacher
pass and once in the quantizer pass. The manifest records byte counts and any
intentional alias rather than claiming an unverifiable physical disk-read count.

### Boundary and sample geometry

`SampleSet` is a generic list of independent sequences, not a model-family type.
The engine materializes embedding rows into boundary A, then for each layer:

1. load the layer once;
2. create fresh per-sample state for that layer;
3. read boundary A in deterministic sequence/time tiles;
4. execute the layer and capture its inputs;
5. write output residuals to boundary B;
6. finalize and spool the layer's calibration records;
7. release layer weights/state and swap A/B.

For recurrent or attention layers, a microbatch is two-dimensional:

```text
rows = active_sequences * time_tile
rows <= configured_or_auto_row_budget
```

Rows are ordered by time round and then sequence so each session's positions are
monotonic. State pointer tables route every row to its own sample-local KV,
convolution, or recurrent state. Ragged sequence tails shrink the active set;
they never concatenate the next sample onto the previous sample's state.

Use F32 host boundaries by default. For 262,144 rows at hidden size 4,096, two
F32 boundaries require about 8 GiB, which is acceptable on the target host and
avoids adding BF16 rounding at every streamed layer. If RAM is constrained,
`BoundaryStore` may mmap its F32 buffers; BF16 storage requires an explicit flag
and a distinct artifact provenance fingerprint.

## Family-neutral capture and telemetry

### Capture descriptors

Replace calibration's implicit `weight_ptr -> String` contract with:

```rust
pub struct CaptureDescriptor {
    pub id: CaptureId,
    pub output_names: Vec<String>,
    pub input_width: usize,
    pub policy: CapturePolicy, // HessianAndImatrix | ImatrixOnly | Skip
    pub layer: usize,
    pub role: ProjectionRole,
    pub expert: Option<usize>,
    pub expert_quota: Option<ExpertCaptureQuota>,
}

pub struct ExpertCaptureQuota {
    pub min_rows: u64,
    pub target_rows: u64,
    pub tile_rows: usize,
    pub sampling: ExpertSamplingPolicy,
}
```

`output_names` supports activation aliases: Q/K/V projections and gate/up pairs
can share one accumulator when their inputs are identical while still emitting
separate canonical artifact entries. This removes duplicated memory and keeps
the artifact compatible with checkpoint tensor names.

Provide three capture entry points:

- dense contiguous rows by `CaptureId`;
- aliased dense rows by one descriptor with multiple output names;
- grouped/segmented rows using expert offsets and sorted row indices.

The current pointer map becomes a compatibility adapter that resolves a pointer
to `CaptureId`. New streamed/fused/grouped code calls the ID-based API directly.

### Generic expert telemetry

Move the data model currently represented by Qwen's thread-local
`MoeRouterHistogram` into a family-neutral `ExpertTelemetry` owned by the
calibration job. It records:

- expert count and runtime K-top;
- per-layer and aggregate top-1/top-k hit counts;
- routed tokens, routed slots, dropped/invalid indices, and routing-weight sums;
- bounded expert co-occurrence pairs;
- per-expert gate-up/down seen, admitted, and quota-skipped row counts;
- routing-weight sum, squared-weight sum, and effective sample size over the
  full seen stream as well as the admitted capture stream.

Adapters feed selections through an explicit job handle. Qwen can keep a thin
compatibility wrapper for serving evidence, but calibration must not depend on
thread-local reset/take globals.

A capture is structurally invalid when an expert had rows admitted by the quota
policy but the corresponding gate-up or down capture is missing. Seeing routes
after saturation with no additional capture is valid. Coverage is incomplete
whenever the admitted count is below the configured floor. The finalizer checks:

```text
sum(expert gate-up seen rows) == valid routed slots
sum(expert down seen rows)    == valid routed slots
admitted rows + quota-skipped rows == seen rows
gate-up admitted rows == min(gate-up seen rows, capture target)
down admitted rows    == min(down seen rows, capture target)
```

Invalid router indices are reported separately and are never counted as seen,
admitted, quota-skipped, or executed routes.

### Minimum expert activation contract

The engine must enforce a coverage floor, not merely report expert hits. The
initial family-neutral policy is:

```text
min_expert_activations = 2048
required_expert_fraction = 1.0
counting_unit = routed rows per (layer, expert, capture role)
capture_roles = gate_up_input, down_input
```

This is evaluated for every routed expert tensor declared by the model. An
expert with no hits has zero rows; traffic in another layer or expert cannot
compensate. Fused gate/up checkpoint tensors may share an accumulator because
their input rows are identical, but both canonical output names inherit the
same verified count.

Before loading weights, the engine applies the necessary capacity bound for
each MoE layer:

```text
minimum_tokens = ceil(num_experts * min_expert_activations / k_top)
```

This cannot prove balanced coverage, but it rejects a corpus that is
mathematically incapable of satisfying the floor. For 512 experts, K=10, and a
2,048-row floor, at least 104,858 routed tokens are required before routing skew
is considered.

The default `strict` policy behaves as follows:

- process the entire frozen calibration sample set while reporting remaining
  per-layer expert deficits;
- finalize only when gate-up and down counts both meet the threshold for every
  required `(layer, expert)`;
- if the corpus is exhausted first, write a resumable coverage report and fail
  the calibration job without publishing the final `.calib.hfq`;
- require a new, deterministically fingerprinted sample set or an explicit
  quantization fallback; never lower the threshold after looking at the result.

An explicit `preserve-undercovered` policy may complete the artifact, but it
must list every shortfall and force those experts to BF16/F16 in the quantizer.
It may not silently apply the requested low-bit format. Zero-hit experts always
stay BF16/F16. Such a candidate is a different mixed-precision policy and needs
its own size, KLD/PPL, and runtime evidence.

The artifact records raw routed rows, admitted capture rows, routing-weight sum,
squared-weight sum, and the derived weighted effective sample size for each
expert. Admitted raw activation rows are the initial hard gate; seen-but-skipped
traffic does not satisfy it. Before declaring the value universal, Astrea should
sweep 512/1,024/2,048/4,096-row floors on smaller MoE models and compare
held-out KLD/PPL. The chosen threshold is frozen in the job manifest before a
production run and cannot be relaxed after seeing held-out results.

### Per-route capture quota and saturation

The coverage floor and the capture saturation target are separate controls:

```text
min_expert_activations = 2048      # low-bit eligibility gate
expert_capture_target = 4096       # initial quality/cost saturation point
expert_capture_tile_rows = 256     # reduction batch; limit is aligned
expert_capture_limit = ceil(target / tile_rows) * tile_rows
```

Call this family-neutral policy **routing-aware tile admission**. It belongs in
the shared calibration scheduler/capture contract, not in Qwen-specific router
code. A family adapter supplies valid route identities and capture roles; the
shared policy decides which routed rows enter the calibration accumulators.

The 4,096-row target is an initial production-plan value, not a universal
quality claim. It is frozen before the run and must be validated by the Astrea
cap sweep below. A faster experimental profile may set the target equal to the
2,048-row floor; it must carry a distinct manifest fingerprint and quality row.
The engine rejects `target < min`. It does not silently change the requested
quality target: `target` and the derived, tile-aligned `limit` are both recorded
in the job and artifact. At most `tile_rows - 1` rows may be admitted above the
target, and only to finish the already-open capture tile. The production default
is aligned, so its target and limit are both 4,096.

For every valid routed `(token, expert)` edge, the grouped MoE executor always
runs gate-up, activation, down, weighted combine, and residual update. Capture
admission is an independent mask at each logical capture role:

```text
seen_rows += 1
execute_route()
if admitted_rows < expert_capture_limit:
    stage_capture_row()
    admitted_rows += 1
    if admitted_rows > expert_capture_target:
        batch_slack_rows += 1
else:
    quota_skipped_rows += 1
```

This mask is per route, not per token. If one token selects a saturated expert
and an undercovered expert, only the saturated route is omitted from capture;
both expert computations and the token's other dense/routed captures remain.
Skipping expert execution would change the layer boundary, later routers, and
KLDREF and is therefore forbidden in the quality path.

Stage admitted rows in per-expert/per-role buffers and launch reductions only
for full `tile_rows` batches during normal processing. When a microbatch crosses
the target, admit at most the rows needed to finish the already-open tile and
classify the rest as quota-skipped. Rows in this bounded slack improve the
statistic but do not alter the frozen coverage target. They must be counted
separately so batch geometry cannot masquerade as corpus coverage. An
undercovered expert may retain or flush a final partial tile at corpus
exhaustion/resume finalization so fallback evidence is not lost; padding must
never be counted as a real activation. Once every expert in a layer reaches its
limit at both roles, disable expert capture hooks for that layer while
continuing the full teacher forward, dense capture, boundary production, and
KLDREF work.

The filter is intentionally **not** a request to form a smaller routed-expert
execution batch. Router output first defines the complete teacher computation;
the capture admission mask then selects which of those already-valid routes
feed calibration staging. Partial per-expert staging is carried across model
microbatches until `tile_rows` is reached, so a temporarily half-full capture
batch is not launched merely because the current model microbatch ended. The
only normal-path cutoff is the derived tile-aligned limit. Surplus routes that
would require a new tile are dropped from capture even when they are in the same
model microbatch. A final partial launch is reserved for corpus exhaustion,
interruption checkpoints, or explicit undercoverage diagnostics, and is
recorded in telemetry.

The admission decision for each `(layer, expert, capture role)` is therefore:

| State | Teacher expert route | Calibration capture |
|---|---|---|
| Below the frozen minimum | Always execute | Always admit |
| At/above minimum but below target | Always execute | Admit |
| Target crossed with an already-open reduction tile | Always execute | Admit only enough rows to fill that tile; count as batch slack |
| Target reached and no tile is open | Always execute | Skip all surplus rows |

This is the batch-economics rule: do not pay another nearly full reduction
launch for a half batch after the expert has enough quality evidence. Tokens
remain in the model microbatch because removing them would alter residuals,
later routing, dense calibration statistics, and KLDREF.

Avoid corpus-order bias. The frozen sample set must be deterministically
shuffled or stratified by corpus source and sequence-position band before
execution, and the admission policy and seed are artifact metadata. The first
`target` routed rows are only acceptable after that deterministic distribution
step. Telemetry continues over all routes after saturation so the artifact
preserves the true routing distribution rather than the quota-capped one.

## Extract the grouped expert substrate from Qwen

The current Qwen path already has the right core algorithm:

1. flatten router selections into routed slots;
2. histogram and pad each expert bucket;
3. build sorted slot indices, inverse permutation, offsets, and tile IDs;
4. run grouped gate-up GEMM;
5. apply activation;
6. run grouped down GEMM;
7. combine by router weights into the residual.

Make those mechanics reusable instead of copying them into calibration.

### Move to family-neutral code

Extract from Qwen's `prefill_batch.rs`, `prefill_chunk.rs`, and `mod.rs`:

- `MOE_GROUPED_BLOCK_M`, total-slot and padded-row bounds;
- the expert counts/offsets/sorted/inverse/tile scratch allocation;
- grouped gate-up/down output scratch;
- scatter, unscatter, activation, and weighted-combine orchestration;
- dtype/architecture capability resolution;
- a configurable K-top contract (test at least K=8 and K=10);
- optional grouped capture calls at the sorted-input and post-activation seams.

Suggested shared types under `hipfire-runtime::moe::grouped`:

```rust
pub struct GroupedMoeScratch { /* extracted moe_* tensors */ }
pub struct GroupedMoeWeights<'a> { /* pointer tables + shapes + dtypes */ }
pub struct GroupedMoeRouting<'a> { /* indices + weights + K-top */ }
pub struct GroupedMoeCapture<'a> { /* gate-up/down CaptureIds */ }
pub struct GroupedMoeExecutor;
```

Kernel wrappers stay in `hipfire-rdna`. Qwen's `PrefillBatchScratch` embeds a
`GroupedMoeScratch`; its FFN keeps Qwen-specific RMSNorm, router normalization,
shared-expert gate, and residual semantics, then calls `GroupedMoeExecutor` for
the routed experts. Other families can supply their own activation/fusion policy
without importing Qwen types.

The existing kernel names ending in `_k8` accept runtime `K_TOP` in several
places. Do not mechanically rename them until tests prove the runtime-K
contract, but remove hard-coded K=8 admission from the new Rust abstraction and
add K=10 channel coverage before using A17B.

### Expert activation capture without per-expert launches

The sorted expert layout is also the best calibration layout:

- gate-up imatrix consumes sorted residual/norm rows segmented by expert;
- down imatrix consumes sorted post-activation rows segmented by expert;
- one segmented sum-of-squares launch updates all live expert accumulators;
- padding/sentinel rows are ignored;
- aliases emit separate gate/up records from one accumulator.

Add a generic segmented reduction kernel only if the existing reduction cannot
be cleanly driven over expert ranges. The first correctness implementation may
launch one reduction per **active expert**, but never one model forward or one
weight page-in per expert. Profile that baseline before fusing it.

## Precision and portability

The source loader accepts BF16, F16, and F32 source tensors. Computation follows
the family adapter's native full-precision path and records effective compute
dtype in the artifact.

- **gfx1151 first:** reuse the existing raw BF16/F16 grouped routed-expert WMMA
  kernels. This is the production path for the 397B target.
- **RDNA4:** compile and channel-test the corresponding instruction form before
  enabling a raw grouped kernel. Until then, use the portable active-expert
  batched GEMM fallback.
- **RDNA3 other than gfx1151:** F16 may use a portable per-active-expert batched
  GEMM fallback unless a validated grouped kernel exists.
- **RDNA2 / cards without BF16:** convert each streamed BF16 layer to F16 once
  during upload and run the F16 fallback. Do not create a second source artifact
  or silently round boundary storage.
- **CDNA:** use its supported dense/batched backend; do not route through RDNA
  WMMA kernels.

The baseline must be correct on the portability matrix before an arch-specific
fast path is admitted. Kernel optimization follows channel test -> coherence
gate -> fresh-process speed gate, with one tuning lever per change.

## KLDREF in the same pass

After the last layer, load the final norm and lm-head once. Process boundary rows
in batches and reuse `kld_tile_topk_lse_f32` to produce exact tiled candidates
and logsumexp components. Merge candidates, truncate to requested top-k (default
64), and append the standard:

- `lm_head.kldref_idx`;
- `lm_head.kldref_logit`;
- `lm_head.kldref_logz`.

Move the duplicated `CalibOpts`, `kldref_extra`, F32 byte packing, artifact list,
and progress reporting out of each arch calibration module into generic
builders. Family adapters provide logits; they do not define KLDREF storage.

Exclude padded rows and positions without a next-token target. Record the exact
sample/position map so eval can prove that a candidate is scored against the
same teacher rows.

## CLI contract

First native source command:

```bash
cargo run --release -p hipfire-coexistence -- \
  calibrate \
  --model /srv/huggingface/models--Qwen--Qwen3.5-397B-A17B \
  --corpus benchmarks/calib/calib-5m.txt \
  --output ~/.hipfire/calib/Qwen3.5-397B-A17B.calib.hfq \
  --sequences 128 \
  --context 2048 \
  --sequence-batch 64 \
  --time-tile 32 \
  --max-rows 2048 \
  --min-expert-activations 2048 \
  --expert-capture-target 4096 \
  --expert-capture-tile-rows 256 \
  --required-expert-fraction 1.0 \
  --expert-coverage-policy strict \
  --kldref --kldref-topk 64
```

The CLI resolves the adapter by registered family metadata, not by a
Qwen-specific flag. `--dry-run` prints:

- resolved family and adapter;
- source dtype and layer/tensor plan;
- boundary bytes and state/scratch estimate;
- initial sequence-batch, time-tile, and row budget;
- expert capture target, derived tile-aligned limit, and maximum free slack;
- minimum expert rows, capture target/tile, sampling seed, required expert
  fraction, and fallback policy;
- expected artifacts and output path;
- logical source byte total and read-ledger rules.

`auto` starts conservatively, probes scratch allocation, and grows within the
declared row and memory budgets. It never changes sample order or corpus rows,
so auto-tuning cannot change the calibration dataset.

## Implementation phases

### C0 — Freeze generic artifact and sample contracts

Files:

- `crates/hipfire-runtime/src/calibration.rs`
- new focused modules under `crates/hipfire-runtime/src/calibration/`
- `crates/hipfire-kld/` where the existing reference types fit

Work:

- introduce `CalibrationJob`, `CalibrationOptions`, `SampleSet`, sample-position
  mapping, `CaptureId`, `CaptureDescriptor`, `ExpertCoveragePolicy`, and
  structured errors;
- centralize KLDREF packing, progress, provenance, and artifact-list assembly;
- add activation aliasing so identical inputs share one accumulator;
- keep compatibility wrappers for existing arch collectors.

Gate:

- no-GPU tests for deterministic sampling, boundary reset semantics, aliases,
  per-layer/per-expert threshold and saturation evaluation, strict/fallback
  behavior, KLD packing, metadata round-trip, and old/new artifact equivalence.

### C1 — Extract family-neutral grouped MoE execution

Files:

- `crates/hipfire-runtime/src/moe/` or its existing nearest shared MoE module;
- `crates/hipfire-rdna/src/dispatch/moe.rs`;
- `crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs`;
- `crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs`;
- `crates/hipfire-arch-qwen35/src/qwen35/mod.rs`.

Work:

- extract `GroupedMoeScratch`, shape math, routing view, dtype capability, and
  gate-up/down executor;
- make Qwen use the shared executor with no output change;
- validate runtime K-top at K=8 and K=10;
- add grouped capture descriptors but leave capture disabled in normal serving.

Gate:

- pure shape/permutation tests including skewed routing, empty experts, invalid
  indices, K=8, K=10, and 512 experts;
- GPU channel parity of grouped versus indexed/reference outputs;
- Qwen prefill tests and `cargo test -p hipfire-arch-qwen35 --lib moe_prefill`;
- `tests/coherence-gate-dflash.sh` because the serving MoE path is touched.

### C2 — Add source plan, read ledger, and boundary store

Files:

- `crates/hipfire-runtime/src/safetensors_source.rs` only for general source
  metadata/view improvements;
- new offline orchestration modules in `crates/hipfire-coexistence/`;
- generic plan/boundary types in `hipfire-runtime::calibration`.

Work:

- build shard-aware `TensorLoadPlan` from `ModelSource`;
- add logical read accounting and tied-weight aliases;
- implement double-buffered F32 RAM/mmap boundary storage;
- implement deterministic sequence/time tiling and ragged-tail scheduling;
- add resumable per-layer spool/checkpoint metadata without claiming a complete
  artifact until every layer and KLD finalization succeeds.

Gate:

- synthetic multi-shard safetensors fixtures;
- duplicate/missing/alias read-ledger failures;
- byte-exact boundary swap and resume tests;
- peak-memory estimator versus allocations on a tiny fixture.

### C3 — Qwen3.5 one-layer adapter

Files:

- new `crates/hipfire-arch-qwen35/src/calibration_stream.rs`;
- extracted single-layer seams in Qwen prefill/state code;
- Qwen loading helpers currently tied to `HfqFile`.

Work:

- implement Qwen tensor-name/layout plan from `ModelSource`;
- adapt existing layer loader logic to BF16/F16 safetensors views;
- expose embedding, one-layer boundary-in/boundary-out execution, and finalizer;
- reuse the existing independent-session pointer-table/state routing instead of
  inventing calibration-only KV or DeltaNet kernels;
- reset state per sample and preserve state across time tiles;
- keep Qwen-specific attention, DeltaNet, router normalization, shared expert,
  and residual semantics inside this adapter.

Gate:

- tiny Qwen3.5 resident-full-stack versus layer-stream comparison on identical
  independent samples;
- per-layer residual, router selection, logits, Hessian, and imatrix tolerances;
- explicit tests that concatenating samples produces a different result and is
  rejected by the sample contract.

### C4 — Large expert microbatching and capture

Work:

- run router and routed experts over `sequence_batch * time_tile` rows;
- feed sorted gate-up/down inputs into grouped capture;
- apply the post-router, per-route capture admission mask without pruning any
  valid routed expert execution or shrinking the model microbatch;
- carry partial capture staging across model microbatches, launch only full
  per-expert reduction tiles during normal processing, admit only free slack to
  finish an already-open tile, and quota-skip rows above the derived tile-aligned
  limit;
- enforce expert seen/admitted/batch-slack/quota-skipped count invariants;
- enforce the frozen 2,048-row default independently for every layer, expert,
  and capture role; emit a sorted deficit report while the layer is resident;
- stop expert capture reductions independently at the frozen target plus its
  bounded tile slack (initially target=limit=4,096), and disable layer
  expert-capture hooks when every role is saturated while leaving teacher
  execution unchanged;
- autotune row geometry over safe candidates (for example 32 through 2,048)
  using allocation success and measured throughput;
- record row geometry, active experts, padding overhead, launches, and peak
  scratch for each layer type.

Gate:

- CPU/GPU segmented sum-of-squares parity;
- expert imatrix parity with a serial per-token/per-expert reference;
- quota-boundary tests where a microbatch crosses the target, proving only the
  free tile slack is admitted, rows requiring another capture tile are skipped,
  and all routed outputs still contribute;
- batch-economics tests proving a partial capture tile is carried into the next
  model microbatch, no undersized normal-path reduction is launched, and model
  row geometry is unchanged as experts saturate;
- multi-route tests where one token selects saturated and undercovered experts;
- K=10 A17B routing and capture-count test;
- synthetic balanced/skewed/zero-hit coverage tests proving aggregate traffic
  cannot hide an undercovered expert;
- batch 1/4/8/16/32 sequence sweep on the target host, reporting context,
  time-tile, rows, peak memory, tokens/s, and routed slots/s;
- no throughput claim from a catalog/index-only load.

### C5 — Batched KLDREF and canonical artifact finalization

Work:

- reuse the GPU tiled top-k/logsumexp reducer over batched lm-head rows;
- remove duplicated per-family KLD packing;
- stream layer records and final extras into one canonical package;
- include source/tokenizer/corpus hashes, sample map, effective precision,
  adapter version, read ledger, coverage threshold/policy, capture
  target/limit/tile/sampling policy, per-expert full-stream, admitted,
  batch-slack, and quota-skipped counts,
  weighted effective sample sizes, and any high-precision fallback list.

Gate:

- exact top-k indices/logits versus CPU reference, bounded logZ tolerance;
- resident collector versus streamed collector artifact comparison;
- quantizer successfully consumes the artifact with expected Hessian/imatrix
  joins and no missing routed expert records.

### C6 — Prove family independence

Choose one already-supported non-Qwen family with a straightforward dense
decoder (Gemma3 is the preferred anchor) and implement only its thin adapter.
Do not add family branches to the engine.

Gate:

- same CLI and sample/artifact contract;
- no Qwen types or tensor-name literals in generic modules;
- a compile-time adapter registry/completeness test catches a registered family
  without a source planner or layer executor;
- old resident collector and new layer-streamed collector agree within the
  defined activation/logit/statistic tolerances.

### C7 — Induction integration and Python retirement

Work:

- make the induction workflow call `hipfire-coexistence calibrate` for pass 1;
- keep `scripts/collect_hessian.py` only as an explicit parity/debug oracle;
- make the two-pass manifest consume the native read ledger and artifact
  fingerprints;
- update `docs/MODEL-INDUCTION.md`, `docs/QUANTIZE.md`, and collector status when
  native production evidence exists;
- delete the Python default only after Qwen 397B and the second family pass.

Gate:

- dry-run plan test;
- interrupted/resumed induction test;
- full two-pass source-read manifest;
- generated calibration and quant artifacts use canonical names.

## Verification and admission ladder

### No-GPU correctness

- sample boundaries, truncation, padding, and fingerprints;
- tensor plan and one-read ledger across multiple shards;
- grouped MoE bounds/permutation at K=8 and K=10;
- capture aliases and expert count reconciliation;
- HFQM streaming/combination and metadata round-trip;
- CLI parsing, dry-run, resume, and failure cleanup;
- `./tests/no-gpu-ci.sh`.

### GPU mechanism tests

Run under `hipfire lock`:

- dense capture reduction versus CPU;
- segmented expert reduction versus CPU;
- Qwen one-layer streamed output versus resident output;
- multi-session state routing versus independent serial sessions;
- grouped expert output versus indexed/reference path;
- batched KLD top-k/logZ versus CPU;
- compile/channel coverage for gfx1010, gfx1030, gfx1100, gfx1151, gfx1201,
  plus relevant CDNA targets.

### End-to-end scale ladder

1. tiny synthetic Qwen fixture: read accounting and exact mechanism tests;
2. Qwen3.5 dense small model: complete streamed artifact parity;
3. Qwen3.5 35B-A3B: routed expert coverage and batch sweep;
4. bounded 397B layer test: one dense and one MoE layer, K=10, memory/read proof;
5. full Qwen3.5-397B-A17B pass: complete calibration + KLDREF artifact;
6. second non-Qwen family: architecture-independence proof.

### Quality/admission evidence

For an identical corpus, sample map, and target quant recipe, compare:

- resident/native or Python oracle calibration where the model fits;
- new layer-streamed native calibration;
- random/no-calibration control where useful.

Quantize identical candidates and report KLD, PPL/NLL, task batteries, and expert
coverage. Do not promote the new collector merely because Hessian diagonals are
consistent. Astrea admission requires matched KLD/PPL evidence; DFlash/CASK and
runtime performance remain separate Atlas/eval gates.

Before freezing the production default, run a controlled expert-coverage sweep
at 512, 1,024, 2,048, and 4,096 minimum rows on a smaller MoE model. Hold the
evaluation corpus fixed and untouched, compare KLD/PPL and low-traffic-expert
sensitivity, then record why the selected floor is sufficient. Do not reduce a
frozen production threshold because the 397B corpus missed it; expand the
calibration sample set or preserve the deficient experts instead.

After selecting the floor, hold it fixed and independently sweep capture targets
of 2,048, 4,096, and 8,192 rows (excluding values below the selected floor).
Compare held-out KLD/PPL, per-expert statistic stability, artifact size, capture
time, and reduction launches. This isolates the value of additional samples
from the low-bit eligibility rule. Astrea selects the default cap from measured
quality/cost evidence; a cap is not declared safe merely because every expert
met the minimum.

## Performance methodology

Measure before tuning. For each batch sweep record:

- GPU architecture and effective source/compute precision;
- sequence batch, time tile, total rows, context length, K-top, active experts;
- layer load/upload time, dense/attention/DeltaNet/MoE/capture time;
- scatter, gate-up, activation, down, combine, and reduction timing;
- padded grouped rows versus real routed slots;
- expert capture seen/admitted/quota-skipped rows, full versus partial
  reduction tiles, and the point at which each layer became fully saturated;
- peak layer weights, state, grouped scratch, capture accumulators, and host
  boundary memory;
- fresh-process median throughput.

Optimize only the measured limiter. Likely first levers are row geometry,
grouped scratch lifetime, segmented capture, and the bounded next-layer source
read lookahead. If warmed loads remain dominant, split source fault/read,
decode, and host-to-device upload before considering pinned host staging or
GPU double buffering. Kernel changes come after those measurements and land
one lever at a time.

## Definition of done

The calibration engine is complete when:

1. `hipfire-coexistence calibrate` accepts a supported safetensors directory and
   resolves its family adapter without a family-specific CLI flag;
2. Qwen3.5-397B-A17B completes one logical teacher read with F32 boundaries and
   writes Hessian, imatrix, expert telemetry, and KLDREF in one artifact;
3. the quantizer completes the second and only other logical target-source pass;
4. every routed expert has at least 2,048 correctly attributed gate-up rows and
   2,048 down rows in every layer, including K=10 routing; undercovered experts
   are either a hard failure or explicitly preserved at BF16/F16;
5. the artifact records the frozen coverage policy, every expert count and
   deficit, capture target/tile/sampling policy, full-stream and admitted-stream
   telemetry, and the quantizer refuses a low-bit plan that violates it;
6. sequence batches larger than one use independent state and the best measured
   safe geometry on the target host;
7. Qwen's reusable grouped-MoE scratch/executor and telemetry have moved to
   shared code, while Qwen math remains in the Qwen adapter;
8. a second family uses the same engine without adding a generic-module family
   branch;
9. resident-versus-streamed mechanism tests pass and matched KLD/PPL evidence
   shows no material quality regression;
10. RDNA2 uses the declared F16 fallback, RDNA3/gfx1151 uses the validated raw
   BF16/F16 grouped path, and RDNA4/CDNA either pass their admitted path or fail
   with an honest capability error;
11. workflow docs name the native path as default and the Python forward as an
    oracle only.

## Explicit non-goals

- Loading the whole 397B BF16 model at once.
- Full Hessians for every routed expert.
- Reusing state across unrelated corpus samples.
- Dropping valid routed expert execution after its calibration quota is full.
- Adding calibration logic to daemon/server hot paths.
- Making the induction orchestrator itself the model-execution engine.
- Claiming quality from calibration consistency alone.
- Porting or fusing a grouped expert kernel before profiling the extracted
  baseline.
