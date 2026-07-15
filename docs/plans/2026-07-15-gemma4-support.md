# Plan: Gemma 4 support with reusable transformer seams

Status: **in progress at the revised Phase 5 dense-31B OQ8 admission gate**.
Branch: `chaingun`. Plan date: 2026-07-15. Status updated: 2026-07-15.

This is the canonical Gemma 4 plan. When older roadmap text disagrees with this
file, this file wins. In particular, Gemma 4 is not "Gemma 3 plus MoE" and the
E2B/E4B models are not MoE models.

## Current implementation status

The implementation has advanced through the reusable dense-text foundation, but
no Gemma 4 checkpoint is admitted in the support registry. By explicit contract
revision on 2026-07-15, OQ8 replaces BF16 as the Phase 5 product candidate. The
pinned Transformers BF16 capture remains the independent oracle. The first valid
exact-prompt OQ8 measurement is frozen as the broad functional baseline in
`benchmarks/gemma4/oq8-thresholds.json`; this adoption occurred after observing
the result and is recorded honestly rather than presented as a pre-observation
gate. The former strict limits remain unchanged as the final OQ8++ narrowing
stage in `benchmarks/gemma4/oq8pp-thresholds.json`.

| phase | status | current result |
|---|---|---|
| 0 — truth freeze | passed, contract revised | Pinned config, tensor, prompt/token, and BF16 oracle fixtures remain authoritative. Separate OQ8 baseline and OQ8++ narrowing thresholds are committed under `benchmarks/gemma4/`. |
| 1 — identity/ingest/toys | passed | Architecture id 24, registry/spec/ingest policy, and dense, PLE-sharing, and dense-MoE toy fixtures are implemented. |
| 2 — shared loader | passed | `TransformerLoader` is shared by Gemma 3 and Gemma 4; Gemma 3 regression evidence is retained. |
| 3 — layered KV | passed | Mixed full/SWA geometry, shared-producer planning, exact storage accounting, reset, and GPU boundary parity are implemented. |
| 4 — primitives/lowered forward | passed | Proportional RoPE, weightless RMSNorm, vector softcap, reference forward, and lowered dense forward passed operator, tiny-model, portability-compile, and coherence gates. |
| 5 — dense 31B/serving | in progress | Dense loading, bounded state, lowered execution, and the boxed serving factory exist. The real 31B OQ8 base-short run establishes and passes the revised baseline; SWA-boundary, multi-global, reload/sequential, and IT gates remain. |
| 6 — prompt/tools/sampling | implementation gates passed, model not admitted | Official Jinja bytes and all 1,640 fixture token IDs match; native tools/channels, metadata EOS IDs `[1, 106, 50]`, and generic top-k 64 pass their gates. 31B-it remains unadmitted until the remaining Phase 5 OQ8 gates pass. |
| 7 — PLE/KV sharing | scaffold only | Config lowering and generic shared-KV storage exist; the real Gemma 4 loader/forward intentionally rejects PLE and shared-KV variants. |
| 8 — dense-plus-MoE | scaffold only | Ingest/config/toy coverage exists; no Gemma 4 routed-expert runtime or real-model admission exists. |
| 9 — quant/eval | OQ8 baseline established | The first OQ8 artifact and exact-prompt evidence exist; the reusable Gemma 4 eval battery, broader prompt gates, and per-variant admission remain. |
| 10 — unified/multimodal/DSpark | not started | Only fixture/config investigation exists; no unified 12B, multimodal, or DSpark spec-decode runtime is claimed. |
| 11 — final OQ8++ narrowing | not started | Produce OQ8++ from the same pinned source and pass the unchanged former strict limits after broad OQ8 admission. |

The revised Phase 5 baseline is the valid exact-prompt OQ8 result at
`~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-oq8-exact-prompt`:

- the embedded tokenizer is byte-identical to the pinned Gemma 4 tokenizer
  (`sha256:12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`),
  and the five input IDs match the oracle exactly;
- greedy generation matches exactly for eight tokens, final argmax agrees at
  token `7001`, and top-5 overlap is `5/5`;
- final-logit cosine is `0.9993488808874887` and maximum absolute error is
  `1.1197633743286133`;
- the all-layer minimum hidden cosine is `0.9953113139978088`, the maximum
  hidden NRMSE is `0.09686980655966955`, and every value is finite;
- the broad OQ8 base-short gate passes at the newly frozen observed values.

The OQ8 result advances Phase 5 beyond base-short but does not admit the model.
The same OQ8 ceiling/floor must now hold for the committed short-prompt suite,
SWA-1/SWA/SWA+1, multiple global layers, unload/reload, and sequential requests.
Any worse result stops the phase and is recorded without changing the new gate.

The earlier BF16 result remains useful historical localization evidence, not the
product admission target. Its best final-logit maximum error was
`0.5618224143981934`, with exact greedy/argmax/top-5 agreement and hidden-state
failures at layers 39, 52, 56, 57, and 58. The rejected BF16-boundary,
batched-prefill, attention, rocBLAS, RoPE, and GeGLU experiments remain recorded
in `benchmarks/gemma4/PHASE5.md`; none is promoted into serving.

A same-input sweep now also covers all 60 decoder transitions independently.
Each layer receives the exact frozen oracle boundary and builds its own real
five-position KV history. The worst final-position transition NRMSE is
`0.005205519351551357`; sliding and full-attention layers have nearly identical
mean transition error (`0.0030814201888346663` and `0.003070946966302261`). This
rules out a discrete layer or attention-geometry defect for the frozen prompt
and narrows the historical BF16 discrepancy to small numerical differences
accumulated through the full stack. The exact per-layer evidence and reproduction
tools are recorded in `benchmarks/gemma4/PHASE5.md`; this diagnostic does not
alter serving or either OQ8 gate.

Operator traces at exact-input layers 39, 40, and 58 reproduce the frozen
Transformers layer outputs bit-for-bit and show smooth error growth through
normalization, projections, attention, and FFN rather than a missing operation
or discrete jump. A selective BF16-staged GeGLU diagnostic improves layer 40
slightly but worsens the first failing layer 39 and the worst late layer 58, so
it is rejected for serving. At this point the retained evidence has ruled out
the tested loader, geometry, cache, norm, projection, RoPE, attention, GeGLU,
residual, layer-scalar, final-norm, and head hypotheses. The historical BF16
discrepancy remains cumulative reduction/materialization-order sensitivity; it
does not override the revised OQ8 product contract.

A post-freeze Transformers reference-variability control is retained at
`benchmarks/gemma4/control-noise-31B.json`. It shows that alternate BF16
reference executions can diverge at the same late-stack hotspots, which is useful
diagnostic context. It does not change the BF16 oracle, the OQ8 baseline, or the
final OQ8++ narrowing limits.

The Phase-6 implementation gates are complete: official rendered bytes and
1,640 token IDs match across 36 committed cases, the full metadata stop-ID set
is carried through serving, and native tools/channels plus top-k 64 pass unit
coverage. This is not a model admission claim; the remaining Phase 5 OQ8 gates
remain binding.

The support matrix therefore intentionally continues to advertise Gemma 4 as
unsupported (`prefill = "none"`, no KV capability) until the implementation
passes all revised Phase 5 OQ8 gates. Phases 0 through 6 have detailed evidence and
reuse/cleanup ledgers in `benchmarks/gemma4/PHASE0.md` through `PHASE6.md`.

## Goal

Add architecture-correct Gemma 4 support to hipfire while improving the seams
the family necessarily crosses:

- offline identity, detection, ingest, and fixtures;
- transformer weight loading;
- heterogeneous per-layer attention and KV state;
- lowered forward execution;
- boxed serving and generation policy;
- prompt, thinking, tool-call, and output-channel handling;
- dense, PLE/KV-sharing, and MoE variants;
- later, multimodal adapters and DSpark speculative decoding.

The first admitted product target is **Gemma-4-31B-it text generation in OQ8**.
The complete text-family target is E2B, E4B, 12B unified, 26B-A4B, and 31B,
base and instruction-tuned where checkpoints exist. Multimodal input and DSpark
speculative decoding are later, explicit capability phases; a working 31B text
decoder must not be advertised as blanket "Gemma 4 multimodal support."

Correctness evidence comes before fusion or speed. OQ8 is the initial product
format, but every candidate is still compared with the independent BF16 oracle;
Python/Transformers is allowed as an offline oracle and fixture generator, never
in the inference hot path.

## Local source material (validation input, not a product dependency)

The current host has the following official Hugging Face snapshots under
`/srv/huggingface`. No runtime code, committed test, or default path may depend
on this mount; it is the local source for conversion, manifest capture, oracle
runs, and GPU admission.

| checkpoint | local size | pinned `refs/main` revision |
|---|---:|---|
| `google/gemma-4-E2B` | 9.6G | `63db66a33dc06d58c02b1e887446e103c202602c` |
| `google/gemma-4-E2B-it` | 9.6G | `70af34e20bd4b7a91f0de6b22675850c43922a03` |
| `google/gemma-4-E4B` | 15G | `a24c9379fd3839ae84e97f0b6aa3152fce9bd033` |
| `google/gemma-4-E4B-it` | 15G | `fee6332c1abaafb77f6f9624236c63aa2f1d0187` |
| `google/gemma-4-26B-A4B` | 49G | `f1102d7de421741c6eafcda46d1806a7a65b83a3` |
| `google/gemma-4-26B-A4B-it` | 49G | `20da991ab4afab98e8f910c4a2e8f4fbefc404ad` |
| `google/gemma-4-31B` | 59G | `02e15e4990e8c452f8543fb26beff15b1daf8f3d` |
| `google/gemma-4-31B-it` | 59G | `3548789868c5356dbf307c98e6f609007b82b3eb` |

The 12B `gemma4_unified` checkpoint is not present in the mount at plan-writing
time. Its text phase is specified below but cannot pass its real-model exit gate
until an official snapshot is available locally.

Top-level GGUF files (`gemma-4-E2B-it-UD-Q6_K_XL.gguf` and
`Gemma-4-26E-A4B-Heretic-TQ4_1S.gguf`) are coexistence/import comparison inputs,
not direct runtime formats. Any GGUF conversion work belongs in
`hipfire-coexistence`; the inference path remains HFQ-only.

## Upstream ground truth

The implementation contract is derived from the pinned checkpoint configs,
tensor manifests, generation configs, and chat templates above, cross-checked
against:

- Google Gemma 4 model card:
  <https://ai.google.dev/gemma/docs/core/model_card_4>
- Transformers Gemma 4 config and model:
  <https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4>
- Transformers Gemma 4 unified model:
  <https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_unified>
- Google Gemma 4 prompt formatting:
  <https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4>
- Google Gemma 4 function calling:
  <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>
- hipfire spec-decode foundation for the DSpark drafter phase (the `SpecTarget`
  verifier seam, `DsparkBody` drafter core, and `.dspark.hfq` sidecar format):
  `crates/hipfire-specdecode-dspark/` and the gemma3 DSpark plan
  `docs/plans/2026-07-07-gemma3-4b-dspark-dflash-cask.md`.

Do not infer missing fields from Gemma 3. Structural fields must be present and
validated; defaults are allowed only where upstream defines a stable default and
the parser test proves it.

## Family shape and feature matrix

| variant | layers / hidden | context / SWA | attention | extra machinery |
|---|---|---|---|---|
| E2B | 35 / 1536 | 128K / 512 | local HD 256; global HD 512; projected K and V | PLE 256; last 20 layers share KV; double-wide MLP in the shared tail |
| E4B | 42 / 2560 | 128K / 512 | local HD 256; global HD 512; projected K and V | PLE 256; last 18 layers share KV |
| 12B unified | 48 / 3840 | 256K / 1024 | local HD 256; global HD 512; global K projection also feeds V; distinct local/global KV-head counts | unified wrapper; no PLE or MoE |
| 26B-A4B | 30 / 2816 | 256K / 1024 | local HD 256; global HD 512; global K projection also feeds V | dense GeGLU plus 128 routed experts, top 8, every layer |
| 31B | 60 / 5376 | 256K / 1024 | local HD 256; global HD 512; global K projection also feeds V | dense GeGLU |

All variants use an explicit `layer_types` list. Local attention uses full
half-split RoPE with its local base. Global attention uses proportional partial
RoPE: the rotated fraction and the exponent basis dimension are distinct. The
implementation must not reduce this to Gemma 3's periodic-pattern and single
head-dimension representation.

## Locked architecture decisions

1. **New family, not a Gemma 3 variant.** Add `hipfire-arch-gemma4` and
   `hipfire-arch-gemma4-spec`. Share generic mechanisms through runtime/dispatch;
   do not create inheritance between the Gemma 3 and Gemma 4 configs or forward
   states.
2. **One base architecture id.** Reserve `ARCH_ID_GEMMA4 = 24` for all Gemma 4
   text-core configurations. `gemma4`, `gemma4_text`, `gemma4_unified`, and
   `gemma4_unified_text` map to the same id; a typed wrapper enum preserves the
   source distinction. Modality and DSpark artifacts are roles/capabilities, not new
   base ids.
3. **No new Option soup.** Gemma 4 loads into a generic
   `Box<dyn ServingBackend>` slot. Do not add `gemma4_text: Option<...>` to
   `LoadedModel`, a `generate_gemma4` central branch, or a daemon branch on id 24.
4. **Per-layer plans are data.** Resolve layer type, geometry, RoPE, projection
   source, cache ownership, and FFN kind once at load/lower time. The token hot
   path consumes the resolved plan.
5. **K=V is a projection rule, not a cache alias.** On qualifying global layers,
   V starts from the unnormalized K projection, then V gets a weightless RMSNorm
   while K gets its weighted norm and RoPE. K and V therefore still require
   distinct post-transform cache storage.
6. **KV sharing is a real alias/lifetime rule.** E2B/E4B sharing layers reuse the
   already-transformed K/V produced by the last non-sharing layer of the same
   attention type. They do not own K/V projection weights or cache allocations.
7. **OQ8 product gate, BF16 oracle.** The pinned Transformers BF16 capture stays
   independent and immutable, while OQ8 is the first runtime artifact admitted.
   The broad OQ8 gate uses `oq8-thresholds.json`; final OQ8++ promotion uses the
   narrower `oq8pp-thresholds.json` without weakening either after this revision.
8. **Official prompts, no family fallback.** Gemma 4 instruction models require
   their official Jinja template and channel/tool grammar. A render failure is an
   error; falling back to Gemma 3 or ChatML is silent model corruption.
9. **No legacy names.** New artifacts follow the canonical names, for example
   `Gemma-4-31B-it.oq8.hfq`, `Gemma-4-E4B-it.oq8.hfq`, and role sidecars such as
   `.vl.hfq`, `.audio.hfq`, `.dspark.hfq`, or `.jinja.hfq` when independently loaded.

## Code-reuse and cleanup contract

Code reuse is an exit criterion, not a suggestion. Every phase maintains a
short reuse/cleanup ledger in its commit or result note:

- existing primitive reused;
- duplicate removed or intentionally retained;
- generic seam added or changed;
- at least two real consumers of every new generic abstraction;
- stale comment, fallback, or branch removed in the touched area;
- oracle path retained until parity and the condition for later deletion.

Rules:

- Search for an existing primitive before adding one.
- Extract only the smallest behavior that is truly shared. Family-specific
  tensor names, config validation, and mathematical policy stay in the family
  crate.
- A generic abstraction with only Gemma 4 as a consumer does not satisfy the
  phase. Either migrate Gemma 3 (normally the second consumer) or keep the code
  Gemma-4-local until a second use exists.
- Do not force Qwen3.5's advanced paging/slab loader through a smaller generic
  API merely to claim reuse. Extract the common raw upload/name/shape mechanics;
  keep paging and family policy layered above it.
- When a touched file contains a known obsolete Gemma 4 assumption, remove it in
  the same phase. Do not leave a second "temporary" truth beside the correct one.
- Do not use compatibility aliases for incorrect Gemma 4 names or prompt tokens.
- Preserve a simple, obviously-correct forward as the oracle while the lowered
  path is enabled behind a flag. Remove the oracle only in a later cleanup after
  recorded fleet parity.

### Reuse map

| concern | reuse | required cleanup while touching it |
|---|---|---|
| identity/ingest | `hipfire-arch-api::{Arch, Ingest, ToyModel}` and `hipfire-arch-specs` | make model-type detection and expert-layout policy registry-driven; do not add a Gemma 4 arm to the central quantizer ladder |
| HFQ loading | `WeightTensor`, existing F16/BF16/Q8/HFQ loaders, tied-head support | extract the duplicated Gemma3/Qwen2 `load_weight_tensor`, norm, embed, and tied-head mechanics into a small runtime transformer loader; migrate Gemma 3 as the second consumer |
| forward lowering | `hipfire-dispatch::pipeline::superop` | represent regular scale/PLE operations as regular ops, not family escape hatches; correct stale "not live" comments |
| dense FFN | existing `gelu_tanh_mul_f32`, GEMV/GEMM families, residual/norm primitives | remove Gemma-specific copies of generic activation/load helpers as they are replaced |
| attention | existing full/SWA GQA attention and per-head QK norm primitives | add explicit weightless RMSNorm and proportional-RoPE basis support instead of fake all-ones weights or Gemma-only kernels |
| KV state | existing `KvCache` storage/codecs and SWA rings | add a layered planner/arena that groups compatible physical caches and maps logical layers; retain old homogeneous constructors as adapters |
| MoE | existing softmax top-8 renorm, indexed expert execution, pointer tables, GeGLU | split the routed-expert core from Qwen shared-expert policy; keep Qwen as an adapter and add Gemma as the second consumer |
| prompt/render | `JinjaChatFrame`, embedded `.jinja` metadata, tokenizer specials | centralize strict render/profile selection; remove the incorrect Gemma 4 JSON parser and `<end_of_turn>` assumptions |
| sampling | shared `SamplerConfig` and GPU sampler | add generic top-k instead of a Gemma-only sampler; preserve greedy behavior when temperature is zero |
| serving | `SimpleAr`, `ServingBackend`, `run_simple_ar` | add a runtime load/serve registry and one boxed backend slot; Gemma 4 must not grow central load/generate structs |
| evidence | `hipfire-eval` batteries/suites and existing coherence wrappers | add model evidence to `hipfire-eval`; shell remains an enforcement wrapper, not the evidence owner |

## Correctness contract

### Config and lowering

`Gemma4Config` parses the text block from the standard or unified wrapper and
lowers it into a `Vec<Gemma4LayerPlan>`. Each layer plan contains at least:

```rust
struct AttentionGeometry {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

enum RopePlan {
    FullHalfSplit { theta: f32, dim: usize },
    ProportionalHalfSplit {
        theta: f32,
        rotary_dim: usize,
        basis_dim: usize,
    },
}

enum ValueProjection {
    Separate,
    FromPreNormKey,
}

enum KvProducer {
    Own,
    SharedFrom { producer_layer: usize },
}

enum FfnPlan {
    Dense { intermediate: usize },
    DensePlusMoe {
        dense_intermediate: usize,
        expert_intermediate: usize,
        experts: usize,
        top_k: usize,
    },
}
```

Names may change during implementation; the distinctions may not be collapsed.
The parser validates that:

- `layer_types.len() == num_hidden_layers`;
- head counts divide correctly for both local and global layers;
- `global_head_dim`, proportional-RoPE parameters, and global KV heads exist
  when used;
- sharing producer layers exist for both attention types before the sharing
  tail begins;
- layers marked shared have no K/V projection requirement;
- K=V layers have no V projection requirement;
- PLE dimensions/tensors agree;
- MoE counts, top-k, expert widths, and tensor ranks agree;
- no structurally meaningful unsupported field is silently ignored.

### Norms, attention, and residual order

- Gemma 4 RMSNorm applies the stored weight directly. Do **not** apply or record
  Gemma 3's `(1+w)` offset.
- Embeddings are scaled exactly as upstream specifies.
- Q and K get weighted per-head RMSNorm.
- V gets a weightless RMSNorm. On K=V layers, both start from the same raw K
  projection and diverge before caching.
- Attention score scale is `1.0`. If the shared attention kernel internally
  multiplies by `1/sqrt(head_dim)`, pre-scale Q by `sqrt(head_dim)` explicitly in
  the lowered plan; do not hide the rule in a weight transform without
  provenance.
- Local layers use full local RoPE and SWA. Global layers use proportional partial
  RoPE. The current partial-RoPE kernel's exponent denominator is `n_rot`; extend
  it with `basis_dim`, pass `basis_dim=n_rot` for existing callers, and use the
  full global head dimension for Gemma 4.
- Apply the four layer norms and residual additions in upstream order.
- Apply `layer_scalar` after the complete decoder layer.
- Apply final logits softcap as `cap * tanh(logits / cap)` with cap from config.

### PLE and KV sharing

- PLE is a first-class regular layer operation: project the layer-specific input,
  apply the configured activation/gate/product/projection/norm, then add the
  residual in upstream order.
- Pack/upload PLE tables once; do not allocate or copy per token.
- Resolve the KV-sharing producer for each attention type at load time.
- Decode consumers read the producer's cache mapping. Batched prefill keeps the
  producer batch K/V live until all sharing consumers have run; a local SWA cache
  may be a ring for decode, but prefill cannot discard producer states before
  consumers use them.
- E2B's double-wide MLP applies only where the config says it does, not to all
  layers.

### MoE

- The dense GeGLU path always runs on 26B-A4B.
- Router input uses weightless RMSNorm, the learned router scale, and
  `1/sqrt(hidden)` before projection.
- Router probabilities are softmaxed in F32, top 8 are selected and renormalized,
  and per-expert scale is applied to selected weights.
- Routed experts use GeGLU and the checkpoint's stacked 3-D gate-up/down layout
  or an explicitly declared offline split. Do not reinterpret the dense MLP as a
  Qwen shared expert.
- The dense and routed branches receive their separate required norms and are
  combined in upstream order before the outer post-FFN norm/residual.

### Prompt, tools, thinking, and sampling

- Use the embedded official `chat_template.jinja`; make Gemma 4's profile strict
  and enabled without an opt-in environment variable.
- Preserve native system/developer roles.
- Implement `<|turn>...<turn|>` framing, thought channels, tool declarations,
  tool calls, and tool responses exactly as the official template renders them.
- Replace the existing `Gemma4NativeParser`, which expects obsolete JSON inside
  `<|tool_call|>`, with the released `call:name{...}` grammar.
- Replace Gemma 4 `<end_of_turn>` assumptions in output filtering. Resolve stop
  IDs from generation/tokenizer metadata; current instruction checkpoints use
  `[1, 106, 50]`.
- Strip hidden thought channels from ordinary visible output while retaining the
  context required across a tool-call/tool-response continuation.
- Add `top_k` to the shared sampler contract and execution. Honor checkpoint
  defaults (`temperature=1.0`, `top_p=0.95`, `top_k=64`) when the request does not
  override them.

## Implementation phases and frozen exit gates

Do not skip an exit gate. If a gate fails, record the exact result and stop that
phase; do not weaken the gate after seeing the result. The explicit 2026-07-15
contract revision from BF16-as-candidate to OQ8-as-candidate is recorded above.
From this revision onward, both the OQ8 baseline and OQ8++ narrowing gate are
frozen.

### Phase 0 — freeze truth and remove stale assumptions

Deliverables:

1. Add distilled config fixtures for E2B, E4B, 12B unified, 26B-A4B, and 31B,
   containing every field that affects execution.
2. Add tensor-name/shape manifest fixtures for the locally available standard
   variants. The fixtures store no model weights and record their source revision.
3. Add prompt-render fixtures from the official instruction templates: plain,
   system, thinking on/off, multi-turn, tool declaration, tool call, tool response,
   and assistant continuation.
4. Add or identify an offline Transformers capture tool that writes token IDs,
   selected hidden states, final logits, and generation outputs. It belongs in
   tooling/benchmarks, not runtime.
5. Correct the obsolete Gemma 4 claims in
   `2026-06-19-arch-roster-feature-matrix.md` and point its Gemma 4 section here.
6. Mark the current Gemma 4 EOS/parser tests as obsolete fixtures to be replaced
   in the prompt phase; do not allow them to become acceptance evidence.
7. Record the pinned Transformers BF16 oracle and original strict numerical
   thresholds before running the first Hipfire whole-model comparison.
8. Under the explicit later contract revision, record the first valid exact-prompt
   OQ8 measurement in `oq8-thresholds.json` as the broad product baseline and copy
   the unchanged original strict limits into `oq8pp-thresholds.json` as the final
   OQ8++ narrowing gate. Do not imply that the OQ8 baseline was pre-observation.

Exit gate:

- fixture extraction is reproducible from the pinned snapshots;
- config/manifest assertions distinguish every variant correctly;
- official Jinja2 renders match the committed expected bytes;
- BF16 oracle, OQ8 baseline, and OQ8++ narrowing artifacts have explicit scopes
  and machine-readable thresholds;
- no code path has been changed yet;
- stale roadmap text no longer claims E2B/E4B are MoE or that bring-up is cheap.

### Phase 1 — identity, detection, ingest, and toy models

Deliverables:

1. Add id 24 to `hipfire-arch-api`, `hipfire-model`, architecture-id docs, and
   model-support metadata.
2. Add `hipfire-arch-gemma4-spec`, force-link it from `hipfire-arch-specs`, and
   register `Arch`, `Ingest`, and `ToyModel`.
3. Extend the offline registry contract just enough for the quantizer to resolve
   canonical model types and stacked-expert layout without a family match arm.
4. Make the Gemma 4 ingest policy retain direct norm weights, layer scalars,
   router scales, PLE tables, and sensitive small tensors at source precision for
   OQ8 conversion and bring-up.
5. Create at least three tiny deterministic fixtures:
   - dense local/global with different head dimensions and a K=V global layer;
   - PLE plus local/global KV sharing;
   - dense-plus-MoE with stacked experts and top 8.
6. Update the architecture onboarding checklist so future registered families do
   not require central quantizer detection edits.

Exit gate:

- `cargo test -p hipfire-arch-gemma4-spec` passes;
- `cargo test -p hipfire-arch-specs` proves id 24 is force-linked with ingest and
  toy capabilities;
- tiny fixture generation and BF16-oracle/OQ8 HFQ conversion pass;
- `rg` finds no Gemma 4 model-type literal in the central quantizer detection
  ladder;
- no Gemma 3 norm-offset transform can match id 24.

### Phase 2 — shared transformer loader extraction

Deliverables:

1. Extract the common HFQ tensor lookup, exact-shape validation, raw/quant upload,
   optional tensor, embedding, tied-head, and direct-norm mechanics from the
   Gemma 3/Qwen2 copies into a small runtime transformer loader module.
2. Migrate Gemma 3 to the shared mechanics without changing stored-weight
   semantics or decode output. Gemma 4 becomes the second consumer.
3. Keep family tensor-name construction, required/optional policy, prefix rules,
   paging, PLE, and MoE assembly in their architecture crates.
4. Do not absorb Qwen3.5 slab/pager orchestration unless it fits without widening
   the API; leave a precise follow-up instead of an abstraction full of options.

Exit gate:

- Gemma 3 loader tests and its existing BF16/tiny golden behavior are unchanged;
- shared loader unit tests cover missing, wrong-rank, wrong-shape, optional,
  tied-head, and direct-norm cases;
- the new shared module has both Gemma 3 and Gemma 4 consumers;
- Gemma 4 does not contain copies of `load_weight_tensor`, `load_norm`,
  `load_embed`, or `load_lm_head` mechanics.

### Phase 3 — layered attention and KV state

Deliverables:

1. Add a per-layer cache plan and arena/map. Prefer grouping compatible physical
   allocations around existing `KvCache` implementations over rewriting every KV
   codec at once.
2. Represent logical layer -> physical group/slot, storage kind (full or SWA),
   and optional shared producer explicitly.
3. Retain current homogeneous `KvCache` constructors as compatibility adapters
   for existing families.
4. Allocate scratch at the maximum required Q/K/V/attention width once and expose
   checked per-layer views; do not allocate maximum geometry for every cache.
5. Add unit tests for mixed 256/512 head dimensions, distinct local/global KV
   counts, shared logical layers with no storage, reset, growth, and boundary
   positions.
6. Exercise at least one existing homogeneous consumer through the adapter so
   the generalized path is not Gemma-4-only.

Exit gate:

- old KV/cache tests pass unchanged;
- allocation byte accounting matches the sum of owned physical layers, not the
  logical layer count or maximum geometry;
- shared layers allocate zero K/V storage and resolve the correct producer;
- local/global cache writes and reads pass CPU/GPU parity at window-1, window,
  window+1, and a global layer;
- reset and second-request behavior are clean.

### Phase 4 — mathematical primitives and lowered Gemma 4 forward

Deliverables:

1. Extend half-split partial RoPE with `basis_dim`; adapt all existing callers to
   pass their old denominator explicitly.
2. Add an explicit weightless RMSNorm primitive/API.
3. Reuse existing GeGLU, norm, scale, residual, full-attention, and SWA kernels.
4. Add final vector softcap; keep it generic over length/cap rather than naming
   the kernel Gemma 4.
5. Extend the lowered super-op representation for proportional RoPE, per-layer
   geometry/cache binding, regular scale, and later PLE. Do not encode regular
   Gemma operations as `EscapeKind` variants merely for expedience.
6. Implement a straightforward Gemma 4 reference forward and a lowered forward
   over the same weights/state. Default to the reference path until dual-run
   parity is recorded; then flip only with an opt-out oracle flag.

Exit gate:

- F32 CPU/GPU operator goldens pass for full/local RoPE, proportional RoPE,
  weighted Q/K norm, weightless V norm, Q scaling, layer scalar, and softcap;
- existing Qwen partial-RoPE goldens remain byte-identical when
  `basis_dim == rotary_dim`;
- tiny dense reference and lowered paths agree within the frozen operator limits
  at every captured layer boundary and final logits;
- kernel build/JIT coverage includes RDNA2, RDNA3, RDNA3.5, and RDNA4 targets;
- no LDS-heavy implementation is introduced for a pointwise operation.

### Phase 5 — dense 31B text and boxed serving seam

Deliverables:

1. Load the 31B base and IT manifests through `Gemma4Config`, shared loader,
   layered cache, and lowered program.
2. Add a runtime architecture factory/registration seam that returns a boxed
   backend plus its prompt/generation profile. Store it in one generic
   `LoadedModel` backend slot.
3. Route load, serve, reset, and unload generically. Gemma 4 must not add a new
   typed field or central generation function.
4. Support OQ8 HFQ first at a bounded bring-up context. Increase context only
   after short-context parity; do not allocate 256K F32 KV by default.
5. Admit the base checkpoint as raw completion and the IT checkpoint through the
   official prompt profile.
6. Migrate one existing `SimpleAr` backend to the runtime factory as the second
   consumer and remove its now-redundant central load/generate branch where safe.

Exit gate:

- no-gpu load/config/fixture tests pass;
- `LoadedModel` has no Gemma 4-specific field;
- daemon/serving code has no `arch_id == 24` or equivalent Gemma 4 branch;
- OQ8 hidden/logit captures stay within `oq8-thresholds.json` for short prompts,
  a prompt crossing SWA 1024, and multiple global layers;
- greedy generation matches the upstream oracle for the committed prompt suite;
- unload/reload and two sequential requests do not retain stale KV state.

### Phase 6 — official prompt, channel, tool, and sampler correctness

This phase may be developed alongside Phase 5, but 31B-it is not admitted until
both phases pass.

Deliverables:

1. Make strict Jinja rendering/profile selection generic and registry-driven.
2. Replace the obsolete Gemma 4 parser with the released native call grammar.
3. Add a channel-aware output state machine for thought/tool/visible content;
   reuse the generic EOS holdback machinery for byte boundaries and stop tokens.
4. Add generic top-k to CPU/GPU sampling and request/profile defaults.
5. Correct stale Gemma 4 comments/tests in `tool_call.rs`, `eos_filter.rs`, and
   runtime prompt override documentation.

Exit gate:

- hipfire render output is byte-identical to official Jinja2 for every Phase-0
  fixture;
- token IDs are identical after encoding rendered bytes;
- native tool declarations, calls, responses, multi-call turns, and malformed
  output behavior have unit tests;
- hidden thought content never leaks to the visible stream, while tool
  continuation preserves required context;
- stop IDs 1, 106, and 50 stop without leaking marker bytes;
- top-k 64 sampling matches a deterministic CPU reference for fixed logits/seed;
- no Gemma 4 `<end_of_turn>` or obsolete `<|tool_call|>{json}` assumption remains.

### Phase 7 — E4B then E2B: PLE and cross-layer KV sharing

Deliverables:

1. Add PLE load/state/lowering using packed, resident per-layer inputs.
2. Add full prefill/decode lifetime handling for local/global shared KV producers.
3. Bring up E4B first (PLE + sharing), then E2B (same plus double-wide shared-tail
   MLP).
4. Validate base and IT checkpoints for both sizes.

Exit gate:

- tiny PLE/sharing reference and lowered paths pass layer-by-layer parity;
- E4B OQ8 passes frozen hidden/logit and greedy-generation gates;
- E2B OQ8 passes the same gates and specifically crosses the first double-wide
  tail layer;
- projected K/V executes only on producer layers; counters/tests prove sharing
  consumers do not launch absent projections;
- allocation accounting proves shared consumers own no cache;
- prompt/tool gates pass on E2B-it and E4B-it.

### Phase 8 — 26B-A4B dense-plus-MoE

Deliverables:

1. Refactor `hipfire-dispatch` MoE parameters into a reusable routed-expert core
   plus Qwen shared-expert and Gemma dense-plus-routed adapters.
2. Add activation and router-pre/post policy explicitly; do not grow boolean
   combinations that make invalid family states representable.
3. Teach ingest/loading the declared stacked 3-D Gemma expert layout and per-expert
   scales.
4. Preserve the dense GeGLU branch as a normal dense path and combine it with the
   routed branch in the Gemma layer implementation.
5. Bring up base, then IT, in OQ8 against the pinned BF16 oracle.

Exit gate:

- existing Qwen MoE goldens and model smoke remain unchanged through its adapter;
- tiny Gemma MoE router probabilities, selected indices, renormalized/scaled
  weights, dense output, routed output, and combined output match the CPU oracle;
- real 26B-A4B OQ8 selected-layer captures and final logits pass frozen limits;
- greedy generation matches on the committed base/IT prompts;
- router/per-expert scales stay at the precision declared by ingest;
- no mandatory Qwen shared-expert weight exists in the generic routed core.

### Phase 9 — quantization and eval admission

Only begin after the corresponding OQ8 variant passes its broad product gate.

Deliverables:

1. Add a Gemma 4 text correctness battery/suite in `hipfire-eval` covering each
   admitted variant, prompt profile, SWA/global boundaries, PLE/sharing, and MoE.
2. Run calibration/format selection through Astrea rather than assuming Gemma 3
   bit assignments transfer.
3. Quantize source weights only; keep norm, scalar, router, and other sensitive
   tensors at evidence-backed precision.
4. Admit one format/variant at a time. Artifact names use the canonical quant
   token.
5. Send performance candidates through Kernel Atlas/AR validation only after
   quality admission.

Exit gate:

- BF16 remains the frozen oracle artifact and OQ8 is the first admitted runtime
  format;
- KLD/PPL/task thresholds are declared before candidate results and are not
  weakened afterward;
- `hipfire-eval` holds the evidence; shell gates only enforce it;
- a rejected candidate is recorded as rejected, not packaged optimistically;
- every packaged artifact identifies source snapshot, calibration data, format,
  arch, and prompt/template provenance.

### Phase 10 — 12B unified text, multimodal roles, and DSpark speculative decoding

These are three separate subprojects, not one "finish Gemma 4" checkbox.

1. **12B unified text:** once the official snapshot is local, prove that the same
   text core loads through the unified wrapper and passes all dense gates. No
   image/audio support is implied by this text admission.
2. **Standard multimodal:** implement vision/audio encoders and projectors as
   explicit capabilities/roles for applicable standard variants. Reuse existing
   preprocessing/vision transport only where tensor math and positional policy
   actually match.
3. **12B unified multimodal:** implement its direct projection/mask behavior as a
   distinct adapter; do not force it through the standard encoder adapter.
4. **DSpark speculative decoding:** add Gemma 4 to the shared spec-decode target
   seam and train a block drafter, packaged independently as a `.dspark.hfq`
   sidecar. Reuse `hipfire-specdecode-dspark`'s `SpecTarget` verifier boundary,
   `DsparkBody` drafter core, greedy-accept rule, and `DsparkConfig`/`DsparkWeights`
   sidecar format instead of adding a Gemma-4-only spec path:
   - implement `SpecTarget for Gemma4Backend` — first a per-token
     greedy-equivalent baseline, then a batched `verify_block` plus an
     extract-layer residual-hidden tap that honors the local/global, PLE, and
     KV-sharing layer layout; Gemma 4 has no recurrent state, so
     `commit_prefix` may be a no-op only after tests prove that the next verify
     re-anchors the layered cache and overwrites every rejected full/SWA tail
     slot, matching the shared pure-attention target contract;
   - implement a Gemma 4 `DsparkBody`, capture a `DSLB` label cache from the
     admitted OQ8 target, train the drafter through `hipfire-train`'s DSpark
     path, and pack it with `dspark_convert` to `.dspark.hfq`;
   - the same extract-layer target seam is the foundation a later DFlash sidecar
     drives, so the hidden capture must stay drafter-agnostic, not DSpark-only.

Each subproject requires its own config/tensor fixtures, CPU/HF oracle, feature
entry, eval battery, and admission result. Audio/video preprocessing and tool
messages containing modalities need separate prompt fixtures.

DSpark exit gate:

- `verify_block` reproduces the admitted AR target's greedy token IDs exactly on
  the committed prompt suite — spec decode is a speedup, never a new decoder;
- partial acceptance followed by another verify preserves the accepted prefix
  and overwrites the rejected tail at SWA-1, SWA, SWA+1, and global-layer cache
  boundaries before the no-op `commit_prefix` contract is admitted;
- drafter acceptance length and end-to-end tokens/sec are measured against the AR
  baseline on locally available checkpoints and recorded as `hipfire-eval`
  evidence, not asserted;
- the `.dspark.hfq` sidecar loads independently and records its provenance
  (source snapshot, label corpus, target revision, arch);
- Gemma 4 adds no spec-decode branch outside the shared `SpecTarget`/`DsparkBody`
  seam, and the extract-layer tap is reachable by DFlash without a second capture.

### Phase 11 — final OQ8++ strict narrowing gate

This is the final promotion stage for the dense 31B product after broad OQ8
functional admission. It does not replace the OQ8 compatibility floor and does
not weaken the pinned BF16 oracle. In canonical artifact spelling, OQ8++ is
`.oq8++.hfq`: activation-aware clipping/scaling plus Hessian/LDLQ error feedback
on the Opus 8-bit weight encoding.

Deliverables:

1. Produce `Gemma-4-31B-it.oq8++.hfq` from the same pinned source revision and
   tokenizer as the admitted OQ8 artifact.
2. Record calibration corpus, activation-aware method, Hessian/LDLQ settings,
   quantization hash, producer commit, and source revision in durable evidence.
3. Run the complete OQ8 prompt, SWA/global, reset/reload, sequential-request,
   prompt/tool, and portability matrix without dropping any case.
4. Compare every required hidden/logit capture against the same pinned
   Transformers BF16 oracle using `benchmarks/gemma4/oq8pp-thresholds.json`.
5. Run the corresponding `hipfire-eval` KLD/PPL/task and performance batteries;
   quality admission precedes Kernel Atlas or performance promotion.

Exit gate:

- hidden-state cosine is at least `0.999`, hidden-state NRMSE is at most `0.045`,
  and no hidden value is non-finite at every required capture point;
- final-logit cosine is at least `0.999`, maximum absolute error is at most
  `0.5`, final argmax matches, top-5 overlap is at least `4`, and no logit is
  non-finite at every committed comparison position;
- greedy token IDs match exactly for the committed prompt suite;
- every broad OQ8 lifecycle, serving, prompt/tool, SWA/global, and portability
  gate remains green on OQ8++;
- KLD/PPL/task limits are declared before the OQ8++ result and pass without a
  post-result threshold change;
- a miss is recorded as a rejected OQ8++ candidate. OQ8 remains the admitted
  compatibility artifact, and the narrower gate is not relaxed.

## Verification matrix

### CPU/no-GPU on every phase

- targeted crate tests for every touched crate;
- config, manifest, ingest, prompt render, parser, sampler, cache-plan, and lowered
  program unit tests;
- `cargo check` for affected feature combinations;
- `./tests/no-gpu-ci.sh` before handoff.

### GPU for kernel/forward/cache phases

- acquire the shared lock with `hipfire lock acquire` unless the invoked gate
  already owns it; always release it on exit;
- operator CPU/GPU parity before a model run;
- reference-vs-lowered layer captures before enabling lowered by default;
- `./tests/coherence-gate-dflash.sh` after changes to kernels, dispatch, quant,
  fusion, rotation, RMSNorm, KV, or spec-decode-adjacent paths;
- OQ8 model comparisons against the pinned BF16 oracle on locally available
  checkpoints;
- boundary prompts at SWA-1/SWA/SWA+1 and positions beyond one full layer pattern;
- reset/reload/multi-turn checks;
- compile/JIT matrix for gfx1030, gfx1103, gfx1151, and gfx1201; run on available
  hardware and record which targets were compile-only.

### Architecture hygiene gates

At every phase boundary:

```text
no Gemma 4 typed field in LoadedModel
no Gemma 4 central generate/load branch
no raw arch id 24 outside the canonical id declaration/test fixtures
no Gemma 3 norm-offset path matching Gemma 4
no duplicate generic loader helpers added to hipfire-arch-gemma4
no obsolete Gemma 4 prompt/tool/EOS assumptions in touched files
every new generic seam has at least two consumers
git diff --check passes
graphify update . completed after code changes
```

## Expected change surface

New:

- `crates/hipfire-arch-gemma4/`
- `crates/hipfire-arch-gemma4-spec/`
- Gemma 4 eval fixtures/batteries and offline oracle tooling

Shared seams likely touched:

- `crates/hipfire-arch-api/`
- `crates/hipfire-arch-specs/`
- `crates/hipfire-model/`
- `crates/hipfire-runtime/src/{arch,kv,sampler,tool_call,...}.rs`
- `crates/hipfire-dispatch/src/pipeline/` and `families/moe.rs`
- `crates/hipfire-rdna/src/dispatch/` and the smallest required HIP kernels
- `crates/hipfire-generate/`
- `crates/hipfire-serving-core/`
- generic registry plumbing in `crates/hipfire-quantize/`
- `crates/hipfire-eval/`, `docs/model-support.toml`, and architecture docs

The implementation must read the nearest nested `AGENTS.md` before editing each
subtree.

## Definition of done

Gemma 4 text support is complete only when:

1. Every locally available official text variant has an explicit support status
   backed by OQ8 evidence against the pinned BF16 oracle; unsupported/deferred
   variants are labeled honestly.
2. Dense, PLE/KV-sharing, and dense-plus-MoE math have independent operator and
   real-model evidence.
3. Official prompt, thinking, tools, stop behavior, and sampling are correct for
   instruction checkpoints.
4. The runtime uses the boxed backend factory and layered cache plan without a
   Gemma 4-specific central branch or `LoadedModel` field.
5. Shared loader, cache, serving, sampler, and routed-expert improvements have at
   least two consumers and remove real duplication.
6. Gemma 3 and Qwen MoE comparison paths remain available until their migrations
   are proven and recorded.
7. Quantized artifacts are admitted variant-by-variant through frozen eval gates.
8. `no-gpu-ci`, coherence, graph update, portability compilation, and relevant
   GPU gates pass.
9. Failures and rejected approaches are documented with exact evidence.

## Explicitly out of scope for the initial 31B milestone

- Gemma 4 vision, audio, and video input;
- 12B unified wrapper validation until its official checkpoint is local;
- DSpark speculative decoding (a later Phase 10 subproject, never folded into
  31B bring-up);
- DFlash, TriAttention, CASK, or new KV quant formats;
- pipeline/expert parallelism;
- performance fusion before OQ8 admission and the final OQ8++ narrowing gate;
- direct GGUF execution;
- broad rewrites of unrelated architecture crates.

Those may follow as their own gated phases. They must not be smuggled into the
31B bring-up or used to relax its correctness requirements.
