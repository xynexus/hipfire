# BLS Mini Code model induction

Status: layered resident serving plus native calibration/CASK implemented; full-model parity and admission open

Owner: architecture bring-up / calibration / CASK

Related: [bundled model induction program](2026-07-22-bundled-model-induction-program.md)

## Source facts

The local source snapshot is
`/srv/huggingface/models--CohereLabs--BLS-Mini-Code-1.0`. Its config declares:

- `Cohere2MoeForCausalLM` / `cohere2_moe`;
- 49 layers, hidden size 2,048;
- 32 query heads, 4 KV heads, head dimension 128;
- sliding window 4,096; and
- 128 routed experts with top-8 routing.

These are planning inputs, not a hard-coded runtime allowlist. The checked-in
registry/spec must own the final architecture identity and tensor policy.

## Current failure mode

The unsafe identity fallbacks and missing calibration producer are fixed. The
remaining gap is full-model evidence: arch 25 now advertises a bounded resident
`ServingFactory`, but the 61 GB source checkpoint has not yet received trusted
full-model BF16/logit/output parity. Consequently the real calibration and CASK
artifacts below are producer evidence, not permission to label an `oq4.25++`
candidate as admitted.

## Implementation evidence: 2026-07-22

Cohere2-MoE now owns stable arch ID 25 in `hipfire-arch-api`, a lean
`hipfire-arch-cohere2-spec` registration for `cohere2_moe`, a split-expert
ingest policy, and a deterministic two-layer fixture that models BLS's dense
first layer, parallel attention/FFN block, sigmoid non-normalized top-k router,
and per-expert gate/up/down tensors. The support matrix deliberately reports no
serving path until BF16 parity exists.

Both unsafe fallbacks are removed. The quantizer refuses missing/unknown
`model_type` instead of treating it as LLaMA, and `SafetensorsSource` resolves
linked specs then returns an explicit invalid-source error instead of defaulting
to Qwen3.5.

The family crate now registers `cohere2-moe-stream-v1`. It streams the real
49-layer checkpoint one layer at a time and executes the production-form
parallel attention/MLP block: one pre-norm, GQA attention, adjacent-pair full-
dimension RoPE on rotated layers, raw-logit top-k routing, non-renormalized
sigmoid route weights, and sequential selected-expert SwiGLU accumulation.
Layer 0 uses the dense FFN. Twelve later full-attention MoE layers are explicitly
unrotated; the remaining attention layers use the configured interleaved RoPE.
The adapter captures attention, dense, router, and admitted expert inputs and
emits per-layer CASK metadata with those exact context/RoPE policies.

The registry/spec tests pass, and the deterministic BLS fixture quantizes to an
arch-25 OQ4 HFQ with all 28 tensors classified (including Q8 router and
compressible split experts). This proves identity and quantizer ingress only;
it is not BF16 runtime or `oq4.25++` admission evidence.

A real uninterrupted one-sequence/two-token producer smoke completed all 49
layers. Its ledger consumed all 18,731 planned logical reads exactly once
(18,730 canonical source tensors, no missing or duplicate reads), totaling
60,968,607,744 source bytes. The output contains 247 Hessian tensors and 2,290
imatrix tensors with maximum consistency error zero. Its canonical HFQM CASK
sidecar is arch 25, schema `hipfire.triattn.v2`, and contains one heterogeneous
F32 center tensor for each of the 49 physical attention layers. A one-row
expert quota intentionally leaves most experts undercovered and KLDREF was
disabled, so this smoke is not a product calibration or quantization gate.
The post-fix structural audit is valid with no errors or warnings
(`fnv64:cd3e7af78abb8191`, 2,537 tensors). The 1,224,704-byte CASK artifact is
`fnv64:cd71bbfcbaca6f13`: 36 sliding and 13 full-context records, with 37
interleaved-RoPE and 12 explicitly unrotated layers.

The shared serving factory now loads arch-25 HFQ embedding, norms, attention,
router, split experts, and tied output weights; executes the same
`execute_cohere2_row` primitive used by native calibration; exposes `SimpleAr`
and KLD evaluation; and unloads all owned GPU state. Resident serving now uses
the common `LayeredKvArena`: full-attention layers own bounded full caches,
sliding layers use head-major rings and the established visibility-stage/SWA
attention kernels, and CASK attaches a `LayeredEvictionCtx` to compact only the
full groups. The no-CASK path bounds logical and physical context together;
the CASK path keeps the requested logical context while allocating the common
budget/beta-derived physical cap. Arch/layer identity mismatches fail closed.
The same two-layer OQ4 fixture now passes loose-versus-embedded CASK daemon
loading, four-token generation, and unload with identical completion events.
Its synthetic tokenizer emits empty decoded text, so full-model readable-output
parity and long-context admission are still required before promotion.

The deterministic two-layer arch-25 fixture loaded and generated four tokens
through the daemon in both BF16 and OQ4 containers before the layered-cache
conversion. A teacher-forced five-token
comparison produced four finite logit rows; OQ4 versus BF16 had maximum absolute
drift `0.29889554`, mean absolute drift `0.06701695`, and 3/4 top-1 agreement.
That is a serving-path smoke, not a BLS quality threshold or full-model parity
claim.

## Scope

Bring up BLS as a registered Cohere2-MoE family through BF16 runtime parity,
quantizer ingress, native streamed calibration, CASK generation, and induction.
Do not implement calibration or quantization against an unverified reused model
class.

## Phase 1: identity and source contract

1. Reserve a stable architecture ID in `hipfire-arch-api`.
2. Add Cohere2 spec/registry entries and a dedicated architecture crate or a
   proven shared implementation behind an arch-owned adapter.
3. Register model-type/architecture aliases through the registry; remove the
   unknown-family fallback for this source.
4. Audit the real Safetensors index into a logical tensor plan, including
   embedding/lm-head tying, the absence of Q/K normalization, interleaved full-dimension RoPE,
   sliding attention, router, expert, and shared-expert tensors.
5. Add config and tensor-shape fixtures from the local snapshot without making
   `/srv/huggingface` a runtime dependency.

## Phase 2: BF16 runtime parity

Implement config parsing, weight loading, state, prefill/decode, logits, and
boxed serving through existing `Architecture`, `ServingFactory`,
`ServingBackend`, and dispatch pipeline seams. Do not add a BLS `Option` field
to central loaded-model structs.

Required evidence:

- tokenizer and prompt fixture parity;
- layer/operator parity for attention, RoPE, router, selected experts, and FFN;
- finite full-model logits and bounded drift against a trusted oracle; and
- coherent greedy output before any quantization work.

## Phase 3: quantizer ingress

Add registry-driven identity and stacked/split expert layout policy. The
quantizer must preserve sensitive tensors according to measured policy and emit
the reserved arch ID. Add fixture-golden coverage for plain and calibrated OQ
formats before running the large model.

The first product candidate requested by induction is `oq4.25++`, using an
audited calibration artifact. Do not infer `+`/`++` from a filename when AWQ or
LDLQ inputs were absent.

## Phase 4: native calibration adapter

Implemented through `CalibrationFamilyAdapter` in the family crate, following the
registered Qwen3.5/Gemma3 contract:

- exact model inspection and resource estimate;
- deterministic tensor read plan and aliases;
- layer-streamed embedding, layer, and finalizer loads;
- attention/FFN/router/expert capture registry;
- routed-expert quotas and undercoverage policy;
- matched-corpus KLDREF generation; and
- complete read-ledger and artifact audit.

The adapter must use the production forward math. A calibration-only
approximation is not sufficient for `oq4.25++` admission.

## Phase 5: family-neutral CASK producer

Implemented by replacing the Qwen-specific `triattn_validate` orchestration with a registered,
offline Rust producer, preferably under `hipfire-coexistence`. The common
producer owns corpus iteration, accumulation, validation, serialization, and
evidence. The Cohere2 adapter owns:

- eligible attention layers;
- per-layer Q geometry and RoPE convention;
- the pre-RoPE query tap; and
- reset/prefill behavior.

Sliding-window layers must be represented explicitly. Do not serialize a
uniform geometry merely because legacy TRIA requires one.

## Phase 6: induction and packaging

Extend induction only after BF16, calibration, and CASK gates pass. With no
DFLASH source supplied, the canonical candidate is:

```text
BLS-Mini-Code-1.0.triattn.oq4.25++.hfq
```

The calibration artifact remains separate evidence and is not embedded in the
shipping model unless a later policy explicitly requires it.

## Gates

1. Unknown Cohere2 fails before the registry work; registered Cohere2 never
   resolves through LLaMA/Qwen fallback.
2. No-GPU config, tensor-plan, registry, tokenizer, and quant fixture tests.
3. BF16 operator, logit, prompt, and output parity.
4. Native calibration dry-plan, one-layer smoke, resume smoke, full audit, and
   resident-oracle comparison.
5. `oq4.25++` finite-logit, coherence, matched KLD/PPL, and size evidence.
6. CASK center validation plus long-context retrieval/recall.
7. Bundled-versus-separate parity after component runtime support lands.
8. Atlas AR rows before promotion.

## Non-goals

- Do not add a generic "unknown MoE is close enough" path.
- Do not claim DFLASH support; no BLS DFLASH source was supplied.
- Do not weaken OQ or CASK admission thresholds after reading results.

## Definition of done

BLS has a registered BF16 runtime, audited calibration artifact, measured
`oq4.25++` candidate, family-owned CASK adapter, and a loadable canonical bundle
with explicit pass/reject evidence.
