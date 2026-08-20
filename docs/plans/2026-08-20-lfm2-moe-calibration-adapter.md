# Scope: upgrading the LFM2-MoE machinery

**Deliverable of this doc:** a scope, not an implementation. Written after the
real-MoE `oq4.25++` measurement was blocked on LFM2.5-8B-A1B; see BUGS.md
"Calibration coverage: three open questions".

---

## The problem, concretely

```
$ hipfire-coexistence calibrate --model <LFM2.5-8B-A1B> ...
Error: InvalidSourcePlan("no native calibration adapter is registered for architecture 11")
```

**LFM2-MoE cannot be calibrated through the production path at all.** Only five
architectures register a `CalibrationFamilyAdapter` — qwen35, gemma3, zaya,
gemma4, cohere2 (`register_calibration_adapter!`). Arch 11 is not one.

That blocks every calibrated format for the family: `oq4+`, `oq4++`,
`oq4.25++`, `oq8+`, `oq8++` all require a Hessian or imatrix, and the second `+`
requires a real Hessian. So the premier quant format is unreachable for LFM2-MoE
on the production path.

### Three gaps, and they are related

1. **No calibration adapter** (above). Blocks calibrated formats.
2. **Routed-expert Hessians are imatrix-only, by memory constraint.**
   `calibration.rs` says so outright: *"routed expert tensors are imatrix-only
   because full per-expert Hessians do not fit for the 8B-A1B model."*
3. **No paged-expert support.** No `WeightPager` use anywhere in
   `hipfire-arch-lfm2moe`, so the whole model must be resident to serve. qwen35
   has paging; LFM2 does not.

(1) and (2) are the same gap seen twice — and **the adapter is what fixes (2)**,
which is the non-obvious part of this scope. See "Why the adapter is the lever".

### What already works — do not rebuild it

- `build_capture_names` covers conv `in_proj`/`out_proj`, attention
  `q/k/v/out_proj`, dense FFN `w1/w3/w2`, and the MoE `gate` router.
- Routed experts are captured explicitly in `forward.rs`, because the fused
  indexed kernels have no one-pointer-per-source-tensor mapping: gate and up are
  byte-fused into `gate_up` while the calibration package needs checkpoint-style
  `w1` / `w3` names. **That fused-vs-split problem is already solved here** — and
  it is the same class as the gemma4 bug in PR #252, solved correctly on this
  family first.
- Indexed MoE GEMVs exist (`gemv_hfq6g256_moe_gate_up_k8_indexed_batched`).
- The tiny gate runs five calibrated `lfm2_moe` cells today, via the harness's
  own collector — which is exactly why this gap was invisible until now (BUGS.md
  §3b: the harness calibrates families production cannot).

---

## Target: register a `CalibrationFamilyAdapter` for arch 11

The trait is ten methods (`calibration/stream.rs:437`):

| method | character |
|---|---|
| `family`, `adapter_version` | trivial |
| `resource_estimate`, `effective_precision` | arithmetic over config |
| `inspect` | read geometry from a `ModelSource` |
| `capture_plan` | **the interesting one** — which tensors, which capture mode |
| `cask_metadata` | pre-RoPE Q geometry; `None` is a legal answer with an explicit refusal |
| `load_embedding`, `load_layer`, `load_finalizer` | **the substance** — stream one unit at a time |

### Why the adapter is the lever for the per-expert Hessian problem

The `load_*` trio is a *streaming* interface: the engine asks for one layer,
calibrates it, and drops it. That bounded residency is precisely what the
current collector cannot do — it arms a whole-model forward and holds every
accumulator at once, which is why per-expert Hessians "do not fit".

So implementing the adapter is not only about unblocking the CLI. It is the
mechanism by which **full per-expert Hessians for LFM2-MoE become affordable**,
upgrading the family from imatrix-only to real `++`. That should be stated as a
hypothesis to test, not a promise — see the exit criteria.

### Effort: most of this is ALREADY written — corrected

The first draft of this section framed the work as writing ~1800–2400 lines of
new machinery, extrapolated from `wc -l` on the existing adapters. **That framing
is wrong and would have led someone to build the wrong thing.**

**A shared calibration library already exists, and it is larger than every
adapter combined:**

| shared module (`hipfire-runtime/src/calibration/`) | lines |
|---|---|
| `layer_stream.rs` | 3991 |
| `contracts.rs` | 2161 |
| `source.rs` | 1582 |
| `expert_capture.rs` | 1068 |
| `boundary.rs` | 722 |
| `stream.rs` | 634 |
| `residual_probe.rs` | 298 |
| **total** | **10,720** |

against **8,745** lines across all five adapters. And the adapters genuinely
consume it — qwen35 imports shared tensor loaders and validators
(`load_source_matrix`, `validate_source_shape`, `PlannedTensorReader`), shared
capture contracts, the shared **`GroupedMoeCalibrationCapture`** for routed
experts, and the shared **`RmsNormLmHeadFinalizer`**, which both gemma4 and
qwen35 use rather than writing their own `load_finalizer` body.

So an adapter is not 2000 lines of framework. It is arch-specific knowledge
wired into an existing framework — which is why M1/M2 below are scoped as
"reuse the naming that already exists" rather than "implement capture".

### Is there a second shared library to extract? Measured: mostly no

Eleven helpers recur by name across all three adapters sampled
(`prepare_capture`, `register_capture`, `write_capture_part`, `release_states`,
`push_required`, `prepare_sequence_group`, …), none of them trait methods. That
looks like duplication, so it was measured rather than assumed.

In gemma4's adapter those six total **218 of 1247 lines — 17%**. And the bodies
are not copies: `write_capture_part` shares a seven-line prefix with qwen35's and
then diverges into expert telemetry that only a MoE arch has. The largest of them,
`register_capture` at 115 lines, is arch-specific *by nature* — it names this
family's tensors.

**Conclusion: the extraction the residual duplication would buy is ~17% of one
adapter, much of which is not actually shareable.** Not worth blocking LFM2 on,
and worth doing only as an independent cleanup with its own justification. The
big extraction already happened; this doc's job is to reuse it, not to redo it.

### Read zaya's body: the corollary was right, and there is a bigger finding

The corollary above (hybrid complexity, not boilerplate) holds. But reading the
file changes the recommendation, so it is recorded here rather than left as a
line-count inference.

Where zaya's 2410 lines actually go:

| item | lines |
|---|---|
| `impl ZayaStreamedCalibrationLayer` | 566 — of which **`forward_position_slice` is 411** |
| `zaya_resource_estimate` | 321 |
| `impl CalibrationLayer` | 212 |
| `impl CalibrationFamilyAdapter` | 174 |
| `load_block_weights` | 151 |
| `zaya_tensor_requests` | 123 |
| everything else | the remainder |

`zaya_resource_estimate` is 321 lines of genuinely arch-specific memory
arithmetic — per-sequence KV, the CCA conv ring and its delayed value, router
and expert widths, position-slice scratch. Nothing generic to extract; it is the
direct analogue of what LFM2 would need for `conv_L_cache`.

**The finding that matters: `forward_position_slice` re-implements the
architecture's forward pass.** It calls kernels directly (`gpu.rmsnorm_batched`,
the CCA attention chain, the MoE chain) with capture taps interleaved. And the
adapter imports exactly two things from its own crate — `ZayaConfig` and
`ARCH_ID_ZAYA`. It does not use zaya's serving forward at all; `gpu.rs` (3623
lines) does not even share the same kernel set.

So **every calibrated arch carries TWO forward implementations**: the serving
one, and a position-sliced one inside its calibration adapter, agreeing only via
a config struct. That is a far larger duplication than the 17% measured between
adapters — and it is *the* reason an adapter costs what it costs.

### What that means for LFM2 — and for whether to write it at all

Writing an LFM2 adapter means writing a second LFM2 forward: 18 double-gated
short-conv layers and 6 GQA layers, sliced by position, with taps. The existing
`forward.rs` cannot be reused as-is, exactly as zaya's could not.

**This is the duplication the v2 daemon plan already proposes deleting.**
`docs/plans/2026-08-09-v2-daemon-module-major-multistream.md` §C makes
calibration a *tap* on the single lowered march — "calibration / imatrix →
`ActStatTap` pre-`Proj`", eliminating `CalibrateDaemonSession` and
`handlers/calibrate.rs` — precisely because the lowered forward is data
(`Vec<SuperOp>`) rather than control flow, and data can be streamed and tapped
without being rewritten per consumer.

So there is a real sequencing decision here, and it should be made deliberately:

* **Write the adapter now.** Unblocks LFM2-MoE calibrated formats today, at the
  cost of adding a *sixth* instance of the duplication v2 intends to remove.
* **Wait for the v2 tap.** No new duplication, but LFM2-MoE stays uncalibratable
  until a large plan lands, and v2's own first target is qwen35, not LFM2.
* **Write it deliberately as throwaway.** Take the adapter, and record in it that
  `forward_position_slice` is expected to be deleted when calibration becomes a
  tap — so the next reader knows it is scaffolding rather than architecture.

The third is probably right if LFM2-MoE calibration is wanted soon, but that is a
judgement about priorities, not something this doc should settle unilaterally.
**What this doc does settle is that the cost is a second forward, not a wiring
job** — which is the opposite of what its first draft implied.

---

## Staging

Each stage exits on something measurable, and each is independently revertible.

### M1 — `inspect` + `capture_plan`, no streaming
Register the adapter with `load_*` returning an explicit unimplemented error.
Reuse `build_capture_names`' naming, including the fused-`gate_up` → split
`w1`/`w3` mapping already solved in `forward.rs`.

*Exit:* `calibrate --dry-run` on LFM2.5-8B-A1B plans a capture set whose tensor
names match what `build_capture_names` produces today, name for name. A
mismatch here is the gemma4 split-vs-fused bug arriving on this family; the
whole point of reusing the existing mapping is that it cannot.

### M2 — streaming `load_embedding` / `load_layer` / `load_finalizer`
The bulk. Both mixer kinds, both FFN kinds.

*Exit:* a calibration artifact for LFM2.5-8B-A1B, and
`artifact audit-calibration` clean. Then the real test: **byte-identical output
against a resident (non-streamed) run** via `artifact compare-calibration
--atol 0 --rtol 0`, which is the established oracle for exactly this. Streaming
must not change the answer.

### M3 — per-expert Hessians
Now that residency is bounded, lift routed experts from imatrix-only.

*Exit:* the artifact carries a full Hessian per routed expert, and peak host +
device memory stays under a declared budget. **This is the stage that can fail
honestly:** if per-expert Hessians still do not fit at 8B-A1B, record the
measured footprint and keep imatrix-only. That is a real answer, and the entire
reason (2) is listed as its own gap rather than folded into M2.

### M4 — the measurement that started this
`oq4.25++` either side of `8357081d3` on a real MoE, the run that is currently
impossible. Note the disk arithmetic from the blocked attempt: HF restore
(~44 GB for 35B-A3B) + bf16 anchor + Hessian + two artifacts ≈ **170 GB**. At
8B-A1B it is roughly a quarter of that and fits this box; **that is a reason to
do LFM2 first rather than wait for a bigger machine.**

*Exit:* a KLD delta on real MoE weights, answering whether the dense −25.7%
result (PR #251) generalizes.

### M5 (separate, optional) — paged experts
Gap (3). Independent of calibration and larger. Only worth starting if LFM2-MoE
needs to serve above resident capacity; scope it separately rather than
bundling.

---

## Risks and unknowns

1. **`cask_metadata` on a hybrid.** The trait's doc is explicit that `None` must
   mean "no native producer, fail explicitly" rather than synthesized uniform
   geometry. With only 6 of 24 layers carrying attention, the honest answer may
   well be `None` — but that is a decision to make deliberately, not by
   defaulting.
2. **Conv-layer capture is not attention capture.** The double-gated LIV
   short-conv (`conv_L_cache = 3`, depthwise causal) has a decode-time state of
   K−1. Whether its `in_proj`/`out_proj` Hessians are meaningful under
   *streaming* — where the state does not persist across the layer boundary the
   way it does in a full forward — is unverified and is the most likely place
   M2's byte-identity exit fails.
3. **The effort figure is a line count**, not an estimate from reading zaya's or
   gemma4's implementation.
4. **M3 may simply not fit**, and the current code says so already. Treat
   "imatrix-only, measured" as a legitimate outcome.

## What this scope does NOT verify

I read the trait signature, the adapter registry, LFM2's config and its existing
`calibration.rs`. I did **not** read gemma4's or zaya's adapter bodies, so the
per-method difficulty is inferred from size and architecture shape alone. Anyone
picking this up should read zaya's first — it is the closest structural analogue
and the largest, which is informative in both directions.
