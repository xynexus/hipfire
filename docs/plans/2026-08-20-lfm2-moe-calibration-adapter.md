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

### Effort proxy, and why LFM2 lands high

Existing adapters, by size:

| arch | lines | shape |
|---|---|---|
| gemma3 | 1057 | dense |
| gemma4 | 1247 | MoE |
| cohere2 | 1892 | dense |
| qwen35 | 2139 | MoE |
| **zaya** | **2410** | **hybrid** |

LFM2.5-8B-A1B is 24 layers of **18 double-gated LIV short-conv + 6 GQA
attention**, with per-layer dense-or-MoE FFN. It is hybrid *and* MoE, and zaya —
the only other hybrid — has the largest adapter in the tree. **Expect the upper
half of that range, ~1800–2400 lines**, not gemma4's 1247.

This is a line-count proxy from `wc -l`, not an estimate from reading the
implementations. Treat it as an order of magnitude.

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
