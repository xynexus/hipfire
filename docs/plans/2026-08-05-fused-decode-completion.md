# Fused decode completion: AWQ coverage, default-on, and the seams behind it

Date: 2026-08-05
Status: proposed
Base: `250414b1e` (PR #215)

## Context

PR #215 landed three things: a guard stopping the fused dense batch path from
silently miscomputing AWQ pre-scaled weights, multi-row decode on by default,
and a repair to `hipfire-quantize --input <org/name>`. It left the measured win
unrealised — `select_qwen35_decode_batch_backend` maps `auto` to
`SerialReference` unconditionally, so no organic traffic reaches the fused path
at all.

This plan finishes that line of work, then clears two pieces of debt it exposed.
Order is deliberate: widen correctness before flipping a default, and flip the
default before proving the seam on a second arch.

### Correction to PR #215's stated mechanism

That PR's commit message says `prefill_batch.rs` "contains zero occurrences of
`awq`, so the fused path computes `(W·s)·x`". The string count is true; the
inference was too broad. The **grouped-MoE** paths in that file already handle
AWQ correctly via `rotate_x_mq_batched_for` and
`fused_rmsnorm_rotate_mq_batched_for` — `_for` helpers that branch on
`next_linear.awq_scale` internally and whose names contain no "awq". Only the
**dense** path is AWQ-blind. The measured corruption and the guard are both
correct; the explanation over-generalised. Phase 1 below is correspondingly
smaller than that framing implies.

---

## Phase 1 — Apply `awq_scale` in the dense fused body

**Goal.** `mq4+`/`q8+` artifacts use the fused path instead of falling back to
serial. Today they are correct but get none of its throughput.

**Everything needed already exists.** Kernels: `rotate_x_mq_awq.hip`,
`fused_rmsnorm_mq_rotate_awq.hip`, `fused_silu_mul_mq_rotate_awq.hip`.
Dispatch: `Gpu::rotate_x_mq_awq_batched`, `Gpu::fused_rmsnorm_rotate_mq_awq`.
AWQ-aware wrappers in `hipfire_runtime::weights`: `rotate_x_mq_batched_for`,
`fused_rmsnorm_rotate_mq_batched_for`, `fused_silu_mul_rotate_mq_for`. The
grouped-MoE half of `prefill_batch.rs` already calls them.

**The two blind sites** (both in `crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs`):

| fn | line | call |
|---|---|---|
| `dense_session_prefill_gemm_full_precision` | ~4578 | raw `gpu.rotate_x_mq_batched(x, &rot, weight.k, n)` |
| `dense_prefill_session_batch_scatter_last_logits` | ~684 | raw `gpu.rotate_x_mq_batched(&normed_rows, &rot, weights.output.k, row_count)` |

The first is the dense body's shared GEMM helper and is the root cause of the
measured corruption; the second is the lm_head/logits path.

**Steps.**

1. Convert both to the `_for` variants. Each already has the downstream
   `WeightTensor` in scope (`weight`, `weights.output`), which is the only
   argument the wrapper needs. Byte-identical on artifacts without an AWQ
   sidecar — that is the wrappers' documented contract and the no-regression
   argument.
2. Audit the dense body's silu-mul path the same way; if it feeds `w_down`
   through a raw rotate, route it via `fused_silu_mul_rotate_mq_for`.
3. Grep for any remaining raw `rotate_x_mq_batched` / `fused_rmsnorm_rotate_mq`
   in the dense body. The audit is the deliverable, not just the two edits —
   a missed site returns to silent corruption, which is exactly the failure
   mode being retired.
4. Narrow the guard in `dense_prefill_weight_unsupported_reason`: drop the
   `awq_scale.is_some()` rejection once every site is covered. Keep the reason
   string and the `fused_dense_supported` capability plumbing — those are
   independently useful and cover the dtype cases.

**Validation.** `smoke-generate-batch-prefill.sh` with
`qwen3.5-0.8b-mq4+.hfq` must now report `backend=fused_dense` and PASS at sizes
2/4/8 — it currently reports `serial_reference` by design. Non-AWQ artifacts
(`Qwen3.5-0.8B--mq4`, `qwen3.5-9b-mq4`) must stay byte-identical. Then the
decode-batch parity smoke on the AWQ artifact.

**Risk.** Low. Additive dispatch branches with an existing byte-identical
fallback. The real risk is an incomplete audit in step 3.

---

## Phase 2 — Default fused decode on — ALREADY DONE, no work required

**This phase was written on a false premise and is closed unbuilt.**

The premise was that `select_qwen35_decode_batch_backend` maps `auto` to
`SerialReference` unconditionally, so no organic traffic reached the fused path.
That function does return `SerialReference` for `auto` — but its caller
immediately overrides it. `qwen35_decode.rs:377-384` selects
`FusedDenseLayerChunked` whenever the request is `auto`, the arch is dense,
`session_count >= 2`, and both `validate_qwen35_fused_dense_decode_model_capability`
and `validate_qwen35_fused_dense_decode_resident_sessions` pass. That has been
there since `08617fbc8` (2026-06-20). The premise came from reading one function
without its caller.

Measured with nothing set (`HIPFIRE_QWEN35_DECODE_BATCH` absent, so `auto`), four
concurrent sessions in lockstep on gfx1103:

    backend = fused_dense_layer_chunked   chunk = 4x1   decode_ms = 1247

against 1229 ms for an explicit `fused_dense` request — the same path. Organic
concurrent traffic has been getting fused multi-row decode all along.

Every step this phase proposed already exists:

| proposed | where it already is |
|---|---|
| consult the capability gate from `auto` | `qwen35_decode.rs:377-384` |
| deliberate envelope (PP / DFlash / eviction) | `validate_qwen35_decode_batch_runtime_surface`, `:352` |
| hierarchical KV forced to serial | `:405` |
| `=serial` kill switch | already works |

**Measurement trap worth recording.** A first check of `auto` reported
`backend=serial_reference` and appeared to confirm the false premise. That is a
telemetry artifact: `last_backend` reflects the final decode step, and when
sessions finish at different lengths the tail steps run one row, which correctly
takes the serial path. Give every session an identical prompt so they stay in
lockstep, or the last-cycle telemetry will describe a one-row step rather than
the batch.

## Phase 3 — Route `resolve_model_path` through `hipfire-hub`

**Goal.** Delete the HuggingFace CLI subprocess entirely.

Already scoped in `docs/plans/2026-08-03-hipfire-hub-downloader.md` ("Status:
scoped, not started") — this phase is that plan applied to the quantizer, not a
new design.

**Steps.**

1. Add the `hipfire-hub` dep to `hipfire-quantize`. This is the decision PR
   #215 deliberately did not make alone: it pulls a network stack into the
   quantizer. Confirm that is wanted before starting.
2. Replace `run_hf_download` and `hf_cache_roots` with the hub client's own
   resolution.
3. Retire `hipfire_env::hf_hub_cache()` / `hf_home()` if the hub client owns
   that resolution — they were added as sanctioned readers precisely because
   the quantizer had to read the env itself.

**Validation.** `--input <org/name>` for a cached and an uncached model; the
`HF_HOME=/srv/huggingface` case must still resolve without downloading.
`env-registry-gate.sh` must still pass.

**Risk.** Low functionally, but it is a dependency-graph change — worth a
second opinion on crate layering.

---

## Phase 4 — Clear the rustfmt backlog

**Goal.** Make the advisory check meaningful again.

It flags **43 files** across `hipfire-coexistence`, `hipfire-hub`,
`hipfire-rdna` examples, `hipfire-xdna`, `hipfire-runtime` examples and others.
Permanently red means it cannot catch new drift — the backlog is hiding signal,
which is the actual cost.

**Steps.** One mechanical `cargo fmt` commit touching nothing else, so it stays
reviewable and `git blame` damage is confined to a single labelled commit. Then
consider promoting the check from advisory to required, which is the only thing
that keeps it clear.

**Risk.** Trivial mechanically; the discipline is keeping it a pure formatting
commit.

---

## Phase 5 — A second arch for continuous batching

**Goal.** Prove the generic seam. Everything above is qwen3.5-only.

`docs/plans/2026-07-18-continuous-scheduler-headline.md` sets the test:
"first non-qwen35 target is deepseek4... If a seam can't be implemented for
deepseek4 without touching the generic layer, the seam is wrong — fix the
abstraction, not deepseek4." That is unproven today. lfm2 has only
`run_generate_batch_prefill_serial_lfm2` and no fused batch decode.

**Steps.**

1. Implement `BatchableSession` for deepseek4 and drive it through the existing
   runner. Change nothing generic at first — treat any forced change to the
   generic layer as a finding about the seam.
2. Port fused batch prefill, then decode.
3. Record what the seam could not express. That record is the deliverable even
   if the port stalls.

**Validation.** The same parity ladder used for qwen3.5: fused == serial at
temp 0, then throughput.

**Risk.** Largest scope here, and the only phase whose output may be "the
abstraction is wrong" rather than a landed feature. That is a legitimate
result — it is the reason to do it before more arches accumulate on the current
seam.

---

## Not doing: batching the Q8 lm_head arm

Considered and dropped. Phase 1 profiling found the Q8 lm_head arm does not
amortize across rows (1.04x vs BF16's 2.20x), which looked like a lever until
the format's status was checked: `--embed-precision` defaults to `source`, and
Q8 is the historical opt-in table that also carries "the largest per-tensor KLD
cost in an otherwise low-bit model" (hipfire-quantize/src/main.rs). Making a
discouraged format batch better is not worth the work.

The live question it leaves is the MQ4 `lm_head.weight` case — `qwen3.5-9b-mq4`
measures 1.54x — which is a supported configuration. Chase that if sub-2.2x
scaling on quantized-lm_head models matters.

## Sequencing

Phase 1 → 2 is a hard dependency: widen correctness before changing the
default. Phases 3 and 4 are independent and can land in any order or in
parallel. Phase 5 is last because it is the only one that may reshape the
generic layer, and doing it after the qwen3.5 path is settled means the seam is
judged against a finished reference rather than a moving one.

Not covered: the ~70 GB/s FastFlowLM question, under investigation on `halo`.
Our own measured ceiling is ~56.5 GB/s (eight columns saturated, XDNA2/Strix
Halo, `docs/npu/npu-memory-bandwidth-cache-characterization.md`), so the most
likely reconciliation is effective-vs-actual byte accounting — a 4x factor on
4-bit weights lands almost exactly on 70.
