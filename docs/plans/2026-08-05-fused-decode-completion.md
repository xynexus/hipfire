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

## Phase 2 — Default fused decode on

**Goal.** Organic traffic reaches the fused path, so the 2.2x measured in
PR #215 (chunk `4x1` 1229 ms vs `1x4` 2710 ms, byte-identical output) is
actually delivered.

**The change** is in `select_qwen35_decode_batch_backend`
(`crates/hipfire-generate/src/lib.rs`): `""`/`"auto"` currently returns
`SerialReference` unconditionally. It should select `FusedDenseLayerChunked`
for dense qwen3.5 when a capability gate passes, and fall back otherwise —
mirroring the prefill arm PR #215 added.

**Steps.**

1. Give decode the same shape prefill now has: consult
   `validate_qwen35_fused_dense_decode_model_capability` from `auto`, not only
   from an explicit request. The capability fn already exists and already
   consults the weights contract.
2. Decide the envelope deliberately. `batch_runner::batch_envelope_ok` already
   excludes DFlash, PP>1 and hierarchical KV from continuous batching; decode
   needs an equivalent, and the KV-mode gate (`fp32`/`q8` only, asym/KVarN
   fall back) already exists in the capability fn. Write the envelope down
   rather than inheriting it by accident.
3. Keep `HIPFIRE_QWEN35_DECODE_BATCH=serial` working as the kill switch, and
   say so in the env doc.

**Validation.** Two-session and four-session parity against serial at temp 0,
byte-identical. Then a throughput number on organic HTTP traffic — the existing
`multirow_bench.py` harness measures exactly this. Coherence across the
qwen3.5 size matrix before flipping, since this changes the default path for
every dense qwen3.5 request.

**Risk.** Highest in this plan — it changes what production does. Phase 1 first
is what makes it safe: with AWQ applied rather than rejected, nothing silently
drops to serial and the fused path is the same path everywhere.

---

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
