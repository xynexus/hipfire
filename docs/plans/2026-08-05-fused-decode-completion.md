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

## Phase 5 — A second arch for continuous batching — BLOCKED, needs re-scope

**Investigated 2026-08-05. The phase cannot run as written, at either end.**

### There is no seam to test

The plan's test was "if a seam can't be implemented for deepseek4 without
touching the generic layer, the seam is wrong". That test presumes a seam. What
actually exists:

- **`ContinuousBatching`** (`hipfire-arch-api`) is declaration-only — its sole
  method is `max_batch_sessions() -> usize`. Declaring it tells the *server* a
  request may be routed to the batch runner. It carries no execution.
- **`BatchableSession`** (`batch_runner.rs:250`) has exactly one impl,
  `DummySession`, in a `#[cfg(test)]` module. No production arch implements it.
  Its only method is `batch_key()`.
- **The daemon dispatches by arch `if`/`else`**, not through a trait:
  `if is_qwen35_family_arch_id(..) { qwen35 } else if ARCH_ID_LFM2_MOE { lfm2 }
  else { "supports qwen35/qwen35-moe and lfm2-moe only" }`
  (`handlers/batch.rs`), for prefill; decode has only
  `run_generate_batch_decode_step_qwen35`.

So adding any arch today means adding a third arm to that chain — touching the
generic layer *by construction*, because there is no generic layer to leave
alone. The seam is not wrong; it is absent.

Note also the population: exactly **one** arch (qwen35) has true fused
multi-session execution. lfm2moe's batch prefill is serial per-session and it has
no batch decode. An abstraction drawn from a single implementation is unproven
regardless of how it is written.

### deepseek4 is not ready to be that second arch

Its batched forwards are single-session and incomplete:

- `forward_prefill_batch` takes one `state`, one `tokens`, one `start_pos` — one
  session — and its body is a per-token `decode_step` loop, commented "Per-token
  fallback until forward_prefill_batch_chunk is end-to-end".
- `forward_prefill_batch_chunk` is "Phase B2 work in progress... Currently a
  partial wiring".

Multi-session batching needs block-diagonal attention over N separate KV/MLA
states. deepseek4 does not yet have batched execution for *one* session.

### Re-scope: two independent pieces of work

**5a — Build the seam, proven against what exists.** Replace the daemon's arch
`if`/`else` with a trait the batch prefill/decode handlers dispatch through, and
implement it for qwen35 (fused) and lfm2moe (serial). Tractable now, and it makes
the abstraction concrete before a second arch has to fit it. Weak evidence on its
own — one real impl plus one degenerate one — but it converts "no seam" into "a
seam with a known-thin proof".

**5b — Finish deepseek4's batched forward.** A kernel project (its own Phase B2),
independent of any batching seam, and a prerequisite for deepseek4 participating
at all.

### 5b investigation (2026-08-05): it is not merely unfinished, it is silently wrong

The in-code status comment on `forward_prefill_batch_chunked` (dated 2026-05-18)
says pure-SWA layers work "end-to-end including the MoE FFN" and that mixed
layers "still bail at the indexer chain". Both halves are stale.

Reproducer — tiny deepseek4 fixture, `--emit-fixture deepseek4 --seed 42`,
quantized `q8f16`, 2 layers, `compress_ratios = [0, 0]` (so **pure-SWA**, the
path the comment claims works), 8 routed experts + 1 shared, top-2:

    profile_prefill_deepseek4 ds4.hfq --prefill 16 --warmup 0 --no-profile
    -> PREFILL_CHECK argmax=0 logit_sum=NaN logit_max=-inf

The same artifact through the per-token path is finite:

    tiny_quant_probe ar-hash --arch deepseek4 --model ds4.hfq --len 16
    -> logit_hash: 0x26a2dc1bd19c368e   (finite, stable)

So this is a path bug, not a property of the fixture's weights. It does not bail
or error — it returns `Ok` and emits NaN, which is why the documented per-token
fallback never triggers: the fallback fires on `Err`, and there is no `Err`.

Localised with the existing `HIPFIRE_DEEPSEEK4_DUMP_STATE` hooks. Every upstream
stage is finite — embedding, HC stream init, `q_lora`, `kv_joint`, tail RoPE, the
whole attention block, `hc_attn_mix`, FFN-side `mhc_pre`. The first non-finite
buffer is `10_l0_ffn_out`, and it is total rather than sporadic: 4096 NaN =
16 tokens x 256 hidden, i.e. every real output element, with the rest of the
buffer untouched zeros.

Narrowed one level further with `HIPFIRE_DEEPSEEK4_MOE=0`, which returns from
`ffn_batched` right after the shared expert:

    HIPFIRE_DEEPSEEK4_MOE=0 -> logit_sum=7.0162 logit_max=0.898018, zero NaN

**The NaN is entirely in the routed-expert MoE half of `ffn_batched`.** The
shared-expert half is clean.

Note the attention/indexer chain — which the stale comment blames — was never
reached by this fixture (`compress_ratios = [0, 0]`) and is therefore still
untested, not exonerated. A fixture with `compress_ratio > 0` is needed to
exercise it.

Next step is to instrument inside the routed-MoE section (router GEMV, top-k,
expert gather, grouped GEMM) the same way. Until this is fixed, "finish the
batched forward" understates the work: the existing path produces wrong numbers
silently, so it needs a correctness fix before any completion work.

They can proceed in either order; 5a does not depend on 5b. Doing 5a first means
5b's author has something to implement against instead of a third `else if`.

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
