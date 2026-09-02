# The FP16 DeltaNet recurrence is not chunk-invariant — batched prefill and speculation advance the state differently from decode

Status: **OPEN, measured.** This is the origin the batched/per-token and spec/AR
divergences were pointing at. It is not the KV cache.

## The result

`crates/hipfire-runtime/examples/gdn_chunk_seq_parity.rs` — no model. Identical
synthetic q/k/v/gate/beta, one GDN state, two schedules:

- **A**: `gated_delta_net_*_batch_seq` with `chunk` tokens per launch — what
  batched prefill and the speculative verify do.
- **B**: one token per launch — what plain decode does. (For FP16 this mirrors
  `speculative.rs` exactly, which loops `f16_batch_seq` with `n_steps = 1`.)

n = 64 tokens, 16 heads, head_dim 128:

| state dtype | chunk | state elements differing | output elements differing | worst \|rel\| |
|---|---|---|---|---|
| FP32 | 7 / 16 / 17 / 32 / 64 | **0 / 262144** | **0 / 131072** | 0.000e0 |
| FP16 | 17 | 0 / 262144 | 33565 / 131072 | **1.989** |
| FP16 | 64 | **262144 / 262144** | 129022 / 131072 | **1.989** |

FP32 is bit-exact at every chunk size tested. FP16 is not: at a single 64-token
chunk **every element of the recurrent state differs** from stepping the same 64
tokens one at a time, and the worst relative error is ~2.0 — sign disagreement,
not rounding noise.

## Why this is the explanation

- The prefill probe reports `FIRST DIVERGING LAYER: 0`, and layer 0 is
  `linear_attn` — a DeltaNet layer that **carries no KV at all** on this model
  (full-attention layers are 3, 7, 11, 15, 19, 23). The divergence appears before
  any KV layer is reached.
- It is bit-width independent across KVarN 2/4/8, because it is not a KV effect.
- `kvarn_attend`'s write AND read paths are bit-exact batch-invariant
  (`2026-09-02-kvarn-write-path-is-batch-invariant.md`), so the cache is not it.
- The daemon's own prefill warning says as much: "the per-token fallback rounds
  the FP16 DeltaNet state once per token where the batched path rounds once per
  chunk ... **the KV mode is not the lever**."

## The code already knows — for one of the two paths

`speculative.rs` handles this correctly on the ROLLBACK REPLAY path, and says
why:

    // Same FP32-batched / FP16-per-token split as `replay_gdn_inner`, and for
    // the same reason: f16 narrows the state once per launch, so replaying an
    // accepted prefix must step token by token to match decode.

So FP16 replay steps token by token, deliberately. But **batched prefill still
calls `batch_seq` with FP16 state**, so a prompt processed in chunks lands in a
different state than the same prompt decoded token by token — which is exactly
what the prefill probe measures, and it switches on at `n = 257`, the first
prompt needing a second `PREFILL_MAX_BATCH` chunk.

The workaround was applied where someone hit the symptom, not where the
non-equivalence lives.

## Consequences

1. **Prefill is not equivalent to decode** for any FP16-state model whose prompt
   exceeds one 256-token chunk.
2. **Speculation is not equivalent to AR**: the verify advances the state over
   `b` rows in one launch, AR over one.
3. It is the mechanism behind
   `2026-09-02-ngram-speculation-drives-prompt-echo.md`, where n-gram
   speculation plus FP16 state deterministically drives the model into echoing
   87% of its prompt while reporting a 3.4x speedup.

## Mitigation and its limit

`2026-09-02-deltanet-redundancy-gate-had-no-caller.md` restores the guard that
forces FP32 state below a redundancy floor, and FP32 is chunk-invariant here, so
models under the floor are covered. `Qwen3.6-35B-A3B` (redundancy 4096) is above
it and still runs FP16 — untested, and this is the reason to test it.

## What this does NOT explain

At decode with FP32 state, speculative output still differs from AR under
quantised KV and matches only at `fp32` KV. If both the GDN recurrence (at FP32)
and the KVarN paths are chunk/batch invariant, that result needs a separate
cause. It is measured on real generations, so it is not an artifact of the
prefill probe. Do not treat this document as closing that question.

## Fix directions

- Make `f16_batch_seq` chunk-invariant (accumulate in f32 within the launch and
  narrow once per token, matching the per-token path), or
- route batched prefill through the per-token path whenever the state is FP16,
  as the replay path already does, or
- make FP32 state the default for every model rather than only below a
  redundancy floor, and treat FP16 state as an explicitly-opted-in approximation
  that breaks prefill/decode equivalence.
