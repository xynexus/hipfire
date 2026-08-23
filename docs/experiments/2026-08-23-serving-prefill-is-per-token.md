# Why serving prefill is 15.4 tok/s — and why it need not be

State box: halo, gfx1151, Qwen3.8-27B oq4.25++ +CASK, kvarn KV, MAX_BATCH=512,
ROCm 7.14, 2026-08-23.

## ⚠️ Correction

An earlier draft of this file claimed the batched-prefill gate "is never
reached" and called this "a serving-integration problem". **Both were wrong.**
The batched path is reached exactly when asked for; it is gated by a deliberate
opt-in that I added myself the day before, in `7224715da`.

## The answer

`hipfire-serving-core/src/generate.rs`:

```rust
if kv.quant_kvarn
    && std::env::var("HIPFIRE_KVARN_BATCHED_PREFILL").ok().as_deref() != Some("1")
{
    // per-token forward_scratch loop
}
```

Every daemon run in this investigation passed `HIPFIRE_KV_MODE=kvarn`, which is
precisely the trigger. The per-token fallback was self-inflicted by the test
setup, not a defect.

`7224715da` (2026-08-22) had already established the guard's stated reason was
obsolete — `prefill_chunk.rs` handles KVarN explicitly ("kvarn_attend owns the
batched write") — measured 15.4 -> 48.1 tok/s, and deliberately left it opt-in:

> Behind HIPFIRE_KVARN_BATCHED_PREFILL=1 rather than default-on: the original
> guard cites a real failure mode, so this wants the full coherence battery
> across the KVarN model set before flipping.

So the honest answer to "why doesn't the daemon reach the batched path" is:
**because it was deliberately left opt-in, pending a coherence battery.**

## What is new today

Two things, and together they are most of the evidence that battery was for.

**1. The batched path is now 310 tok/s, not 48.1.** The 6.4x since 08-22 is this
session's kernel work — overlay unroll + LDS-transposed store, GEMM grid
swizzle, zero-seeded accumulators, hoisted k-major transpose.

| | prefill | TTFT | overall |
|---|---|---|---|
| per-token (default) | 15.4 tok/s | 46.7 s | 3.3 tok/s |
| batched (flag on) | **310.2 tok/s** | **2.3 s** | 12.4 tok/s |

**20.1x prefill, 20x TTFT.** Decode unchanged at 14.5, as expected.

**2. Output is now BYTE-IDENTICAL, where on 08-22 it differed.** Two prompts,
200 greedy tokens each, `coherence_probe`, both verdicts OK:

    lp2 @200 greedy tokens: IDENTICAL CONTENT
    lp3 @200 greedy tokens: IDENTICAL CONTENT

That is a real change in behaviour, and the cause is identifiable. `8ea5a303e`
("attend each KVarN segment BEFORE flushing it, not after the whole batch")
landed AFTER the guard commit and is the only change to `kvarn_attend` since.
The guard commit had observed "wording differs because batched and per-token
export measurably different hidden states"; that divergence is now gone.

## What remains before flipping the default

The bar `7224715da` set was "the full coherence battery across the KVarN model
set". Today's evidence is stronger in kind (byte-identical, not merely fluent)
but narrower in scope: ONE model, two prompts, greedy only. To close it:

1. Run the coherence battery across the KVarN model set, not just Qwen3.8-27B.
2. A KLD comparison batched-vs-per-token would be better than token equality,
   since greedy equality can mask small logit drift that sampling would expose.
3. Check non-greedy sampling, which the identical-token test does not cover.

Until then the flag stays opt-in — but note the cost of leaving it there is now
20x on serving prefill, not the 3.1x it was when the decision was made. That
changes the balance enough to be worth revisiting deliberately rather than by
default.

## Unrelated, still true

`hipfire eval`/`coherence_probe` reach the daemon, so any quality work on the
BATCHED path (e.g. validating W4A4 activations) must set
`HIPFIRE_KVARN_BATCHED_PREFILL=1` or it silently measures the per-token path
instead. That is how the W4A4 quality check failed earlier today.
