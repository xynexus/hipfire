# Continuous batching never delivers throughput (measured 2026-09-05)

Three prefix-sharing requests, sent 0.4 ms apart (inside the 10 ms gather window),
Qwen3.5-9B--oq4.25++ on gfx1103:

| | wall clock |
|---|---|
| 1 request | 14.1 s |
| 3 concurrent | 50.9 s (3.6x) |
| 6 concurrent | 99.1 s (7.1x) |

Linear. Meanwhile `/health` reports `prefill_batch.enabled = true` and capability
`supported`. Two independent defects, in series.

## 1. The probe refuses, so nothing is even eligible

```
[batch-route] enabled=true eligible=false arch=Some("qwen3_5") capable=Some(false)
[batch-probe] REFUSED: dflash drafter present
```

`Qwen35BatchExecutor::probe` refuses any speculative-decode model. **Two** independent
config defaults set `m.dflash`: `ngram_spec: true` and `dflash_draft: "auto"` (which finds
a drafter by sibling discovery). Either alone is enough. With both off the route flips:

```
[batch-probe] OK pp=1 dflash=false eviction=false
[batch-route] eligible=true capable=Some(true) -> BATCH
```

So the shipped default config silently disables continuous batching entirely. Whether
speculative decode should lose to batching is a policy question, but it should be a
*decision*, not an accident: nothing in the logs or `/health` says batching is off because
a drafter is attached.

## 2. Even when eligible, the fused kernel refuses these weights

With the drafters off, everything upstream works. The scheduler groups correctly
(`lease workloads=3`), the runner coalesces (`coalesced 3 request(s) into one batch`), and
the daemon makes all three sessions co-resident within 4 ms. It is still 3.6x.

`HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=fused` surfaces what `auto` swallows:

```
qwen35 fused dense prefill unsupported weights: dense session fused prefix layer 0 has
unsupported weights (AWQ pre-scaled weights: the fused body applies awq_scale only on
the FWHT-rotated arms (MQ4G256, Oq8G256, OqCompactG256))
```

`select_qwen35_prefill_batch_backend` under `auto` maps that `Err` to `SerialReference`,
which computes each session in turn — hence exactly Nx. Every locally served Qwen artifact
is `oq4`/`oq4.25`, so none of them can ever take the fused path.

The silent downgrade is the reportable part. A serial fallback is a correct answer to an
unsupported model; being unable to tell it happened is not. `auto` should record the
refusal reason where an operator sees it — `/health` already has `fallback_reason` plumbed.

## What would actually fix it

1. Apply `awq_scale` on the non-rotated arms in the fused dense body, so oq4.x artifacts
   are fused-eligible. This is the real fix and unblocks every model served here.
2. Failing that, quantize the swarm models to one of the supported arms and measure.
3. Independently: surface both refusals. Batching that reports "enabled" and "supported"
   while running strictly serially cost a day to characterise from the outside.

## Not a factor

`prefill_batch.selected_batch_size = 0` in `/health` looks damning and is not evidence —
that counter is uninstrumented, as its own sibling `counters` field says. The trustworthy
signals are `[prefill-eligible]` (one line per prefill), the runner's `coalesced N` debug
line, and daemon session-creation timestamps.
