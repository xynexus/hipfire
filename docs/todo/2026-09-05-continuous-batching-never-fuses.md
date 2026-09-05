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


## Update: the AWQ guard had drifted from its own call sites

`dense_prefill_weight_unsupported_reason` refused any AWQ-sidecar weight whose dtype was
not `MQ4G256 | Oq8G256` — while the refusal string it returns names **three** arms,
including `OqCompactG256`. The guard and its own message disagreed.

`OqCompactG256` belongs in the list: all three of its dispatch sites (lm_head,
non-residual GEMM, residual GEMM) call `rotate_x_mq_batched_for`, which applies
`x /= awq_scale` when the sidecar is present — the same path `Oq8G256` and `MQ4G256`
take. This is exactly the drift the comment above the guard warns about.

It matters because `oq4.25++` is `OqPlusCompact` on disk and loads as `OqCompactG256`,
so **every** locally served Qwen artifact hit this and silently ran `serial_reference`.
Adding the dtype (one line) makes the fused body run on them. Measured, same KV=q8 /
DeltaNet=FP32 config, only the backend differing:

| N | serial | fused |
|---|---|---|
| 1 | 4.5 s (4.51 s/req) | 4.7 s (4.70 s/req) |
| 2 | 9.1 s (4.54 s/req) | 8.4 s (4.18 s/req) |
| 3 | 14.1 s (4.69 s/req) | 10.4 s (3.45 s/req) |

Serial is flat at ~4.5 s/req — no batching benefit at all. Fused improves with N: 1.36x
throughput at N=3. N=4 OOMs (`hipMalloc` 136 MB) at `max_seq: 131072`, so residency, not
the kernel, caps the batch here.

### It does not meet the parity bar

**6/9 sessions byte-identical to serial** across three repeats. Divergences are near-tie
flips (`$n-1$` vs `$n/2$`; a reordered clause), not gross corruption — the same signature
that keeps `Oq4G256` on serial two comments below this guard ("3 of 8 sessions produced a
different first token ... most likely an activation-precision difference"). Parity is this
file's stated bar, so **this change should not land as-is**: the fix is correct about
which arms apply the scale, and it exposes the same numerical gap that is already known
and unexplained for the neighbouring dtype. Same investigation, now with a second dtype
reproducing it.

## Speculative decode does not have to exclude batching

The probe refuses on `m.dflash.is_some()`, and the wording in `run_generate_batch_prefill_serial_qwen35`
is "does not support DFlash-loaded models **yet**" — unimplemented, not impossible. vLLM and
TRT-LLM run both together; the real tension is batch-size dependent. At small batch the GPU
is underutilised and speculation is close to free; as batch grows the GPU saturates and
rejected draft tokens start costing more than they save. That argues for a batch-size
threshold, not a hard refusal.

The n-gram case is stronger still, and is currently a plain defect:

```
[batch-probe] REFUSED: dflash drafter present (ngram=true dspark=false)
```

`ngram_spec: true` with `dflash_draft: off` still refuses. `NgramState` is a separate
field whose own comment says it is "**deliberately NOT inside `dflash`**: it needs no
drafter" — yet enabling it populates `m.dflash` (the shared `spec_step_dflash` verify
engine), so the drafter-free path is refused by a check written for draft models. N-gram
drafting costs a table lookup, no second forward pass, so it should compose with batching
at any batch size. Fixing this is narrower than the general spec-decode question: the
probe should test for an actual drafter, not for the verify engine they share.
