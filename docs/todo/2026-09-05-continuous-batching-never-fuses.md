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


## Update 2: the drafter check, and KV=kvarn4

### The probe tested for the wrong thing

`DflashState::draft_weights` is the drafter, and its own field comment says the state is
carried with `draft_config`/`draft_weights` = `None` "when this state exists only to carry
n-gram speculative decode, which drafts from statistics and needs no drafter model".
Three sites tested `m.dflash.is_some()` instead — the probe, the prefill operation, and
the decode gate's call site — so the drafter-free path was refused by a check written for
draft models. `batch_executor::has_draft_model` is now the single predicate all three use.

Measured: `ngram_spec: true` with no drafter now routes `-> BATCH`
(`draft_model=false ngram=true`), where it previously refused.

Fixing the probe alone was WRONG and briefly made things worse: the probe admitted models
the decode gate still refused, turning a silent fallback into `fail_all` for every session
("generate_batch_decode_step is not supported on DFlash-loaded models"). The probe's own
comment warns about exactly this. All three sites must move together.

### `auto` could select a backend that then hard-fails

Admitting OqCompactG256 exposed a second hazard. `select_qwen35_prefill_batch_backend`
picks FusedDense from `validate_qwen35_fused_dense_prefill_model_capability`, which
checked only WEIGHTS — while the dense fused prefix also requires FP32 DeltaNet state and
refuses FP16 per-session at execution, inside the batch op, where it is fatal to the whole
cycle. The family default has been FP16 since f5b32ea32, so this was the common case. The
capability check now refuses FP16 state, so `auto` degrades to serial as the contract
promises. Verified: default config (kvarn4, ngram on, FP16 state) runs clean at
4.42/4.45/4.47 s/req for N=1/2/3 — serial, but no errors.

### KV=kvarn4 gets no benefit

| config | N=1 | N=2 | N=3 |
|---|---|---|---|
| kvarn4, DeltaNet FP16 (defaults) | 4.42 | 4.45 | 4.47 s/req |
| kvarn4, DeltaNet FP32 | 4.43 | 4.45 | 5.78 s/req |
| q8, DeltaNet FP32, serial | 4.51 | 4.54 | 4.69 s/req |
| q8, DeltaNet FP32, fused | 4.70 | 4.18 | **3.45 s/req** |

KVarN is admissible for the fused PREFILL (`allow_kvarn`), but the fused DECODE requires
FP32 or plain Q8 KV and excludes it. Decode dominates a 60-token completion, so the prefill
win is invisible and N=3 is slightly worse than N=1. Only q8 — which is deprecated and
needs `HIPFIRE_KV_ALLOW_DEPRECATED=1` — reaches the fused decode.

**CORRECTION — see Update 3.** The conclusion drawn here, that the recommended KV mode and
continuous batching are mutually exclusive, was wrong. The fused decode already accepts
KVarN. What refused in the runs above was the DeltaNet state precision, which is FP16 by
default and blocks the fused path for every KV mode alike. With FP32 state, KVarN reaches
the same throughput as q8.


## Update 3: KVarN needed no widening; the blocker was DeltaNet precision

`validate_qwen35_fused_dense_decode_model_capability` already admits KVarN:

```rust
let kvarn_ok = kv_mode.starts_with("kvarn") && qwen35::qwen35_kvarn_fused_batch_enabled();
```

and `qwen35_kvarn_fused_batch_enabled()` is ON by default (only `=0` disables it). So
there was no kernel to write. `HIPFIRE_DECODE_BACKEND_TRACE=1` gives the real refusal:

```
fused dense declined: qwen35 fused dense decode requires FP32 DeltaNet state;
loaded state=FP16
```

The same FP16 DeltaNet default that blocks the fused PREFILL blocks the fused DECODE, for
every KV mode. Attributing it to KVarN in Update 2 was wrong: the q8 runs there had
`deltanet_state_precision=fp32` applied and the KVarN runs did not, so the variable that
actually changed was the state precision, not the KV mode.

With KVarN + FP32 DeltaNet state (and the OqCompactG256 admission, which the decode gate
also needs — it calls the same weights validation):

| N | serial | fused |
|---|---|---|
| 1 | — | 4.60 s/req |
| 2 | — | 4.15 s/req |
| 3 | 13.6 s (4.53 s/req) | 10.1 s (**3.46 s/req**) |

1.35x at N=3 — the same figure q8 reached (3.45 s/req), confirming the KV mode was never
the discriminator. Parity is 2/3, diverging on the identical `$n-1$` vs `$n/2$` near-tie as
under q8, so the OqCompactG256 parity question is KV-independent too.

### What actually gates batching today

1. A real DFlash drafter (`dflash_draft: auto` finds a sibling for the 9B) — correctly
   refused; drafter-based speculation with batching is unimplemented.
2. FP16 DeltaNet state — the family default since f5b32ea32. `deltanet_state_precision=fp32`
   clears it, at the cost of doubling per-sequence state.
3. OqCompactG256 weights — fixed here, parity pending.

None of these is the KV mode.

### Config bug: per-model overrides silently ignored

`deltanet_state_precision` set under `model_overrides.<model>` had no effect — the daemon
loaded FP16 and the trace said so; only the top-level key worked. `dflash_draft` behaves
the same way. Both are declared `GLOBAL_MODEL_RUNTIME` in the schema, so a per-model value
is accepted by validation and then dropped, which is the worst of the three options.
