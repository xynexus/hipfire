# Batched MoE prefill panics on `Oq4G256` experts — and it kills the daemon

Status: found 2026-08-27, master `556249d8c`, nix1 (`gfx1103`), freshly
`cargo clean`ed release build. **Reproducible, `rc=101`.**

## Symptom

```
thread 'main' panicked at crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:1282:30:
prefill_moe_ffn_body_batched: unsupported experts[0].gate_up dtype Oq4G256
  — admit predicate should have rejected this layer
```

The process exits 101. For a serving daemon that means **every co-resident model
dies**, not just the request that triggered it.

## Reproduce

```jsonl
{"type":"load","model":"~/.hipfire/models/Qwen3.6-35B-A3B--oq4.hfq",
 "params":{"max_seq":2048,"dflash_mode":"off","kv_cache":"kvarn"}}
{"type":"generate","id":"g0","prompt":"<~2.4 KB of text>","max_tokens":4,"temperature":0.0}
```

`hipfire-daemon < repro.jsonl` → panic. A `kld_eval build_ref` with
`n_ctx: 1024` panics identically and was how this surfaced.

## Cause

`prefill_chunk.rs` path-2 (scatter + grouped WMMA) matches only:

```rust
raw @ (DType::F16 | DType::BF16) => gpu.gemm_raw_moe_grouped(…),
DType::ParoQ4G128 => { … },
other => panic!("… unsupported experts[0].gate_up dtype {other:?} \
                 — admit predicate should have rejected this layer"),
```

`Oq4G256` is not in that set, and the admit predicate gating path 2 does not
check the routed experts' `gate_up` dtype against it. The panic message is
correct about its own cause.

## Why it did not show up before

**The KV tier decides whether batching is even attempted.** With an f32 KV cache
the batched path declines and falls back:

```
qwen35 forward_prefill_batch: -> per-token forward_scratch loop
  (batched prefill declined)  [base=true kv_f32=true … arch=gfx1103]
```

That is the trace recorded on 2026-08-26 while benching **this same artifact** at
pp512 = 20.9 t/s — no panic, because the fast path was never entered. Selecting
`kvarn` (the non-deprecated family; `q8` now errors as deprecated) admits
batching and walks straight into the unhandled arm.

So the trigger is `Oq4G256 MoE experts` **+** `a KV tier that admits batching`
**+** `a prefill long enough to batch`. Any two of the three look fine.

## Why it matters beyond this artifact

`oq4` MoE is the target family for the induction work (`oq4.25+++`), and
`kvarn` is the KV family hipfire is standardising on. This combination is
therefore the *intended* configuration, not an exotic one.

It also blocks measuring the batched KLD path: the whole reason for loading this
model was to compare batched scoring against the 4.2 tok/s per-token blanket
impl (`docs/experiments/2026-08-27-induction-quantum-wcet-nix1.md`). That
comparison is still unmeasured because the batched path crashes.

## FIXED 2026-08-27 — and the first two attempts were both wrong

**The defect is path SELECTION, not a missing dtype.** Path 1 (the indexed route
*inside* the batched body) already has an `Oq4G256` arm:

| path | dtypes |
|---|---|
| path-2 (grouped WMMA) | `F16, BF16, ParoQ4G128` |
| path-1 (indexed) | `Oq4G256, Oq8G256, OqCompactG256, ParoQ4G128` |

`mod.rs:2651` declares `DType::Oq4G256 => arch.starts_with("gfx11")` — directly
beneath a comment warning *"Do NOT add a dtype here just to satisfy an admission
check… A dtype declared here without a grouped-WMMA arm is a latent panic."* It
was declared for the **mixed** routing profile, but `use_path2` defaults on, so a
**uniform** oq4 profile also became path-2 eligible and hit the `other` arm.

**Two wrong attempts, recorded because each looked right:**

1. *Decline all Oq4 batching in the admit predicate.* Correct behaviour, wrong
   reason — it would have dropped a working artifact to the 4.2 tok/s per-token
   loop on the false premise that Oq4 is unsupported.
2. *Route uniform Oq4 to path 1 and return `Err` when unavailable.* The `Err`
   fails the **request** rather than declining to the fallback; verified by
   running it (`"prefill failed: … batched MoE prefill declined"`). Declining
   must happen in the admit predicate, before dispatch.

**Shipped:** the admit predicate declines uniform `Oq4G256` **by default**, so the
forward drops to the per-token path; `HIPFIRE_MOE_OQ4_UNIFORM_PATH1=1` opts into
path 1. Both `panic!`s became error returns as a backstop.

### Why the default is the slow route

Path 1 runs and is **deterministic** (run-to-run identical), but its 96-token
greedy output **DIFFERS from the per-token reference**, diverging after 255 of
433 chars. That is an accumulation-order difference between two kernels, not
instability — but it is *unverified*, not known-good.

⚠️ **Correction 2026-08-27.** This paragraph originally continued: *"and
tiny-prefill-gate SKIPS qwen3_5_moe … so batched MoE prefill has no parity
coverage at all."* **Wrong.** That SKIP is deliberate regression cover for the
admission guard (top-2-of-8 vs the required k_top ∈ {8,10}), and
`qwen3_5_moe_indexed` covers batched MoE prefill and passes. The real gap is
narrower — every gate cell is `--format fp16`, so none exercises `Oq4G256`
experts — and it cannot be closed by adding an oq4 cell, because
`is_batchable_la` rejects Opus dtypes on the *attention* projections, so an
all-Opus fixture declines batching before MoE is reached. See
`docs/todo/2026-08-27-oq4-moe-prefill-coverage.md`.

The Oq8 precedent still stands as the reason the default is conservative: that
grouped kernel "ran 1.8x faster and emitted garbage".

Measured after the fix:

| configuration | vs per-token reference |
|---|---|
| default (declined) | **IDENTICAL** (423 chars) |
| `HIPFIRE_MOE_OQ4_UNIFORM_PATH1=1` | DIFFER (433 chars, diverges at 255) |

**To remove the flag:** make `tiny-prefill-gate` cover batched MoE prefill and
show path-1 parity. Until then the fast path is opt-in and the default is right.

## Original analysis — the fix is a decline, not an arm

Two separable changes:

1. **Stop panicking.** An unsupported dtype in a *fast* path must decline to the
   existing per-token fallback — which already exists and already works for this
   model — not abort the process. A `panic!` here converts a missing
   optimisation into a total serving outage.
2. **Fix the predicate** so `Oq4G256` experts are either admitted properly (an
   `Oq4G256` arm in the grouped GEMM) or rejected before dispatch. Note the
   decode side already handles `Oq4G256` routed experts via the indexed path, so
   the capability exists; it is prefill's grouped arm that lacks it.

(1) is the urgent half and is small. (2) is the performance half and can follow.

## Untested here

- Whether other `Oq*` MoE artifacts (`oq4.25++`, `oq8`) hit the same arm — likely,
  since the match is on `DType` and only three variants are handled.
- Whether `ParoQ4G128` MoE artifacts are affected (they have an explicit arm, so
  presumably not).
- gfx1151, where the grouped WMMA path differs.
