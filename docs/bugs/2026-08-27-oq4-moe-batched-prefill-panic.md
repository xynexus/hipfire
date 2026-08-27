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

## The fix is a decline, not an arm

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
