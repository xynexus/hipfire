# Session: put a drafter on the `Speculator` seam

**Blocked on:** nothing, but verification needs a real GPU target — which is what
makes it session-sized rather than a patch.

## Objective

`Speculator` (in `hipfire-specdecode-dspark/src/spec.rs`) has **zero
implementors**. Port one drafter onto it, so the seam is proven and the five
hand-rolled acceptance loops have somewhere to collapse.

## Why

Acceptance is implemented five times, independently, all computing longest
matching prefix + bonus token:

| path | where |
|---|---|
| DFlash chain | `speculative.rs:8088` (greedy + rejection sampling for temp>0) |
| `spec_step_greedy` | `speculative.rs:6435` |
| MTP | `mtp_spec.rs` — and the **only** GPU implementation |
| lfm2moe | `lfm2moe/dflash.rs:668`, own fn, own three tests |
| DFlash2 / DDTree | `spec_step_ddtree*`, tree-shaped |

Only MTP calls the GPU accept kernel (`greedy_accept_from_argmax_i32`); everything
else round-trips argmaxes to the host. And **tree accept generalises linear
accept** — a spine is a degenerate tree — so one implementation could replace all
five.

The trait's own doc describes exactly the target: *"let a model-free speculator
(n-gram / PLD) drive any arch's target without knowing its internals: the target
owns ALL verify mechanics … while the speculator owns only policy (drafting +
acceptance)."* `SpecTarget` already has three implementors (LlamaBackend,
Gemma3Backend, a test double). The speculator half was never built.

This also unblocks n-gram spec decode outside qwen35: `NgramState` was hoisted
onto `LoadedModel` (`563ff6f02`) so any decode path can reach it, but
verification still runs through `spec_step_dflash`.

## Why it needs a real target

`Speculator` requires `prefill`, `step`, `reset`, `block_size`, `ctx_capacity`,
`free`, and `step` is the whole cycle: draft → `target.verify_block` → accept →
commit. Every method takes `&mut Gpu`.

**The existing test double cannot drive it.** `impl SpecTarget for Bare` in
`spec.rs` has `unimplemented!()` for `verify_block`, `spec_advance`,
`new_spec_scratch` and `commit_prefix` — it exists only to prove default trait
methods decline gracefully. So an implementation cannot be verified without
wiring it to a real model, and an unverified spec-decode loop is worse than none.

## First moves

1. Start with the **n-gram** speculator: model-free, and what the trait was
   designed for.
2. Verify against a qwen35 target where the existing chain path gives a reference
   token stream — the bar is identical tokens, since spec decode is lossless.

## Do NOT

Retry the DFlash2 candidate selector as an acceptance lever. Already implemented,
gated behind `HIPFIRE_DFLASH2_SELECTOR=1`, and **measured worse**: tau 2.421 ->
2.25, decode 6.14 -> 5.92.
