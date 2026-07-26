# A4b — Loader Dispatch: Blocked, and Probably Mis-Scoped

Date: 2026-07-26

## Status

**Not started. Deliberately.** A4b was scoped as "convert `load.rs`'s four
`arch_id` chains into dispatch over the resolved identity". Two things surfaced
while preparing it:

1. Its stated justification — a silent fallthrough into the qwen35 body — is
   **unverified on `master` and partly wrong as written**.
2. Its stated payoff — "one exhaustive `match`" — is **not achievable** with the
   identity design that A1–A3 shipped.

Neither is fatal, but together they mean A4b should be re-scoped before anyone
cuts a 2,384-line function's dispatch. This document records what is known, what
is not, and why the review's ordering probably supersedes A4b's.

Prerequisite context: `docs/plans/2026-07-26-arch-identity-structure-descriptor.md`
(the A-phase plan). A1–A3 and A4a are landed.

## Correction 1: the fallthrough claim

Repeatedly asserted, including in the A-phase plan and PR #194:

> an unhandled arch id falls through to the qwen35 body and panics on a
> `None.unwrap()`

That came from an architecture review run against `feat/daemon-state-hoist` and
was **never checked on `master`**. On `master`:

- `load.rs` contains **no** explicit refusal for an unhandled arch. *(verified)*
- But `load.rs:2405` guards the qwen35 body with
  `if is_qwen35_family_arch_id(hfq.arch_id) { … }`. The body is **not**
  unguarded. *(verified)*
- What an unregistered `arch_id` actually does past that point is **unknown**.
  Execution continues into a llama construction path; whether it errors,
  mis-loads, or panics has not been established.

**A4b's justification therefore has to be re-established before the work is
worth doing.** The first task in a new session is a probe, not a refactor.

## Correction 2: string families are not exhaustively checkable

A1 made identity nominal: `ArchRef { family: &'static str, … }`. The A-phase plan
and PR #194 both promise A4b turns the chains into "one exhaustive `match`".

`match family { "llama" => …, _ => … }` has **no compiler exhaustiveness check**.
Adding a family cannot break the build the way a new enum variant would.

So the achievable win is narrower than advertised: an **explicit refusal** for an
identity no arm handles, replacing whatever currently happens by accident. That
is still worth having — it is the difference between a loud failure and a silent
misload — but it is not compiler-enforced coverage.

If compiler exhaustiveness is genuinely wanted, that is a **separate decision**:
generate a `Family` enum from the registry and match on it. That was considered
and rejected during the A-vs-B decision as unnecessary; it should not be
smuggled in as part of A4b.

## The bigger problem: there is nothing clean to dispatch *to*

This is the substantive finding, and it comes from the architecture review.

`load_model_inner` has ten per-arch early-return blocks, each constructing a
differently-populated `LoadedModel`:

| line | arch |
|---|---|
| 850 | `ARCH_ID_LFM2_MOE` (nested pre-check, not a top-level arm) |
| 1107 | `ARCH_ID_EMBEDDINGGEMMA` |
| 1375 | `ARCH_ID_ZAYA` |
| 1469 | `ARCH_ID_NEMOTRON_H` / `ARCH_ID_MAMBA2` (4 nested sub-branches) |
| 1598 | `ARCH_ID_GEMMA3_TEXT` |
| 1699 | `ARCH_ID_GEMMA3_VL` |
| 1813 | `ARCH_ID_DOTS_OCR` |
| 1914 | `ARCH_ID_DEEPSEEK4_FLASH` |
| 2036 | `ARCH_ID_MINIMAX_M2` |
| 2164 | `ARCH_ID_LFM2_MOE` |
| 2405 | qwen35 family, then llama fall-through |

*(line numbers verified on `master` at commit `cf1fc8a1c`, i.e. after A4a.
`load.rs` line numbers shift on almost every edit to the file — re-grep rather
than trusting these. Writing this document already caught four stale entries
that had been captured minutes earlier, before A4a's wrapper was inserted.)*

Converting `if arch_id == X` into `match identity.family` at each of these
**renames the branching without removing it**. The reason ten arms exist is that
`LoadedModel` carries ~37 per-arch `Option` slots and each arm populates a
different subset. Dispatch is a symptom; the god-struct is the cause.

The review's own ordering says as much: collapsing `LoadedModel` is only safe
once the arch slots have a single consumer, and they only have a single consumer
once generation routes through the `ServingBackend` seam.

**Doing A4b first buys a cosmetic improvement and a large, risky diff.**

## What the architecture review found

Numbers below were measured by review agents against **`feat/daemon-state-hoist`**,
not `master`, and are **not re-verified**. Given Correction 1 came from exactly
this source, treat every figure as a lead to confirm, not a fact.

### Verified on `master` (by direct grep, this session)

| fact | value |
|---|---|
| `arch_id ==` sites | `load.rs` 20, `session.rs` 18, `quantize/main.rs` 16, `daemon/main.rs` 15, `generate.rs` 8 |
| four chains in `load.rs` | `load_model` 661 (A4a wrapper) → `load_model_inner` 690, `load_model_safetensors` 3087, `load_model_pp` 3554, `unload_model` 3915 |
| `LoadedModel` | struct at `model.rs:271`, **no `Default` derive**, 20 construction sites (17 `load.rs`, 2 `model.rs`, 1 `session.rs`) |
| qwen35 guard | `load.rs:2405` (two other `is_qwen35_family_arch_id` sites: 286, 3594) |
| explicit unhandled-arch refusal | **none** |

### Inherited, NOT re-verified on `master`

| claim | figure |
|---|---|
| `LoadedModel` shape | 60 fields, 44 `Option`, 37 per-arch slots, 4 one-line methods |
| external access to those slots | 320 sites, **56 of them `.unwrap()`** |
| `unload_model` | 195 lines, 35 `if let Some(…)` teardown arms |
| `generate*` family | 13 fns, 208 params, 6,319 body lines |
| duplication | `generate_nemotron` ≡ `generate_zaya` **96%**; gemma3 pair **91%** |
| callers | **12 of 13** `generate*` have zero callers outside their own file |
| `SimpleAr` | 4 methods hiding ~460 lines, 8 impls — the one genuinely deep module |
| `ServingBackend` | 8 impls, all delegating to `run_simple_ar`; **1 of 13** `generate*` calls `.serve()` |
| `ServingFactory` | 2 impls of 20 archs |
| `SessionServingBackend` | 10 methods, **1 impl**, and it is `LoadedModel` |
| `ArchCaps` (runtime) | **0 `.caps()` call sites**; 7 of 8 impls return `default()` |
| `SpecDecodeChain` | **0 implementors** outside a test |
| family-named code outside `crates/hipfire-arch-*` | 16,681 LOC |
| cost of adding gemma4 | 133 files, **70 outside the new crate** (6.4:1) |

The single sentence the review converged on:

> `ServingBackend` / `SimpleAr` is the right seam. Eight backends implement it.
> Exactly **one of thirteen** `generate*` functions and **zero of four**
> load-path arch chains route through it.

## Recommended re-scope

Do these in order. Each is independently shippable.

### Step 0 — establish the failure mode (small, blocking)

Load a container whose `arch_id` is registered nowhere and record what happens:
error, mis-load, or panic. Until this is known, A4b has no measured
justification. If it errors cleanly today, A4b's safety argument evaporates and
only the cosmetic argument remains — which is not worth a 2,384-line diff.

### Step 1 — re-verify the inherited numbers on `master` (small)

Particularly `ArchCaps` having zero call sites and `SessionServingBackend`
having one impl. Both drive later decisions and both come from the source that
produced Correction 1.

### Step 2 — route `generate*` through the seam (review candidate 1)

The review's top recommendation, and it adds nothing: `ServingBackend::serve`
already delegates to `run_simple_ar` in all 8 impls. Deleting the nine shallow
`generate_arch.rs` modules is the change. This is what gives `LoadedModel`'s
arch slots a single consumer.

### Step 3 — collapse `LoadedModel` (review candidate 2)

`Box<dyn ServingBackend>` plus the genuinely shared state (KV cache, eviction,
conversation bookkeeping, chat template). `unload_model`'s teardown arms
collapse into one drop.

### Step 4 — then the loader dispatch, which is now small

With one backend slot instead of 37, `load_model` becomes: resolve identity →
registry yields a `ServingFactory` → call it. The ten arms become one, and the
"dispatch" A4b set out to rewrite has mostly deleted itself.

`ServingFactory` already exists (`hipfire-runtime/src/arch.rs:532`, accessor
`serving_factory` at 553) with 2 of 20 impls — it is the seam step 4 should route through, not a new one.

### Step 5 — the remaining sweeps

`session.rs` (18), `daemon/main.rs` (15), `generate.rs` (8), `quantize/main.rs`
(16), migrating onto `LoadedModel::identity()` / `is_family()` / `is_variant()`
from A4a.

## Corrections owed to other documents

Both live on branch `worktree-arch-id-plan` (PR #193) and could not be edited
from the A-phase code branch:

1. **A2's row** says "HFQ v3 … `version >= 3` → JSON authoritative". A2 shipped
   **without** a version bump: header `version` selects structural layout
   (`per_entry_tail`, tensor/data offsets) and is read by two independent
   parsers, so presence of the `identity` key is the signal instead. A bump
   would have been safe but would churn every new artifact's header for nothing.
2. **A4's row** promises "one exhaustive `match`". See Correction 2 — not
   achievable with string families, and not intended to be.

## Unrelated, still open

`Zaya` (arch 16) is absent from `worker_key_is_state_arena_conservative` in
`hipfire-scheduler`, the same gap that `Mamba2` had before PR #191. The
fail-closed backstop added there does **not** cover it, because Zaya now
resolves. Whether Zaya needs the conservative treatment was never established —
it needs someone who knows its state model, not a guess.

## Lesson worth keeping

Correction 1 exists because a measured-sounding claim was carried across three
documents and a PR body without being checked on the branch it was being applied
to. The numbers in this document are split into verified and inherited for that
reason. When a plan's justification is a number, re-measure it on the branch
you are about to change.
