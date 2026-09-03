# Handover — issue #383, prefix caching inert on non-qwen3.5 families

Status: **not started.** Written 2026-09-03 from code reading only — the gfx1103
box was wedged (see `docs/bugs/2026-09-03-server-stop-wedges-gfx1103-sdma-ring.md`),
so **nothing below was executed**. Every claim is a code-path reading with a
`file:line`; the ones that need a GPU to confirm are collected in §7. Treat §1 as
the thing to check first, because it changes what the ticket is asking for.

Related: #385 (resident-session cap — collides with the fan-out use case, §6),
`docs/plans/2026-06-29-session-serving-backend.md` (the pre-existing plan that
covers most of the real work).

## 1. The ticket's diagnosis is true but not the operative blocker

#383 attributes `cached_tokens: 0` to `state_arena_backend=unsupported` for zaya
and MiniCPM. That is real and it is a blocker. It is not the *first* one, and
fixing only it would move nothing.

**Prefix reuse is not driven for any family today — including qwen3.5, which has
the full arena.** Four independent gates sit between a repeated prefix and a
cache hit, and the arena is the third:

| # | Gate | State | Where |
|---|---|---|---|
| 1 | A client must ask for reuse | **nothing does** | §2 |
| 2 | A checkpoint must have been stored first | **hardcoded off** | §3 |
| 3 | The family needs `AttachCheckpoint` | qwen3.5 only ← *the ticket* | §4 |
| 4 | The scheduler must allow the batch | qwen3.5 + 3 declared families | §5 |

So a qwen3.5 model — `Qwen35Wrapped`, every operation supported — is expected to
report `cached_tokens: 0` on a repeated prefix for exactly the same reason zaya
does. If that prediction is wrong, §1 is wrong and the ticket is right; **run the
experiment in §7.1 before planning any of this work.**

## 2. Gate 1 — nothing requests reuse

Two request shapes reach the daemon, and neither asks for a cached prefix.

**The plain path.** `GenerateTextRequest::session_id` is the field that names a
conversation, and the OpenAI-shaped constructor hardcodes it to `None`
(`crates/hipfire-generate/src/lib.rs:193`), with a comment stating the intent:

```rust
// OpenAI-shaped callers resend the whole `messages` history each
// turn, so they are stateless by protocol and want no session.
session_id: None,
```

That is a correct decision, not an oversight — the OpenAI protocol has no session
concept, so the server cannot honestly name one. The consequence is that
session-addressed reuse is unavailable on the public surface **by design**, and
the only reuse that can work there is **content-addressed** — by prefix hash.

Content-addressed reuse is exactly what the daemon already implements:
`prefix_hash_preflight` (`crates/hipfire-daemon/src/handlers/batch.rs:130`) takes a
prompt, returns candidate prefix hashes at semantic boundaries, and
`requested_prefix_hash` on a fork/attach names the one to reuse. **No production
client sends `prefix_hash_preflight`.** Its only callers are the daemon handler
and smoke tests; the scheduler merely counts it in health
(`crates/hipfire-scheduler/src/lib.rs:2167`), which is why grepping for the name
suggests more wiring than exists.

**The batch path** (`HIPFIRE_SERVER_PREFILL_BATCH`, which the reporter had on)
builds its prefill request at `crates/hipfire-server/src/batch_runner.rs:310`
with all three reuse inputs pinned to nothing:

```rust
"state_handle": {
    "state_kinds": s.state_kinds,
    "logical_position": 0,
    "cached_prefix_tokens": 0,
},
```

It does set `session_id` — but only on the *decode* step
(`batch_runner.rs:372`), to address sessions that are already resident inside the
same batch. Nothing reattaches a prior request's state.

**Note on the ticket's own caveat.** #383 correctly observes that
`cache_write_tokens` is derived as `prompt_tokens - cached_tokens`
(`crates/hipfire-server/src/routes/chat.rs:1147`) and so restates the miss. The
same applies to `cached_tokens` itself: it is read straight off the daemon's done
event (`chat.rs:1124`). With `cached_prefix_tokens: 0` going in, `0` coming back
is the request being echoed, not a measurement of the cache.

## 3. Gate 2 — nothing is stored to reuse

Reuse needs a checkpoint to attach *to*, and checkpoint creation is gated per
session by `semantic_boundary_checkpoints`. The batch runner sends `false`
(`batch_runner.rs:327`), and both families' checkpoint builders return empty
immediately on it:

- `qwen35_semantic_boundary_checkpoints` — `crates/hipfire-serving-core/src/qwen35_prefill.rs:499`
- `lfm2_semantic_boundary_checkpoints` — `crates/hipfire-serving-core/src/lfm2_prefill.rs:240`

So even a client that asked for a prefix (gate 1) would find nothing stored.

**There is no env escape hatch.** `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS`
(`qwen35_prefill.rs:502`) is opt-**out** only — it recognises `0/false/off/no` and
has no "force on" arm. Flipping the batch runner's literal to `true` is a
one-line change and is the cheapest way to make gate 2 testable (§7.1).

## 4. Gate 3 — the ticket's ask, in detail

`SequenceStateArenaBackend::for_worker_parts(arch_id, pp)` is a hardcoded arch
allowlist (`crates/hipfire-state/src/lib.rs:90`):

```rust
if is_qwen35_family_arch_id(arch_id) && pp == 1 {
    Self::Qwen35Wrapped
} else if matches!(arch_id, ARCH_ID_MINIMAX_M2 | ARCH_ID_LFM2_MOE | ARCH_ID_NEMOTRON_H) && pp == 1 {
    Self::BackendOwned
} else {
    Self::Unsupported
}
```

zaya is `ARCH_ID_ZAYA = 16` and MiniCPM is `ARCH_ID_LLAMA_MISTRAL = 0`
(`crates/hipfire-arch-api/src/lib.rs:66,79`), so both land in `Unsupported`,
matching the `/health` payload in the ticket. `AttachCheckpoint` and
`ForkCheckpoint` are listed only for `Qwen35Wrapped`
(`hipfire-state/src/lib.rs:78`).

**This is not a missing switch — it is a serving-tier boundary.** Per
`docs/plans/2026-06-29-session-serving-backend.md`, hipfire has two tiers:

- **Simple tier** — llama, qwen2, gemma3, **zaya**, nemotron — implement `SimpleAr`
  and serve through `run_simple_ar`: *stateless, one-shot, no sessions*.
- **Rich tier** — qwen3.5 (5/6) and lfm2-moe (11) — multi-session KV, prefix-hash
  prompt cache, semantic checkpoints, fork/save.

`LoadedModel` carries exactly two session registries — `q35_registry` and
`lfm2_registry` (`crates/hipfire-serving-core/src/model.rs:464,541`). A simple-tier
family has **no per-session state object at all**, so there is nothing for an
arena to hold. Giving zaya an arena means promoting it to the rich tier, which is
what that plan is for, and it is marked "design / not started".

**Read that plan before designing anything.** Its §SALVAGE is the important part:
the arch-agnostic half already exists. `SequenceState` (KV + recurrent, driven by
`MixerProfile`) is already generic in `crates/hipfire-runtime/src/sequence_state.rs:73`,
`SessionRegistry<S>` is already generic
(`crates/hipfire-serving-core/src/session.rs:67`, explicitly "S0 of the
`SessionServingBackend` hoist"), and the prefix hash itself is arch-generic in its
inputs — `compute_qwen35_prefix_hash` takes `(arch_id, kv_mode, state_kinds,
assistant_prefix, max_think_tokens, tokens)` and nothing else
(`crates/hipfire-generate/src/lib.rs:484`). Its name is the only qwen35-specific
thing about it.

What is *genuinely* Qwen-specific is smaller than it looks: the boundary markers
are hardcoded Qwen chat-template tokens — `<|im_end|>`, `<|vision_end|>`,
`<|tool_call_end|>`, `<|tool_response_end|>` (`qwen35_prefill.rs:404`) — and
`run_prefix_hash_preflight_qwen35` refuses any other arch outright
(`qwen35_prefill.rs:554`). A new family needs its own boundary marker set, not a
new hashing scheme.

## 5. Gate 4 — the scheduler predicate

`prefill_session_multi_session_batchable`
(`crates/hipfire-scheduler/src/lib.rs:1552`) admits a session if it has the
`multi_session_state_batch` / `fused_state_batch` feature flag, or is qwen3.5, or
is *not* state-arena-conservative and *not* token-ordered-recurrent — then fails
closed on an unresolved arch family (`:1572`). The fail-closed branch is correct
and should stay: fusing two sessions of a model that carries recurrent or
arena-private state corrupts it.

Worth knowing: this gate is about **fusing several sessions into one prefill
batch**, which is a different optimisation from **reusing a stored prefix**. The
ticket's fan-out scenario wants the second, and would also benefit from the
first. They are separable, and the first is much cheaper — it needs no
checkpoint storage, only a safe batchability answer.

### The MiniCPM arch-resolution defect is real, small, and worth splitting out

The ticket suggests splitting this; agreed. Mechanism, precisely:

`model_arch_family_from_str` (`crates/hipfire-model/src/lib.rs:285`) parses a
numeric arch_id, else resolves the tag through the arch registry.
`ArchRegistry::resolve` (`crates/hipfire-arch-api/src/lib.rs:380`) matches a
declared `model_types()` entry, then falls back to the spec's `family()`, both
under `normalize_arch_tag` — which strips `-`, `_`, `.`, spaces and lowercases
(`:288`).

`model_types()` **defaults to `&[]`** (`hipfire-arch-api/src/lib.rs:129`), and
most specs never override it — llama, qwen2, minimax, nemotron, dots-ocr,
embeddinggemma and gemma3-vl all declare only `id()` and `family()`. They still
resolve through the family fallback (`"llama"` → `"llama"` ✓), so this is not
broken in general.

`qwen3` is the gap. Nothing declares it as a `model_type`, and the family that
should own it — `ARCH_ID_QWEN3_QWEN2_LEGACY = 1` — has runtime label
`"qwen3-legacy"`, which normalizes to `"qwen3legacy"` and does not match
`"qwen3"`. **No spec claims arch id 1 at all.** Hence `Unknown`, hence the
fail-closed backstop.

Two things to check before fixing, because guessing wrong here is a correctness
bug and not just a missed optimisation:

1. **`qwen3` normalizes to `qwen3`, and `qwen3.5` normalizes to `qwen35`** — they
   do *not* collide, so adding `"qwen3"` somewhere cannot accidentally route a
   legacy Qwen3 model into the qwen3.5 rich tier. Verified by reading
   `normalize_arch_tag`; worth a unit test pinning it, since the two families
   differ by one character and a future normalizer change could merge them.
2. **Which family should own `qwen3`?** The local artifact
   `MiniCPM5-1B.oq4.25++.hfq` reports `architecture: llama`,
   `model_type: llama`, `input_arch_id: 0` — *not* `qwen3`. The ticket's
   `arch_id=qwen3` came from a differently-built artifact
   (`MiniCPM5-1B--oq4.25++.hfq`, the `--`-renamed one). **Get the reporter's
   artifact and read its header before assigning the tag**; do not infer the
   family from the model name.

## 6. Two reporting defects found while reading

Neither is #383, both are the same shape as the health-honesty work in #384, and
both are cheap.

**(a) `BackendOwned` advertises operations the gate refuses.**
`loaded_model_state_arena_operations` special-cases LFM2 to report
`attach_checkpoint, fork_checkpoint, release_state, describe_state`
(`crates/hipfire-serving-core/src/session.rs:1266`), while
`SequenceStateArenaBackend::require_supported` accepts `BackendOwned` **only**
for `describe_state` and errors on everything else
(`crates/hipfire-state/src/lib.rs:104`). So `/health` claims three capabilities
that every call would be rejected for.

Not currently a live break: the `sequence_state_arena_*` wrappers are called only
from `qwen35_prefill.rs` (checked — no other production caller), so for LFM2 the
advertised ops are never exercised. It becomes a real break the moment the
`SessionServingBackend` hoist routes LFM2 through them.

There is a second, subtler mismatch underneath it: the advertised operation
*names* and the strings the call sites pass are different vocabularies. Call
sites pass `"release_sessions"`, `"activate_session"`, `"fork_session_state"`,
`"checkpoint_session_state"` (`session.rs:2505-2590`); the advertised names are
`"release_state"`, `"attach_checkpoint"`, `"fork_checkpoint"`. This is invisible
today only because `Qwen35Wrapped` returns `Ok(())` without comparing the string
at all. Any future arm that actually matches on the op name will silently reject
everything.

**(b) minimax and nemotron are labelled `BackendOwned` with no session state.**
`for_worker_parts` returns `BackendOwned` for `ARCH_ID_MINIMAX_M2` and
`ARCH_ID_NEMOTRON_H`, but neither has a registry on `LoadedModel` (only
`q35_registry` and `lfm2_registry` exist), and the session-serving plan lists
nemotron in the *simple* tier. So `/health` reports a non-`Unsupported` arena for
two families that have no sessions. The label is aspirational; it should either
be `Unsupported` until a backend exists, or the aspiration should be stated in
the payload rather than implied.

## 7. Experiments — in this order

None of these have been run. §7.1 is the gate: if it fails, §1 is wrong.

1. **Does qwen3.5 report a prefix hit today?** Serve
   `qwen3.5-0.8b--oq4++.hfq`, batch runner on, and send the ticket's exact
   4-request sequence (three sharing a 418-token prefix, one control). §1
   predicts `cached_tokens: 0` on all four, identical to zaya. Confirming this
   means the arena is not the first blocker and reorders the whole plan.
2. **Flip `semantic_boundary_checkpoints` to `true`** in `batch_runner.rs:327`
   and repeat. Predicted: still `0`, because gate 1 is untouched — nothing
   requests a prefix — but checkpoints should now appear in the daemon's
   prefill events (`HIPFIRE_DEBUG_PREFIX_BOUNDARIES=1` logs the candidates,
   `qwen35_prefill.rs:511`). This separates "not stored" from "not requested".
3. **Drive the daemon directly** over stdin JSON — `prefix_hash_preflight`, then
   a `generate_batch_prefill` carrying a `requested_prefix_hash` — on qwen3.5.
   This is the only way to exercise the reuse path end to end without touching
   the server. `tests/smoke-generate-batch-prefill.sh` has the wire shapes;
   `tests/orphaned/smoke-server-prefix-checkpoint-reuse.sh` and
   `smoke-server-prefix-boundary-reuse.sh` look like prior attempts at this and
   are worth reading before writing a new one. If this path works, the whole of
   #383 for qwen3.5 is a *client-side* wiring job, not an arena job.
4. **Only then** size the zaya work, against `2026-06-29-session-serving-backend.md`.

## 8. The fan-out use case collides with #385

The ticket's motivating workload is a fan-out of N subagents sharing a
byte-identical prefix. Two facts make N matter:

- **Fork deep-copies.** `Qwen35RequestSessionState::fork_from`
  (`crates/hipfire-serving-core/src/session.rs:284`) clones the KV cache, the
  DeltaNet state and the logits tensor; `copy_on_write_attach` is `false` in
  every policy (`crates/hipfire-state/src/lib.rs:577`). So forking a shared
  prefix N ways saves the *compute* and none of the *memory*.
- **Resident sessions are capped at 8.** `resident_session_limit`
  (`session.rs:2346`) defaults to 8, clamped 1..64, from
  `HIPFIRE_SCHED_RESIDENT_STATE_MAX` — and since #385 that cap is now
  *enforced* by LRU eviction. A fan-out of 16 cannot hold 16 forks.

So "make prefix reuse work" and "make a 16-way fan-out fast" are not the same
deliverable. The second needs either a raised cap with the VRAM to back it, or
copy-on-write attach so N forks share pages until they diverge — which is a
much larger piece of work and is listed as `copy_on_write_attach: false` rather
than as a TODO anywhere.

Also relevant from #385's own write-up: eviction orders by `allocation_epoch`
(creation), not last use, which is FIFO. The `ponytail:` note there says that
coincides with LRU only while nothing re-activates a parked session — and
"nothing does today: no production caller sets a `session_id`". **Making prefix
reuse work is precisely what invalidates that assumption.** Whoever lands #383
owes #385 a touch timestamp.

## 9. What I did not check

- Whether the reporter's `zaya1-8b-native.oq4++` and `MiniCPM5-1B--oq4.25++.hfq`
  headers actually report the tags the ticket quotes. My local MiniCPM artifact
  disagrees (§5), and I could not load either.
- Anything requiring a GPU: every prediction in §7 is unrun.
- Whether `run_simple_ar` could carry a *read-only* attached prefix without the
  full rich-tier promotion. If it can, that is a much cheaper path to the
  ticket's ask than the session-serving hoist, and it is the first design
  question I would put to whoever picks this up.
- The three `tests/orphaned/smoke-server-prefix-*.sh` scripts. They are named
  exactly for this problem and may already encode a prior attempt and its
  failure — read them before starting.
