# The prefix state cache is fully built and never engages

Status: found 2026-08-30 on nix1, master `3351ec70d`. **NOT fixed** — the wiring
is two hardcoded values, but flipping them needs correctness and memory work
that is not a one-line change. Scope at the end.

## Symptom

An identical prompt sent twice reports no cache hit either time:

```
call 1: prompt=31 cached=0 cache_write=31
call 2: prompt=31 cached=0 cache_write=31
```

and `/health` shows the cache empty after both:

```
state_cache:    entries=0 bytes=0 metadata_hits=0 runtime_hits=0
prefill_batch:  cache_hits=0 cache_misses=0 resident_runtime_sessions=0
                semantic_boundary_checkpoint_entries=0
                prefix_hash_preflight_requests=0
```

Fourteen concurrent sessions against one model left
`resident_decode_sessions=0` and `state_cache_evictions_total=0`. Every request
prefills its whole prompt from scratch.

## Cause

Every component exists. Two hardcoded values keep them from meeting.

**Nothing captures a checkpoint.** `semantic_boundary_checkpoints` is set in
exactly one place, and it is a literal:

```rust
// crates/hipfire-server/src/batch_runner.rs:327
"params": {
    "assistant_prefix": s.assistant_prefix,
    "max_think_tokens": s.max_think_tokens,
    "semantic_boundary_checkpoints": false,
},
```

**Nothing performs the lookup.** `prefix_hash_preflight` is implemented end to
end — `hipfire_generate::validate_prefix_hash_preflight`, the daemon handler
(`handlers/batch.rs:130`), the executor seam (`batch_executor.rs:166`), and the
qwen35 implementation (`qwen35_prefill.rs:547`) — and **no caller anywhere in
`hipfire-server` or `hipfire-daemon-adapter` ever sends it.** The consumer of a
hit exists too: `qwen35_prefill_suffix_batch` (`qwen35_prefill.rs:574`), which
prefills only the prompt suffix.

So the daemon can answer "which of these prefix boundaries do you already have
state for", and can prefill only the remainder, but is never asked and has
nothing stored to answer with.

## Two things this is NOT

**`cache_write_tokens` is not evidence of caching.** It is derived, not
measured:

```rust
// crates/hipfire-server/src/routes/chat.rs:1079
let cache_write_tokens = prompt_tokens.saturating_sub(cached_tokens);
```

With `cached_tokens` always 0 it always equals the prompt length, which reads
like "31 tokens were cached" and means "31 tokens were not served from cache".
This is what made the gap look like a cache miss rather than an absent cache.

**`cached_tokens` cannot be non-zero on this arch.** It is emitted from one
place, and that place is the DeepSeek-V4-Flash path:

```rust
// crates/hipfire-serving-core/src/generate_arch.rs:696
let cached_tokens: usize = lcp;
```

`grep -c lcp crates/hipfire-serving-core/src/generate.rs` is **0** — the qwen35
generate path has no longest-common-prefix logic at all. For a qwen35 or llama
model the field is structurally always zero regardless of what the cache does.

## What else this explains

**Session/state eviction has never run, and is not itself broken.**
`resident_state_limit=8`, `resident_checkpoint_max=4`, and the policy is unit
tested (`session::eviction_tests`: overflow oldest-first, the active session is
never evicted, a limit of zero evicts every eligible session). It has nothing to
evict because nothing is retained. Fix the cache and this starts working
unchanged — measure it, do not modify it.

**Disk spill is working as designed, not broken.**

```rust
// crates/hipfire-scheduler/src/lib.rs:1359
let disk_spill_allowed = parse_server_prefill_policy_controls(env).state_cache_disk
    && priority >= disk_spill_min_priority;   // default 128
```

Normal traffic runs at priority 64, so spill is opportunistic-only. That is
deliberate — spilling is slow and hipfire's banded scheduler puts background work
at ≥128. `disk_spill_allowed=false` in `/health` on a chat request is correct
output. It is also moot until state is retained.

## Scope of the fix

Roughly 1-2 days, and the wiring is the easy half.

1. Stop hardcoding `semantic_boundary_checkpoints`; drive it from policy.
2. Send `prefix_hash_preflight` before a batch prefill and route a hit into
   `qwen35_prefill_suffix_batch`.
3. Report `cached_tokens` from the qwen35 path so the API contract stops lying.

The risk is correctness and cost, not wiring. Qwen3.5/3.6 is 48 of 64 layers
**linear attention**, whose recurrent state is not positionally truncatable the
way KV is — reuse needs a checkpoint captured AT the boundary, which is what
`semantic_boundary_checkpoints` does, and checkpoints cost VRAM
(`resident_checkpoint_max=4`). Before enabling by default this needs:

* a multi-turn equivalence check — a reused prefix must produce byte-identical
  output to a cold prefill, at temp 0;
* a memory measurement of checkpoint residency against the model residency cap
  added in #391 and the derived VRAM budget in #393.

Highest value of the open residency work: prefix reuse beats both eviction paths,
and shared-prefix KV reuse is load-bearing in hipfire's own scheduler design.
