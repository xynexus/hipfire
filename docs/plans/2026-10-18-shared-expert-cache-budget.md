# Shared expert-cache budget across resident models — scope

Status: SCOPED, not started. 2026-10-18, halo (Strix Halo gfx1151, 124 GiB UMA).

`expert_cache_mb` (this PR) is one setting for every paged-expert family, but
each resident model still gets its OWN cache of that size. This doc scopes making
it one budget shared by all of them. Line numbers are against `origin/master`
at 4530b2cc1.

## What a shared budget would cover

- **One daemon, at most `max_resident_workers` models, one running at a time.**
  The server spawns a single daemon (`hipfire-server/src/lib.rs`
  `spawn_daemon_for_serving`). It holds one active `model` plus parked
  `resident_models` (`hipfire-daemon/src/state.rs:42`), and
  `activate_model_worker` (`hipfire-serving-core/src/session.rs:1539`) parks one
  to swap in another. `max_resident_workers` defaults to **2**
  (`hipfire-config/src/lib.rs:165`). A parked model keeps its weights AND its
  pager's resident experts.
- So today's worst case is `max_resident_workers × expert_cache_mb` of expert
  cache (16 GiB at defaults), of which only one model's share is in use.
- **Only three families page:** qwen35 (opt-in, `qwen35_paged_experts`),
  deepseek4 (opt-in, `HIPFIRE_DEEPSEEK4_PAGED_EXPERTS`), qwen4_exp (always,
  when the artifact has a module table). minimax, zaya, lfm2-moe, nemotron,
  gemma4-moe and cohere2 keep every expert resident; a shared budget does not
  touch them.

## Bugs found while scoping (independent of sharing)

1. **The scheduler's accounting is wrong for every paged model except qwen35 in
   modules mode.** The daemon's `planned_usage_for_load`
   (`hipfire-daemon/src/lib.rs:~655`) charges a load at FULL FILE SIZE unless
   `residency_mode == qwen_moe_modules`. Flash-Next is charged 170 GB while it
   actually uses trunk plus 8 GiB.
2. **Family detection is a path-name check:** `qwen_moe_module_capable_model`
   (`hipfire-server/src/routes/chat.rs:965`) is `contains("qwen3") &&
   contains("a")`. Almost any qwen3* path matches — including
   `Qwen3.8-Flash-Next` (the "a" in "Flash"), which is qwen4_exp, not qwen35.
   It should be a capability the arch declares (`hipfire-arch-api` caps), not a
   filename.
3. **Modules mode claims the whole VRAM budget for one model**
   (`chat.rs:~803`, `effective_module_budget`). The ledger then charges that
   amount and evicts nearly every other worker. This PR caps the cache the
   loader actually gets at `expert_cache_mb` (`capped_expert_cache_bytes`), but
   the ledger still charges the whole budget.
4. **The deepseek4 pager may not be freed on unload.** No `free_all` call in its
   `free_gpu` path (qwen35 `layout.rs`, qwen4exp `serving.rs` both have one).
   Unverified whether dropping the `GpuTensor`s frees the memory anyway; check
   before relying on unload to give memory back.

## Why a true shared pool is hard

Each pager is owned by its model: `RefCell<WeightPager>` in qwen35 and deepseek4,
`Arc<Mutex<WeightPager>>` in qwen4exp. There is no global pager state, and a
pager is free to evict only from itself. Cross-pager eviction would need:

- **Pointer tables.** Evicting must null the owning model's pointer-table slot
  BEFORE freeing (`evict_lru_until`, `weight_pager.rs:~2332`). Only that pager
  can do that, with `&mut` to itself and to the `Gpu`.
- **Per-layer residency.** qwen35 checks `would_fit_expert_module_set`, then
  admits a layer's experts one by one (`moe_decode.rs:~476-512`). If another
  model's activity could shrink the budget in the middle of that loop, "fits"
  would stop being true.
- **The daemon is arch-agnostic.** It holds `LoadedModel`, not pager types, so
  it would need a backend method to reach a pager.

## Recommended plan (cheapest first)

**Phase 1 — trim on park (S–M). Gets ~all of the benefit.** Only one model
runs at a time, so "shared" can mean "the active model gets the budget, and a
parked one gives its cache back":

- Add `WeightPager::shrink_to(bytes, gpu)`, built on the existing
  `evict_lru_until`, so it uses the same pointer-nulling path. A `free_all` that
  keeps the pager usable would also do.
- Add a default-no-op backend method, e.g. `shrink_expert_cache(bytes)`, on the
  serving backend trait. Implement it in qwen35, deepseek4 and qwen4exp.
- In `activate_model_worker`, call it on the OUTGOING worker (to 0, or a small
  low-water mark) before swapping.
- Result: total expert cache ≤ `expert_cache_mb` + parked low-water, with no
  shared mutable state and no cross-pager locking.
- Cost: a re-activated model starts its cache cold. On UMA the page cache may
  still hold the bytes; measure the re-activation TTFT on Flash-Next before
  choosing the low-water mark.

**Phase 2 — honest accounting (S).** Fixes bugs 1–3.
- Charge every paged-expert load as trunk + `expert_cache_mb`, not file size.
- Key "pages experts" off an arch capability, not a path.
- Make the ledger charge the capped cache, not the whole budget.
- This is what lets `max_resident_workers` > 1 admit a second model next to a
  paged one instead of evicting it.

**Phase 3 — heatguard hook (S, after Phase 1).** On TIGHT memory, shrink the
ACTIVE model's cache to low-water through the same backend method. Today
heatguard can only unload whole workers (`hipfire-server/src/heatguard.rs`).

**Phase 4 — a real shared pool (M–L). Only if workers ever run concurrently.**
- An `Arc<ExpertCachePool { cap, used: AtomicU64 }>` owned by the daemon. Each
  pager charges and credits it, and evicts itself first when over.
- When its own LRU cannot cover the shortfall, it asks the daemon to
  `shrink_to` the others.
- Not worth it while execution is serial: Phase 1 already gives one budget.

## Not in scope

- Families that do not page (listed above). Making them page is a per-family
  loader change, tracked separately.
- The n-gram table on Flash-Next. It has its own host budget
  (`NGRAM_HOST_BUDGET_BYTES`) and is never GPU resident.
