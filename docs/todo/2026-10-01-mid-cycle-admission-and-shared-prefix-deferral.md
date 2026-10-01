# TODO: admit requests into a running batch; defer siblings that share an uncached prefix

> **CLOSED 2026-10-01** — both parts built and verified live on the 27B:
> staggered B (same prefix, arriving 5 s into A's 600-token generation) now
> starts decoding at 5.0 s and answers at 6.0 s (was 46.6 s), reusing 4260
> tokens; the D/E/F batch prefills `p3` once (one 1294-token prefill, two
> 23-token attaches; was 3 x 1294), 16.1 -> 7.7 s. Throughput unchanged at 16/64
> sessions; mixed-length bursts correct.

Status: DONE (see banner).
Branch: `feat/swarm-batching-paged-kv`.

## Measured behaviour (live, Qwen3.8-27B--oq4.25++, 2026-10-01)

Requests built as `[system p1][user p2|p3][assistant][user pX]`:

| job | prefilled | reused | |
|---|---|---|---|
| A (cold) | 4268 | 0 | mints `p1` and `p1+p2` |
| B, C (same batch as A) | 23 | 4244 | deferred one round, attach A's `p1+p2` |
| D, E, F (next batch) | 1294 each | 2973 | attach `p1`; **each re-prefills `p3`** |

Staggered: A generating 600 tokens, B (same prefix, same priority) arriving 5 s
later — B's prefill started only when A's whole batch finished, ~41 s after it
arrived, then reused 4260 tokens.

So prefix reuse works across batches and stacks at chat-turn boundaries, but:

1. **No admission into a running batch.** A batch cycle (`run_batch_cycle`) runs
   until every request in it finishes. A same-priority request waits for all of
   it; a strictly-higher-priority one parks it (cooperative preemption). For a
   swarm, whose tasks arrive staggered (follow-ups emitted mid-turn, the coder
   after research), late arrivals sit idle behind a long-running batch.
2. **Siblings with a hit don't share a deeper miss.** `plan_prefix_reuse` defers
   sessions to a second round only when they have no cached prefix at all
   (B, C). D/E/F had a hit (`p1`), so all three attached it immediately and
   each prefilled `p3` — 3x the work one mint plus two attaches would do.

## Part 1 — mid-cycle admission (continuous batching proper)

Between decode steps, a running text cycle takes queued requests it can run and
prefills them into the batch:

- **Which.** Queued `TokenPrefill` workloads with the cycle's microbatch key (the
  worker key: same model/worker) and priority at least as urgent as the batch's
  (`priority <= running_priority`), up to `batch_max() - active`. Less urgent
  work keeps waiting — admitting it would slow the running sessions, which is
  what the bands exist to prevent. Order: priority bucket, then the scheduler's
  own order within it.
- **Scheduler API.** New `ContinuousWorkScheduler::take_microbatch_compatible(now,
  key, class, max_priority, limit) -> Vec<WorkloadSpec>`: removes matching queued
  workloads without granting a lease. Text workloads carry zero resources, so
  there is nothing to account; the running cycle's lease covers the GPU turn.
- **Prefill.** `prefill_with_prefix_reuse` for the newcomers alone (prefix index
  shared with the cycle, so they attach whatever the batch already cached),
  then fold their positions / remaining (KV-capacity clamp) / txs / specs into
  the cycle's maps; they decode from the next step. Running sessions stall for
  that one prefill call (bounded by the row budget) — the usual continuous-
  batching trade; chunked interleaving is a later refinement.
- **Failure.** A newcomer prefill error fails (and releases) only the newcomers;
  the running sessions continue.
- **Interaction with parking.** Admission runs before the preemption check, so a
  higher-priority request on the same model joins instead of parking the batch;
  parking still applies to work that cannot join (other model, other class,
  batch full).
- **Switch.** `HIPFIRE_SERVER_MIDCYCLE_ADMIT=0` restores cycle-granular batching.

Tests: scheduler unit tests for the take (key / class / priority filters, limit,
order, leaves the rest queued). Live: the staggered A/B run above — B must start
decoding within a step or two of arriving (not ~41 s) and still reuse the prefix;
plus a mixed-length cold burst and a 64-session run for correctness.

## Part 2 — defer siblings that share an uncached deeper prefix

In `plan_prefix_reuse`, after each session's longest cached hit is known: group
sessions by the deepest boundary hash they share **beyond** their hit. In each
group of 2+, one session runs now (prefilling and minting the shared boundary);
the rest defer to the next round and attach it. Same two-round mechanism that
already serves the no-hit case (`deferred`), extended to "hit, but a deeper
shared miss".

Test: unit test of the planner (D/E/F shape: shared hit `p1`, shared miss
`p1+p3` -> one runs, two deferred); live: D/E/F must show one 1294-token prefill
and two ~25-token attaches.

## Out of scope

Chunked prefill interleaved with decode steps (newcomer prefill split across
steps), preempting lower-priority sessions out of a running batch to make room,
and cross-worker admission.
