# The resident model swap is free — 0 ms vs 8–29 s for a `load`

Status: measured 2026-08-26 on nix1 (`gfx1103`, 42.0 GB GTT).
Prior: `2026-08-26-two-model-coresidency.md` (halo, `load` switching costs a full
reload), `2026-08-26-three-model-coresidency-nix1.md` (3 models co-reside).

## The open number

Both co-residency experiments switched models by re-issuing `load`, which costs
a full reload by design — `lifecycle.rs` unloads the parked copy before
re-reading. The cost of the *real* swap, `activate_model_worker`, was left
unmeasured. It is the number that decides whether multi-model serving is
worthwhile, so this measures it.

## Setup

Three workers resident (`resident_workers: 3` confirmed at the end):

| worker | artifact | on disk |
|---|---|---|
| `wA` | `Qwen3.6-35B-A3B--oq4.hfq` | 17.79 GB |
| `wB` | `Qwen3-4B--bf16.hfq` | 5.01 GB |
| `wC` | `Llama-3.2-3B-Instruct--bf16.hfq` | 3.99 GB |

**A `generate` carrying `worker_key_id` activates that worker itself**
(`handlers/generate.rs:27, :89`), so the swap can be measured as generate
latency with no separate activation call. Every worker is warmed once first so
nothing is first-touch.

## Result: the swap costs nothing measurable

Same worker repeatedly (`wC`, no swap):

```
2.289  2.289  2.286  2.284   -> mean 2.287 s
```

Alternating workers, so **every** generate forces a swap:

| worker | run 1 | run 2 | correct output |
|---|---|---|---|
| `wA` | 1.021 s | 1.024 s | yes |
| `wB` | 1.923 s | 1.921 s | yes |
| `wC` | **2.284 s** | **2.289 s** | yes |

The comparison that matters is `wC` against itself: **2.287 s with no swap,
2.286 s average when every generate swaps into it.** The difference is below
measurement noise. The spread across workers (1.02 / 1.92 / 2.29 s) is just each
model's own generate speed, not swap cost.

| switching mechanism | cost |
|---|---|
| re-issue `load` | **8.3 – 28.6 s** (full reload) |
| `generate` with `worker_key_id` | **~0 ms** |

`activate_model_worker` (`serving-core/session.rs:1368`) explains it — the whole
body is a park-and-move between the active slot and a `HashMap`:

```rust
park_active_model(model, gpu, active_worker_id, resident_models)?;
if let Some(m) = resident_models.remove(worker_id) {
    *active_worker_id = worker_id.to_string();
    *model = Some(m);
```

No allocation, no upload, no re-read. Pointer movement plus a session save.

### The correctness check, which a timing test alone would not give

A swap that silently failed to activate would also be fast. So each generate's
output is compared against that worker's own warmed baseline, and the three
baselines are plainly distinct models:

```
wA: "<think>\nHere's a thinking process:\n\n1. "
wB: "<think>\nOkay, the user is asking for the cap"
wC: '{"name": "wikipedia", "parameters": {"action'
```

All 6 alternating generates matched their own worker's baseline. The right model
served every time — fast *and* correct, not fast *because* nothing happened.

## Consequence

Multi-model serving on one daemon is viable: N models resident, swap between
them for free, bounded only by GTT (and by the ledger caveat below). What is not
viable is switching with `load`, which pays 8–29 s *and* holds the memory
anyway.

Two things still bound it:

- **Memory, not time.** Per `2026-08-26-three-model-coresidency-nix1.md`, three
  models put real GTT at 39.79 GB of 42 GB (95%), while the daemon's ledger read
  31.25 GB. Size a multi-model deployment against `mem_info_gtt_used`, not
  against `total_model_weight_bytes`.
- **Concurrency.** This is fast *serial* switching — one active model at a time.
  It is not two models executing concurrently.

## ⚠️ Correction carried back

`2026-08-26-two-model-coresidency.md` stated `activate_model_worker` is "called
only from `handlers/sessions.rs` and `handlers/batch.rs`". **It is also called
from `handlers/generate.rs` (:27, :89)**, which the original grep missed. The
omission mattered: it implied switching required the session/batch control
plane, when a plain `generate` with a `worker_key_id` does it. That doc is
corrected.

## A discarded first attempt, and why its number was not trustworthy

The first probe timed `release_sessions` (which also activates) and reported
0.000–0.001 s. Two problems made that unusable as evidence:

1. It returned `error` for two of the three workers — activation happens
   *before* the request parse that failed, so the swap did occur, but the timing
   was of an error return.
2. Since `generate` *also* activates, a following generate would have covered
   for a swap that never happened — the measurement could not distinguish them.

Rerunning as generate latency removes both ambiguities: the same worker is
compared with and without a preceding swap, so any real cost would appear as a
difference, and the output check proves the swap took effect.

## Reproducing

`swap2.py` in this session's scratch directory: loads three workers, warms each,
then contrasts repeated same-worker generates against strictly alternating ones,
comparing every output to its baseline.
