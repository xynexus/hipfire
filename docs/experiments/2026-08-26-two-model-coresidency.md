# Two fully-resident models co-reside — but `load` is the wrong way to switch

Status: measured 2026-08-26 on halo (`gfx1151`, 128.8 GB, ~120 GB GTT).
Models: `Qwen3.5-35B-A3B--oq4.25++.hfq` (17.94 GB, MoE) as worker `wA`, and
`Qwen3.5-27B--oq4.25++.hfq` (15.00 GB, dense) as worker `wB` — ~33 GB total,
deliberately different architectures.

## Question

Rounds 1 and 2 of the M8 work interleaved **one** served model with a trainer.
This asks the adjacent question: can two models that both fit stay resident
together, and what does switching between them cost?

## Answer: they do co-reside, and switching via `load` costs a full reload

| event | elapsed |
|---|---|
| cold load `wA` | 16.38 s |
| cold load `wB` | 14.61 s |
| switch back to `wA` | **12.16 s** |
| switch to `wB` | **13.02 s** |
| `wA` | **12.16 s** |
| `wB` | **13.47 s** |
| `wA` | **12.23 s** |
| `wB` | **13.51 s** |

Solo baselines for reference: `wA` cold 16.32 s / generate 0.56 s; `wB` cold
20.25 s / generate 2.39 s.

Every switch costs 12–13 s — indistinguishable from a cold load net of page
cache, and 20–30× the cost of a generate. Yet at the end:

```
{"type":"worker_status","resident_workers":2,"active_worker_key_id":"wB",
 "total_model_weight_bytes":35298560700, ...}
```

**Both models are resident** — 35,298,560,700 B ≈ 32.9 GB is `wA` + `wB`
together — while switching still pays a full reload. Those two facts look
contradictory. They are not, and the reason is worth knowing.

## Why: `load` destroys the parked copy on purpose

Three pieces of code, in the order they matter:

1. **Parking keeps GPU state.** `park_active_model`
   (`hipfire-serving-core/src/session.rs:1263`) saves session state and then
   `resident_models.insert(active_worker_id, m)` — it *moves* the `LoadedModel`,
   with its live GPU allocations, into the resident map. Nothing is freed. So a
   parked model really is still on the device, and `resident_workers: 2` is a
   truthful count, not an accounting artifact.

2. **`load` then unloads it again.** `handlers/lifecycle.rs`, immediately after
   parking the outgoing worker:

   ```rust
   if let Some(m) = daemon_state.resident_models.remove(&requested_worker_id) {
       daemon_state.generic_state_arena.release_worker(&requested_worker_id);
       unload_model(m, &mut daemon_state.gpu);          // <- parked copy destroyed
       daemon_state.resource_reservations.remove_worker(&requested_worker_id);
   }
   ```

   A `load` for a worker that is already parked **removes and unloads it**, then
   re-reads the artifact from disk. That is deliberate — `load` means "load
   fresh" — but it means `load` can never be the fast path for switching.

3. **The fast path is elsewhere.** `activate_model_worker` — the function that
   swaps a parked worker back into the active slot without reloading — is called
   from `handlers/generate.rs` (:27, :89), `handlers/sessions.rs` (:17, :96) and
   `handlers/batch.rs` (:18, :134). **It is not called from `lifecycle.rs` at
   all.** Switching without a reload goes through those paths, not through
   `load`.

   ⚠️ **Corrected 2026-08-26:** this list originally said "called *only* from
   `sessions.rs` and `batch.rs`" and omitted `generate.rs`. That omission
   mattered — `generate` carrying a `worker_key_id` activates the target worker
   by itself, so it is the simplest way to switch, and no separate activation
   call is needed. Measured cost in
   `2026-08-26-resident-swap-cost-nix1.md`: **free**.

## What this means

- **For multi-model serving:** a client that switches models by re-issuing
  `load` pays ~12–13 s per switch on a ~16 GB artifact *even though both models
  are already resident on the device*. The memory is spent and the reload is
  paid anyway — the worst of both. Route switches through the session/batch
  path.
- **For the M6/M8 residency claims:** co-residency itself is real and now
  directly evidenced (2 workers, 32.9 GB, both live). ~~What has not been
  measured here is the cost of the actual resident swap.~~ **Now measured** on
  nix1 — it is free; see `2026-08-26-resident-swap-cost-nix1.md`.

## Two measurement traps hit while getting this

**A `load` with no worker id always targets `__default__`.** The first attempt
sent plain `load` frames with no `worker_key_id`. Result: `resident_workers: 1`,
`total_model_weight_bytes` = model A alone, and each load evicting the previous
model — the two never co-resided at all. `parse_model_worker_id`
(`hipfire-model/src/lib.rs:140`) reads a **top-level** `worker_id` /
`worker_key_id`; without it every model lands in the same slot. Any multi-model
test that omits it is measuring sequential reloads and will conclude,
incorrectly, that the daemon cannot co-reside models.

**A driver that blocks on `readline()` will hang forever.** The first harness
waited for `{"type":"loaded"}` with an unbounded read. It deadlocked for 68
minutes — both processes `S`, daemon at 0.0% CPU — looking exactly like a slow
run. The daemon emits a stream of `{"type":"load_progress",...}` frames during a
load; any driver must skip those and, more importantly, must not use a blocking
read with no deadline. The working harness logs every stdout line with a
timestamp and paces with fixed sleeps, which is also what made the per-switch
timings above readable.

(Also: heredocs through `ssh` lose a quoting layer and mangled two scripts
before they were base64-encoded in transit.)

## Reproducing

On halo, `/tmp/claude-1000/m8b/`: `sw2.py` (two-worker probe, distinct
`worker_key_id`s), `sw2.stdout` (timestamped frame log), `show3.py` (timeline
extractor), plus `soloA.json` / `soloB.json` baselines and `probe_switch.py`
(the single-`__default__` negative case).

Follow-up, done: `2026-08-26-resident-swap-cost-nix1.md` measures the resident
swap against this 12–13 s `load` figure. A `generate` carrying `worker_key_id`
activates the worker itself, so no separate session/batch call is required.
