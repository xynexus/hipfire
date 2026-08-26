# `load` eviction semantics, the GTT ceiling, and why M6's guard must not use the ledger

Status: measured 2026-08-26 on nix1 (`gfx1103`, **42.0 GB GTT**).
Companions: `2026-08-26-three-model-coresidency-nix1.md` (3 models co-reside),
`2026-08-26-resident-swap-cost-nix1.md` (the swap is free),
`2026-08-26-two-model-coresidency.md` (halo, `load` reloads).

## Does loading a new model evict the current one?

**It depends entirely on the worker id.** `handlers/lifecycle.rs`:

```rust
if requested_worker_id == daemon_state.active_worker_id {
    …
    unload_model(m, &mut daemon_state.gpu);                    // SAME id -> evicted
} else {
    park_active_model(…, &mut daemon_state.resident_models)?;  // NEW id  -> parked, kept
}
if let Some(m) = daemon_state.resident_models.remove(&requested_worker_id) {
    unload_model(m, &mut daemon_state.gpu);   // a parked copy OF THE TARGET is destroyed
}
```

| action | effect on the model currently active |
|---|---|
| `load` with a **new** `worker_key_id` | **parked** — stays resident, GPU state intact |
| `load` with the **same** id, or none (→ `__default__`) | **evicted** |
| `load` targeting an id that already holds a parked copy | that copy destroyed, artifact re-read (8–29 s) |
| `unload_worker` | memory properly returned |

Eviction is **per slot, not global**: a load only ever destroys whatever occupies
the id it is loading into. Omitting `worker_key_id` is the trap — everything
lands in `__default__`, so every load evicts the last and the daemon looks
incapable of co-residency.

## There is no pressure-driven eviction — a load that does not fit OOMs

Three models resident, then a fourth:

| load | result | resident | ledger | GTT |
|---|---|---|---|---|
| `wA` 35B-A3B | loaded | 1 | 17.77 GB | 23.41 GB |
| `wB` Qwen3-4B | loaded | 2 | 25.27 GB | 32.31 GB |
| `wC` Llama-3.2-3B | loaded | 3 | 31.25 GB | 39.47 GB |
| `wD` | **error**, 3.3 s | 3 | 31.25 GB | 41.97 GB |

```
hipMalloc(49807360 bytes = 47.50 MiB), free=41.9 MiB of total=43008.0 MiB (hipError=2)
```

Nothing is evicted to make room — `resident_models` has no eviction policy, and
`plan_model_residency` (which *does* evict) is called only from
`hipfire-server`, never from the daemon. The daemon finds out a model does not
fit by allocating until the driver refuses.

**The survivors are unharmed.** `resident_workers` stays 3, the ledger is
unchanged, and a generate on `wA` after the failure returns text **bit-identical**
to its pre-failure output. The failed load fails in isolation.

## The retained 2.5 GB is pool reuse, not a leak

After the failure GTT sits at 41.97 GB — up 2.5 GB — and stays there through
+20 s, +60 s and beyond. That looks like a leak. It is not. Three consecutive
failing loads:

| step | GTT | Δ |
|---|---|---|
| after 3 loads | 39.47 GB | |
| fail #1 | 41.97 GB | **+2.50** |
| fail #2 | 41.97 GB | **+0.00** |
| fail #3 | 41.97 GB | **+0.00** |
| `unload_worker wC` | 34.82 GB | **−7.15** |

The first failure fills the tensor pool to the ceiling; later failures **reuse**
it. Not cumulative, so repeated failed loads cannot exhaust the box. And
`unload_worker` returns memory properly — 7.15 GB for a model the ledger counted
as 5.98 GB, the difference being KV and runtime state.

Corroborating detail: fail #1 died on a 47.5 MiB allocation, fails #2 and #3 on
a **742 MiB** one. With the pool already warm they got further into the load
before hitting the wall — which is what pool reuse looks like from outside.

*(A "leaks 2.5 GB per failed load" claim and "retains 2.5 GB in a reusable pool"
are indistinguishable from a single failure. They differ enormously in severity,
so the repeat is what separates them.)*

## What this means for M6's VRAM guard

The failure mode is **safe but late**. Nothing consults a budget before a load,
so the daemon streams weights until the allocator refuses. A pre-flight check
would refuse in milliseconds with "needs ~7.5 GB, 2.5 GB free" instead.

⚠️ **The guard must size against real GTT, not the daemon's ledger.** At three
models the ledger read **31.25 GB** while actual GTT was **39.47 GB** — the
ledger counts model weights only, not KV, runtime state, scratch, or transient
load buffers. A ledger-based guard would have seen 31.25 of 42 GB, concluded
there was 10.7 GB spare, and cheerfully approved the load that OOM'd.

Use `/sys/class/drm/card0/device/mem_info_gtt_used` against
`mem_info_gtt_total`. Note `rocm-smi` is useless on these APUs — it reports only
the tiny dedicated carveout (0.2 GB on nix1, 512 MB on halo), not the GTT pool
where these models actually live.

## Reproducing

In this session's scratch directory: `cap.py` (fourth-load capacity test),
`leak.py` (retention over time plus a survivor-correctness check), `leak2.py`
(three consecutive failures plus `unload_worker`).
