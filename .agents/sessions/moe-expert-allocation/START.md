# Session: one GTT allocation per routed expert

**UNBLOCKED 2026-09-01 — the decision is made: per-layer arena.**
Design: `docs/todo/2026-09-01-moe-expert-pair-allocation.md`.
Decision record: `docs/todo/DECISIONS-PENDING.md` item 7.

## The payoff, computed from real artifacts

gfx1151 rounds every GTT allocation, and the qwen35 loader allocates each
expert's two projections separately. Through `gtt_alloc_cost`:

| model | separate | one/expert | saving |
|---|---|---|---|
| Qwen3.6-35B-A3B oq4.25++ | 30.0 GiB | 20.0 GiB | **10.0 GiB (33.3%)** |
| Qwen3.5-122B-A10B oq4.25++ | 75.8 GiB | 74.5 GiB | 1.3 GiB (1.7%) |

**Model-shaped, and NOT a 122B unlock** — that model's amplification was the
compact->Oq8 expansion, already closed by compact-resident being default. The
benefit depends where per-expert size sits on the 2 MiB grid.

## The shape, and why — measured, not assumed

The owner asked whether ROCm actually has to round to 2 MiB before choosing.
Re-measured live (`hipfire-rdna` example `gtt_granularity`):

| allocation shape | requested | GTT consumed | ratio |
|---|---|---|---|
| `down_proj` alone | 1 064 960 | 2 092 958 | **1.965x** |
| `gate_up` alone | 2 129 920 | 4 194 304 | **1.969x** |
| one per expert | 3 194 880 | 4 194 304 | 1.313x |
| **arena, 128-expert layer** | 408 944 640 | 408 944 640 | **1.000x** |

The granule is real but charged **per allocation**, so an arena amortises it to
nothing. On Qwen3.6-35B-A3B: separate 72.0 GiB, pair 48.0, **arena 36.6** — the
raw byte count exactly, and 11.4 GiB better than pairing.

## What the implementation still has to solve

**The pointer tables.** `expert_gate_up_ptrs` / `expert_down_ptrs` hold
independent device addresses today; under an arena they become base + offset, and
the indexed MoE kernels dereference those slots. Measure against those kernels
rather than assuming the layout is neutral.

**Ownership.** `GpuTensor::sub_offset` yields a `NonOwning` buffer, and
`dispose()` carries:

```rust
BufferOrigin::NonOwning => {
    debug_assert!(false, "dispose() on a non-owning buffer — aliasing bug");
}
```

That guard exists because of **#262, where an alias went back to the pool and
came back as scratch over live weights**. The fork:

An arena still hands out views, so this still applies — but it is now ONE owned
allocation per layer with N views, rather than N allocations each with a view.
The arena outlives every view by construction, which is a much simpler lifetime
than the pair shape would have had.

## Ruled out — do not re-test

Routing expert loads through the GPU pool. `pool.rs::alloc` recycles whole
buffers rather than sub-allocating from slabs, so per-tensor rounding applies
either way. The allocation SHAPE is the only lever.

## Verification bar, once unblocked

1. `HIPFIRE_ALLOC_REPORT=1` on `Qwen3.6-35B-A3B--oq4.25++`: the two 10240-count
   allocation lines become one, consumed GTT drops ~10 GiB.
2. `./tests/tiny-affected-gate.sh` — MoE quant and state cells bit-identical
   (18 state hashes).
3. A real decode on the 35B: same argmax as before.
