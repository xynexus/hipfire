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

## ⭐ The mechanism already exists — wire it, do not build it

`MoeFfnWeights.raw_expert_storage: Option<RawExpertStorage { gate_up, down }>`
is the arena: **two owning allocations per LAYER** holding all E experts, with
each `ExpertWeights` a non-owning slice alias. Built today by
`calibration_stream.rs:1747` for stacked safetensors; teardown already correct
(`free_moe_ffn_maybe_slab` frees the two backings once, and `free_moe_ffn`
debug-asserts against running on an ffn that has one).

**The HFQ loader sets it to `None` (`loading.rs:1890`) and allocates 2xE buffers
per layer.** That is the whole bug.

Two-per-layer is already optimal — stacked `gate_up` is 260.0 x 2 MiB and
stacked `down` is 130.0 x 2 MiB, both exact, both 1.0000x. Nothing to gain from
a single whole-layer arena.

### Do it without touching the loaders

`load_moe_expert` has three exits, two through `load_weight_tensor` which every
dense tensor shares — do NOT restructure it to return bytes. Instead: allocate
the two stacked buffers up front (sizes known from shapes and E), then per
expert load normally, `memcpy_dtod` into the stacked buffer at its offset, free
the original. Peak overhead is the arena plus ONE expert. Then build each
`ExpertWeights` via `sub_offset` and set `raw_expert_storage: Some(..)`.

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
