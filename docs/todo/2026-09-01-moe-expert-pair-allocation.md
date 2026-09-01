# TODO: one GTT allocation per routed expert, not two

**Status:** DECIDED 2026-09-01 — **per-layer arena**, on the owner's call after
the 2 MiB granule was re-measured on the live driver. Not implemented.

The title says "pair" because that is where the investigation started. The pair
shape is now the REJECTED option: it recovers a third of the tax where an arena
recovers all of it.

## The win, measured

gfx1151 rounds every GTT allocation: ≤512 KiB to 4 KiB pages, ≤2 MiB to the next
power of two, above that to the next multiple of 2 MiB (`gtt_alloc_cost`). The
qwen35 MoE loader allocates each expert's two projections SEPARATELY, and on some
models both land just over a boundary.

Computed from the real artifacts' routed-expert tensor sizes (pure arithmetic, no
GPU needed):

| model | expert pairs | raw | separate | one/expert | saving |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B oq4.25++ | 10240 | 16.5 GiB | 30.0 (1.814x) | 20.0 (1.209x) | **10.0 GiB (33.3%)** |
| Qwen3.5-122B-A10B oq4.25++ | 12288 | 63.0 GiB | 75.8 (1.204x) | 74.5 (1.184x) | 1.3 GiB (1.7%) |

**It is model-shaped, not universal.** The 35B's pairs are ~1.6 MB and sit just
over a boundary; the 122B's are ~5.25 MB and already well above the grid. Do not
sell this as a 122B unlock — the 122B's amplification was the compact->Oq8
expansion, and that is already closed (compact-resident is default-on).

`weight_pager` already uses the packed shape for PAGED experts — one module holds
gate_up and down back to back, which is why `ResidentExpertViews` hands out byte
offsets instead of two tensors, and its comment says the rounding is "paid once
per module rather than per projection". The RESIDENT loader never adopted it.

Also ruled out: routing expert loads through the GPU pool does NOT help.
`pool.rs::alloc` recycles whole buffers, it does not sub-allocate from slabs, so
the per-tensor rounding applies either way. The allocation SHAPE is the only lever.

## Why it is not a small patch

Two approaches, both blocked on the same thing.

**A. Restructure the loaders.** `load_moe_expert` has three exits and two of them
go through `load_weight_tensor`, which every dense tensor also uses. Making it
return bytes-for-the-caller-to-upload widens the blast radius to the whole
loading path.

**B. Pack after loading** (preferred: loaders untouched, correctness is a
device-to-device copy). Allocate the combined size, `memcpy_dtod_at_auto` both
regions in, free the originals, and hand back two `WeightTensor`s over one
buffer.

B is blocked on ownership. `GpuTensor::sub_offset` produces a **`NonOwning`**
buffer, and `dispose()` carries

```rust
BufferOrigin::NonOwning => {
    debug_assert!(false, "dispose() on a non-owning buffer — aliasing bug");
}
```

so an alias reaching `free_tensor` panics in debug builds by design — the tag
exists because of #262, where an alias went back to the pool and returned as
scratch over live weights. So packing means `ExpertWeights` must learn that
`down` is a view and must not be freed, which is a lifetime change to a struct on
the load path of every MoE model.

## What to decide first

Whether `ExpertWeights` should own one buffer with two views, or keep two owned
buffers and pack only at a higher level (e.g. a per-layer arena for all experts,
which would amortise the rounding even further — 512 experts in one allocation
rather than 512 allocations).

The arena option is probably better and is not much more work than the pair
version, but it changes the pointer tables' construction (`expert_gate_up_ptrs` /
`expert_down_ptrs` become offsets into one base rather than independent
addresses), so it wants measuring against the indexed MoE kernels first.

## How to verify a fix

1. `HIPFIRE_ALLOC_REPORT=1` on `Qwen3.6-35B-A3B--oq4.25++`; the two 10240-count
   allocation lines should become one, and consumed GTT should drop ~10 GiB.
2. `./tests/tiny-affected-gate.sh` — the MoE quant + state cells must stay
   bit-identical (18 state hashes).
3. A real decode on the 35B: same argmax as before the change.

Do not measure this with a first run after rebuilding — kernels JIT-compile
inside the timed window and cost up to 3.45x. `dflash_spec_demo` now warns
(`hipfire_rdna::jit_compiles()`).


---

## DECIDED: per-layer arena (2026-09-01)

The owner asked the right question first — *does ROCm actually have to round to
2 MiB?* Re-measured on the live driver with `hipfire-rdna` example
`gtt_granularity`, which allocates N buffers of a given size and reads the real
GTT delta:

| allocation shape | bytes requested | GTT consumed | ratio |
|---|---|---|---|
| `down_proj` alone | 1 064 960 | 2 092 958 | **1.965x** |
| `gate_up` alone | 2 129 920 | 4 194 304 | **1.969x** |
| one per expert (both projections) | 3 194 880 | 4 194 304 | 1.313x |
| arena, 8 experts | 25 559 040 | 27 262 976 | 1.067x |
| **arena, 128-expert layer** | 408 944 640 | 408 944 640 | **1.000x** |

**The granule is real and mandatory — but it is charged PER ALLOCATION, so a
large enough arena amortises it to nothing.** That is what decides the fork.

### Payoff on Qwen3.6-35B-A3B (256 experts x 48 layers)

| shape | resident | vs raw | saving |
|---|---|---|---|
| separate (today) | 72.0 GiB | 1.969x | — |
| one per expert | 48.0 GiB | 1.313x | 24.0 GiB |
| **per-layer arena** | **36.6 GiB** | **1.000x** | **35.4 GiB** |

The arena saves **11.4 GiB more than pairing — 48% more**, and lands on the raw
byte count exactly.

### What the arena still has to solve

The pointer tables. `expert_gate_up_ptrs` / `expert_down_ptrs` currently hold
independent device addresses; under an arena they become base + offset. The
indexed MoE kernels dereference those slots, so the change must be measured
against them, not just assumed to be layout-neutral.

The ownership question that blocked the pair shape **does not go away** — an
arena hands out views too, and `dispose()` still `debug_assert!`s on a
`NonOwning` buffer (guard added after #262, where an alias returned from the pool
as scratch over live weights). But it is now ONE owned allocation per layer with
N views, rather than N allocations each with a view, which is a simpler lifetime
to reason about: the arena outlives every view by construction.

### Do not re-test

Routing expert loads through the GPU pool. `pool.rs::alloc` recycles whole
buffers rather than sub-allocating from slabs, so per-tensor rounding applies
either way.


---

## ⭐ The mechanism ALREADY EXISTS — this is a wiring job, not a new allocator

Checked the tree before writing an arena, and found one.

`MoeFfnWeights` already carries `raw_expert_storage: Option<RawExpertStorage>`:

```rust
pub struct RawExpertStorage {
    pub gate_up: GpuTensor,
    pub down: GpuTensor,
}
```

Its doc: *"Owning storage for source safetensors whose routed experts are stacked
as `[E, M, K]`. `experts` then contains **non-owning slice aliases** used by the
existing executor; the two backing allocations are freed once here."*

That is the arena. **Two owning allocations per LAYER** holding all E experts,
with each `ExpertWeights` a slice alias. It is built today by
`calibration_stream.rs:1747` for the stacked-safetensors source, and the teardown
is already correct — `free_moe_ffn_maybe_slab` frees the two backing allocations
once, and `free_moe_ffn` carries a `debug_assert!` refusing to run on an ffn that
has one, precisely so the aliases cannot be double-freed.

**The HFQ loader sets `raw_expert_storage: None`** (`loading.rs:1890`) and
allocates 2xE buffers per layer instead. That is the entire bug.

### Two allocations per layer is already optimal

| | bytes | multiple of 2 MiB | gtt | ratio |
|---|---|---|---|---|
| stacked `gate_up` (256 experts) | 545 259 520 | **260.0** | 545 259 520 | **1.0000x** |
| stacked `down` (256 experts) | 272 629 760 | **130.0** | 272 629 760 | **1.0000x** |

Both land on exact multiples, so there is nothing to gain from a single
whole-layer arena over the existing two-buffer shape. 35B routed experts:
**72.0 -> 36.6 GiB, the raw byte count exactly, saving 35.4 GiB.**

### The implementation, without touching the loaders

`load_moe_expert` has three exits and two go through `load_weight_tensor`, which
every dense tensor shares — restructuring it to return bytes has a wide blast
radius. It is not necessary. Allocate the two stacked buffers up front (sizes are
known from the shapes and E), then per expert: load normally, `memcpy_dtod` into
the stacked buffer at its offset, free the original. Peak overhead is the arena
plus ONE expert.

Then build each `ExpertWeights` as a `sub_offset` alias and set
`raw_expert_storage: Some(..)`. The existing free path handles the rest.

### Why this is the safe version of the idea

The ownership hazard that made this a decision — `dispose()` debug-asserts on a
`NonOwning` buffer, a guard added after #262 put pool scratch over live weights —
is already solved on this path, by code that ships. Reusing it is strictly safer
than introducing a second aliasing scheme beside it.
