# TODO: one GTT allocation per routed expert, not two

**Status:** open, designed, NOT implemented (2026-09-01). Blocked on an ownership
question that deserves a decision rather than a 4 a.m. patch.

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
