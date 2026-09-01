# Session: one GTT allocation per routed expert

**BLOCKED — needs an owner decision before any code.** Registered as item 7 in
`docs/todo/DECISIONS-PENDING.md`. Design: `docs/todo/2026-09-01-moe-expert-pair-allocation.md`.

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

## What must be decided first

Packing after load is the low-risk shape (loaders untouched, correctness is a
device-to-device copy). But `GpuTensor::sub_offset` yields a `NonOwning` buffer,
and `dispose()` carries:

```rust
BufferOrigin::NonOwning => {
    debug_assert!(false, "dispose() on a non-owning buffer — aliasing bug");
}
```

That guard exists because of **#262, where an alias went back to the pool and
came back as scratch over live weights**. The fork:

- **one buffer with two views** — smallest diff; `ExpertWeights` must learn that
  `down` is a view, plus an explicit exemption from the alias invariant;
- **a per-layer arena** — amortises further (512 experts in one allocation), but
  changes `expert_gate_up_ptrs` / `expert_down_ptrs` from independent addresses to
  offsets into one base, so it wants measuring against the indexed MoE kernels
  first.

Either branch deliberately routes around a guard added in response to memory
corruption. **Do not start without the call.**

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
