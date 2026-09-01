# Session: stack the HFQ loader's routed experts into per-layer storage

**DONE 2026-09-01 — `484af81bd`.** Landed on `docs/moe-alloc-decision`.
Measured on `Qwen3.6-35B-A3B--oq4.25++`: peak GTT **23.08 → 19.13 GiB**
(−3.96 GiB), decode 60.3 → 60.0 tok/s, output byte-identical, tiny gate green.

⚠️ **The −35.4 GiB payoff below did NOT hold. The real number is −3.96 GiB.**
Mechanism was right, magnitude was 9x optimistic — see "Why the prediction was
wrong" at the bottom. Everything from here down is the original brief, kept for
the reasoning; read the correction first.

---

**UNBLOCKED 2026-09-01 — decided: reuse `RawExpertStorage` (two owning
allocations per LAYER).** Not "an arena per expert", and not a new allocator.

Design: `docs/todo/2026-09-01-moe-expert-pair-allocation.md` ·
Decision: `docs/todo/DECISIONS-PENDING.md` item 7 ·
Granule evidence: `docs/plans/2026-09-01-gtt-slab-suballocator.md`

## The one-line bug

`MoeFfnWeights.raw_expert_storage` is the mechanism, it ships, and **the HFQ
loader sets it to `None` and allocates 2×E buffers per layer instead.**

## Payoff — Qwen3.6-35B-A3B, 256 experts × 48 layers

| shape | resident | vs raw |
|---|---|---|
| today: 2×E allocations per layer | **72.0 GiB** | 1.969x |
| `RawExpertStorage`: 2 per layer | **36.6 GiB** | **1.0000x** |

**Saves 35.4 GiB**, landing on the raw byte count exactly.

⚠️ **Model-shaped.** The benefit depends where per-expert size sits on the 2 MiB
grid. It is **not** a 122B unlock — that model's amplification was the
compact→Oq8 expansion, already closed by compact-resident being default.

## Why two-per-layer is already optimal

Measured live (`hipfire-rdna` example `gtt_granularity`, reading the driver's own
`mem_info_gtt_used`). **The rule is ALIGNMENT, not size** — any exact multiple of
2 MiB costs exactly its size:

| requested | consumed | ratio |
|---|---|---|
| `down_proj` alone, 1 064 960 B | 2 092 958 | **1.965x** |
| `gate_up` alone, 2 129 920 B | 4 194 304 | **1.969x** |
| one per expert, 3 194 880 B | 4 194 304 | 1.313x |
| stacked `gate_up`, 545 259 520 B = **260.0 × 2 MiB** | exact | **1.0000x** |
| stacked `down`, 272 629 760 B = **130.0 × 2 MiB** | exact | **1.0000x** |

Both stacked buffers land on exact multiples, so a single whole-layer arena buys
nothing over the two-buffer shape that already exists.

## The reference implementation to copy

`calibration_stream.rs:1675-1715` builds exactly this for stacked safetensors:

```rust
let gate_up_stride = 2 * intermediate * dim * dtype.size();
let down_stride    = dim * intermediate * dtype.size();
for expert in 0..experts {
    let gate_up_ptr = (pending.tensor(gate_up_storage).buf.as_ptr() as usize
        + expert * gate_up_stride) as *mut c_void;
    // ... same for down
    gate_up_ptrs.push(gate_up_ptr as u64);          // the pointer table slot
    expert_weights.push(ExpertWeights {
        gate_up: alias_weight(gate_up_ptr, gate_up_stride, dtype, 2*intermediate, dim),
        down:    alias_weight(down_ptr,    down_stride,    dtype, dim, intermediate),
    });
}
// ... raw_expert_storage: Some(RawExpertStorage { gate_up, down })   (line 1747)
```

Note `alias_weight` and the explicit pointer arithmetic — the pointer tables are
built from the same addresses as the aliases, so they stay consistent by
construction.

## The call sites to change

| file:line | what |
|---|---|
| `loading.rs:6784` | `for x in 0..n_exp` — the HFQ routed-expert loop (two `load_moe_expert` calls per expert) |
| `loading.rs:6813` | `gu_ptrs` / `dn_ptrs` built from `e.gate_up.buf.buf.as_ptr()` — becomes base + offset |
| `loading.rs:6911` | `raw_expert_storage: None` — the one to populate |
| `loading.rs:1794`, `1850`, `1890` | the PARO loop and its tables — same shape, likely a second instance |
| `loading.rs:2550`, `7050` | two more `for x in 0..n_exp` / `None` sites — check whether they are the same path |

## Do it WITHOUT touching the loaders

`load_moe_expert` has three exits and two go through `load_weight_tensor`, which
every dense tensor shares. **Do not restructure it to return bytes** — that is a
workspace-wide blast radius for no gain.

Instead:

1. Allocate the two stacked buffers up front. Sizes are known from the shapes and
   `n_exp` before any expert is read.
2. Per expert: `load_moe_expert` as today → `memcpy_dtod_at_auto` into the stacked
   buffer at `x * stride` → `free_tensor` the original.
3. Build each `ExpertWeights` as an alias (see the reference above) and populate
   the pointer tables from the same addresses.
4. Set `raw_expert_storage: Some(RawExpertStorage { gate_up, down })`.

Peak overhead is the stacked buffers plus ONE expert.

**You must ADD the stride check — qwen35 does not have one.** ⚠️ Verified
2026-09-01: the HFQ loop calls `load_moe_expert` per expert independently with no
uniformity assertion, and `expert_gate_up_dtypes` / `expert_down_dtypes`
(`loading.rs:1864-1865`) are per-expert VECTORS, so a mixed-dtype layer is
structurally representable today. A stacked buffer needs one stride for all E.

Copy the guard from `qwen4exp/src/trunk_gpu.rs:295-320`, which does exactly this
for its own stacking: take expert 0's dtype and byte length as the stride, require
every later expert to match both, and on mismatch fall the WHOLE projection back
to the unstacked path with a named warning. Its comment is the reason — "a
differing stride would silently misalign every later expert". Falling back per
expert is not an option; the layout is all-or-nothing.

## What this still has to solve

**The pointer tables.** `expert_gate_up_ptrs` / `expert_down_ptrs` hold
independent device addresses today; they become base + offset. The indexed MoE
kernels dereference those slots, so **measure against those kernels** rather than
assuming the layout is neutral.

**Ownership — already solved on this path, do not reinvent it.**
`GpuTensor::sub_offset` yields a `NonOwning` buffer and `dispose()` carries
`debug_assert!(false, "aliasing bug")` — a guard added after **#262**, where an
alias went back to the pool and came back as scratch over live weights.

`RawExpertStorage` is exactly how that is handled correctly today:
`free_moe_ffn_maybe_slab` frees the two backings once, and `free_moe_ffn`
debug-asserts against running on an ffn that has one. **Reuse that teardown; do
not add a second aliasing scheme beside it.**

## Ruled out — do not re-test

Routing expert loads through the GPU pool. `pool.rs::alloc` recycles whole
buffers rather than sub-allocating from slabs, so per-tensor rounding applies
either way. The allocation SHAPE is the only lever.

## Verification bar

1. **`HIPFIRE_ALLOC_REPORT=1`** on `Qwen3.6-35B-A3B--oq4.25++`: the two
   high-count expert allocation lines (10240 × 2129920 B and 10240 × 1064960 B)
   collapse to 2 per layer, and consumed GTT drops ~35 GiB.
2. **`mem_info_gtt_used`** delta across the load, against the artifact's on-disk
   size — should approach 1.0x for the expert share.
3. **`./tests/tiny-affected-gate.sh`** — MoE quant and state cells bit-identical
   (18 state hashes). This is the correctness bar; the aliases must produce the
   same bytes the separate allocations did.
4. **A real decode on the 35B**: same argmax as before the change.

## Traps

- **Discard the first run after any rebuild** — kernels JIT-compile inside the
  timed window (3.45x, with tau bit-identical).
- **Rebuild the example, not just the workspace.** `cargo build --workspace` does
  not relink examples; a stale binary once produced a confident "no effect"
  reading that only the pager counters exposed.
- GTT never appears in RSS, and exhaustion surfaces as a kernel `page allocation
  failure`, not an OOM kill.
- ⚠️ `mamba2/fp16` in the tiny-state gate is currently RED for unrelated reasons
  (`docs/bugs/2026-09-01-mamba2-tiny-state-drift.md`) — logit-only, token stream
  identical. Do not read it as caused by this change; check the other 17 cells.


---

## Why the prediction was wrong (measured 2026-09-01, after the change landed)

The payoff table said −35.4 GiB at 1.969x. The load shows **−3.96 GiB at
1.247x**. Three compounding errors, all in the same direction:

**1. The per-expert size was wrong for this artifact.** The brief's 2 129 920 B
gate_up / 1 064 960 B down are twice the real ones. `Qwen3.6-35B-A3B--oq4.25++`
carries **1 114 112 B gate_up and 557 056 B down**, and 40 layers, not 48.

**2. `gtt_granularity` measures ONE SIZE AT A TIME, and that is not the shape
the loader allocates in.** Re-run against the real sizes, the example still
reproduces its own numbers:

| requested | per-alloc consumed | ratio |
|---|---|---|
| 1 114 112 (gate_up) | 2 096 103 | 1.881x |
| 557 056 (down) | 698 352 | 1.254x |

But the driver **suballocates from 2 MiB blocks** — it is not a flat 2 MiB per
allocation, which is why 557 056 measures 1.254x and not 3.76x: three of them
pack into one block. In a uniform stream only ONE 1 114 112 B gate_up fits per
block, so it measures 1.881x. **The loader's stream is not uniform.** It
alternates gate_up, down, gate_up, down — and `1 114 112 + 557 056 =
1 671 168`, which fits ONE 2 MiB block. The pair costs 2 MiB, i.e. **1.255x**,
and the load measured **1.247x**. The isolated 1.881x was never available to
win back.

**3. So the rule is not "alignment, not size" — it is packing.** An exact 2 MiB
multiple still costs exactly its size (the stacked buffers here are
285 212 672 B = 136.0 x 2 MiB and 142 606 336 B = 68.0 x 2 MiB, both exact), so
the fix is right. But a sub-2 MiB allocation does not automatically pay a whole
granule; it pays for the fraction of a block its NEIGHBOURS cannot use.

**What to take forward.** Before predicting a GTT win from allocation shape,
measure the sizes the loader actually emits (`HIPFIRE_ALLOC_REPORT=1`) and feed
the interleaved stream, not one size in a loop. `gtt_granularity` answers "what
does N of THIS size cost", which is a different and more pessimistic question.

The change is still worth having: −3.96 GiB is 17% of this model's footprint,
throughput is unchanged, and it collapsed two divergent MoE teardown paths into
one that no longer leaks the AWQ sidecars or the stride tensors.

**Follow-on, if a bigger win is wanted:** the residual tax is now in the
2 MiB-block packing of everything ELSE, not the experts. `HIPFIRE_ALLOC_REPORT`
on the post-change binary shows the next-largest repeated shapes are 10 561 x
8 192 B and 10 445 x 2 048 B — small, numerous, and unstacked.
