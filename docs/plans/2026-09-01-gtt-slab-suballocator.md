# Should hipfire manage its own GTT? Measured answer: suballocate, do not reserve

**Date:** 2026-09-01 · **Box:** halo, gfx1151, 128 GB UMA, ROCm 7.14
**Status:** design, not started. Step 1 (per-layer expert arena) is already
decided separately — see `2026-09-01-moe-expert-pair-allocation.md`.

## The question

Should the per-layer expert arena generalise into hipfire owning **all** GTT —
reserving a chunk at startup and suballocating everything from it?

## The measurement that answers it

`hipfire-rdna` example `gtt_granularity` allocates N buffers of a given size and
reads the driver's own accounting (`mem_info_gtt_used`), so these are committed
bytes, not virtual reservations.

| requested | GTT consumed | ratio |
|---|---|---|
| 1 064 960 B (`down_proj`) | 2 092 958 | **1.965x** |
| 2 129 920 B (`gate_up`) | 4 194 304 | **1.969x** |
| 3 194 880 B (one expert, both) | 4 194 304 | 1.313x |
| **2 MiB** | 2 MiB | **1.000x** |
| 3 MiB | 4 MiB | 1.333x |
| **4 / 6 / 8 / 12 / 16 / 64 / 256 / 1024 MiB** | exact | **1.000x** |
| 1 / 4 / 8 / 16 GiB (single alloc) | exact | **1.000x** |

**The rule is alignment, not size.** Any request that is an exact multiple of
2 MiB costs exactly its size. A 2 MiB slab is already optimal; a 3 MiB one is
not. Today's tax exists purely because OQ tensors land a hair over a boundary —
every OQ block is a 4-byte scale on a power-of-two payload (260 = 4 + 256), so
2 129 920 B is 2 MiB + 32 KiB and pays a full 4 MiB. **1.6% over the line, 97%
tax.**

Single allocations up to 16 GiB succeed at 1.000x, so large contiguous GTT is
viable on this hardware.

## Recommendation

**Yes to suballocating from 2 MiB-aligned slabs. No to a fixed startup
reservation.** They are separable, and all of the win is in the first.

### Why not reserve at startup

- **Sizing has no good answer.** hipfire serves 0.5 GB fixtures and a 170 GB 180B
  on one box, and paged experts deliberately grow and shrink against a budget.
  Too large starves the OS, too small fails later, and today the kernel arbitrates
  that dynamically at no cost.
- **On UMA, reserved GTT is taken from the page cache.** There is direct
  precedent: mmap'ing weights filled the page cache with pages GTT could not
  reclaim (118 GiB buff/cache vs 1 GiB via `pread`, and `pread` was FASTER). A
  static reservation is the same mistake by a different mechanism.
- **The failure mode is invisible.** GTT never appears in RSS, and exhaustion
  surfaces as a kernel `page allocation failure`, not an OOM kill.
- **It buys nothing the slab does not.** A 2 MiB-aligned slab already measures
  1.000x. Reservation would only pre-empt competition for memory, which is a
  scheduling policy, not an allocation win.

Keep it as an optional cap (`--gtt-reserve`) for dedicated serving hosts. Never
the default.

### Arenas per LIFETIME CLASS, not one global arena

The three classes have genuinely different lifetimes, and mixing them is the
standard route to fragmentation:

| class | lifetime | shape |
|---|---|---|
| model weights | load-once, free at unload | large, uniform, never freed individually |
| routed experts (paged) | LRU churn | uniform, high turnover |
| scratch | per-op | small, bursty, reused |

Weights are the right first target: largest, longest-lived, most uniform, and the
class where the measured tax actually lands.

## The hazard this invites

**Every suballocation is a view**, and the view/ownership machinery exists
*because* of #262 — an alias went back to the pool and came back as scratch over
live weights. `dispose()` carries `debug_assert!(false, "aliasing bug")` on a
`NonOwning` buffer for that reason.

A suballocator makes views the normal case rather than the exception. Whatever
shape it takes must make "this pointer is borrowed from a slab that outlives it"
a type-level fact, not a convention.

Note also that today's `GpuPool` does **not** suballocate — `pool.rs::alloc`
recycles whole buffers or `hipMalloc`s at the requested size. This replaces it
rather than extending it.

## Staged path

Each step is independently valuable and derisks the next.

1. **Per-layer expert arena** — decided; 72.0 -> 36.6 GiB on Qwen3.6-35B-A3B.
   One call site, one lifetime class, and it settles the ownership question in
   the small.
2. **Round every slab request up to 2 MiB** wherever hipfire allocates in bulk.
   Cheap, mechanical, and captures the tax with no allocator.
3. **A slab suballocator for load-once weights.** Measure against
   `HIPFIRE_ALLOC_REPORT=1`, which already attributes requests by size and count.
4. **Scratch, only if measured to matter**, and in its own arena.
5. `--gtt-reserve` as an opt-in cap, last.

## How to verify any step

- `HIPFIRE_ALLOC_REPORT=1` before/after: allocation count should fall and
  consumed GTT should approach the raw byte count.
- `mem_info_gtt_used` delta across a load, against the artifact's on-disk size.
- `./tests/tiny-affected-gate.sh` — quant and state cells bit-identical.
- A real decode: same argmax.

⚠️ Do not measure the first run after a rebuild — kernels JIT-compile inside the
timed window (3.45x, with tau bit-identical). And rebuild the *example*, not just
the workspace: `cargo build --workspace` does not relink examples, which once
produced a confident "no effect" reading from a stale binary.
