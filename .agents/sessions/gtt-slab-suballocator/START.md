# Session: stop paying the 2 MiB GTT granule outside the expert path

**Blocked on:** step 1 (per-layer expert arena) landing first — it settles the
ownership question on one call site. **Est:** step 2 is small; step 3 is a session.
Design: `docs/plans/2026-09-01-gtt-slab-suballocator.md`.

## The finding this rests on

Measured live with `hipfire-rdna` example `gtt_granularity`, which reads the
driver's own `mem_info_gtt_used` (committed bytes, not virtual):

**The rule is ALIGNMENT, not size.** Any request that is an exact multiple of
2 MiB costs exactly its size:

| requested | consumed | ratio |
|---|---|---|
| 1 064 960 B (`down_proj`) | 2 092 958 | **1.965x** |
| 2 129 920 B (`gate_up`) | 4 194 304 | **1.969x** |
| 2 MiB | 2 MiB | **1.000x** |
| 3 MiB | 4 MiB | 1.333x |
| 4 / 8 / 64 / 1024 MiB, 1–16 GiB | exact | **1.000x** |

A 2 MiB slab is already optimal — you do not need a big arena. Today's tax exists
because OQ tensors land a hair over a boundary: every OQ block is a 4-byte scale
on a power-of-two payload (260 = 4 + 256), so `gate_up` is 2 MiB + 32 KiB and pays
a full 4 MiB. **1.6% over the line, 97% tax.**

## Objective, in order

**Step 2 — round bulk slab requests up to 2 MiB.** Mechanical, no allocator, and
it captures the tax anywhere hipfire allocates in bulk. Start by finding those
sites with `HIPFIRE_ALLOC_REPORT=1` on a real load — it already attributes
requests by size and count.

**Step 3 — a slab suballocator for LOAD-ONCE WEIGHTS ONLY.** Largest,
longest-lived, most uniform class, and where the measured tax lands.

## Do NOT

- **Do not reserve a chunk at startup.** Sizing has no good answer (0.5 GB
  fixtures to a 170 GB 180B on one box, plus paged experts that deliberately grow
  and shrink), and on UMA a reservation is taken from the page cache — the same
  mistake that made mmap'ing weights a disaster (118 GiB buff/cache vs 1 GiB via
  `pread`, and `pread` was faster). It also buys nothing a 2 MiB-aligned slab does
  not already measure at 1.000x. Keep `--gtt-reserve` as an opt-in cap for
  dedicated hosts, last, never default.
- **Do not build ONE global arena.** Weights, paged experts and scratch have
  different lifetimes; mixing them fragments. One arena per lifetime class.
- **Do not extend `GpuPool`.** It recycles whole buffers and does not
  suballocate; this replaces that path rather than growing it.

## The hazard to design against

**Every suballocation is a view.** The view/ownership machinery exists because of
**#262** — an alias went back to the pool and came back as scratch over live
weights — which is why `dispose()` carries
`debug_assert!(false, "aliasing bug")` on a `NonOwning` buffer.

A suballocator makes views the normal case rather than the exception. Make "this
pointer is borrowed from a slab that outlives it" a type-level fact, not a
convention, or this reintroduces #262 at scale.

## Verification bar

- `HIPFIRE_ALLOC_REPORT=1` before/after: allocation count falls, consumed GTT
  approaches the raw byte count.
- `mem_info_gtt_used` delta across a load vs the artifact's on-disk size.
- `./tests/tiny-affected-gate.sh` — quant and state cells bit-identical.
- A real decode: same argmax.

## Traps

- **Discard the first run after any rebuild** (JIT inside the timed window,
  3.45x, tau bit-identical).
- **Rebuild the example, not just the workspace** — `cargo build --workspace`
  does not relink examples, and a stale binary once produced a confident
  "no effect" reading.
- GTT never shows in RSS; exhaustion surfaces as a kernel `page allocation
  failure`, not an OOM kill.
