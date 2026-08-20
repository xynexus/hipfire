# Plan: P3 — make allocation and free symmetric

Split out of `2026-08-20-v2-prerequisites-autonomous.md`, which re-scoped P3 as
its own project after finding it is not a queue item.

## The defect

`Gpu::upload_raw` allocates with `hip.malloc` directly; `Gpu::free_tensor`
unconditionally returns the buffer to `GpuPool`:

```rust
pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
    self.bind_thread()?;
    self.pool.free(tensor.buf);   // <- pooled, regardless of where buf came from
    Ok(())
}
```

Pair those in a loop and every allocation is fresh while every free piles into a
list nothing draws from. Not theoretical: it leaked ~9.6 MB per page-in and OOM'd
a 122B mid-generation, and turned a 32768-token KLD run into a 1-minute failure.
Both call sites are fixed; **the API shape that produced them is not.**

## Constraints, measured

* **~757 `upload_raw` call sites.** Most are load-once/free-at-unload and
  perfectly correct.
* **The asymmetry is a borrow constraint.** `GpuPool::alloc(&mut self)`, `Gpu`
  holds the pool by value, `upload_raw` is offered behind `&self`. A pooled
  upload cannot be offered on that signature.
* **`GpuPool` buckets by power-of-two rounded size**
  (`free_lists: HashMap<usize, Vec<DeviceBuffer>>`), so a 1.59 MiB routed expert
  occupies a 2 MiB bucket — ~26% waste.
* **`DeviceBuffer` is `{ ptr, size }`** — it carries no provenance today.

## Candidate designs

### A — provenance on the buffer (leading candidate)

Tag `DeviceBuffer` with how it was allocated; `free_tensor` routes on the tag —
pooled buffers to `pool.free`, direct ones to `hip.free`.

**Why it leads:** it makes the bug *unrepresentable* without touching a single
one of the 757 call sites, and it fixes the direction that actually leaks. Cost
is one enum-sized field on a struct that is already `{ptr, size}`.

**The trap that must be handled first:** `DeviceBuffer::from_raw` constructs a
**non-owning** buffer. That is a third provenance, and it must be freed by
NEITHER path. Before writing anything, establish whether such a buffer can reach
`free_tensor` today — if it can, this plan has found a double-free/aliasing bug
that outranks the leak and should be reported before proceeding.

### B — `&mut self` on `upload_raw`

Mechanically correct, ~757 call sites, and some callers hold `&Gpu` deliberately.
Large diff, high review cost, no behavioural win over A.

### C — interior mutability on `GpuPool`

A `RefCell` is arguably redundant (the daemon threads `Gpu` as `&mut` anyway) and
can panic on re-entrancy; a `Mutex` puts a lock on an allocation hot path.
Changes the allocator's aliasing story for no gain A does not deliver.

### D — fixed-frame slabs

The parent plan's original idea, and still worth doing — `ExpertShape` says every
routed expert in a layer is one size, so admission is `free.pop()` and eviction
`free.push()`, which also recovers the ~26% bucket waste. **Orthogonal to A:**
slabs make churn cheap, A makes mispairing impossible. Do A first; it is smaller
and it is the correctness half.

## Staging

**M1 — establish the provenance cases.** Enumerate every way a `DeviceBuffer` is
constructed (`malloc`, `pool.alloc`, `from_raw`, any others) and determine which
can reach `free_tensor`. *Exit:* a written list, and an explicit answer on
whether a non-owning buffer can reach it. **If it can, stop and report.**

**M2 — tag and route.** Add the provenance field, route `free_tensor`, and make
the non-owning case a loud refusal rather than a silent skip. *Exit:*
`cargo check --workspace --all-targets` clean; `./tests/no-gpu-ci.sh` exit 0;
`pool_churn_upload_raw` still PASSes.

**M3 — prove the bug is now unrepresentable.** Extend `pool_churn_upload_raw`
with an arm that does exactly what the pager used to: `upload_raw` in a loop,
freed with `free_tensor`. *Exit:* VRAM flat across 4000 cycles. Today that arm
strands 400 MiB per 200 cycles, so it is a real before/after.

**M4 (optional, separate PR) — fixed-frame slabs.** Only after M3.

## Verification

Every stage: `./tests/no-gpu-ci.sh`, and `./tests/tiny-affected-gate.sh
--require-coverage` for the runtime change — reporting honestly if it answers
"no tiny coverage selected", which it likely will, since the tiny fixtures do not
churn allocations. `pool_churn_upload_raw` is the real coverage here.

Do not change serving numerics. This is an allocator-routing change; if any
logit moves, something is wrong.

## Stop and report rather than proceed

* a non-owning buffer can reach `free_tensor` (M1);
* the fix would require touching more than ~20 call sites — that means design A
  has failed and the choice should be re-made, not forced;
* `pool_churn_upload_raw` regresses in any arm;
* anything that would alter serving numerics.
