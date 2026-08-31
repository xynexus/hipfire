# `free_tensor` does not know what it is freeing

> **CLOSED 2026-08-31 — provenance exists and disposal routes on it.**
>
> The complaint below quotes `free_tensor` calling `self.pool.free(tensor.buf)`
> unconditionally. It now calls `self.dispose(tensor.buf)`
> (`hipfire-rdna/src/dispatch/mod.rs:2401`), and `dispose` matches on
> `buf.origin()`:
>
> - `BufferOrigin::Pooled` -> `self.pool.free(buf)`
> - `BufferOrigin::Direct` -> `self.hip.free(buf)`
> - `BufferOrigin::NonOwning` -> neither, which is the correct disposal for a
>   view — plus a `debug_assert!`, a `NON_OWNING_DISPOSES` counter, and a
>   one-shot warning that cites this document by name.
>
> The tag is stamped at allocation, not inferred: `GpuPool::alloc` marks
> `Pooled`, `DeviceBuffer::from_raw` / `alias` mark `NonOwning`
> (`hip-bridge/src/lib.rs:89`). The `NonOwning` arm's comment records what this
> prevented — issue #262, "where the alias went to the pool and came back as
> scratch over live weights", which is exactly the failure this document was
> written to stop.
>
> Kept as the record of why the tag exists.

`Gpu::free_tensor` returns every buffer to `GpuPool`, regardless of how that
buffer was allocated:

```rust
pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
    self.bind_thread()?;
    self.pool.free(tensor.buf);   // <- pooled, always
    Ok(())
}
```

`DeviceBuffer` is `{ ptr, size }`. It carries no provenance, so `free_tensor`
cannot distinguish three cases that must be handled differently:

| provenance | constructed by | correct disposal |
|---|---|---|
| pooled | `pool.alloc` / `upload_raw_pooled` | `pool.free` |
| direct | `hip.malloc` / `upload_raw` | `hip.free` |
| **non-owning** | `DeviceBuffer::from_raw` (slab alias, `sub_offset` view, stacked-expert slice) | **neither — `mem::forget`** |

Every call site has to remember which it holds. That is an invariant the type
system could enforce and currently does not, and it has already produced one
corruption bug, one OOM, and several leaks.

## What this has cost so far

| symptom | site | status |
|---|---|---|
| **silent weight corruption** — a slab alias freed into the pool, re-handed as scratch, written over live weights | `shard_moe_experts` | FIXED (#262) |
| **VRAM leak, ~9.6 MB/page-in**, OOM'd a 122B mid-generation; separately turned a 32768-token KLD run into a 1-minute failure | expert pager page-in (`upload_raw` + `free_tensor`) | FIXED (#253 / `c6c06b27a`) |
| whole teardown path unguarded, safe only because a loader hardcodes `slab_storage: None` | `free_gpu_multi` / `free_moe_ffn` | ASSERTED, not fixed (#263) |
| `awq_scale` sidecars leak on the `pp > 1` teardown | `free_moe_ffn` frees only `.buf` | open |
| sidecars leak — frees `l.wq.buf` instead of `WeightTensor::free_all` | `hipfire-runtime/src/dflash.rs:1230` | open |
| VRAM leak on the error path — `state.*.take()` before a `?`, originals dropped as locals, and `GpuTensor` has no `Drop` | `hipfire-arch-deepseek4/src/forward.rs:6634` | open |
| `sub_offset` views handed out by value; safe only because the drafter has no teardown | `hipfire-serving-core/src/load.rs:4275` | latent |
| conditionally-owning `hin`, guarded correctly but fragile | `hipfire-arch-zaya/src/gpu.rs:2955` | correct, fragile |
| no callers; wiring it up would admit caller-supplied tensors into the eviction free path | `weight_pager.rs:1748 insert_resident` | latent |

The pattern is not that people are careless. `free_moe_ffn` already
`mem::forget`s the `paro_shared` per-expert views — the same function that
mishandles slab aliases *knows* freeing a non-owning view is wrong. The
information needed to get it right is simply not on the value.

## The fix, and why it is not just "add a field"

Scoped in `docs/plans/2026-08-20-p3-upload-raw-allocator-symmetry.md`. Tag
`DeviceBuffer` with its provenance and route `free_tensor` on the tag. That makes
the bug unrepresentable without touching any of `upload_raw`'s ~757 call sites.

Two things make it more than a field addition:

1. **The non-owning variant must be loud.** A `free_tensor(NonOwning)` that
   silently skips is a leak; one that panics turns today's corruption into a
   crash. Panicking is correct — but it is a behaviour change, so it needs a
   decision rather than a default.
2. **It subsumes an existing mechanism.** `ModelGpuStorage::contains_tensor`
   (`layout.rs`) answers "is this mine" with a linear walk over slabs on every
   free. A tagged buffer answers it in O(1), so the hand-rolled
   `*_maybe_slab` helpers largely collapse. That is a simplification worth
   having and a bigger diff than the tag itself.

## Why the guards are not enough on their own

`free_tensor_maybe_slab` and `free_moe_ffn_maybe_slab` are correct. The problem
is that using them is optional, and the two paths that skipped them
(`shard_moe_experts`, `free_gpu_multi`) both did so by *predating* them rather
than by disagreeing. Every new teardown site is another chance to miss, and the
failure is silent.

The assertions added in #263 are an interim: they make `free_gpu_multi`'s
assumption fail loudly instead of corrupting, but they do not make the path
slab-safe. Doing that means threading `slabs` through ~40 frees, which is worth
doing only if the provenance tag is rejected.

## Where to start

`M1` of the P3 plan — enumerate the provenances — is already done, and it is what
found the corruption bug. The remaining stages are M2 (tag and route) and M3
(prove it: a `pool_churn_upload_raw` arm that does `upload_raw` in a loop freed
by `free_tensor`, requiring VRAM flat over 4000 cycles; that arm strands 400 MiB
per 200 cycles today).
