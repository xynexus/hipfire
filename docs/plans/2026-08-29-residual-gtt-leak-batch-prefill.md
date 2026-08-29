# RESOLVED: the ~95 MB/request GTT leak in the qwen3.5 batch-prefill path

Found and fixed 2026-08-29. Root cause: **`DeltaNetState`'s FP16 `s_matrices`
were allocated with a direct `hip.malloc` but freed through the pool**, so every
session parked ~48 buffers the allocator would never look at again.

This was the remainder after `4db8dd954` fixed two larger session leaks. The
sequence on a 27B (arch 5), plain chat completions:

| state | per request |
|---|---|
| before any fix | ~1100 MB |
| after `4db8dd954` (release frees; no unusable checkpoint) | ~91 MB |
| after this fix | **0 — flat over 39 requests** |

## Root cause

`crates/hipfire-arch-qwen35/src/qwen35/state.rs`, the `StateQuant::FP16` arm of
`DeltaNetState::new_with_quant`, allocated outside the pool:

```rust
let buf = gpu.hip.malloc(s_size * 2)?;   // direct — the pool never sees it
```

while `DeltaNetState::free_gpu` returns those same tensors via
`gpu.free_tensor(t)` → `pool.free(buf)`.

**An unpooled allocation paired with a pooled free is a one-way street.** The
buffer lands on a free list keyed by its power-of-2 bucket, but the allocator
that would reuse it is never consulted, so it sits there forever. Each session
stranded ~48 buffers / ~75.5 MB — one per DeltaNet layer.

The hand-rolled `malloc` existed for a real reason: the tensor is f16 storage
carried as `DType::Raw`, and `Raw::size()` is 1, so `alloc_tensor(&[s_size],
Raw)` would have reserved half the bytes needed. The fix asks the pool for the
byte count directly and restores the shape the kernels index by:

```rust
let mut t = gpu.zeros(&[s_size * 2], DType::Raw)?;   // pooled, exact bytes
t.shape = vec![s_size];                              // shape kernels expect
```

Observable shape, dtype, size and zeroing are unchanged — only the buffer's
provenance. The identical bug in the multi-GPU variant was fixed in the same
change.

## Why every symptom followed from it

- **Fixed size per request, not decode-scaled** (95.0 MB/req at 1 token out vs
  96.8 at 19): one buffer per DeltaNet layer, allocated at session construction,
  independent of prompt or generation length.
- **Driver/GTT, never process RSS** (MemAvailable −983 MB over 10 requests while
  summed hipfire RSS moved +1 MB): `hipMalloc` memory is driver-managed, and on
  a UMA APU it comes out of the same physical pool as everything else.
- **Invisible to `/health`**: the pool cannot count what it did not allocate, and
  `resident_sessions` / `runtime_session_bytes` stayed at 0 throughout.
- **Batch-prefill path only** — the sharpest clue, and the one that made the
  bisect decisive. `HIPFIRE_SERVER_PREFILL_BATCH=0` was perfectly flat, because
  the non-batch path reuses the legacy session and therefore builds
  `DeltaNetState` once, while the batch path constructs a fresh session per
  request.

## What found it

Pool accounting, added in the same change and worth keeping:
`GpuPool::stats()` → `GpuPoolStats` (reused / new / cumulative allocated / live
free-list buffers, bytes, buckets) and `Gpu::pool_stats()`, which also reports
deferred-free mailbox depth. `qwen35_prefill.rs` logs them per prefill at debug
level.

The discriminating read, and the reason this took one measurement instead of
more guessing:

```
new=837 reused=2176  allocated=2199.2MB free_buffers=  1 free=  0.0MB   <- request 1
new=837 reused=4967  allocated=2199.2MB free_buffers= 49 free= 75.5MB
new=837 reused=7758  allocated=2199.2MB free_buffers= 97 free=151.0MB
...                                                  +48    +75.5MB per request
```

`total_new` frozen while `free_buffers` climbs linearly says precisely one
thing: buffers are being returned to a pool that never allocated them. After the
fix, `free_buffers=1` and `free=0.0MB` hold constant while `reused` climbs.

**Rule this leaves behind: allocate and free a `GpuTensor` through the same
allocator.** `GpuTensor` has no `Drop`, so a raw `hip.malloc` must be paired with
`hip.free`, and anything freed with `free_tensor` must have come from the pool.
`upload_raw` (`dispatch/mod.rs`) is the other unpooled constructor and documents
the same hazard — it is correct only for load-once/free-at-unload weights, which
is what its ~757 call sites are. Anything per-request must use the pool.

Refs: xynexus/hipfire#385.
