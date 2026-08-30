# Routed KVarN attention is hardcoded to 4-bit — `kvarn8` and `kvarn2` read garbage

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1. Confirmed at
both the dispatch and kernel level, then fixed by threading `bits` through both —
the value was already in scope as `KvarnBatchFlushContext::bits`. `parity_kvarn_routed`
passes unchanged at max-abs-err 3.73e-8, proving the default 4-bit path is
byte-for-byte unaffected. NOTE: this was unreachable until the same day's
[uncompilable-kernel fix](2026-08-29-kvarn-routed-attention-uncompilable.md).

## Symptom

Serving a qwen35-family model with `--kv-mode kvarn8` (or `kvarn2`) through the
fused multi-session prefill path. The flush writes KVarN K records at the
cache's real width; the attention kernel reads them as if they were 4-bit. Every
record after the first is read from the wrong offset, and the codes inside it
are unpacked at the wrong width.

## Cause

Two halves disagree about the same geometry.

**The cache** (`crates/hipfire-runtime/src/kv.rs:1180`) sizes a K record from
`bits`:

```rust
pub fn kvarn_k_record_bytes_bits(head_dim: usize, bits: usize) -> usize {
    let (r, c) = (head_dim, Self::KVARN_GROUP);
    let cpb = 8 / bits;
    (r * c).div_ceil(cpb) + r * 2 * 2 + c * 2
}
```

**The attention dispatch** (`crates/hipfire-rdna/src/dispatch/attention.rs:1203`)
recomputes it with `cpb` pinned to 2:

```rust
const GROUP: usize = 128;
let rec_bytes = (head_dim * GROUP).div_ceil(2) + head_dim * 2 * 2 + GROUP * 2;
```

At `head_dim = 128`:

| kv-mode | bits | real record | dispatch assumes | |
|---|---|---|---|---|
| `kvarn2` | 2 | 4864 B | 8960 B | stride **1.84x too large** |
| `kvarn` / `kvarn4` | 4 | 8960 B | 8960 B | correct by coincidence |
| `kvarn8` | 8 | 17152 B | 8960 B | stride **1.91x too small** |

`attention_kvarn_routed_batched` takes no `bits` parameter at all, so the value
cannot reach it.

The kernel repeats the same assumption. `kernels/src/attention_kvarn_routed_batched.hip:98`
unpacks two codes per byte, unconditionally:

```c
const float q4 = (float)((idx & 1) == 0 ? (byte & 0xf) : (byte >> 4));
```

So even record 0 decodes wrong at 2 or 8 bits, before the stride error compounds.

## The tell: the sibling kernel already does this correctly

`kernels/src/attention_flash_kvarn_tile_batched.hip:118-126` takes `bits` as a
runtime parameter and derives the unpack from it:

```c
const int cpb = 8 / bits;                  // K codes per byte
const int cpb_shift = (bits == 8) ? 0 : ((bits == 4) ? 1 : 2);
const int cpb_mask = cpb - 1;
const int codemask = (1 << bits) - 1;
```

So the tree contains both a bits-aware and a bits-blind KVarN attention kernel,
and the routed batched path uses the blind one.

## Why it did not show up before

**The write side had exactly this bug and was fixed; the read side was not.**
`kvarn_batch_bits` (`crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs:853`)
carries the repair, and its doc comment describes the identical failure:

> `bits` feeds `kvarn_k_record_bytes_bits`, so it sets the record STRIDE. Reading
> `HIPFIRE_KVARN_BITS` here wrote records at the env default (4) into a cache
> allocated for whatever `kv_cache` asked for — correct only by coincidence at
> `kvarn4`, and a stride mismatch at `kvarn2` / `kvarn8`. The cache is the single
> source of truth for its own geometry.

That reasoning applies verbatim one call later, on the read. The flush now takes
`bits` from the cache and the attention still does not.

Defaults also hide it: `kvarn_bits_from_mode` (`crates/hipfire-serving-core/src/load.rs:577`)
maps bare `kvarn` and `kvarn4` to 4, and 4 is the only width where the hardcoded
arithmetic is right. Only an explicit `--kv-mode kvarn8` or `kvarn2` reaches the
broken combination.

## Fix

Thread `bits` through `attention_kvarn_routed_batched` (dispatch and kernel),
compute `rec_bytes` with `KvCache::kvarn_k_record_bytes_bits`, and copy the
`cpb` / `cpb_shift` / `codemask` unpack from `attention_flash_kvarn_tile_batched.hip`.

As a stopgap, `load.rs` could refuse `kvarn2` / `kvarn8` when the routed batched
path is eligible, so the mode fails loudly instead of serving wrong numbers.

A cheap regression pin: assert in the dispatch that the computed `rec_bytes`
equals `kvarn_k_record_bytes_bits(head_dim, cache.kvarn_bits)`. That fails today
at `kvarn8` and would have caught the original write-side bug too.
