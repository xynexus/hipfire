# Routed KVarN prefill: rows in a wrapped block attend to FUTURE tokens

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1. Confirmed 3/3
by independent kernel read, then fixed by attending per segment: the kernel took
a `row_offset` argument and both call sites moved inside the segment loop.
`parity_kvarn_routed` passes unchanged at max-abs-err 3.73e-8. NOTE: this was
unreachable until the same day's
[uncompilable-kernel fix](2026-08-29-kvarn-routed-attention-uncompilable.md).
Still owed: a >=129-token fused-vs-serial A/B that would FAIL before the fix —
no such test exists yet, so the fix is verified as non-regressive, not as
curing a measured divergence.

Severity: **critical**. This is a causality violation in the default KV mode
(`kvarn`) on the fused multi-session prefill path. For a 2048-token prefill
roughly **94% of rows** read keys they must not see.

## Cause

The fused path writes the KVarN window in segments, flushing at each block
boundary, then attends **once for all rows after the loop**
(`crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs:2984-3028` dense,
`:4447-4491` grouped MoE):

```rust
let mut seg_start = 0usize;
for (seg_idx, &split) in kvarn.splits.iter().enumerate() {
    // write rows [seg_start, split) into the window
    // flush the completed block
    seg_start = split;
}
prefill_session_batch_attention_kvarn_layer(…, row_count)?;   // ALL rows, after
```

**The window is a 128-slot ring.** `crates/hipfire-runtime/src/kv.rs:1486`
allocates `[group * kv_dim]` with `KVARN_GROUP = 128` (`kv.rs:1170`), and the
writer stores at `slot = positions[b] % group`
(`kernels/src/kv_cache_write_kvarn_window_routed_batched.hip:74`). That kernel's
header states the caller contract it needs, at `:27-31`:

> The window physically holds `group` tokens, so a session writing more than
> `group` rows in one call would overwrite tokens the flush has not yet quantized.

**The attention kernel derives its window base PER ROW**, which is the fact that
decides this. `kernels/src/attention_kvarn_routed_batched.hip:56-58`:

```c
const int seq_len  = positions[b] + 1;      // b = blockIdx.y, the row
const int n_full   = seq_len / group;
const int win_base = n_full * group;
```

and its own header at `:14-15`:

> token `t < n_full*GROUP` comes from record block `t/GROUP`, and the trailing
> partial block (`t >= n_full*GROUP`) comes from the f32 window. **Each row's
> `n_full_blocks` is derived from its own `positions[b]`.**

Consumption at `:87-113` sends `t < win_base` to the records and everything else
to `window + (t - win_base) * kv_dim`. The only mask is `t < seq_len` (`:85`) —
causal against the row's own position. **Nothing masks against what the window
actually holds now.** There is no global flushed-block count in the kernarg list
(`crates/hipfire-rdna/src/dispatch/attention.rs:1223-1234`).

So the kernel assumes the window still holds *that row's* trailing partial block,
while the segment loop has already overwritten it with a later block's tokens.

## Worked example

One session prefilling positions 0..299 in a single launch. Splits cut after 127
and 255. At attention time the ring holds:

- slots 0..43 → tokens 256..299 (written by the third segment)
- slots 44..127 → tokens 172..255 (left by the second)

Row at position 100: `seq_len = 101`, `n_full = 0`, `win_base = 0`. So `t < win_base`
is never true and **all of t = 0..100 read the window** — returning tokens
256..299 and 172..228. Every one of them is in the future relative to position 100.

A single block boundary is enough; ≥129 tokens in one launch triggers it.

## Which rows are correct

Two classes survive:

1. rows whose position exactly ends a block (`win_base == seq_len`), which read
   only flushed records; and
2. every row in the session's **final** in-flight block for that launch, whose
   window slots no later segment has overwritten.

Everything else — every row in a block a later segment wrapped — is corrupt.

## The tell: the single-session path already has the fix

`crates/hipfire-rdna/src/dispatch/kv.rs:1817` passes the block count as a launch
**argument** to `attention_flash_kvarn_batched_masked`, and the caller attends
**inside** the segment loop, before the flush (`:1824-1836`). That is commit
`8ea5a303e`. The routed batched dispatch takes no such parameter and attends
after the loop — the same fix, not carried across.

## Why existing verification missed it

BUGS.md records `[FIXED 2026-08-11] Routed KVarN prefill wrote K in the WRONG
BASIS — both arms`, which re-verified all three write paths numerically with
`HIPFIRE_KVARN_DUMP`. That work compared **cache contents**, which are identical
either way here, and it ran on 19–25 token prompts — well under one 128-token
window group, so it could not have exercised a wrap at all.

## Fix

Move the attention call inside the segment loop, after that segment's write and
flush, restricted to rows `[seg_start, split)`. The write helper
`kv_cache_write_kvarn_window_routed_batched` already carries `row_offset` /
`row_count` for exactly this reason; give the attention dispatch the same pair,
mirroring what `dispatch/kv.rs:1817-1836` does for the single-session path.

To pin it: prefill ≥129 tokens through the fused path and diff the attention
output against the serial path. Any per-row assertion that
`win_base + slots_read <= tokens_currently_resident` would also have caught it.
