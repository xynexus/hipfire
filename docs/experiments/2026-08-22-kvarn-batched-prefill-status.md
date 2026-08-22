# KVarN batched prefill: the 57x gap is stale — it is 2.3x, and it is in the window path

State box: halo, gfx1151, 128 GB UMA. Qwen3.8-27B oq4.25++, `kv_mode=kvarn`,
`compare_prefill_hidden_paths`, hidden states diffed layer-by-layer against the
same `HiddenStateRingBuffer` both paths write.

## Headline

The recorded defect — "batched prefill + KVarN is 57x less faithful than
per-token (0.93 vs 0.016 vs an fp32-KV ref)" — **no longer reproduces at that
magnitude.** Current measurement:

| | worst-case rel. error vs fp32-KV ref |
|---|---|
| batched | 2.766e-2 |
| per-token | 1.203e-2 |

**2.3x, not 57x.** Most of the gap closed as a side effect of this branch's
earlier KVarN batched-prefill work (`fa_kv_ok` admitting `quant_kvarn`, the
`KvTierPlan::derive` block skipped under KVarN, and `kvarn_quantize_tile`
actually reaching the batched path — the kernel trace shows 256 calls, exactly
16 complete blocks x 16 FA layers for a 2059-token prompt).

**Practically it is currently invisible.** 64 greedy tokens after a 2059-token
prompt are **byte-identical** (sha256 match) whether the prompt was prefilled
batched or per-token.

## ⚠️ CORRECTION: the fp32 "control" used below was DEGENERATE

An earlier revision of this note argued from `--kv-mode fp32` reporting
**IDENTICAL across all layers (0.00e0)** that "the batched machinery is exactly
right and every bit of the divergence is KVarN's own". **That inference was
wrong.** fp32 KV fails `fa_kv_ok`, so it never takes the batched path at all —
measured independently: prefill stays at 15.0 tok/s with or without
`HIPFIRE_PREFILL_BATCHED=1`. The "control" was comparing per-token against
per-token, so 0.00e0 was guaranteed and proved nothing.

The valid control is a quantized KV that actually batches. `--kv-mode q8` (the
tool builds the cache directly, bypassing the load-path deprecation gate):

| kv mode | batched vs per-token, worst rel |
|---|---|
| q8 | **1.58e-2** |
| kvarn | **1.62e-2** |

Essentially equal. **The residual is not KVarN-specific.** It is the generic
batched-vs-per-token divergence, and the first diverging layer is 0 — a
**LinearAttention** layer with no KV at all (the first FullAttention layer is 3;
layers 0-2 have empty KV buffers). A KV-mode difference cannot explain a
divergence in a layer that has no KV.

This also reinstates the prior finding that batched and per-token prefill export
different hidden states for every dtype. That note was right; the "correction"
made from the degenerate fp32 control was not.

Divergence vs prompt length, batched vs per-token, first diverging layer 0 in
every case:

| n | worst rel. error |
|---|---|
| 64 | 1.62e-2 |
| 127 | 2.29e-2 |
| 128 | 3.90e-2 |
| 200 | 3.90e-2 |

**The n=64 row is the important one.** KVarN stores complete 128-token blocks as
4-bit records plus an **f32 recent-window ring** for the partial trailing block.
At n=64 no block completes, so every K row is still f32 in the window — *no
quantization has happened at all* — and batched still disagrees with per-token.
The error then roughly doubles at n=128 when the first block flushes, and
plateaus.

Conclusion: **the defect is in the KVarN f32 window path under batched prefill,
not in the 4-bit codec.** The codec flush adds a second, smaller increment on
top.

Prime suspect: `attention_flash_kvarn_tile_batched`'s handling of the window
region. The batched path writes all `row_count` rows into the window in
segments and *then* runs one attention call for all rows, where per-token writes
and attends one position at a time. That ordering is provably fine under fp32 KV
(0.00e0 control), so what differs is how the KVarN kernel reads/masks the window
half of the frame.

## Instrument caveat, recorded so the next reader does not misread it

The tool's two "against the fp32-KV reference" numbers are a **max over all
layers and rows**, not a mean. They saturate: they are bit-identical under
`HIPFIRE_KVARN_ROTATE=1` and `=0` even though the batched-vs-per-token figure
moves 1.62e-2 -> 2.83e-2. That is consistent with the worst-case element being
dominated by V's Q8 error, which the K-side Hadamard rotation deliberately does
not touch. Use the layer-wise batched-vs-per-token number as the live signal;
treat the two summary rows as a saturating worst case.

Separately: `kld_eval` is **blind to KV mode**. fp32-KV and kvarn score
1.87e-10 against each other — the float floor — because teacher-forced scoring
reads prefill's own logits and batched attention computes from fresh Q/K/V
without reading the cache back. Do not use `kld_eval` to measure a KV change.

## Root cause found and FIXED: flush-before-attend in `kvarn_attend`

The kernel's masking is correct — `seq_len = positions[global_bid] + 1` bounds
each row by its OWN position, and `tile_end = min(tile_start + group, seq_len)`,
so a row never reads a future token. The bug was one level up, in `kvarn_attend`.

It did: write **every** row into the window (flushing completed blocks as it
went), then run **one** flash for the whole batch with
`n_full_blocks = seq_len / GROUP`. That scalar is derived from the batch's FINAL
length but applied to every row, while the kernel bounds rows individually. So a
row at t=50 inside a 128-row batch read block 0 as a **4-bit record**, where the
per-token path reads that same K as **f32 window**: the batch's trailing flush
retroactively quantized K the row should have seen unquantized.

Fix: segment at block boundaries and **attend each segment before flushing it**,
passing that segment's own `n_full_blocks = block`. This is the same
segment-then-flush shape the session-batched path already used
(`kvarn.splits` / `kvarn.flushes`). Tree-verify keeps the single-call path,
because its bias is indexed `tree_bias[global_bid * block_cols + ...]` and
splitting the batch would renumber `global_bid`.

Result — the flush-boundary step is gone:

| n | before | after |
|---|---|---|
| 64 | 1.62e-2 | 1.62e-2 |
| 127 | 2.29e-2 | 2.29e-2 |
| 128 | **3.90e-2** | **2.29e-2** |
| 200 | **3.90e-2** | **2.29e-2** |

Free: prefill 216.7/217.5 tok/s (was ~217), decode 14.2, decode text sha256
unchanged, and the `--kv-mode fp32` control still IDENTICAL across all layers.

## Residual: real, but NOT a KV bug

A length-dependent divergence remains (1.62e-2 at n=64, 2.29e-2 at n>=127). It is
present at n=64 where no block completes and every K row is still f32 — and
`--kv-mode q8` shows the same 1.58e-2, and the first diverging layer is 0, a
LinearAttention layer with no KV. So it is the generic batched-vs-per-token
difference, not KVarN and not the window path.

`dump_kvarn_window` (added with this work) shows the K window and Q8 V plane
differing on every FullAttention layer at n=64 — but both sides are fully
populated with equal magnitudes (max 4.964 vs 4.964, all 65536 live floats
nonzero on both), a ~0.9% relative difference in every element. That is the
*downstream consequence* of hidden states that already differ by layer 0, not an
independent write-side defect: identical inputs would produce identical writes.

Chasing it further means the DeltaNet / LinearAttention chunked path, not the KV
cache. `HIPFIRE_KVARN_ROTATE=0` does not remove it, so it is not the Hadamard
rotation either.
