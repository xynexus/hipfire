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

## But there is a real residual, and it is NOT the 4-bit codec

Control, same tool, `--kv-mode fp32`:

> **IDENTICAL across all layers (worst 0.00e0)**

So the batched machinery — chunking, batched attention, the write ordering — is
exactly right when KV is unquantized. Every bit of the KVarN divergence is
KVarN's own.

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

## Next step

Diff the window-region read in `attention_flash_kvarn_tile_batched` against the
per-token `kvarn_attend` path, at n=64 where the records are empty and only the
window is live. A single-layer, single-head dump at n=64 should localize it to
either the window write layout or the kernel's masking of the window half.
