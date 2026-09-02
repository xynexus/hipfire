# The KV divergence: a batched read of a quantised KV cache ≠ the same reads done one row at a time

Status: **LOCALISED 2026-09-02**, mechanism named, exact kernel not yet
identified. This is the common root under two symptoms that were being chased
separately.

## The finding

Reading a quantised KV cache for N rows in one batched call does not produce the
same result as N single-row reads of the same cache. Unquantised (fp32) KV does
not have the problem. The KV *bit width* does not materially change it.

It surfaces in two places, which is why it looked like two bugs:

| symptom | batched reader | single-row reader |
|---|---|---|
| prefill hidden states differ | batched prefill (256-row chunks) | per-token prefill |
| speculative output ≠ AR output | speculative verify (b rows) | AR decode (1 row) |

Both vanish at fp32 KV. Neither responds to KV bit width.

## Evidence

**Decode (spec vs AR).** 300 greedy tokens, same prompt, b=16, via
`HIPFIRE_KV_MODE` (see the caveat below about `kv_cache`):

    kvarn 4-bit   AR 77ab36b1  spec 675ebc8a   differs
    kvarn 8-bit   AR 86f354c7  spec 2db61eab   differs
    fp32          AR c27c9545  spec c27c9545   BYTE-IDENTICAL

Batched prefill is NOT the cause here: with `HIPFIRE_PREFILL_BATCHED=0` and
kvarn4, spec still differs from AR (`e1ae445f` vs `6a735f9c`). The KV mode is the
variable that matters.

**Prefill (batched vs per-token).** `compare_prefill_hidden_paths`, real 0.8B,
sweeping `--n` across the batch boundary (`PREFILL_MAX_BATCH = 256`):

    n=256   IDENTICAL          (batched arm ran — confirmed)
    n=257   FIRST DIVERGING LAYER 0, worst 6.18e-1
    n=264   ... 6.18e-1
    n=512   ... 6.18e-1

A step function exactly at the point prefill needs a SECOND batched call, and
already at full magnitude when the second batch holds a single token. Not
accumulation.

## What is ruled out

Each of these was tested, not argued:

- **KV bit width.** 2/4/8-bit KVarN give 6.18e-1, 6.18e-1, 6.18e-1. Per-layer the
  width moves L3 (4.89e-1 / 5.08e-1 / 5.46e-1) — deterministic across repeat runs
  — but it moves the WRONG WAY, growing with precision. Quantisation error would
  fall ~64x from 2 to 8 bits.
- **DeltaNet state precision.** The whole sweep at `HIPFIRE_DN_STATE_FP16=0`
  (`dn_quant=FP32` confirmed in-log) reproduces the same numbers, and spec/AR
  still diverges with FP32 forced.
- **KVarN block-flush causality.** `grouped_moe_prefill_session_batch_kvarn_block_flushes`
  cuts a segment after every row that completes a block and flushes only then, so
  rows before a flush read the fp16 window and rows after read the sealed record.
  It is correct, and the dense and grouped-MoE paths share it.
- **Hidden ring buffer.** Capacity is exactly `n` with `max_batch = n`, so no
  wrap; `commit_staging_to_ring` copies exactly `n` rows at `head`. No overwrite
  of earlier batches.
- **Batched prefill, as the cause of the spec/AR symptom.** Disabling it leaves
  the divergence (above).
- Earlier and separately: `repeat_penalty` (arch default is 1.0 for qwen3.5, the
  same value speculation uses) and the rollback replay path (serial and batched
  diverge identically).

## The open puzzle

At `n=257` the worst-diverging rows are **2, 9, 66, 148, 169, 220** — all in the
FIRST batch, not the lone second-batch row. Batch 1 is processed identically
whether or not a batch 2 follows, and at `n=256` the two arms agree with the
batched path confirmed running. A row's hidden state changing because a later
token exists is a causality violation, and the three obvious carriers (flush
planning, ring wrap, ring commit) are all ruled out above.

That is the thread to pull next: instrument which layer's K/V read differs for a
batch-1 row between the one-batch and two-batch cases.

## Methodological warning, earned three times

This probe reports `IDENTICAL (worst 0.00e0)` in at least three situations where
nothing was compared:

1. `--n <= 256` — prefill is one chunk and attention never reads the KV cache.
   The probe warns about this; `tiny-prefill-gate.sh` ran at `--n 32` for its
   whole life and every per-KV-mode row it printed was structurally zero.
2. **fp32 KV** — the batched arm silently declines (no `[features] prefill
   batched` line) and both arms run per-token. So "fp32 is exact" is NOT
   established by this probe; it is established by the decode measurement above,
   which is independent.
3. Any configuration where the batched path declines for its own reasons.

Always confirm the batched arm actually ran before believing a zero. Twice in
this investigation a conclusion was drawn, and once a correct conclusion was
retracted, on a number produced by a comparison that did not happen.

## Related

- `2026-09-02-kv-cache-setting-never-reaches-the-loader.md` — why every "KV mode"
  comparison before this was secretly the same mode.
- `2026-09-01-spec-decode-not-output-equivalent-to-ar.md` — the decode symptom.
