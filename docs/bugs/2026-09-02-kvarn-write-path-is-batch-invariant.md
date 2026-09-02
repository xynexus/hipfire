# The KVarN write and read paths ARE batch-invariant — the divergence is not there

Status: **NEGATIVE RESULT, measured.** Closes the hypothesis left standing by
`2026-09-02-kv-batched-vs-single-row-read-divergence.md`, and redirects the hunt.

## The test

`crates/hipfire-runtime/examples/kvarn_write_batch_parity.rs` — no model, no
generation. Two KVarN caches with identical geometry, identical synthetic K/V/Q,
driven through `kvarn_attend` on two schedules:

- **A**: batched, `--batch` rows per call (what speculation does with `b` rows)
- **B**: strictly one row per call (what AR decode does)

Then it compares the stored bytes (`k_gpu` records, `k_window`) AND the
per-row attention outputs.

## Result: identical, everywhere tested

    n=300 batch=17 bits=4   records 0/143360 differ   attend 0/300 rows differ
    n=300 batch=16 bits=4   records 0/143360 differ   attend 0/300 rows differ
    n=300 batch=17 bits=8   records 0/274432 differ   attend 0/300 rows differ
    n=300 batch=17 bits=2   records 0/ 77824 differ   attend 0/300 rows differ
    n=400 batch=32 bits=4   records 0/143360 differ   attend 0/400 rows differ

Aligned (16, 32) and unaligned (17) batches, every supported bit width, both the
write and the read. Bit-exact.

**Non-vacuity guard.** A KVarN block seals every 128 rows, so `--n` below 128
leaves the record buffer untouched and comparing two never-written buffers is a
guaranteed pass. The test counts non-zero bytes and exits INCONCLUSIVE rather
than PASS when nothing was written:

    --n 40    records 0/143360 differ (0 non-zero)      INCONCLUSIVE
    --n 300   records 0/143360 differ (71616 non-zero)  PASS

That guard is not decoration. This investigation was misled three separate times
by a probe reporting `IDENTICAL (0.00e0)` for a comparison that never happened.

## Where the divergence is NOT

Four carriers are now eliminated by measurement:

| candidate | verdict | how |
|---|---|---|
| sealed KVarN records perturbed by rejected drafts | no | change only on block completion, 0 exceptions in 137 steps |
| stale window rows read by attention | no | poisoning them changes nothing; the control that also poisons committed rows DOES change output |
| KVarN write path batch-dependence | no | this document |
| KVarN read path batch-dependence | no | this document |

## Where it probably is: the DeltaNet path, not the KV cache

The prefill probe reports `FIRST DIVERGING LAYER: 0`. On this model **layer 0
has no KV cache at all**: the full-attention layers are 3, 7, 11, 15, 19, 23 (6
of 24), and every other layer — including 0 — is `linear_attn`, the DeltaNet
recurrence.

So the batched-vs-per-token prefill divergence appears *before any KV layer is
reached*, and cannot be caused by the KV cache. That is consistent with its
bit-width independence, and with the daemon's own prefill warning, which names
the DeltaNet path rather than the KV: "the per-token fallback rounds the FP16
DeltaNet state once per token where the batched path rounds once per chunk ...
the KV mode is not the lever".

FP32 state was tested and did not close the gap — but rounding cadence is only
one way a chunked recurrence can differ from a sequential one. The recurrence
math itself is the untested half.

**Next test, same shape as this one**: drive the DeltaNet layer with N rows in
one chunked call and with N single-row calls, and compare the resulting state and
outputs. If they differ, that is the origin.

## One result this does NOT explain, and should not be assumed away

At decode, `fp32` KV makes speculative output **byte-identical** to AR while every
quantised width diverges — measured through the daemon on real generations, so it
does not depend on the prefill probe. If the KV cache is batch-invariant, that
result needs another explanation: most likely that selecting `fp32` KV also
changes which attention and prefill routes execute, rather than only the storage
format. Worth confirming before anyone concludes "fp32 KV is the fix".
