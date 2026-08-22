# RETRACTION: the FFN GEMV is NOT slower under KVarN

This retracts the central claim of
`2026-08-22-kvarn-regression-is-not-kv.md`, which said the KVarN decode
regression was "the FFN gate/up GEMV, ~7-10% slower, deterministic, with no KV
data involved" and called it the largest unexplained item.

**The FFN GEMV is identical in both KV modes.** The claim came from comparing
MEANS over a heavy-tailed distribution.

## The distribution, which I should have looked at first

`gemv_oq_compact_grouped_v3`, grid_x=524288 (FFN gate/up), same 3000-token run:

| | min | p10 | p50 | p90 | p99 | **mean** | max |
|---|---|---|---|---|---|---|---|
| f32 | 199.3 | 200.7 | 202.1 | 221.0 | 243.7 | **209.5** | 2949.9 |
| kvarn | 199.4 | 200.8 | 202.7 | 221.1 | 247.5 | **230.3** | 3279.9 |

Identical through p90. Splitting body from tail:

| | bottom 99% | top 1% |
|---|---|---|
| f32 | **209.05 us** | 2717 calls @ 253.3 us — 0.688 s total |
| kvarn | **209.31 us** | 2738 calls @ **2310.8 us** — 6.327 s total |

**Bottom-99% ratio 1.0012.** The kernel is the same. The tail's 5.6 s excess is
the entire +6.13 s I had attributed to the kernel being slower.

## What is actually there

Filtering kvarn's gate/up dispatches over 1500 us gives **2122 calls — exactly
the number of forward passes in the run, i.e. ONE PER TOKEN** — and **every one
of them is immediately preceded by `mq_rotate_x`**, with a median inter-kernel
gap of only 2.3 us.

So the real effect is a **once-per-token ~2.1 ms stall in the KVarN path**, which
lands on whatever kernel is dispatched next. It is not a property of the FFN
GEMV; the FFN just happens to be standing there. f32 has the same structural
position and the same tail COUNT (2717) but its tail calls cost 253 us, not 2310.

2122 x ~2.1 ms = ~4.5 s of the run's 148.6 s (~3%), which is the right order for
the ~3.4% end-to-end gap originally measured.

`mq_rotate_x` is KVarN-specific (70,026 calls vs f32's 7). Candidate mechanisms,
none yet confirmed: a first-touch GTT page fault as the records/window buffers
grow, or a stall behind the synchronous window-append memcpy. Worth noting that
end-to-end decode is now at parity (14.5 = f32's 14.5) after the attention
kernel fix, so whatever this is, it no longer shows in tok/s on this workload.

## Lesson

**Never compare kernel means across configurations without looking at the
distribution.** A heavy tail — here 1% of calls at 11x — moved the mean 10% while
the kernel itself moved 0.12%, and it survived six ablations precisely because
there was nothing in the kernel to ablate. Percentiles would have shown it in one
query. Ratio-of-means is the wrong statistic for GPU dispatch timings, which are
routinely heavy-tailed.

Alignment was also checked and cleared on the way: with `HIPFIRE_ALLOC_DUMP` the
large tensors (including the 47.3 MB gate/up weights) are 2 MB-aligned with
identical sizes and allocation order in both modes.
