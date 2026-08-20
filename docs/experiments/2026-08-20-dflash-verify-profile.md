# Where DFlash verify actually spends its time — DeltaNet is 0.7 %

**Status:** measured 2026-08-20, rocprofv3 kernel trace of a real DFlash2 run on
Qwen3.8-27B oq4.25++ + CASK, q8 KV, 64 tokens greedy.
**This retracts the mechanism claimed in
`2026-08-20-dflash2-qwen38-27b-performance.md`.** The tok/s numbers there stand;
the explanation for them was wrong.

## What was claimed, and why it was wrong

That writeup said batched verify cannot amortize because 48 of 64 layers are
DeltaNet and its recurrence is sequential per token. It was inferred from a code
comment in the batched-prefill path ("the inner `gated_delta_net` batch_seq loop
is still sequential per token"), not measured. The comment is true and irrelevant:

    138,430 dispatches, 53.37 s span, 29.71 s GPU busy (55.7 % duty)

    kernel                                calls        ms   % GPU   us/call
    gemm_oq8_grouped_wmma                 10933     20567    69.2    1881.2
    gemv_oq8_grouped_v2                    8268      3415    11.5     413.1
    fused_gate_up_oq8_gemv                 4480      3315    11.2     739.9
    fused_qkvza_oq8_gemv                   2688      1084     3.6     403.2
    gemm_dflash_oq4_plain_dp4a_staged_8w    966       598     2.0     619.4
    gated_delta_net_f32                    3744       219     0.7      58.6

**DeltaNet is 0.7 % of GPU time.** Nothing done to it can move this workload. A
chunkwise-parallel DeltaNet was built and verified against the recurrence before
this profile was taken; switching it on changes verify by ~1 %, which is noise.
It is kept, default-off, behind `HIPFIRE_GDN_CHUNK=1` — correct, tested, and not
the lever. See `hipfire_rdna::gdn_chunk`.

## What the profile actually says

`gemm_oq8_grouped_wmma` is 69 % of GPU time at **1881 us/call**. That is the
batched Opus W8A8 GEMM verify runs over its 9 positions. Per cycle:

    520 GEMM calls x 1881 us = 978 ms   (matches the measured verify=910 ms)

Per position that is 109 ms, against a 133 ms/token plain decode. So batching 9
positions buys **18 %**, not the order of magnitude a weight-bandwidth-bound GEMM
should give — the weights are read once for the whole batch either way.

Put the two Opus kernels side by side on the same weights:

| | bytes/call | us/call | effective |
|---|---|---|---|
| `gemv_oq8_grouped_v2` (N=1) | ~89 MB | 413 | **215 GB/s** |
| `gemm_oq8_grouped_wmma` (N=9) | ~89 MB | 1881 | **47 GB/s** |

Same weight payload. The batched kernel moves it at **4.5x lower bandwidth than
the GEMV**. It is not compute-bound either — 800 MFLOP of int8 at N=9 is ~16 us
against ~55 TOPS, so 1881 us is ~100x off compute peak. It is simply a badly
tiled read at small N.

## The revised ceiling

Cycle cost is `draft 74 ms + verify 978 ms + replay 131 ms x accepted`, committing
`accepted + 1` tokens.

* at the measured accept=2: 1314 ms / 3 tokens = 438 ms/token (matches 2.27 tok/s)
* at PERFECT acceptance (accept=7, no replay): 1052 ms / 8 = **131 ms/token**

which is the 133 ms/token baseline. **The real ceiling is 1.0x, not the 1.22x
previously computed** — with a perfect drafter, DFlash exactly ties plain decode.
Acceptance is not the problem and never was.

## What would actually move it

1. **`gemm_oq8_grouped_wmma` at small N.** 69 % of GPU time running at 47 GB/s
   against the GEMV's 215 GB/s on the same bytes. If it reached GEMV bandwidth,
   verify would drop 978 ms -> ~135 ms and the cycle would be dominated by replay
   instead. This is an ordinary kernel-tuning problem (tiling / staging for N in
   8..17), not an architectural one — cf. `docs/` on the gfx1151 iu4 GEMM work,
   where wave64 + double-buffered LDS + N-heavy tiling took a similar kernel to
   ~50 % of peak.
2. **`replay`**, 131 ms per accepted token. With (1) done it becomes the majority
   of the cycle. Needs per-position DeltaNet checkpoints during verify so rollback
   is a restore, not a re-run.

With both: at accept=2, `(74 + 135) / 3` = 70 ms/token = **14 tok/s against a 7.5
baseline**. That is where speculation starts paying on this model, and neither
step is about the drafter or the recurrence.

## Method

    HIPFIRE_KV_MODE=q8 HIPFIRE_KV_ALLOW_DEPRECATED=1 HIPFIRE_DFLASH_ALLOW_OPUS=1 \
    rocprofv3 --kernel-trace --stats -d prof -o run \
      -- ./target/release/hipfire-daemon < req.jsonl

`prof/run_results.db` is SQLite. The `top_kernels` view's `total_duration` is
scaled — for real time, sum `end - start` over `rocpd_kernel_dispatch` joined to
`rocpd_info_kernel_symbol`. The percentages in the view are trustworthy; the
absolute durations are not.
