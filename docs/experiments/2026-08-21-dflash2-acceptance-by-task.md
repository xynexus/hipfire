# DFlash2 acceptance is strongly task-dependent — and my benchmark prompt was its worst case

Qwen3.8-27B oq4.25++ / DFlash2 oq4+, halo (gfx1151), 96 tokens per row, one
model load via `--prompts-file`, batched rollback on. Measured after the
`fa_batched_ok` fix.

| task | tok/s | tau | accept_rate |
|---|---|---|---|
| translate | 12.32 | 5.00 | 0.714 |
| json | 11.83 | 4.88 | 0.696 |
| math | 10.05 | 3.90 | 0.557 |
| repetitive | 9.92 | 3.88 | 0.554 |
| code | 8.81 | 3.57 | 0.510 |
| sql | 7.89 | 3.08 | 0.441 |
| factoid | 6.93 | 2.84 | 0.406 |
| prose | 6.20 | 2.19 | 0.313 |

A **2x spread**, and two conclusions.

## 1. On structured output the drafter is already at spec

The DFlash2 paper reports 4.80 mean acceptance length for Qwen3.8-27B. We measure
**5.00 on translation and 4.88 on JSON**. There is no drafter defect to chase on
those tasks — the published number is a mean over a benchmark mix, and our
structured-output rows meet or beat it.

What drags the average is open-ended generation: prose 2.19, factoid 2.84. That
is the drafter having genuinely less to predict, not a bug.

## 2. Every spec-decode number taken this session used the WORST task

The prompt used throughout was *"Explain the theory of relativity in simple
terms."* — a **prose** task, the bottom row. So the whole session's spec-decode
figures were measured on the drafter's worst case, and the fixes landed here look
smaller than they are. Use a task-mixed set, not one prose prompt.

## Where spec-decode still stands

Plain decode is **15.1 tok/s**. Spec-decode loses on every task, best case 12.32.
The reason is no longer acceptance and no longer the drafter (5.3% of profile):

    gemv_oq_compact_multicol              12912 calls   80.7%   <- the verify
    gemm_oq_compact_grouped_wmma            496          6.2%
    gemm_dflash_oq4_plain_dp4a_staged_8w    736          5.3%   <- the drafter
    gemv_oq_compact_grouped_v3              123          4.1%   <- was 33.7%

`gemv_oq_compact_multicol` sustains roughly **35 GB/s against a 233 GB/s
ceiling** and is 80.7% of the profile. That is the whole remaining gap.

Routing around it does NOT work — sending B <= 16 to the WMMA GEMM measures 6.72
tok/s against multicol's 11.84 (tau identical). The half-empty 16-wide tile at
B=8 is worse than multicol's low bandwidth. **The lever is making multicol
faster**, and at 15% of achievable bandwidth there should be a lot of room.

If multicol reached even 100 GB/s the verify would be ~3x cheaper, which on the
translate/json rows would clear plain decode comfortably.
