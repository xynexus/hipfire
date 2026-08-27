# DFlash block-size sweep on gfx1103 — the parked Qwen3.8-27B drafter wins after all

Date: 2026-08-27. Host: nix2 (gfx1103 Phoenix APU, 32 GB UMA, `amdgpu.cwsr_enable=0`).

## Summary

The `Qwen3.8-27B--dflash2.oq4+` drafter, parked on halo as
`*.parked-slower-than-plain-decode`, is **1.75× faster than plain decode** on
gfx1103 when driven through `dflash_spec_demo` with `HIPFIRE_DN_STATE_FP16=1`
and `--block-size 10`. The parking verdict is real but misattributed: the
drafter is fine; the **daemon's speculative path** is what loses to plain
decode. See the companion scope doc for the merge plan.

## Setup

- Target: `Qwen3.8-27B--oq4.25++.hfq` (15 GB, pulled from halo)
- Draft: `Qwen3.8-27B--dflash2.oq4+.hfq` (1.2 GB, 5-layer DFlash, oq4+)
- Driver: `cargo build --release -p hipfire-runtime --example dflash_spec_demo`
- Env: `HIPFIRE_DN_STATE_FP16=1 HIPFIRE_DFLASH_ALLOW_OPUS=1`
- Prompt: "Write the numbers 1 through 30 as a comma-separated list.",
  `--max 256 --ctx 2048`, greedy
- Search: ternary search over integer block sizes [2, 20] on `decode_tok_s`
  (unimodal — confirmed by the measured curve), exhaustive over the final
  ≤4-wide bracket. Script: session scratchpad `bsweep.sh`, one warmup run,
  GPU lease via `hipfire lock acquire bsweep --watch-pid $$`.

## Results

Plain decode baseline (daemon, drafter off): **5.3 tok/s**.

| B  | decode tok/s | τ (committed/cycle) |
|----|--------------|---------------------|
| 5  | 7.87         | 2.63                |
| 8  | 8.45         | 5.27                |
| 9  | 8.86         | 5.27                |
| **10** | **9.29** | **5.90**            |
| 11 | 9.26         | 5.90                |
| 12 | 7.46         | 6.26                |
| 14 | 7.70         | 5.27                |

Shape: below B=10 the block caps acceptable runs (τ collapses to 2.63 at
B=5); above it the drafted tokens past the accepted run are wasted verify
work — B=12 has the *highest* τ yet worse throughput. B=10/11 is a plateau;
pick 10.

## The daemon contradiction

Same target, same drafter, same machine, same day, via the daemon
(`HIPFIRE_DFLASH_ALLOW_OPUS=1`, `params.draft`):

| Path | decode tok/s | τ | accept rate |
|------|--------------|-----|-------------|
| daemon speculative | 1.31 | 2.24 | 28% |
| demo, B=10         | 9.29 | 5.90 | — |
| plain decode       | 5.3  | —    | — |

The daemon's in-code gate ("Opus oq* … CORRECT now but measured slower than
plain decode on this family, so it stays behind HIPFIRE_DFLASH_ALLOW_OPUS=1")
and the artifact parking both encode a measurement of the *daemon path*, not
of the drafter. The v1 drafter (`--dflash.oq4+`) is genuinely poor (9% accept,
0.54 tok/s via the daemon) and stays parked.

## Operational notes

- The `.parked-slower-than-plain-decode` suffix is load-bearing: with the
  canonical name, sibling auto-discovery finds the drafter and the daemon
  load **hard-fails** on oq targets unless `HIPFIRE_DFLASH_ALLOW_OPUS=1` is
  set. Keep parked names until the daemon path is fixed.
- dflash2 config block_size is what the daemon ran (mean_draft_len 8.0,
  fixed); the demo's win came from sweeping past it. Whatever merges into
  serving-core must carry the block-size knob (and ideally the demo's
  adaptive sizing) with it.
