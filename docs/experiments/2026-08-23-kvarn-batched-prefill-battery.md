# Coherence battery: KVarN batched prefill is now the default

State box: halo, gfx1151, ROCm 7.14, kvarn KV, `--ctx 512`, corpus
`benchmarks/calib/calib-multi-8m.txt`, 2026-08-23.

Closes the bar set by `7224715da`: *"wants the full coherence battery across the
KVarN model set before flipping."*

## Result — batched vs per-token, 150 greedy tokens

| model | text | verdict | prefill tok/s |
|-------|------|---------|---------------|
| qwen3.5-2b--bf16 | IDENTICAL | OK / OK | 20.8 -> **1572.7** |
| qwen3.5-4b--bf16 | IDENTICAL | OK / OK | 16.6 -> **734.5** |
| Qwen3.8-27B--oq4.25++ | IDENTICAL | OK / OK | 15.3 -> **308.8** |
| Qwen3.5-27B--oq4.25++ | IDENTICAL | OK / OK | 15.4 -> **310.1** |
| Qwen3.6-27B--oq4.25++ | IDENTICAL | WARN / WARN | 15.4 -> **308.0** |
| Qwen3.6-35B-A3B--oq4.25++ (MoE) | identical, but flag INERT | WARN / WARN | 58.9 -> 59.2 |
| Qwen3.5-35B-A3B--oq4.25++ (MoE) | FAILED TO LOAD, both arms | - | - |

Byte-identical output on **every model where the batched path engages**. The
WARN verdicts on the two Qwen3.6 artifacts are present in the PER-TOKEN baseline
too — pre-existing, not caused by batching.

## Two rows that are not passes, and should not be read as passes

**Qwen3.6-35B-A3B: the flag is INERT on MoE** (58.9 -> 59.2 tok/s, i.e. no
change). MoE models fail the `all(DeltaNet|FullAttn)` arm of the batched gate by
construction, so this guard never reaches them. That row is UNTESTED, not
passing — but it also means flipping the default cannot regress MoE, because it
does not apply to MoE.

**Qwen3.5-35B-A3B fails to load** (`[probe] expected loaded, got error`) in BOTH
arms, independent of the flag. Pre-existing and unrelated; worth its own
investigation.

## A false all-clear that had to be discarded

The first battery ran `--battery smoke,perplexity` and reported perplexity
**identical to four decimals on all seven models** — which looked like perfect
evidence and was worthless. The perplexity battery runs in a SEPARATE BINARY
(`resolve_perplexity_bin`) and never enters `generate.rs`'s kvarn branch, so the
flag cannot affect it. The giveaway: perplexity `elapsed_ms` was unchanged
between arms (95223 vs 94688 on the 27B; 31582 vs 31676 on the 2B) — if the flag
had touched that path, the batched arm would have been faster.

Only the smoke rows were valid there, because their `ttft_ms` did move (2.5x to
10.3x). The generation comparison above is what actually exercises the guard.

**Lesson: when an A/B shows a suspiciously perfect null, check that the knob
reaches the code under test before believing it.**

## What changed

`generate.rs` now takes the per-token path only on an explicit `=0`:

    unset -> batched prefill, drafter OFF   (new default)
    =1    -> batched prefill, drafter ON    (unchanged opt-in)
    =0    -> per-token prefill, drafter OFF (full rollback)

The flag is deliberately NOT symmetric with `kvarn_specdecode_ok`, which still
requires an explicit `=1`. Engaging a DFlash drafter under KVarN measured 5.77
tok/s against plain decode's 14.4 — **2.5x slower**, because verify runs
per-token. Flipping both together would have quietly regressed every drafter
user.

Verified on Qwen3.8-27B: unset 313.0 / =1 312.1 / =0 15.4 tok/s.

## Still open

- MoE batched prefill is gated separately and remains per-token. Unaffected here.
- Qwen3.5-35B-A3B load failure.
- `hipfire eval --battery quality` needs `--reference`; a KLD comparison
  batched-vs-per-token would be stronger than greedy token equality, which can
  mask logit drift that sampling would expose.
