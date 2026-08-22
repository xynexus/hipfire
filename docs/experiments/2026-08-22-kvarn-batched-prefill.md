# KVarN batched prefill: the guard was stale — prefill 15.4 -> 48.1 tok/s

`generate.rs` declined batched prefill for KVarN outright, with this reason in
place:

> KVarN ... require the per-token attention dispatch: the batched
> `forward_prefill_batch` runs its own batched attention and **never populates the
> KVarN window/records**, so the prompt KV is wrong and decode degenerates.

**That is no longer true.** `prefill_chunk.rs:4836` handles KVarN explicitly —
"kvarn_attend owns the batched write (window append + 128-block flush)" — and it
also wires `tree_verify` (`kvarn_tree_bias`, `kvarn_block_start/cols`). The
capability was added under the batched path and the guard was never lifted.

## Measured

Qwen3.8-27B--oq4.25++ on halo, `--kv-mode kvarn`, CASK off:

| | prefill | decode | ttft |
|---|---|---|---|
| per-token (guard on) | 15.4 tok/s | 14.5 | 15566 ms |
| **batched (guard off)** | **48.1 tok/s** | 14.4 | **4993 ms** |

**3.1x prefill, 3.1x faster time-to-first-token.** Decode is unchanged, as
expected — the guard only ever affected prefill.

Coherence checked directly rather than assumed, same prompt, greedy:

```
per-token: "...First five prime numbers: 2, 3, 5, 7, 11.  Definition of..."
batched:   "...The first five prime numbers are: 2, 3, 5, 7, 11.  A prime number is..."
```

Both correct and fluent. Wording differs because batched and per-token export
measurably different hidden states (a known property of every dtype here, not a
compact/KVarN quirk).

Behind `HIPFIRE_KVARN_BATCHED_PREFILL=1` for now rather than flipped by default:
the original guard cites a real failure mode (unpopulated window/records ->
garbage), so this wants the full coherence battery across the KVarN model set
before it becomes the default.

## What it unblocks, and what it does NOT

The same flag lifts the spec-decode gate (`!kvarn_active` in the DFlash routing
condition), and the drafter then genuinely engages under KVarN — prefill reads
48.1, and decode changes, so it is no longer silently running plain AR.

**But spec-decode is still 5.77 tok/s against plain decode's 14.4 — 2.5x SLOWER.**
That ratio is the signature of verify running per-token: ~K weight sweeps per
accepted token at block_size 8.

The cause is one line, `mod.rs:2122`:

```rust
let allow_compact = !tape_in_play;   // tape_in_play includes tree_verify.is_some()
```

So a tree-verify forward excludes compact Opus from the batched path **by
construction**, and verify falls to per-token — exactly the root cause recorded in
`2026-08-2x-dflash2-qwen38-perf`. Batched prefill does not fix it because verify
takes a different branch.

**Next domino: let tree-verify batch for compact.** `prefill_chunk.rs:4870`
already threads `kvarn_tree_bias` / `kvarn_block_start` / `kvarn_block_cols`
through the KVarN attention, so the machinery may already be there — but the
in-code comment warns this is exactly what once took DFlash2's accept_rate to
**0.000** with the draft emitting random vocab ids. Verify acceptance, not just
tok/s, before believing any speedup there.

Also noted while measuring: the DFlash2 checkpoint carries a
`candidate_selector` (rank=256, top_k=16) that **is not applied** — the draft path
still takes a per-position argmax, so acceptance is below what the checkpoint can
do. Independent of the batching problem, and worth its own look.
