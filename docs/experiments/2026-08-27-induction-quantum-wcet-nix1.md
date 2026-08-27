# Per-quantum WCETs for induction work — the admission gate's missing input

Status: measured 2026-08-27 on nix1 (`gfx1103`, 42.0 GB GTT).
Motivation: `2026-08-27-non-disruptive-induction-scope.md` §2.2 — `admit_realtime`
exists but has no caller, and no induction stage's per-quantum cost was known.
This measures them.

## Setup

| role | artifact | notes |
|---|---|---|
| served | `Qwen3-4B--bf16.hfq` | `kv_cache: kvarn`, `max_seq: 2048` |
| trainer | `Llama-3.2-1B--bf16.hfq` | fp32-widened; `.hfq` base via PR #363 |
| QAT | `HIPFIRE_KVNOISE=1` | KVarN-4bit + CASK merge, STE |
| corpus | 391 KB local slice | `n_ctx: 1024`, `max_chunks: 6` |

Serving is a 16-token greedy generate. Each induction quantum is one daemon
request (`quantum: 1` for training).

## Results

| measurement | n | min | mean | **max (WCET)** |
|---|---|---|---|---|
| serving baseline | 5 | 2.450 | 2.455 | **2.463 s** |
| **QAT quantum** (1B fp32) | 10 | 10.033 | 10.611 | **15.323 s** |
| serving *between* QAT quanta | 10 | 2.872 | 2.884 | **2.893 s** |
| **KLD chunk** (n_ctx 1024) | 6 | 246.295 | 272.940 | **280.760 s** |

`kld_eval build_ref` total: **1,637.7 s (27 min)** for six chunks, with serving
blocked throughout — it is one monolithic call.

## What this says about admission

§1.1's contract is a **200 ms** entry-latency budget. Against it:

| quantum | WCET | × over budget |
|---|---|---|
| QAT step (1B) | 15.3 s | **77×** |
| KLD chunk | 280.8 s | **1,404×** |

**Neither is admissible under the contract as written, and no scheduling policy
fixes that.** `admit_realtime` can only decide *whether to start* a quantum; once
started, nothing preempts it. A realtime request arriving one millisecond after a
KLD chunk begins waits ~280 s.

So the scope doc's item 3 ("give `kld_eval` a chunk quantum") is **necessary but
not sufficient**. A 280 s chunk is not a quantum in any useful sense. Both stages
need *sub-quantum yielding* — the ability to stop mid-step at a layer or
token-block boundary — which is a stronger requirement than "expose a step()".

The one genuinely encouraging number: **serving between QAT quanta costs
+17.5 %** (2.455 → 2.884 s), and remarkably tightly (max 2.893). That is the
steady-state tax of interleaving, and it is a number a policy could trade
against — as distinct from the blocking figures above, which are what you pay if
you land mid-quantum.

## ⚠️ The KLD chunk cost may be a fallback, not the intrinsic cost

280.8 s for 1024 tokens is **0.274 s/token (~3.6 tok/s)** — decode-speed, not
batched-prefill speed. For reference the same box served 16 tokens in 2.455 s.

Two observations, neither conclusive:

- the run logged `KV cache: Q8` during KLD despite the model being loaded with
  `kvarn`, consistent with the note at `qwen35/loading.rs` near
  `forward_chunk_scored` that the batched prefill's F4 guard rejects an f32 KV
  cache, so the KLD path pins its own tier;
- `HIPFIRE_KERNEL_TRACE` was **not** enabled for this run, so the
  "batched prefill declined → per-token forward_scratch loop" fallback site
  cannot be confirmed or excluded.

KLD scoring does more per position than decode (full-vocab softmax + divergence),
so some of the gap is real work. **Whether the rest is a batching fallback is
open, and is worth resolving before anyone designs around 280 s** — if it is a
fallback, fixing it could move KLD from "hopeless" to merely "needs a quantum".
Re-run with `HIPFIRE_KERNEL_TRACE=1` to settle it.

## Harness notes (three of my own errors, each caught before it became data)

- **Counting frames in the wrong stream.** Protocol frames go to *stdout*, which
  the driver consumes; I was grepping `.err`, which holds only logs, and read
  "0 frames" as "stuck". The daemon was at 100 % CPU doing real work.
- **A 3B fp32 trainer was too large to measure.** The first run ground for 29
  minutes of CPU on a single quantum. Dropped to 1B.
- **Results only at the end.** The first harness dumped JSON on completion, so a
  failed load 2 s in was invisible for 25 minutes. Rewritten to stream every
  record to JSONL immediately — which then surfaced the `q8` failure instantly.
- **A stale local artifact.** `Llama-3.2-1B--bf16.hfq` existed locally at
  578,813,952 B against `/srv`'s 1,654,914,061 B. An `ls || cp` short-circuited
  on the filename. Always compare sizes to source.

## `kv_cache: "q8"` is deprecated

The first corrected run failed at load with:

> `kv_mode=q8 is deprecated. hipfire is retiring KV storage down to two families:
> kvarn (kvarn2 / kvarn / kvarn4 / kvarn8) and unquantized (fp32).`

`HIPFIRE_KV_ALLOW_DEPRECATED=1` runs it during migration. Any doc or memory
saying "use Q8 KV for batched prefill" is stale — `kvarn` is the family now.

## Reproducing

`wcet2.py` in this session's scratch directory: loads the served model, times
five solo generates, then alternates QAT quanta with generates, then runs
`kld_eval build_ref` collecting per-chunk frame timings. Every record is appended
to `wcet2.jsonl` as it happens.
