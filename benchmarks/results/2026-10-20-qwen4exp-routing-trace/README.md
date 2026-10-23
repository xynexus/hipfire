# Qwen3.8-Flash-Next routing trace (512 tokens, top-32 alternatives)

A teacher-forced record of every MoE routing decision Qwen3.8-Flash-Next
makes over 112 grouped documents. It is the input for offline simulation of
expert scheduling, caching, prefetch and substitution policies — none of which
needs the model to be re-run, because teacher-forced routing does not depend
on how the work is scheduled.

The binaries (~420 MB) are not committed. Regenerate them with:

```sh
python3 benchmarks/scripts/routing_locality_corpus.py . docs.jsonl
hipfire lock run routing-trace -- \
  target/release/examples/routing_trace_qwen4exp \
  Qwen3.8-Flash-Next-180B--oq4.hfq docs.jsonl <out_dir> 512 32
```

(2.8 h at 5.7 tok/s on gfx1151.) `provenance.txt` here records the md5s of
the run this README describes.

## Shape

`meta.json`: `layers` L=48 MoE layers, `k`=10 routed experts per token,
`experts` E=512 per layer, `n_tok` T=512 tokens per stream, `top_n` N=32
alternatives, `norm_topk_prob`.

Streams S=112: 7 groups × 16 documents, in `docs.jsonl` order (`group`, `id`).
Groups: `rust`, `docs`, `samedoc` (16 consecutive chunks of one document),
`wiki` (unrelated articles, control), `math`, `pycode`, `chinese`.

| File | dtype | Shape | Content |
|---|---|---|---|
| `routing.bin` | u16 | [S, T, L, k] | experts the GPU routed to (unordered) |
| `alt_ids.bin` | u16 | [S, T, L, N] | top-N experts by router probability, descending |
| `alt_p.bin` | f16 | [S, T, L, N] | their softmax-over-all-E probability |
| `tokens.bin` | u32 | [S, T] | input token ids |

All little-endian, row-major. `alt_ids[..., :k]` equals `routing.bin` as a
set (the probe checks this on every call) except at 7 of 2.75M calls, all
exact ties between ranks k-1 and k where the GPU took the other expert.
Routed outputs are weighted by `alt_p[..., :k] / alt_p[..., :k].sum()`
(`norm_topk_prob`). The router is flat: the routed top-10 carry 17% of the probability mass on
average, the top-32 30%.

Python: `benchmarks/scripts/routing_trace.py` (`load(dir)` → arrays above).

## Dependency graph

Nodes are `(s, t, l)`: token t of stream s at MoE layer l.

1. **Layer order.** `(s, t, l)` → `(s, t, l+1)`.
2. **Token order.** `(s, t, L-1)` → `(s, t+1, 0)`. Token t+1 of a stream cannot
   start until token t finishes every layer (autoregressive decode; teacher
   forcing fixes *which* token, not *when*).
3. **Expert work.** `(s, t, l)` needs module `(l, e)` for each
   `e ∈ routing[s, t, l]`. The k expert calls are independent and accumulate
   into one output, so each can run whenever its module is available; the node
   completes when all k have run.
4. **Always-resident work.** The attention/GDN mixer, the shared expert and the
   router of `(s, t, l)` have no module dependency; the router output is what
   rule 3 consumes.
5. **No cross-stream edges.** Streams only interact through shared modules
   (one load of `(l, e)` can serve every node that needs it while resident).

A schedule is any order respecting 1–3; its cost is the module loads it
issues under a residency budget. `routing_locality_analyze.py` (lockstep,
LRU/Belady) and `routing_locality_defer.py` (pause-until-queued) are two such
schedules.

## Limits

- Teacher-forced human text, not model-generated continuations.
- Every stream starts at position 0 (no shared long prefix).
- Policies that change the selection (substituting an `alt_ids` expert for a
  routed one) change the hidden state, so routing *after* the substitution is
  only approximated by the trace.
- Flash-Next stands in for other MoEs (e.g. GLM-5.3: 256 experts, top-8).

## Check against the 256-token trace

`lru.txt` / `belady.txt` here are `routing_locality_analyze.py --warmup 64
--draws 24 [--belady]` on this trace. They match the 256-token results
(`../2026-10-18-qwen4exp-routing-locality/`) to within 0.025 in miss rate p
at every group, B and cache size: e.g. B=16 at 22% cache, LRU / Belady,
samedoc 0.227 / 0.122, MIXED 0.405 / 0.239. Doubling the stream length does
not move the steady state.

## Structure (`structure.txt`)

`routing_trace_analyze.py <trace>`: even streams fit popularity and layer
co-occurrence, odd streams are scored; cache 22% of E per layer. Values below
span the four layer bands (early → late).

1. **Routed weight is flat.** Renormalised weight by rank: 0.22, 0.15, 0.12,
   0.10, …, 0.056. The bottom 4 picks still carry 25%. Misses are not
   concentrated on the light picks: under a static hot set, 50% of picks miss
   and they carry 48% of the routed weight.
2. **A stream reuses its own recent experts.** S(t) shares 32–40% of its
   experts with S(t-1), 19–26% with S(t-4), 10–13% with S(t-256). Another
   stream at the same position shares 8–10% if it is the same subject and
   3.5–4% (= the popularity baseline) if not. Streams share a *distribution*,
   not token-by-token picks.
3. **Token id predicts routing.** The same token id in another document
   shares 24–39% of experts (strongest in layers 0–11), in the same document
   36–53%; random pairs 3.5–4.0%. The token id is known before layer 0, so this
   is a whole-token lookahead.
4. **Next-layer experts are predictable from this layer.** Recall of S(t,l)
   in a predicted set, before layer l's router runs: co-occurrence with
   S(t,l-1) 37–42% @10, 70–75% @40, 84–87% @80; plus decayed previous-token
   picks 37–43% @10, 77–80% @40, 89–91% @80. Static popularity: 10% @10.
5. **But the picks that miss are the unpredictable ones.** Per-stream LRU
   with top-m prefetch one layer ahead (§8): with no prefetch the demand miss
   rate is 0.14–0.21. Top-20 lowers it by ≤0.02, top-40 by 0.02–0.06 while
   issuing 0.21–0.32 extra loads per pick, of which only 15–23% are used before
   eviction. Total loads more than double. Prefetch can hide latency
   (today's pager is synchronous) but does not reduce bytes; when NVMe
   bandwidth binds, it loses.
6. **Hot sets are subject-specific.** A static 22% hot set built from the
   same subject hits 65–74% of picks, built from all subjects 47–51%, from a
   different subject 28–30%. Dynamic LRU does better than any static set
   (B=1: 83%), but the gap says to batch streams by subject and to re-warm on a
   subject switch.
7. **Substitution has candidates but costs probability.** Against the static
   hot set, 73–79% of misses have a resident expert in the token's top-32
   alternatives, at 0.38–0.43× the missed expert's probability. The trace
   cannot say what that does to output quality (routing after a substitution
   diverges); that needs a forward-pass KLD test.
8. **Per-stream working set.** One stream touches 13–15% of a layer's experts
   in 16 tokens, 26–32% in 64, 46–55% in 256, 57–66% in 512 (uniform routing:
   27%, 72%, 99%, 100%).
