# GLM-5.3 on Strix Halo: batched-throughput viability

Date: 2026-10-18
Box: halo — Strix Halo gfx1151, 124 GiB UMA (~115 GiB usable), 256 GB/s DRAM
(~250 measured), NVMe cold read 6.2 GB/s, page-cache read ~37 GB/s
(see `reference_qwen4exp_pager_profile`).
Status: config arithmetic plus measured halo bandwidths; routing locality
**measured on Qwen3.8-Flash-Next as a stand-in** (§8). GLM-5.3 itself was not
loaded or run.

## Question

hipfire now targets several models resident at once. Is GLM-5.3
(`zai-org/GLM-5.3`, `glm_moe_dsa`) a viable candidate for batched
(microbatched) decode on halo, and does the attention state cache make large
batches infeasible?

## Verdict

**Not viable at 4-bit experts; marginal at best with ~2-bit experts.** Expert
weights never fit; with per-expert-module microbatching and load/compute
overlap they cost NVMe *bandwidth*, and the measured routing locality (§8)
says that bandwidth is not enough. The NVMe ceiling is ~0.52/p tok/s, where p
is the share of routed picks that miss the resident set: ~2.4 tok/s aggregate
(4.3 with ideal eviction) for same-subject batches at the ~22% residency 4-bit
allows. Same-subject batching helps (1.8× fewer misses than mixed), pausing
streams to queue tokens per module makes it worse (§8.5), and the miss rate
only falls with batch size once B reaches the hundreds — where the KV state no
longer fits beside the expert cache (§8.6). The lever that matters is the
resident fraction: at 50% residency same-subject misses drop ~7–10×. Four
pieces of implementation are also missing (§7).

An earlier pass of this analysis concluded "not viable". That was wrong, and
the reasons are recorded in §3.3 so nobody re-derives it.

## 1. Model shape

From `config.json`:

| | |
|---|---|
| Total params | 753.3B (743.4B main + ~9.9B MTP layer) |
| Layers | 78 (3 dense, 75 MoE) |
| Hidden | 6144 |
| Experts | 256 routed, top-8, 37.75M params each |
| Attention | MLA: `q_lora_rank` 2048, `kv_lora_rank` 512, rope 64, 64 heads |
| Sparse attention (DSA) | indexer 32 heads × 128, `index_topk` 2048, indexer on 21 `full` layers (rest `shared`) |
| Active / token | ~40.2B |
| Max positions | 1,048,576 |

## 2. Weights

| Format | Total | Routed experts | Trunk |
|---|---|---|---|
| FP8 (as shipped) | 777 GB | 747 GB | 19 GB |
| oq4.25 | 400 GB | 385 GB | 9.9 GB |
| 3-bit | 306 GB | | |
| 2-bit | 212 GB | 204 GB | 5.2 GB |

Fully resident would need ~1.06 bpw. Experts are therefore paged
(`weight_pager`, LRU). At oq4.25 one expert module is ~20 MB, a layer's
256 experts ~5.1 GB, and B=1 decode touches ~12 GB of experts per token
(Flash-Next: ~1.2 GB).

## 3. Expert paging under microbatching

### 3.1 Cost model

Microbatching groups tokens **per expert module**: each module touched in a
step is read once and applied to every token routed to it. Per layer per step,
let U = distinct experts touched, M = those not resident.

- RAM read: 20 MB / 250 GB/s ≈ 0.08 ms per module.
- NVMe read: 20 MB / 6.2 GB/s ≈ 3.2 ms per module.
- Misses are loaded **overlapped with compute**, so

  `step ≈ max( Σ_layers U·0.08 ms + compute,  Σ_layers M·3.2 ms )`

  — the maximum, not the sum. A miss is free until the miss stream saturates
  NVMe.

Budget at oq4.25: ~85 GB of expert cache (after trunk, ~15 GB KV, scratch)
≈ 56 of 256 experts per layer (~22%). If a batch's union stays within that
set, a step is ~0.34 s — B=32 ≈ 94 tok/s aggregate, ~3 tok/s per stream;
compute is not the limit. In that 0.34 s NVMe delivers ~2.1 GB ≈ 105 modules
(~1.4 per layer per step) for free. Beyond that the ceiling is

  `aggregate tok/s ≈ 6.2 GB/s ÷ (unique miss bytes per step / B)`

which improves with batch size *and* with routing locality, since one fetched
module serves every token routed to it.

### 3.2 Difference from Qwen3.8-Flash-Next

Same mechanism; Flash-Next does not hold all its experts in GTT either. What
differs is the **tier a miss falls to**: Flash-Next's ~59 GiB of oq4 experts
all fit in page cache (~37 GB/s), whereas ~78% of GLM-5.3's 385 GB can only
come from NVMe (6.2 GB/s) — 6.5× the expert bytes, ~6× slower per miss.

### 3.3 Retracted first-pass reasoning

The first pass computed the expert union under **uniform, independent
routing** (B=16 → 102/256 experts/layer = 153 GB/step; B=64 → 222/256 =
335 GB/step), then (a) treated the step's union as a set that must be
co-resident and (b) summed miss latency onto the step. All three are wrong:

- streams on similar subjects share experts heavily, so the uniform union is a
  loose upper bound, not an estimate;
- per-module microbatching streams modules one at a time; residency only buys
  reuse *across* steps;
- misses overlap compute (§3.1).

That pass predicted an aggregate plateau of ~1.5–3 tok/s. Disregard it.

## 4. State (KV) size — the batch-count constraint

GLM-5.3 has no fixed-size recurrent state; state is MLA latent KV plus DSA
indexer keys, linear in tokens.

Per token, all 78 layers (latent 512 + rope 64 per layer; indexer keys 128 on
21 layers):

| Layout | Per token |
|---|---|
| bf16 latent + bf16 indexer | 93.0 KiB |
| FP8 latent (FlashMLA-style: fp8 nope + scales + bf16 rope), FP8 indexer | 52.7 KiB |
| KVarN-4 latent, bf16 rope, FP8 indexer | 33.3 KiB |
| KVarN-2 latent, bf16 rope, FP8 indexer | 23.6 KiB |
| Per-head decompressed K/V (non-absorbed MLA) | 4.88 MiB (53× bf16 latent) |

The kernel **must** attend over the latent with absorbed projections; a
per-head cache makes batching impossible.

Per sequence (FP8): 4K 211 MiB · 8K 422 MiB · 32K 1.65 GiB · 128K 6.6 GiB.

Sequences per KV budget (FP8, no sharing):

| Context | 16 GiB | 32 GiB | 64 GiB |
|---|---|---|---|
| 4K | 77 | 155 | 310 |
| 8K | 38 | 77 | 155 |
| 32K | 9 | 19 | 38 |
| 128K | 2 | 4 | 9 |

KV and the expert cache share one UMA pool of ~100 GiB: every GiB of KV is a
GiB of hot experts pushed to NVMe. This assumes KV allocated by actual length;
reserving max length per slot makes the long-context rows apply to every
request.

## 5. Shared-prefix workload

Target: one 500K-token shared prefill context, 32 jobs each adding a 4K
private prefill and 8K of output.

| | Tokens | FP8 | KVarN-4 | KVarN-2 | bf16 |
|---|---|---|---|---|---|
| Shared prefix (stored once) | 500K | 25.1 GiB | | | 44.3 GiB |
| 32 × (4K + 8K) private | 32 × 12K | 19.3 GiB | | | 34.1 GiB |
| **Total** | | **44.5 GiB** | **28.1 GiB** | **19.9 GiB** | 78.5 GiB |

Without sharing: ~823 GiB. With FP8 ~55 GiB remains for experts (~15% of
them); KVarN-4 frees another ~16 GiB, KVarN-2 ~25 GiB. bf16 leaves ~22 GiB
and is not workable.

### 5.1 Indexer cost dominates per-step attention

DSA attends to 2048 selected tokens per layer, so attention cost is
context-independent (~105 MB per sequence per step in FP8). The **indexer**
scans every key each step on 21 layers:

- per-sequence kernel: ~1.4 GB per sequence per step at 512K → ~45 GB/step for
  32 sequences, ~0.18 s at 250 GB/s — rivals the expert reads;
- prefix-aware (cascade) kernel: shared keys read once and scored against all
  32 queries as one GEMM → ~1.4 GB/step (~6 ms); compute becomes ~2.8 int8
  TOPs/step (~60–100 ms).

A shared-prefix indexer and attention kernel is therefore a requirement for
this workload, not an optimisation.

### 5.2 Prefill and persistence

The 500K prefill is a one-off ~20–40 min (estimate: ~4e16 FLOP of weights
plus ~2e16 of indexer). Experts are fully amortised during prefill
(compute-bound). It only pays if the prefix KV survives across job batches; at
25 GiB FP8 it reloads from NVMe in ~4 s.

## 6. KVarN

KVarN fits GLM-5.3 natively: the paper (2606.03458) targets DeepSeek MLA
latent KV with R=512, which is GLM-5.3's `kv_lora_rank`. The latent is both K
and V, so one tile covers both, where GQA needs two.

hipfire's codec core (`crates/hipfire-kvquant/src/kvarn.rs`) is a generic
`[R × C]` tile with a 128-token group (`crates/hipfire-runtime/src/kv.rs:1205`),
but is wired only per (layer, kv_head) for GQA. GLM-5.3 needs MLA wiring that
tiles the 512-wide latent, and a latent-attention kernel that reads KVarN
tiles. KVarN-8 is slightly *larger* than FlashMLA-style FP8; skip it.

**Caveat — scattered gather.** DSA's top-2048 picks tokens spread across the
context. At 500K most picks land in distinct 128-token tiles, and each drags
that tile's ~2 KB of per-channel scale/zero-point alongside its 256-byte
payload: ~389 MB per sequence per step vs ~105 MB for FP8 (~3.7×), ~12 GB/step
(~50 ms) at B=32. Tolerable next to expert reads. Mitigations: a smaller token
group on long-context layers, or exploiting overlap between the 32 sequences'
picks on the shared prefix.

Keep the indexer keys FP8: they are scanned, not gathered, and top-k
selection depends on ranking precision.

## 7. Implementation gaps

1. **MLA family.** hipfire has no MLA. `hipfire-arch-deepseek4` is DeepSeek-V4
   (compressed-KV / SWA, no `kv_lora_rank`). Needs a new family with
   absorbed-MLA decode and prefill kernels. The deepseek4 indexer kernels
   (`indexer_compressed_k_score.hip`, `indexer_kv_gather.hip`) are a partial
   head start on DSA.
2. **Async pager.** Load/compute overlap is assumed above, but the pager is
   synchronous today (`pread` then `stream_synchronize`,
   `crates/hipfire-runtime/src/weight_pager.rs:556-558`). Overlap is a
   documented TODO there. Within a layer, computing ~56 resident experts
   (~4.5 ms) hides only ~1 miss (3.2 ms); hiding more needs next-layer routing
   prediction or two half-batches pipelined a layer apart.
3. **Shared-prefix (cascade) attention and indexer kernels** (§5.1), plus
   prefix-KV persistence (§5.2).
4. **KVarN MLA wiring** (§6).

## 8. Measured routing locality (Qwen3.8-Flash-Next stand-in)

### 8.1 Method

- Probe: `routing_trace_qwen4exp` (teacher-forced prefill; a `ROUTING_TRACE`
  sink in `moe_forward` records every token's top-10 routed experts per layer).
  Trace is bit-reproducible across runs.
- Model: `Qwen3.8-Flash-Next-180B--oq4.hfq` — 48 MoE layers, 512 experts,
  top-10. 5.8 tok/s through the probe.
- Corpus (`benchmarks/scripts/routing_locality_corpus.py`): 7 groups × 16
  documents × 256 tokens. `rust` (largest source file of 16 crates), `docs`
  (16 hipfire plan docs), `samedoc` (16 consecutive chunks of ONE document —
  the stand-in for jobs sharing a context), `math`, `pycode`, `chinese` (from
  the labelled calib mix), `wiki` (unrelated wikitext articles). `MIXED` draws
  across all groups.
- Analysis (`benchmarks/scripts/routing_locality_analyze.py`): B documents
  decoded in lockstep; per layer per step the batch needs the union of its
  experts. A per-layer cache holds a fraction of E; misses counted in steady
  state (t ≥ 64), under LRU and under offline-optimal (Belady) eviction.
  `p` = misses / (B·k) = the share of routed selections that reach the slow
  tier, after batch sharing. Raw output:
  `benchmarks/results/2026-10-18-qwen4exp-routing-locality/`.

### 8.2 Results

Batch union, distinct experts per layer per step (ratio to uniform
independent routing):

| B | same-subject groups | samedoc | MIXED |
|---|---|---|---|
| 4 | 33.6–36.6 (0.87–0.94) | 34.9 (0.90) | 37.8 (0.97) |
| 16 | 96–113 (0.70–0.82) | 98.5 (0.71) | 126 (0.91) |
| 112 | — | — | 381 (0.84) |

Per-routing miss rate `p` at B=16 (LRU / Belady):

| Cache (fraction of experts) | same-subject | samedoc | MIXED |
|---|---|---|---|
| 22% | 0.22–0.33 / 0.12–0.19 | 0.22 / 0.12 | 0.40 / 0.24 |
| 50% | 0.03–0.10 / 0.012–0.041 | 0.033 / 0.012 | 0.15 / 0.06 |

`p` vs batch size at 22% (LRU):

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 112 |
|---|---|---|---|---|---|---|---|---|
| samedoc | 0.17 | 0.20 | 0.22 | 0.23 | 0.22 | | | |
| MIXED | 0.16 | 0.23 | 0.30 | 0.35 | 0.40 | 0.39 | 0.32 | 0.24 |

Static hot set (in-sample, optimistic): a group's own top 22% of experts
carry 67–78% of its routings (MIXED: 51%).

### 8.3 What it means

- **Similar-subject reuse is real but modest.** At B=16 the union is 70–82%
  of the uniform prediction (mixed: 91%). Topic granularity barely matters:
  chunks of one document share no more than "16 math problems". Against
  mixed traffic, same-subject batching cuts misses 1.8× at 22% residency and
  4.5× at 50%.
- **Batching cuts the miss rate only at large B.** `p` rises with B up to
  B≈16 (the union grows almost linearly and outruns the cache), then falls
  once the union saturates the layer (MIXED 0.40 → 0.24 from B=16 to 112).
  Microbatching amortises RAM reads at any B; it only amortises NVMe misses
  when B·(per-token pick probability of a cold expert) ≳ 1, i.e. B in the
  hundreds (§8.6).
- **NVMe ceiling.** For GLM-5.3 at oq4.25 (12 GB of experts per token at
  6.2 GB/s):

  `aggregate tok/s ≈ 6.2 / (12 · p) ≈ 0.52 / p`

  | Residency, B=16 | same-subject p (LRU / Belady) | ceiling | mixed p | ceiling |
  |---|---|---|---|---|
  | 22% (oq4.25 in ~85 GB) | 0.22 / 0.12 | 2.4 / 4.3 tok/s | 0.40 / 0.24 | 1.3 / 2.2 tok/s |
  | 50% (needs ~190 GB at oq4.25) | 0.033 / 0.012 | 16 / ~36 (RAM-capped) | 0.15 / 0.06 | 3.5 / 8.7 tok/s |

  The free-overlap budget from §3.1 (~1.4 misses per layer per step) is
  exceeded by ~10–20× at 22% residency.
- **Residency is the lever, not batch shape.** 50% residency on ~100 GB
  means ~2-bit experts, and 2-bit has not held quality in hipfire so far
  (`project_lowbit_quant_findings`); KV then competes for the same pool.
- **Eviction policy is worth ~2×.** LRU vs offline-optimal: B=1 at 22%
  0.17 vs 0.09; B=16 same-subject 0.22 vs 0.12. A frequency-aware policy or
  router-predicted prefetch could recover part of that — relevant to
  Flash-Next's own pager regardless of GLM.

### 8.5 Pausing a stream until a module has queued tokens

Tested directly (`benchmarks/scripts/routing_locality_defer.py`): rounds
sweep layers in order; a token whose routed module is not resident waits at
that layer (served experts stay served — outputs accumulate), and a missing
module is loaded only once `min_waiters` tokens are queued on it or its oldest
waiter has waited `max_wait` rounds. `min_waiters=1` is lockstep. 22% cache,
steady state (30 000 rounds):

| Batch | lockstep p | deferred p (min_waiters / max_wait) | rounds per token |
|---|---|---|---|
| samedoc B=16 | 0.22 | 0.26 (4 / 64) | 1 → ~2600 |
| MIXED B=112 | 0.25 | 0.50 (4 / 16) · 0.40 (8 / 64) · 0.26 (16 / 256) | 1 → 760–11 000 |

**Pausing is worse than lockstep.** Tokens queued on module (l, e) can only
come from streams whose *current* token is at layer l. A paused stream stops
feeding every other layer, and paused streams scatter across all 48 layers,
so queues rarely fill: streams end up waiting `max_wait` at nearly every layer,
and modules are still loaded for ~1 waiter after the timeout, by which time the
cache has moved on. (An early, short run showed p=0.09 at 8 / 64; that was a
start-up transient and did not survive to steady state.)

The sharing pausing is after is exactly what a layer-synchronised batch
already gets: every in-flight stream is at layer l when its modules are
loaded. The only way to put more tokens behind each module load is more
streams in flight.

### 8.6 Large B: where the state cache binds again

Extrapolating lockstep to large B for GLM-5.3 (22% residency ≈ 200 cold
experts per layer; uniform cold pick probability q ≈ 1.76/200 per token per
layer; cold bytes per step ≈ 300 GB · (1 − (1 − q)^B)):

| B | cold bytes / step | step | aggregate | KV (500K shared + 4K each, KVarN-4) |
|---|---|---|---|---|
| 16 | 40 GB | 6.4 s | 2.5 tok/s | 18 GiB |
| 64 | 130 GB | 21 s | 3.1 tok/s | 24 GiB |
| 256 | 269 GB | 43 s | 5.9 tok/s | 49 GiB |
| 1024 | 300 GB | 48 s | 21 tok/s | 149 GiB — does not fit |

Throughput only climbs once B reaches the hundreds, at tens of seconds per
token per stream — and there the KV state eats the expert cache (49 GiB of
KV at B=256 leaves ~14% residency, not 22%), which pushes `p` back up. **The
state cache is the binding constraint after all**, in exactly the regime that
would make expert paging efficient.

### 8.4 Caveats

- Flash-Next is a stand-in: 512 experts top-10 (2% per token) vs GLM-5.3's
  256 top-8 (3.1%). Locality per cache fraction may differ.
- 256-token traces include compulsory first-touch misses after t=64, which
  inflate `p` most at large caches and small B (B=1 at 50% shows ~5.5%).
  Long-running streams would do somewhat better at 50%; at 22% capacity
  misses dominate.
- Teacher-forced human text, not model-generated continuations; all streams
  start at position 0 rather than inside a shared 500K context.
- Equal cache share per layer; a per-layer budget weighted by locality could
  help.
- §8.6 assumes uniform picks among cold experts; real skew saturates the union
  more slowly, so the large-B rows are optimistic.
- The deferral sim (§8.5) runs to ~11 000 rounds on 256-token documents, so
  slow configurations cover only the first few tokens of each document.

Multi-model note: even when viable, GLM-5.3 takes most of the UMA pool and all
NVMe bandwidth while it is active, so it coexists poorly with other resident
models.
