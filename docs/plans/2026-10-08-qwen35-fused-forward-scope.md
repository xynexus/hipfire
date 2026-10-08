# Scope: a fused forward for Qwen3.5-family dense and MoE

2026-10-08. Halo / gfx1151, master `7485eca27`. Targets Qwen3.8-27B (dense, 64
layers, 48 Gated DeltaNet + 16 full attention) and Qwen3.6-35B-A3B (MoE, 40 layers,
30 + 10, 256 experts top-8 + a shared expert), both `oq4.25++`, KVarN KV, DFlash on.

Two earlier fusion efforts set the rules here, so they come first:

- **Prefill megakernel, 2026-08-22 — killed.** Prefill was 99.2% GPU-busy: removing
  launches could recover at most 0.8%
  (`2026-08-22-prefill-fused-megakernel-scope.md`).
- **ZAYA decode cooperative megakernel, 2026-07-24 — correct and flat.** Launches fell
  4.3x (1403 -> 323 per token) for +3% tok/s. Every `grid.sync` phase that ran on one
  block (rmsnorm, router, qk-prep) idled the other 159, so DRAM traffic never stayed
  continuous (`docs/perf/zaya-decode-optimization.md` EXP-17/18).

**So the target is not launch count.** A fused forward pays only where it (a) keeps
weight bytes in flight across op boundaries, (b) deletes an activation round trip
through memory, or (c) deletes a host round trip. Every phase below is sized by
which of those three it buys, from measurements taken today.

## Where the time goes (measured 2026-10-08, rocprofv3 kernel traces)

### Decode, one session, DFlash on, 192 tokens (traced; tracing inflates gaps)

| | 27B dense | 35B-A3B MoE |
|---|---|---|
| untraced decode | 14.5 tok/s (69 ms/token) | 57.7 tok/s (17.3 ms/token) |
| weight bytes/token (4.25 b/w) + DeltaNet state r/w | ~13.6 + 0.15-0.30 GB | ~1.56 + 0.06-0.13 GB |
| bandwidth floor at 248.5 GB/s | ~56 ms (17.9 tok/s) | ~6.8 ms (~147 tok/s) |
| **share of the floor achieved** | **~81%** | **~39%** |
| dispatches/token (target + drafter) | 1,861 | 1,775 |
| GPU busy / span (traced) | 91% | 70% |
| kernels < 10 us | 70%, 0.72 s of 15.9 s | 84%, 0.86 s of 5.1 s |
| gaps < 5 us (launch bubbles) | 0.73 s | 0.69 s |
| gaps > 1 ms (host round trips) | 107, 0.53 s (~4.9 ms each) | 149, 0.69 s (~4.6 ms each) |

The A3B is far from its floor and its time is in exactly the three places fusion
can reach: tiny kernels, bubbles between them, and a host stall per DFlash round.
The 27B is already at ~81% of its floor; its upside is bounded at ~1.23x and
realistically ~1.1x.

**Those decode traces start from a 15-token prompt, which understates KVarN.** At
Corrode's depth (decode after a 13.3K-token prompt, traced):

| | 27B | A3B |
|---|---|---|
| decode at 13.3K, untraced rate from the run | 11.3 tok/s (88 ms; **+28%** vs short) | 35.0 tok/s (28.6 ms; **+65%** vs short) |
| KVarN attention, share of busy | **15%** | **27%** (the A3B's largest kernel) |
| `attention_flash_kvarn_tile_batched` | 11.4% | 17.5% |
| `attention_kvarn_routed_batched_gqa*` | 1.8% | 5.2% |
| ideal KV bytes per verify step (4-bit K + Q8 V) | ~0.34 GB (~1.4 ms) | ~0.11 GB (~0.4 ms) |
| traced tile-path time per token | ~9.7 ms | ~4.1 ms |

DFlash verify rows (2-16 per step) sit under `KVARN_PREFILL_WMMA_MIN_ROWS` and take
the per-row tile path, which reads K/V once per query head and per row: ~6x (27B,
24/4 heads) and ~8x (A3B, 16/2) redundant KV reads, plus a dequant per row. The
cost grows linearly with context, so it is the part of decode that Corrode's long
agent prompts feel most.

Top decode kernels, A3B: `moe_gate_up_k8_indexed_batched_splitk` 17%,
`grouped_v3_splitk` 16%, `multicol_w2` (verify) 12%, `moe_down_k8_indexed_expanded`
12%, `grouped_v3` 8.5%, `gemm_q8_0_batched` 5.6% (router / shared-expert gate).
27B: `grouped_v3` 74% -- one big GEMV family, already streaming.

### Prefill (cold 13.3K-token prompt; 7.8K step on the cached 13.3K)

| share of kernel time | 27B cold | 27B step* | A3B cold | A3B step |
|---|---|---|---|---|
| dense compact GEMM | 58% (w64) | 39% | 26% (wave32 `iu4x2_wmma`: K=2048 < the w64 route's 5120) | 22% |
| MoE grouped GEMM | -- | -- | 40% | 35% |
| attention (KVarN WMMA) | 13% | 41% | 11% | 24% |
| overlay correction | 10% | 7% | 1.8% | 1.5% |
| Gated DeltaNet | 6.8% | 4.7% | 5.5% | 4.8% |
| glue: act quantize, kvarn tile quantize, silu-mul, norms, adds, transposes, MoE route/unscatter/combine | ~8.5% | ~6% | ~12% | ~10% |
| GPU busy / span | 96% | 98% | 97% | 97% |

\* traced before #464: 21 of its 41 attention points were the short-segment tile path
that #464 removed (27B step 32.8 -> 27.0 s). The A3B columns are after it.

Prefill stays compute-bound and busy. Fusion there is worth the glue it deletes
plus the activation bytes it stops writing; the rest of prefill's gap to halogen
(22.9 s vs 38.3 s cold on the 27B) is GEMM format and attention, which have their
own plans (overlay-free NF4 format, `study/nf4-codebook-format`; attention at depth).

## The three dimensions every kernel must cover

A kernel tuned for one session at short context is tuned for the wrong workload:
Corrode runs many tasks at once, at 13-21K context, DFlash on. Every kernel and
work item below is specified -- and benchmarked -- across all three.

**1. Batch: B sessions x R rows per session.** Rows are DFlash verify positions
(2-16) or a prefill chunk; sessions are concurrent requests. Weights are shared by
all B x R rows; KV, DeltaNet state and positions are per session. Aggregate decode,
measured today against each model's byte floor:

| | B=1 | B=4 | B=8 |
|---|---|---|---|
| 27B measured / floor | 14.5 / 17.9 (81%) | 47 / ~66 (71%) | 80 / ~122 (66%) |
| A3B measured / floor | 57.7 / ~147 (39%) | 102 / ~280 (36%) | 141 / ~340 (41%) |

(27B floor: shared weights + per-session DeltaNet state. A3B floor: shared dense
weights + the UNION of the experts the B tokens route to + per-session state.)
The dense model loses efficiency as B grows (state traffic, per-session attention);
the MoE model never had it.

**2. KVarN.** Append, 128-token block flush, and attention over 4-bit K records +
f32 window + Q8 V, at every (B, R): each record tile dequantized once per session
and shared by all of that session's rows and all query heads of the KV group. The
per-row tile path that does neither is 15% / 27% of decode at 13.3K (see above).

**3. MoE: expert grouping and microbatching.**
- *Grouping.* A token picks 8 of 256 experts, so at B tokens a layer touches the
  union -- ~30 experts at B=4, ~57 at B=8, ~101 at B=16 -- and expert bytes per step
  grow ~4x / ~7x / ~12x over one token. Each distinct expert's weights must be read
  ONCE per step for every row routed to it (sort rows by expert, one item per
  expert), not once per (row, expert) pair: -11% expert bytes at B=8, -21% at B=16.
  The batch scheduler can also bias toward fewer distinct experts per step only by
  choosing WHICH requests share a step -- a serving-policy lever, out of kernel scope.
- *Microbatching.* Split each step's rows into two microbatches offset by one phase:
  while microbatch A runs its serial parts (norms, router, top-k, attention, the
  DeltaNet recurrence), microbatch B streams expert weights, and vice versa. That is
  the direct answer to ZAYA's flat result -- its serial phases idled the GPU. In the
  counter-scheduled persistent kernel it costs nothing structural: two microbatches
  in flight just means there is always a bandwidth item ready. Microbatch size is a
  tunable set from the expert-union curve (see "MoE: large batches" below): it
  re-reads experts, so it pays at small and moderate B and not at large B; B=1 has
  nothing to split, so there the overlap must come from cross-layer prefetch.

## MoE: large batches are where the A3B pays, and microbatching has a price

A3B decode-step byte floor by batch width. Dense weights are shared; experts are
the union the step's tokens route to, MEASURED with `HIPFIRE_MOE_ROUTE_DUMP` on
21K tokens of Rust source (B tokens sampled far apart, standing in for B sessions);
DeltaNet state ~0.1 GB and, at 13.3K, KVarN ~0.11 GB per session per step; 248.5 GB/s:

| B | union: uniform / measured | step GB (short) | aggregate floor | per-session floor | aggregate @13.3K |
|---|---|---|---|---|---|
| 1 | 8 / 8 | 1.66 | 149 tok/s | 149 | 140 |
| 8 | 57 / 46 | 4.90 | 405 | 51 | 345 |
| 32 | 163 / 111 | 11.7 | 683 | 21 | 528 |
| 64 | 222 / 146 | 17.2 | 925 | 15 | 663 |
| 128 | 252 / 184 | 26.1 | 1,218 | 10 | 800 |

Routing is skewed: ~119 of 256 experts are effectively in use per layer (layer 0:
~203) and the 16 hottest take ~40% of slots, so unions grow well under the uniform
estimate; one session's adjacent tokens share even more (37 at B=8, 116 at B=64).
And routing is FLAT across the top 8: renormalized weight by rank averages 0.236,
0.164, 0.132, 0.113, 0.100, 0.091, 0.085, 0.080 -- the 8th expert carries 8% of a
token's mix, under 0.05 for only 4% of token-layers.

What it says:

- **Per-token cost falls ~6x from B=1 to B=64.** Concurrency pays most on the MoE;
  the 27B's floor is 575 tok/s at B=64 and its per-token cost ~8x the A3B's at B=1.
- **At large B per-session traffic is the majority.** At B=64, 13.3K: experts 9.6 GB,
  DeltaNet state ~6.4 GB, KVarN ~7 GB of ~23 GB. DeltaNet state precision and KVarN
  attention efficiency become first-order for large-batch MoE.
- **Latency and throughput trade directly** (51 tok/s per session at B=8, 15 at B=64).
  Which work gets the wide steps is serving policy: realtime work in small steps,
  opportunistic batch-filling work (speculative review, exploration) in large ones.
  A batch-filler is NOT free at depth -- each added session costs ~0.2 GB/step
  (~0.8 ms) in state + KV even where its experts are already being read.
- **Microbatching re-reads experts.** Splitting B into m microbatches reads
  m x union(B/m) instead of union(B); with the measured unions, two microbatches cost
  x1.17 at B=8, x1.26 at B=16, x1.52 at B=64. It pays where phases are narrow and idle
  the GPU and costs where they are already wide, so m is a function of B: 2 at small
  B, 1 at large B -- not a constant.
- **Per-expert queues, largest first, with aging.** The MoE stage keeps a queue per
  (layer, expert); items run largest queue first -- the hot experts, multi-row and
  WMMA-friendly -- and lone-token items fill the gaps between them. Within a lockstep
  step the order changes overlap and latency, not bytes: the union is fixed by
  routing. Bytes move only by (a) more tokens at the same layer at the same time
  (wider steps) or (b) deferrable opportunistic tokens waiting at a queue for an
  expert being read for realtime tokens anyway, released by an age deadline so a lone
  token never starves. (b) is limited: a token needs all 8 of its experts at every
  layer before it can move on, so waiting stalls its whole session; with the skew a
  hot expert is usually free and a cold one rarely is.
- **Dropping a lone low-weight expert does not pay** (measured, see "Already measured").
  Because routing is flat, a lone (token, expert) under 0.05 weight is rare: ~1% of
  expert reads saved at B = 8-64. At 0.08 it saves 7-14% but strips ~5% of the mix
  from ~40% of tokens (up to 36% from one), a model change, not an approximation.
- **Kernel consequence.** Past B ~ 32 the decode MoE should be expert-grouped (sort
  rows by expert, one item per distinct expert), i.e. the prefill grouped MoE GEMM at
  1-16 rows per expert (B x 8 / union: 2.3 at B=64, 8 at B=256). Today's decode kernels
  work per (row, expert) pair and the wide multicol stops at 16 rows; there is no hard
  cap on decode batch width in serving (`qwen35_decode_batch_max_chunk_size` defaults
  to the session count), so the kernels, not the scheduler, are the limit.

## Architecture

One description covers both models: the four `LayerProgram`s in
`qwen35/lowered.rs:98` (DeltaNet 7 super-ops, FullAttn 5, DeltaNetMoe 6,
FullAttnMoe 4) already name every op. The fused forward is a different EXECUTOR
for those programs, not a new model description.

### D. Decode: persistent layer kernel with a dependency-counter scheduler

- **One persistent kernel per decode step** (all layers, or per layer as the first
  cut), grid = full residency. Not `grid.sync`: each super-op is split into work
  items (a GEMV row tile, an expert's tile, a head's recurrence), and items wait on
  per-op completion counters in global memory. A workgroup that finds nothing
  ready does not idle -- it **prefetches the next op's weights** (touch lines into
  L2/MALL), which is the exact gap ZAYA's design left open.
- **Prefetch window = the MALL.** An A3B layer is ~18 MB of dense weights + ~13 MB
  of routed experts, about the 32 MB MALL: the dense part of layer L+1 (DeltaNet /
  attention projections, shared expert, router) can be in flight while layer L's
  serial phases run. Routed experts cannot be known before their router runs, but
  popularity is skewed and stable: the 8 hottest experts of a layer (~13 MB) cover
  ~26% of its slots, the 12 hottest (~20 MB) ~33%, the 16 hottest ~39% -- measured
  out of sample (hot set from one half of the tokens, hit rate on the other) -- so
  prefetching layer L+1's hottest experts alongside its dense part turns a guess into
  a MALL hit for about a third of the routed reads -- the
  overlap B=1 can still get when there is no batch to microbatch. Everything else
  starts the moment top-k lands.
- **Small ops become work items, not kernels**: rmsnorm + rotate, qk-prep, conv1d,
  the gated norm, top-k + renorm, silu-mul, adds. Each runs on the few workgroups it
  needs while the rest stream weights -- the inversion of ZAYA's single-block phases.
- **Batch rows are a parameter.** DFlash verify (2-16 rows) and batched serving
  (up to 16 rows on the wide multicol) use the same items over B columns; batched
  decode is the throughput mode Corrode runs, and the design must not assume B=1.
- **KVarN is three kinds of item.** (1) append: the V Q8 row write and the K row into
  the f32 window, folded into the QKV projection item that produced them; (2) flush:
  every 128 tokens the completed window block is gathered and quantized into a 4-bit
  record (`kvarn_gather_k_tiles` + `kvarn_quantize_tile`), an item that runs off the
  critical path before the block is next read; (3) attend: split-K items, one per
  (KV head, context slice), each covering ALL query heads of its group and ALL verify
  rows, dequantizing each record tile once, merged by a reduce item. Attend items are
  bandwidth like weights and interleave with weight streaming; at 20K context they
  are ~4% (27B) and ~10% (A3B) of the bytes per step.
- **DeltaNet state** (A3B 31-63 MB, 27B 75-151 MB per token at f16/f32) stays a per-head work item
  reading and writing its own state; it is bandwidth like a weight.

### V. Decode loop: speculative accept on the device

The >1 ms gaps are a host round trip per DFlash round (~4.6-4.9 ms traced). Move
draft -> verify -> accept/reject -> KV/state commit into a device-side loop that
returns to the host only every N accepted tokens or on a stop condition. This is
worth ~13% of traced A3B decode and ~3% on the 27B, independent of D, and can land
first.

### P. Prefill: fused prologues and epilogues, not a megakernel

- **Pre-GEMM**: rmsnorm + FWHT rotate + AWQ + activation quantize in one kernel per
  activation (today `fused_rmsnorm_mq_rotate_awq` + `quantize_act_oq8` + transpose).
  With the NF4 format and an f16 staging GEMM the quantize and transpose disappear.
- **gate/up epilogue**: silu(gate)·up + rotate (+ quantize) written straight from the
  GEMM tile -- the 2·M·B f32 intermediate (71 MB per 512-row 27B chunk) is never
  stored.
- **o / down epilogue**: residual add in the epilogue.
- **KVarN write in the QKV epilogue**: V rows to Q8 and K rows to the window straight
  from the projection tile, and the 128-token block flush (gather + 4-bit quantize,
  2.3-3% of prefill today) done by the attention kernel that already has the block's
  K in registers when it attends the segment that completes it.
- **MoE**: fold `moe_gate_up_unscatter` / `moe_down_combine` / `rotate_x_..._indexed`
  into the grouped GEMM's prologue/epilogue.
- **A3B dense GEMM at K=2048**: give it the tuned w64 structure (it runs the older
  wave32 kernel today).

## Phases, sized and gated

| # | work | buys | expected | exit | kill |
|---|---|---|---|---|---|
| 0 | Untraced decode accounting: per-op bytes and time per token, both models, B = 1/4/8, short and 13.3K context, A3B expert-union size per step | the baseline every later number is judged against | -- | a per-op table per (B, context) | -- |
| K0 | GQA-shared KVarN attention for verify / multi-row decode: one workgroup per (session, KV head, context slice) over all query heads and that session's rows, B sessions per launch, each record tile dequantized once (extend `attention_kvarn_routed_batched_gqa*` to causal multi-row, or route verify to the WMMA kernel per KV head) | redundant KV reads | at 13.3K: 27B ~-8 ms/token (~-10%), A3B ~-3.5 ms/token (~-12%); more at depth | parity vs tile path, all verify widths, tree bias | < 5% at 13.3K |
| V | Device-side DFlash accept loop | host round trips | A3B -10% decode time, 27B -3% | bit-identical tokens vs host loop; tok/s gain | < 5% on A3B |
| D1 | Persistent kernel for ONE A3B MoE layer (DeltaNetMoe), counters + idle-prefetch (dense + hottest experts of the next layer), per-expert queues largest-first with aging, m microbatches | bubbles, tiny kernels, cross-op streaming, expert reuse, serial-phase overlap | layer at >= 70% of its byte floor at B = 1, 8, 32, 64, m in {1, 2} (today ~36-41% whole-model) | per-layer parity (cos 1.0); bytes moved / layer time at each B | < 1.3x over today's 6 super-ops at every B (ZAYA-style flat) |
| D2 | All four layer programs, whole decode step, B in 1..16 | the rest of D | A3B 57.7 -> ~100 tok/s; 27B 14.5 -> ~16 | greedy text identical; DFlash acceptance unchanged | A3B < 80 tok/s |
| P1 | gate/up silu epilogue + pre-GEMM fused quantize (current format) | activation round trips | 27B prefill -3..5% | parity; served A/B | < 2% |
| P2 | A3B K=2048 dense GEMM on the w64 structure | GEMM efficiency | A3B prefill -8..12% (26% share) | kernel bench vs `iu4x2_wmma` | < 1.2x kernel |
| P3 | MoE route/combine folded into the grouped GEMM | MoE glue | A3B prefill -2..3% | parity | < 2% |

Order: 0, K0 (independent of everything else and the largest sure win at depth), V,
D1 (the decisive experiment -- it either breaks ZAYA's flat result or
confirms it for this hardware), then D2 or stop; P1-P3 are independent and small.

## What D1 must prove that ZAYA did not

ZAYA reached full cooperative residency (160 workgroups) and stayed at ~48 GB/s,
because phases were serial and `grid.sync`-separated. D1's single question: **with
dependency counters and idle-workgroup prefetch, does one A3B layer sustain >= 70%
of its byte floor?** Measure it directly -- bytes moved / layer time -- not tok/s,
before building the rest. If it does not, the decode megakernel is dead on this
hardware for the second time and the plan reduces to V + P.

## Risks

- **Counters across workgroups need device-scope atomics and memory ordering**
  (release/acquire on the counter, `s_waitcnt` + `buffer_gl*_inv` on the consumer).
  ZAYA's one hang was a variable shadow, but a wrong fence here is a silent race.
- **Wedging the APU GPU**: a spinning persistent kernel that never sees its counter
  holds the GPU; every wait needs a bounded spin with an error exit (the 397B kworker
  deadlock is the precedent).
- **Weight format churn**: if NF4 replaces `oq4.25++`, D's GEMV items change their
  decode step; keep the item interface format-agnostic (a per-format decode function).
- **Per-model kernels**: dims are compile-time for performance; the 27B and A3B are two
  instantiations, plus each future Qwen3.5-family size.

## Already measured -- do not redo

| | result |
|---|---|
| prefill megakernel for launches | 99.2% busy; <= 0.8% recoverable |
| ZAYA coop decode megakernel (grid.sync) | launches -77%, tok/s +3% |
| overlay fused into the GEMM: K-loop, epilogue, group fold | all three slower (parked branches) |
| hipGraph for decode | ~0 (ZAYA EXP-12/15) |
| drop a lone low-weight routed expert (A3B) | ~1% of expert reads at weight < 0.05; routing is flat (rank-8 mean 0.080) |
| run unrouted tokens through an expert to fill its queue | no benefit: a non-selected expert's weight is 0 after top-k; computing it is waste, adding it changes the model |
