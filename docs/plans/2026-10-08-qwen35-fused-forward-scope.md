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
  serial phases run. Routed experts cannot be prefetched before their router runs;
  they start the moment top-k lands.
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
| 0 | Untraced decode accounting: per-op bytes and time per token, both models, B = 1/4/8 | the baseline every later number is judged against | -- | a per-op table | -- |
| K0 | GQA-shared KVarN attention for verify / multi-row decode: one workgroup per (KV head, context slice) over all query heads and rows, each record tile dequantized once (extend `attention_kvarn_routed_batched_gqa*` to causal multi-row, or route verify to the WMMA kernel per KV head) | redundant KV reads | at 13.3K: 27B ~-8 ms/token (~-10%), A3B ~-3.5 ms/token (~-12%); more at depth | parity vs tile path, all verify widths, tree bias | < 5% at 13.3K |
| V | Device-side DFlash accept loop | host round trips | A3B -10% decode time, 27B -3% | bit-identical tokens vs host loop; tok/s gain | < 5% on A3B |
| D1 | Persistent kernel for ONE A3B MoE layer (DeltaNetMoe), counters + idle-prefetch | bubbles, tiny kernels, cross-op streaming | layer at >= 70% of its byte floor (today ~39% whole-model) | per-layer parity (cos 1.0); layer time | layer < 1.3x faster than today's 6 super-ops (ZAYA-style flat) |
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
