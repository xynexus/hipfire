# Qwen3.5-family decode: Phase 0 accounting and K0 (GQA-shared KVarN attention)

Phases 0 and K0 of `docs/plans/2026-10-08-qwen35-fused-forward-scope.md`, gfx1151
(Strix Halo, 40 CUs / 20 WGPs, 248.5 GB/s), Qwen3.8-27B `oq4.25++` and
Qwen3.6-35B-A3B `oq4.25++`, KVarN-4 KV, DFlash on for B=1, batched n-gram
speculation for B>1. "Deep" = a ~13.3K-token shared system prefix (CAE sources)
plus a one-line question, minted once so every session attaches it; "short" = a
one-line prompt. Rates are untraced `phase0_probe` runs, 128 tokens per session.

## Decode rate before K0 (master 2026-10-08, aggregate tok/s)

| B | 27B short | 27B deep | A3B short | A3B deep |
|---|---|---|---|---|
| 1 | 14.6 | 11.5 | 57.3 | 42.9 |
| 4 | 46.8 | 34.5 | 99.8 | 71.8 |
| 8 | 78.8 | 50.8 | 137.9 | 96.7 |
| 16 | 100.2 | 63.4 | 146.5 | 101.0 |
| 32 | 128.9 | 73.4 | 173.4 | 115.3 |
| 64 | 141.1 | 73.4 | 252.2 | 155.0 |

## What K0 shipped

| PR | change | effect |
|---|---|---|
| #468 | `attention_flash_kvarn_tile_batched` and `attention_flash_asym_reduce_batched` compiled with head_dim 256 / bits 4 as constants (their loops were rolled and indexed registers through `v_movrel`) | tile+reduce per FA layer at 13.4K: 27B 489 -> 251 us, A3B 364 -> 178 us; B=1 deep 27B 12.3 -> 12.9, A3B 43.9 -> 47.7 tok/s |
| #469 | `kvarn_quantize_tile` raced on its best-iteration snapshot (3-4% of identical tiles quantized differently) | K flush deterministic |
| #470 | `attention_decode_kvarn_wmma` (+ `_routed`): one workgroup per (KV head, context split, 16 query rows = G heads x the session's rows), each K/V sub-tile dequantized once for the group; device-sized splits; routed entry for multi-session decode | see below |

K0 v1 -- a VALU tile kernel shared across the query group (`attention_flash_kvarn_tile_gqa*`,
branch `perf/kvarn-tile-gqa`, parked) -- was correct but 0.73x the per-head tile kernel
warm and slower in situ: L2 already absorbed the per-head re-reads and G-fold fewer
workgroups serialized the per-token reductions.

Decode after #470, tok/s (tile/routed path -> decode WMMA; three A/B rounds agree):

| | 27B | A3B |
|---|---|---|
| B=1 deep | 12.44 -> 13.21 (300-token generation) | 47.8 -> 52.3 |
| B=4 deep | 34.4 -> 38.7 | 75.0 -> 87.9 |
| B=8 deep | 49.5 -> 55.6 | 96.0 -> 111-113 |
| B=1 short | 14.7 -> 14.7 | 57.6 -> 58.0 |

K0 as a whole at 13.3K, B=1: A3B 42.9 -> 52.3 tok/s (-4.2 ms/token, +22%); 27B 11.5 ->
12.9 with #468 on the 128-token probe, and a further +6% from #470 on a 300-token
generation (the probe cannot show it, below). B=4-8: +12% (27B) and +17% (A3B). The
gate was "< 5% at 13.3K": passed on both models.

The 27B's 128-token probe reads 12.9 -> 12.8 because f16 staging changes the greedy
text and with it DFlash's acceptance; traced, its attention fell 12.3 -> 5.4
ms/token and the step span 108 -> 102 ms.

## Per-op accounting after K0 (rocprofv3, 64 tokens/session, ms per decode step)

"Step" = one token per session. Tracing inflates gaps; B=1 "short" traces are
dominated by idle between the probe's warm and measured requests and are omitted.

| category | 27B B=1 deep | A3B B=1 deep | 27B B=8 short | 27B B=8 deep | A3B B=8 short | A3B B=8 deep |
|---|---|---|---|---|---|---|
| span | 102.2 | 32.2 | 105.8 | 90.6 | 60.2 | 46.1 |
| dense GEMV/GEMM | 74.0 (72%) | 9.3 (29%) | 73.1 (69%) | 46.1 (51%) | 17.0 (28%) | 10.8 (23%) |
| MoE experts | -- | 7.7 (24%) | -- | -- | 23.9 (40%) | 15.5 (34%) |
| KVarN attention | 5.5 (5%) | 1.9 (6%) | 0.8 | 18.3 (20%) | 0.3 | 4.3 (9%) |
| DeltaNet | 2.4 | 1.1 | 7.5 (7%) | 4.6 | 3.3 | 2.0 |
| norm/rotate/glue | 3.3 | 1.7 | 4.8 | 3.3 | 1.8 | 1.4 |
| gaps | 15.9 (16%) | 9.2 (29%) | 15.6 (15%) | 15.7 (17%) | 10.7 (18%) | 10.0 (22%) |
| dispatches / step | 2161 | 1796 | 3364 | 2117 | 2262 | 1428 |

Bandwidth floors for reference: 27B weights ~13.6 GB/step (~56 ms); A3B dense ~1.56
GB/step (~6.3 ms) plus routed experts; KVarN K+V at 13.3K ~21.6 MB/layer/session
(27B, 16 FA layers) and ~10.8 MB (A3B, 10).

## Findings that steer the next phases

1. **Host round trips are now the A3B's largest B=1 item** (29% of the traced deep
   step). Per DFlash round the host reads the draft argmax (to build verify tokens:
   the noise embedding takes each token as a kernel scalar) and the verify argmax (to
   scan acceptance, pick the replay length and the next positions), and issues ~90
   blocking `hipMemcpy` device-to-device copies -- DeltaNet snapshot save/restore,
   the hidden-row scatter, the drafter's K/V concat -- each of which drains the GPU.
   `commit_staging_to_ring` adds an explicit `stream_synchronize`. Phase V.
2. **Batched A3B decode reads each token's experts separately.**
   `gemv_oq_compact_moe_*_k8_indexed_batched*` launch over (row tile, k, token): the
   expert bytes per step grow as B x 8, not as the union of distinct experts. The
   step time grows ~5.5 ms per added row from B=1 to 32 (59 / 106 / 187 ms at
   B=8/16/32, short); the union floor is ~12 / 20 / 30 ms. The "B=16 cliff" in the
   matrix above is this linear growth, not a cliff (re-measured: 137 / 151 / 172
   tok/s at B=8/16/32). This is D1's per-expert queue item.
3. **At depth and batch, attention is again a fifth of the 27B** (B=8 deep, 18.3
   ms/step). `attention_decode_kvarn_wmma_routed` runs one workgroup per ROW, so a
   session's n-gram verify rows (widths 8-10 for 8 sessions here) each stream that
   session's 21.6 MB/layer again; grouping a session's rows into one workgroup's 16
   (as the single-session entry does) would read it once.
4. Not a cliff either: batched n-gram speculation drafts little on prose at any B
   (acceptance floor 0.3); the B=16 multicol widths were 16 almost throughout.

## Pre-existing, unrelated

The tiny-quant gate's `qwen3_5_moe/kld:oq8` (0.0141 vs 0.0016) and
`qwen3_5_moe_indexed/kld:oq8++(calib)` cells drift identically on clean master.
