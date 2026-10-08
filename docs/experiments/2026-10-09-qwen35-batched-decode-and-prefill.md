# Qwen3.5-family: batched decode and prefill after K0

Follows `2026-10-09-qwen35-decode-accounting-k0.md` through the next phases of
`docs/plans/2026-10-08-qwen35-fused-forward-scope.md`. gfx1151, Qwen3.8-27B and
Qwen3.6-35B-A3B `oq4.25++`, KVarN-4. Decode rates are untraced `phase0_probe` runs
(aggregate tok/s, 128 tokens/session); prefill is `prefill_reqs.py` (a cold
13.3K-token prompt, then the same conversation +7.8K attached).

## Phase V does not apply to the served models

V was "move the DFlash accept loop onto the device". Neither served model runs
DFlash: the 27B's drafters are parked (`*.parked-slower-than-plain-decode`) and the
A3B has `dflash_draft: off`. Both decode B=1 through the batched n-gram path. The
round's ~90 blocking `hipMemcpy` D2D copies were made stream-ordered anyway
(branch `perf/dflash-async-copies`, parked): output bit-identical, decode unchanged
on both models, because that code path is not on.

The A3B's B=1 gaps (29% of the traced deep step) are ~1,800 launches/token through
the n-gram path -- phase D's ground, not V's.

## Batched decode

| PR | change | A3B | 27B |
|---|---|---|---|
| #473 | `gemm_q8_0_batched` spreads batch rows over waves (bit-identical). The MoE router (256x2048) and shared-expert gate (1x2048) were one wave per output row, ~230 us each per layer at 32 rows | B=8 138.9 -> 150.8, B=16 150.3 -> 163.3, B=32 171.3 -> 185.7, B=8 deep 108.9 -> 115.0 | -- |
| #474 | batch-serving dense projections with K <= 4096 stay on the wide multicol GEMV to 32 rows (was 16; the BN=32 tile's win was measured at the 27B's K) | B=16 162.4 -> 188.0, B=24 169.1 -> 200.7, B=24 deep 126.1 -> 136.5 | unchanged |
| #472 (open, needs a decision) | multi-session batches take the grouped W4A8 expert GEMM from 16 rows (was 64) | B=16 +9%, B=32 +24% | -- |

#472 changes numerics: routed experts in 16-63-row batches would run W4A8 (int8
activations), as prefill and >= 64-row batches already do -- and as batched decode's
DENSE projections already do (multicol takes int8 activations). The bit-exact f32
grouped kernel is no faster at decode widths: it saves the weight bytes but every
row still re-reads each slot's f32 activations (0.50-0.74x the GEMV at 1-2 tokens
per expert).

## Prefill

| PR | change | A3B cold / attached (s) | 27B cold / attached (s) |
|---|---|---|---|
| (before) | | 16.9 / 11.6 | 38.2 / 27.0 |
| #475 | wave64 compact GEMM at every K (was K >= 5120; the 27B wo's 0.75x was stale -- re-benched it is 1.01-1.28x, A3B shapes 1.23-3.09x) | 14.8 / 10.3 | 37.7 / 26.7 |
| #476 | prefill chunk 1024 for MoE, 512 dense (was 256 for both); DFlash ring staging sized from the same resolver and the forward clamps chunks to it | 10.2 / 7.2 | 35.6 / 25.0 |

A3B prefill: 16.9 -> 10.2 s (-40%). Scratch: +0.5 GB (A3B), +0.2 GB (27B).

Chunk sweep (cold / attached, s), 256 / 512 / 1024 / 2048:
A3B 14.7 / 11.5 / 10.3 / 10.2 and 10.2 / 8.1 / 7.2 / 7.2;
27B 37.6 / 35.9 / 37.0 / 41.5 and 26.7 / 25.0 / 25.4 / 28.2.

### Where A3B prefill goes now (cold 13.3K, 10.3 s traced)

| | share |
|---|---|
| MoE experts (`gemm_oq_compact_iu4x2_moe_grouped` 3.15 s) | 35% |
| KVarN prefill attention | 19.5% |
| dense GEMM (wave64) | 15% |
| DeltaNet | 12% |
| activation quantize / transpose / overlay | 10% |
| `kvarn_quantize_tile` | 5% |

The grouped expert GEMM costs per 16-slot tile, not per token: synthetic routing at
M=1024 K=2048 over 256 experts takes 2.6 / 2.9 / 4.3 / 7.0 ms at 8 / 16 / 32 / 64
slots per expert -- each extra tile re-reads all 256 experts (~200 GB/s). At chunk
1024 an expert averages ~32 slots, two tiles. A 32-slot tile that applies each
weight load to both halves (scatter padded to 32, a kernel variant) would bring
that to ~one tile: est. ~1 s of the 10.3 s. Launching tile-major instead (so an
expert's tiles hit cache) measured slower (4.3 -> 5.3 ms).

`kvarn_quantize_tile`: hoisting its per-element `__expf` changes nothing (same
record hash, same 50 ms / 1024 tiles) -- it is bound by its strided passes over the
128 KB tile; a register-resident tile with cross-thread reductions would change
summation order.

27B prefill (35.6 s): dense GEMM ~57%, attention ~16%, `oq_compact_overlay_correct_tr3`
~10% (fusion attempts were slower, see the scope's already-measured list).

## D1 baseline

Per A3B layer, today vs its byte floor (dense ~39 MB + the step's expert union +
state): B=1 ~49%, B=8 ~33%, B=32 ~21%, B=64 ~20% (before #473/#474). Removing all
launch gaps alone would be worth at most 1/busy: ~1.4x at B=1, ~1.28x at B=8,
~1.16x at B=32 -- under D1's 1.3x kill line at batch, so D1 has to win on overlap and
expert reuse, not launches. Not started.

## Also found

- Multi-session decode is not run-to-run deterministic (batch composition varies
  with arrival timing); per-session output hashes differ between identical runs.
- One master run of single-session A3B decode at 13.3K produced a different greedy
  continuation on one of four identical requests: some nondeterminism remains
  outside the KVarN quantizer fixed in #469.
- The tiny-quant gate's `qwen3_5_moe/kld:oq8` and `qwen3_5_moe_indexed/kld:oq8++`
  drifts reproduce on clean master.
