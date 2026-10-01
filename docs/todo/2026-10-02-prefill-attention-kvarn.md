# TODO: a real prefill attention kernel for KVarN (query-tiled, WMMA)

Status: v1 DONE (`f8ab1c334`) — see "Result". GQA sharing and double buffering
are the open follow-ups.
Date: 2026-10-02
Model: Qwen3.8-27B--oq4.25++ (16/64 layers carry KV, KVarN 4-bit K + Q8 V), gfx1151.

## Measured

`rocprofv3 --kernel-trace` of `hipfire serve`, one cold ~8.3K-token prompt (a real
Corrode tool-step prompt), 1 output token. 43.6 s of kernel time (~190 tok/s):

| kernel | time | share |
|---|---|---|
| `attention_flash_kvarn_tile_batched` | 18.4 s | **42%** |
| `gemm_oq_compact_iu4x2_w64` | 13.5 s | 31% (~33 int8 TOPS) |
| `oq_compact_overlay_correct_tr` | 5.8 s | **13%** |
| `gated_delta_net_f16` | 1.7 s | 4% |
| everything else | ~4 s | 10% |

The GEMMs are not the problem at this length. Attention is, and it grows with the
square of the context: an attached 9.8K tail at position 8K ran at ~98 tok/s, and
Corrode tool loops reach 17-28K tokens.

## Why attention is slow

The prefill path dispatches the DECODE kernel per query row: grid
`(n_heads x 128 threads, kv_tiles, rows_in_sub_batch)` — e.g. `(3072, 64, 64)`,
17 ms per launch — one workgroup per (head, 128-position KV tile, query row), with
`sub_batch` capped by the `partials` buffer (64 rows). Every query row re-reads and
re-dequantizes every K/V tile of its prefix on VALU, and a separate
`attention_flash_asym_reduce_batched` pass merges the per-tile partials. That is
O(rows x tiles) tile loads with no reuse across rows. Rough count for 8.3K tokens:
~13.6 TFLOP of attention math in 18.4 s = ~0.7 TFLOPS.

## Scope

A causal flash-attention prefill kernel over KVarN storage:

- Query block of 64-128 rows x KV tile of 128 positions per workgroup; dequantize
  the K tile (4-bit var-norm records) and V tile (Q8) into LDS ONCE per query block.
- QK^T and PV on WMMA (f16), online softmax across KV tiles in registers — no
  per-tile partials buffer and no reduce pass.
- GQA: one workgroup per KV head serves its G query heads (G = 6 on the 27B), as
  the decode GQA kernel already does — K/V loaded once for all G.
- Causal mask only on the diagonal tile; tiles past the query block skipped.
- The f32 recent window (the last positions not yet in a full 128-block) handled
  as a tail tile.

Target: attention at 8.3K from 18.4 s to <= 2 s, i.e. this prompt from ~190 to
~300+ tok/s end to end, more at longer contexts.

Second, smaller: `oq_compact_overlay_correct_tr` at 30-40% of the GEMM's own time
at prefill widths — the same overlay-at-weight-decode idea as candidate 1 in
`2026-10-01-decode-width-opus-gemm.md`, which would help both.

## Acceptance

- Parity vs the current path within a stated tolerance (logits, 8K prompt), and
  the tiny-prefill gate.
- Kernel trace of the same prompt: attention share and total time recorded here.
- Corrode CAE tool-step prefill (attached 9-28K contexts) re-measured.

## Result (v1, `f8ab1c334`)

`attention_prefill_kvarn_wmma` — 64 rows x 1 head per workgroup, 16-token K/V
sub-tiles dequantized into LDS, f16 WMMA, online softmax, causal by `positions`;
routed inside `attention_flash_kvarn_batched_masked` (>= 32 rows, no tree bias,
head_dim 256, gfx11). Parity: `parity_kvarn_prefill_wmma` (2.6e-5..4.8e-4 vs an
f64 reference, ~2x the f32 path's error from f16 staging).

End to end, cold, Qwen3.8-27B, same first output token both ways:

| prompt | old path | WMMA | tok/s |
|---|---|---|---|
| 8.3K-token Corrode tool step (64 out) | 47.8 s | 31.2 s | ~175 -> ~270 |
| 18K-token captured conversation (1 out) | 148.2 s | 66.5 s | ~121 -> ~271 |

Prefill throughput no longer falls with context at these lengths.

Open:
- GQA: each of the G=6 query heads of a KV head re-dequantizes the same K/V
  sub-tile. One workgroup per KV head (G heads x 16 rows, or 64 rows looped over
  heads) would cut the dequant work 6x.
- No double buffering: stage, barrier, compute, barrier per 16-token sub-tile.
