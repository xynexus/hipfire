# Batched vs per-token prefill: a uniform ~1.7%, not KVarN and not quantization

State box: halo, gfx1151. `compare_prefill_hidden_paths`, worst relative
divergence over all layers/rows, one process, one `HiddenStateRingBuffer`.

## What it is not

Chasing the residual left after the KVarN flush-before-attend fix, every
plausible cause was eliminated by measurement:

| hypothesis | test | result |
|---|---|---|
| KVarN-specific | `--kv-mode q8` vs `kvarn`, 27B oq4.25++ | 1.58e-2 vs 1.62e-2 — **same** |
| the 4-bit codec | n=64, no block flushes, all K still f32 | still 1.62e-2 |
| the Hadamard K/Q rotation | `HIPFIRE_KVARN_ROTATE=0` | still diverges (2.83e-2) |
| activation quantization (batched W4A8 vs per-token wider GEMV) | **bf16 2B model**, no quantization anywhere | **1.72e-2 — still there** |
| accumulated reassociation in a chunked scan | sweep n = 2,4,8,16,32,64 | **flat**, 1.56e-2 → 1.58e-2 |
| the LinearAttention path | n=2 on 27B | LA layers 0-2 agree; divergence starts at layer **3**, the first FullAttention layer |

## What it is

A **uniform ~1.6-1.7% divergence between the two prefill implementations**,
independent of KV mode, weight quantization, activation precision, batch size,
and model:

| model | kv | n=2 | n=8 | n=64 |
|---|---|---|---|---|
| Qwen3.8-27B oq4.25++ | q8 | 1.56e-2 | 1.58e-2 | 1.58e-2 |
| Qwen3.8-27B oq4.25++ | kvarn | — | — | 1.62e-2 |
| qwen3.5-2B **bf16** | q8 | 1.72e-2 | 1.72e-2 | 1.72e-2 |

Flat in n and flat across two different models and dtypes. That profile — a
constant relative offset that ignores every knob — reads as a **systematic
difference between the two implementations**, not accumulated float error, which
would vary with depth, width and batch size.

This **reinstates the prior finding** that batched and per-token prefill export
different hidden states for every dtype, including a bf16 control. An earlier
revision of the KVarN note "corrected" that claim on the strength of an
`--kv-mode fp32` control reading 0.00e0; that control was degenerate — fp32 KV
fails `fa_kv_ok` and never batches, so it compared per-token against per-token.

## Why this stops here rather than continuing

- It does **not** affect served output. 64 greedy tokens after a 2059-token
  prompt are byte-identical (sha256) between the two paths, and the batched
  route's KLD sits inside the envelope already accepted for batched prefill.
- It is **pre-existing**, not a regression from any work on this branch.
- The instrument's own docstring warns that this comparison has historically
  been confounded by the two paths capturing at different call sites. A constant
  offset that ignores model, dtype and batch size is exactly the shape a
  capture-point/semantic difference would take, so the next step is to verify the
  two capture points are the same quantity **before** treating it as a numerical
  bug.

## Where it does matter

DFlash spec-decode. The drafter consumes exactly this ring buffer, and enabling
compact-resident Opus for a hidden-capturing forward reproducibly took DFlash2's
accept_rate 0.468 -> 0.000. Any drafter retrain or accept-rate work should treat
"which path captured the hidden states" as a first-class variable until this is
resolved.

## Next step when it is picked up

Confirm the two capture points export the same quantity — same tensor, same
point in the layer, same residual state — before assuming arithmetic. If they do
match, bisect the FullAttention layer at **n=2 on the 27B**, where LA layers
still agree and divergence first appears at layer 3: two tokens and one layer is
as small as this reproducer gets.
