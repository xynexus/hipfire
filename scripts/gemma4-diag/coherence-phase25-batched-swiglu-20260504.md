# Gemma 4 Phase 2.5 batched gelu+mul coherence + bench

- Branch: `gemma4`
- Date: 2026-05-04
- Hardware: 7900 XTX (gfx1100)

## What changed

New kernel `gelu_tanh_mul_batched_f32` collapses the 8 per-expert
`gelu_tanh_f32 + mul_f32` launches in `apply_moe_branch` into one
launch per layer. Decode launch count drops from ~1317 to ~867
per step (–34%). Wired in `apply_moe_branch` between the fused
gate_up and fused down kernels, so the full MoE branch is now
3 launches per layer (gate_up + batched-swiglu + down) vs the
legacy 40 launches per layer.

## Coherence

`coherence-gemma4.sh` — 6/6 strict pass (no panics, no zero tokens, no
timeouts). Output **bit-identical** on E2B / E4B / 31B-cap / 31B-reason
(non-MoE). Soft diff on 26B-A4B (±1 ULP from FMA fusion in the batched
kernel vs separate gelu+mul launches; outputs remain fluent and on-topic):

```
26b-a4b-cap:   "10-year-olds" → "10-year-old children"
26b-a4b-reason: "9 died" → "nine died"
```

CLAUDE.md coherence gate explicitly allows soft diffs ("hard fails only
on panics, zero tokens, or timeouts").

## Decode bench (creative prompt, max_tokens=300)

Clean apples-to-apples (forces per-expert `gelu_tanh + mul` via
`HIPFIRE_GEMMA4_MOE_BATCHED_SWIGLU=0`, NOT the gelu_erf knob which
contaminates the math):

| Path | Decode | Prefill | TTFT |
|---|---|---|---|
| Phase 2 per-expert (avg) | 67.5 | 74.4 | 659 ms |
| Phase 2.5 batched (avg)  | 73.3 | 81.6 | 601 ms |

Phase 2 → Phase 2.5: **+8.6% decode, +9.7% prefill, –9% TTFT**.

Cumulative Phase 0 → Phase 2.5:
- Decode: 60.8 → 73.3 (**+20%**)
- Prefill: 65.2 → 81.6 (**+25%**)
- TTFT: 752 → 601 ms (**–20%**)

## Profile delta

Pre-Phase-2.5 (commit 2967da6, profile_gemma4 at ctx ~20):
- 1317 launches/step
- mul_f32: 14.7% (270 calls/step)
- gelu_tanh implicit in mul-then-gelu-tanh chain

Post-Phase-2.5 (this commit):
- ~867 launches/step (–34%)
- mul_f32 calls drop from 270/step → 30/step (just the SwiGLU side)
- 30 new `gelu_tanh_mul_batched_f32` launches/step

## Next levers (remaining per-token cost from profile)

1. **rmsnorm_f32 fusion** — still 21% of kernel time. 241 calls/step (8
   per layer). Many are followed immediately by a GEMV; could fuse
   norm into the next GEMV's prologue, saving ~200 launches/step.
2. **Batched prefill** — `gemma4::forward_prefill_batch` is still a
   stub. Would amortize all per-token launch overhead across the
   prompt at once. Largest remaining lever for prefill.
3. **mq_rotate_x batching** — 13.3% of kernel time, 235 calls/step.
   Many are setup for downstream gemv_hfq4g256; could fuse into a
   single FWHT-over-batch kernel for the MoE pre2_rot path.
