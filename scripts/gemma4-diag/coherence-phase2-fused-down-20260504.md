# Gemma 4 Phase 2 fused-down coherence + bench

- Branch: `gemma4`
- Date: 2026-05-04
- Hardware: 7900 XTX (gfx1100, 25.8 GB VRAM, HIP 7.2)

## Coherence

`scripts/gemma4-diag/coherence-gemma4.sh` — 6/6 strict pass, output **bit-identical** to
`scripts/gemma4-diag/coherence-baseline-fused-default-20260502.md` (Phase 1 baseline) on all
models including the two 26B-A4B tests. Only wall times differ (run-to-run noise).

## Decode bench (creative prompt, max_tokens=300)

| Path | Decode tok/s | Prefill tok/s | TTFT (ms) |
|---|---|---|---|
| PHASE0 legacy (`HIPFIRE_GEMMA4_MOE_FUSED=0`) | 60.8 | 65.2 | 752 |
| PHASE1 fused gate_up (`FUSED=1 DOWN_FUSED=0`) | 66.0 | 71.1 | 689 |
| PHASE2 + fused down (default) | 66.9 | 74.2 | 661 |

- PHASE0 → PHASE1: **+8.6% decode**, +9% prefill, –8% TTFT
- PHASE1 → PHASE2: **+1.4% decode**, +4.4% prefill, –4% TTFT
- PHASE0 → PHASE2: **+10% decode**, +14% prefill, –12% TTFT

## Note on plan vs reality

The pre-Phase-2 plan (`docs/plans/gemma4-moe-indexed-gemv.md`) cited a 5-7 tok/s
baseline for 26B-A4B and predicted Phase 2 would push to 30-40 tok/s. **The actual
baseline is ~60 tok/s decode** — the original 5-7 number was either from a much
earlier kernel state or a different hardware setup. The MoE down launch overhead
was real but not the dominant cost; the parallel dense MLP branch + 30 layers of
attention contribute the bulk of per-token work.

Phase 2 still ships: it's a real +1.4% decode on top of Phase 1, bit-identical
output, no extra VRAM (one 32-byte H2D per layer per token for the fused weights
buffer), and removes 7 launches per layer from the hot path. The TTFT improvement
(–8% PHASE1, –12% PHASE0→2) is the standout — first-token latency drops from 752
to 661 ms.
