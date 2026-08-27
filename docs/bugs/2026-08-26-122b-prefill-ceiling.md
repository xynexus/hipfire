# 122B prefill is capped at 29 tok/s by a kernel asymmetry

Status: **OPEN**. Opened 2026-08-26. Summary entry in `BUGS.md`.
Full measurement log: `docs/plans/2026-08-26-122b-perf-findings.md`.

## Symptom

Prefill is FLAT at 29.0–29.1 tok/s for n=61, n=92 **and** n=658. Flat in n is the
signature of no amortisation at all: ~34 ms/token, ~8.6 GB read per token against
~5.3 GB of active weights.

| | 122B | 35B-A3B | 27B |
|---|---|---|---|
| prefill tok/s | **29.1** | 69.6 | 306 |
| decode tok/s | 23–27 | 61.5 | 69.9 |

## Two coupled causes — either alone is a no-op

**1. The admission gate declines the model.** The artifact is mixed-precision per
layer (1288 `Oq8G256` expert tensors against 23,288 `OqPlusCompact`; 37 of 48
layers). `classify_routed_expert_dtypes` (`qwen35/mod.rs:1651`) has three
variants — `Uniform`, `QuantWithFullPrecisionFallback{quant, full}` (quant +
F16/BF16 only) and `Invalid` — so a two-QUANT mix matches none, and
`routed_supported` maps `Invalid => false` (`:2106`). Meanwhile
`routed_oq_mixed_compact` (`:1711`) is computed on the adjacent lines, is TRUE,
and says the compact family can serve the layer from its per-expert stride table.
The admission never consults it. Instrumented:

    [moe-admit] router=BF16 router_ok=true scalar_gate=BF16 shared_gate_ok=true
                profile=Invalid oq_mixed_compact=true gu=OqCompactG256 down=OqCompactG256

**2. The grouped prefill GEMM cannot serve a mixed layer.**
`gemm_oq_compact_moe_grouped_wmma.hip:53` takes `block_stride` as a LAUNCH-WIDE
scalar. The decode GEMV `oqc_moe_row_dot`
(`gemv_oq_compact_moe_indexed.hip:76`) takes a PER-EXPERT stride table with
`block_stride == 0` as the Oq8 sentinel. Mixed layers are therefore dispatchable
at decode and not at prefill.

## The experiment that proves they are coupled

Patching only (1) to admit mixed-compact gives `[pbs-gate] verdict=true` — and
prefill 29.0 → 29.0 (n=92), 29.1 → 29.1 (n=658). rocprofv3 says why: every
dominant kernel is a GEMV at **one call per token per layer**
(31,680 = 658 × 48):

    gemv_oq_compact_grouped_v3                            213840   5624.6 ms  26.8%
    gemv_oq_compact_moe_gate_up_k8_indexed_batched         31680   4121.7 ms  19.7%
    gemv_oq_compact_moe_down_k8_indexed_batched_expanded   31680   3040.4 ms  14.5%
    gemv_oq_compact_grouped_v3_splitk                      31680   2393.2 ms  11.4%

## Fix

Both halves together: a per-expert stride table in the grouped GEMM, and an
admission that consults `routed_oq_mixed_compact`.

Sizing: grouped MoE measured +31% prefill on the 27B (`43653ea3d`), but the 122B
starts from FULLY per-token and its uniform-compact sibling prefills at 69.6 vs
29.1. Note E=256/K=8 makes tokens-per-expert 8x lower than the 27B at equal n, so
the grouped crossover sits well above that commit's N>=64 threshold.

⚠️ Do this AFTER the coherence bug. A faster incoherent model is not progress,
and the mixed-layer path is the region that bug lives in.
