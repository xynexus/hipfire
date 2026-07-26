# DFlash 9B composed cached block wall + M_TILE double-stream verdict (Phase 0 task #38)

nix1 / npu1 (Phoenix gfx1103 APU), W4A8 multicore, `dflash_body_native` bring-up
harness. Config: `--gemm multicore --cpu-primitives --ctx-cache --pipeline-glue
--attn flash`, l_ctx=32, B=16, tot=48, `r14_1x2x128_nb128`
(M_TILE=16 / N_TILE=64 / K_CHUNK=2048), flash kv_tile=48. Same-lock, warm-only
(block 0 cold excluded; block 1 is the non-cached verify; blocks >=2 are the
timed warm-cached cycles).

## Headline: the composed cached 9B wall (previously UNMEASURED)

| run | cold blk0 | non-cached blk1 | **cached warm (blk>=2)** | npu_busy warm |
|---|---|---|---|---|
| A | 290.8 ms | 102.4 ms | **82.0 ms** (82.0/83.2/83.4/83.9/85.5/85.6) | ~45 ms |
| B | 236.4 ms | 98.4 ms | **81.9 ms** (81.9/82.2/83.0/84.8/86.1) | ~47 ms |

**Composed cached 9B warm block wall ≈ 82 ms** (min 81.9, mean ~83.6, spread
~3% = the r135 noise floor). This is **below the ~100 ms batched-16 verify wall**
and below the earlier no-cache pipelined 97 ms. The 9B draft now fits its real
verify budget under overlap without any kernel change. ctx_misses=0. Context
cache bit-identical (max|Δ| K=V=0 all 5 layers, final cos=1.000000000).

cos vs golden = 0.899149 — the expected W4A8-by-construction value (documented
0.898399/0.897333); gate is acceptance rate, not cosine. No code changed, so no
regression is possible; this is a pure measurement.

## M_TILE=16 double-stream verdict: MOOT in the cached steady state

Per-op dispatch table, WARM cached blocks — **every** r14 GEMM runs at `rows16`:

| GEMM (op) | shape | rows | mean/dispatch |
|---|---|---|---|
| o proj | N4096_K4096 | **16** | 2.67 ms |
| gateup | N24576_K4096 | **16** | 0.39 ms |
| down | N4096_K12288 | **16** | 0.47 ms |
| qkv | N6144_K4096 | **16** | 0.60 ms |
| attention | flash q16_kvt48 | — | 1.48 ms |

rows16 == M_TILE == GRID·LM·MR (4·1·4). m_blocks = rows/M_TILE = 16/16 = **1**,
so each GEMM streams its weights **exactly once** — no double-stream. The only
GEMMs that run at 32 rows (`fc` N4096_K20480, per-layer `kv`, ceil(32/16)=2 =
double-stream) are the L-scaling context projections, and the ctx-cache removes
them from the per-cycle path entirely — they do not appear in the warm table.

**Conclusion: the M_TILE=16 double-stream penalty (the ~40 ms the brief flagged)
lives ONLY on the no-cache path (the 111.9/97 ms numbers). In the deployed cached
loop it is MOOT. Do NOT rebuild the kernel to chase it.** Confirmed both by
geometry (`gemm_r14.rs`: m_tile()=GRID·LM·MR=16; weight passes = rows/M_TILE) and
by the harness's own per-op instrumentation.

## The real remaining GEMM floor for the cached loop

Warm npu_busy ~46 ms is at the genuine int4 weight-bandwidth floor: per cached
cycle the four projections stream, per layer, qkv 12.6 + o 8.4 + gateup 50.3 +
down 25.2 = 96.5 MB int4, ×5 layers = **482 MB / cycle**; at ~10.4 GB/s ≈ 46 ms.
So the cached loop is already bandwidth-bound with a single weight pass. The real
levers left, in order:

1. **Fewer weight bytes** — lower-bit weights, gated on acceptance rate (Phase F:
   oq4.25+ costs 1.25% of τ). This scales the 46 ms floor directly.
2. **Route-concurrency ~1.25×** (r135 memory) — two orthogonal weight routes push
   the topology ceiling ~10.4 → ~13 GB/s.

M_TILE is not on this list. The ~36 ms of wall above npu_busy is host glue +
submit/serial overhead — task #39's target, not a GEMM-kernel concern.
