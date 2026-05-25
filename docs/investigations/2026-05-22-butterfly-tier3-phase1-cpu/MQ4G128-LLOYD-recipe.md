# MQ4G128+Lloyd — PARO-class flat-MQ4 recipe (0.8B preliminary, 2026-05-24)

**Recipe:** uniform 4-bit, **group-128**, **Lloyd-Max codebook**, FWHT rotation,
AWQ α0.55. No learned signs.

⚠️ **IMPLEMENTATION STATUS: proxy-validated only, NOT shippable.** Numbers are
the Python pseudo-quant trainer (`--group-size 128 --quant-mode lloyd`). Rust has
`quantize_mq4g128` (UNIFORM, no Lloyd, no kernel). g128+Lloyd needs: (1) a
mq4g128-Lloyd quantizer (combine 128-pt FWHT + per-128 Lloyd codebook), (2) an
FWHT-128 + codebook-lookup dequant GEMV kernel. Neither exists. Production kldref
unmeasured. Treat as a lever-direction, not a deliverable.

## 0.8B ladder (Python pseudo-quant KLD, theta=0, 32-seq, AWQ α0.55)

| recipe | KLD nats | Δ |
|---|---:|---:|
| MQ4 g256 uniform (current ship) | 0.0567 | — |
| g128 uniform | 0.0492 | −13% |
| g128 + signs | 0.0479 | −16% |
| **g128 + Lloyd** | **0.0383** | **−32%** |
| g128 + Lloyd + signs (trained) | 0.0386 | signs FAIL |

Real PARO 0.8B ≈ 0.035 KLD-class. **g128+Lloyd ≈ parity, no training.**

## UPDATE: g256-layout + g64 subscales + Lloyd = BEST (Kaden's helical idea)
| g256 uniform | g128+Lloyd | g256/4×g64-subscale+Lloyd | g256/2×g128-sub+Lloyd |
|---:|---:|---:|---:|
| 0.0567 | 0.0383 | **0.0268** | 0.0369 |
Packed g64-in-g256 (4 sub-scales/block + per-sub Lloyd) = 0.0268 vs g256-uniform
0.0567 (−53%), all in the SAME Python proxy. CAVEATS: proxy-only, 0.8B (easy,
outlier-bound), no kernel exists, no production eval; 0.035 PARO is a different
real-model/harness number — NOT directly comparable, do not claim "beats PARO".
g256 occupancy is hypothesized (4-way subscale unpack, unbuilt — not measured).
Promising lever-direction. Sim: --group-size 256 --quant-mode lloyd --subscale-size 64.

## What works, what doesn't
- **g128**: finer scaling, −13% (1 extra bit of scale precision). Targets granularity.
- **Lloyd**: non-uniform 16-level codebook → the big lever (−32% combined). Places
  levels where mass sits; PARO/uniform can't.
- **Learned signs**: SATURATED once g128+Lloyd tighten range — they overlap (both
  range-fixers), signs go FAIL. 0.8B-only, drop them.
- AWQ outlier lever taps out on big models; g128+Lloyd attack granularity instead.

## 9B MEASURED (beat the projection): proxy uniform 0.354 → subscale64+Lloyd 0.036
Granularity-bound, so the lever drops harder than 0.8B (−90% vs −53%), beating the
~0.10 projection. NOTE units: 0.179 was PRODUCTION kldref uniform; 0.354 is the
PROXY uniform — all subscale numbers here are PROXY. A3B still unmeasured; mq4g128
+ subscale have NO runtime kernel — needs FWHT + 4×g64-min/scale + Lloyd-lookup
GEMV for prod eval.

## Cost (Bjorn's call): g128 ≈ +0.25 bpw (4.25→4.5) + Lloyd codebook lookup.
Both decode-hit per his prior sub-block measure. Mixed-block (g128/Lloyd only on
ffn_down/v_proj via kmap) recovers most KLD at fraction of perf. Python proxy ~2x
harsher than prod kldref. Built: quantize_mq4g128 (0e5d1482), HFQ4G128 kernels.

## 9B scaling (Python proxy, confirms hypothesis)
uniform 0.354 → g256+Lloyd 0.054 (−85%) → +g64-subscale 0.036 (−90%). Scales
STEEPER than 0.8B (granularity-bound model, Lloyd is the lever). PARO-class in
proxy. CAVEATS unchanged: proxy/0.8B-9B/no-kernel/no-prod-kldref. Recipe: g256
layout + 4×g64 subscale + Lloyd. A3B + production kernel = next.
## A3B-35B (bf16, proxy): g256+Lloyd 0.0193 → +g64-subscale 0.0149 (−23%). MoE most
compressible. NOT comparable to PARO 0.0347 (diff harness/slice, pseudo-quant, no kernel
/prod-eval). Subscale helps all trunk in-proxy: 0.8B 0.027 / 9B 0.036 / A3B 0.0149.
