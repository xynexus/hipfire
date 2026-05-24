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

## Projection (NOT yet measured — 9B HF download pending)
9B+ is granularity-bound (4bit 0.41, 6bit 0.088 same rotation). g128+Lloyd attacks
exactly that → expect LARGER % drop than 0.8B's −32%. Lloyd alone gave −10% (9B) /
−6.5% (A3B); +g128 should stack. Plausibly: 9B 0.179→~0.10, A3B 0.103→~0.06.
Unproven; mq4g128 has NO runtime kernel — needs FWHT-128 dequant for prod eval.

## Cost (Bjorn's call): g128 ≈ +0.25 bpw (4.25→4.5) + Lloyd codebook lookup.
Both decode-hit per his prior sub-block measure. Mixed-block (g128/Lloyd only on
ffn_down/v_proj via kmap) recovers most KLD at fraction of perf. Python proxy ~2x
harsher than prod kldref. Built: quantize_mq4g128 (0e5d1482), HFQ4G128 kernels.
