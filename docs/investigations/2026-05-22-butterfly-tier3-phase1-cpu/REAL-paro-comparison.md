# Real PARO measured vs hipfire MQ4 — the goal reality check (2026-05-23)

The earlier RECIPE doc claimed MQ4-SLfwht+Lloyd "beats the PARO mechanism" in
Python KLD. That is a 6–15× harsher pseudo-quant proxy and does NOT translate.
Measured against REAL PARO + the production eval pipeline, hipfire's uniform MQ4
**loses** to PARO. This is another synth-win → prod-falsify (CLAUDE.md rule).

## Real PARO is fully characterized (no 70 GB download)

z-lab published official PARO for all 4 trunk models. `quantization_config` =
**`{quant_method: paroquant, bits: 4, group_size: 128, krot: 8}`**, uniform quant,
router/gates kept high-precision. Paper (2511.10645): PARO targets **+2.4% on
reasoning tasks vs AWQ**, optimized for reasoning accuracy, NOT KLD.

shisa-ai A3B PARO quality table (tx4, 129,921 tokens, KL nats vs BF16):

| A3B | bpw | PPL | KL nats | Top-1 |
|---|--:|--:|--:|--:|
| PARO full4096-e5 | 4.68 | 6.622 | **0.0347** | 92% |
| GGUF Q4_K_S | 4.78 | 6.578 | 0.0128 | 95% |
| GGUF Q4_K_M | 5.06 | 6.564 | 0.0108 | 95% |

PARO is **beaten by stock Q4_K** on KLD. It wins on reasoning accuracy (its axis).

## hipfire MQ4 measured (eval_hipfire, q8 KV, same metric/ctx)

- **A3B MQ4 = 0.1027 nats** (60 chunks). 9B MQ4 = 0.1788. Both ~3–5× real-PARO.
- hipfire MQ4 = uniform 4-bit **g256** + FIXED-FWHT. PARO = 4-bit **g128** +
  LEARNED krot8. Coarser groups + fixed rotation. MQ4 is Q4_0-class, not the
  Q4_K_M that beat PARO. learnable-FWHT (+8–19% Python) can't close ~3×.
- Confound: hipfire kldref slice ≠ shisa tx4; ~2× possible. Airtight verdict needs
  same-harness eval (eval_hipfire is .hfq-only; PARO is paroquant — blocked).

## Verdict

**Goal "MQ4-SLfwht ≥ PARO" not met, lever exhausted.** Gap-closer is g128/mixed
precision, not rotation. hipfire HAS g128 (HFQ4G128 quantizer+kernels, plain 4-bit)
— pivot is built, but outside the MQ4 scope. lm-eval (reasoning) is PARO's home
axis, untested. See [[project_real_paro_measured_vs_hipfire_mq4_2026_05_23]].
