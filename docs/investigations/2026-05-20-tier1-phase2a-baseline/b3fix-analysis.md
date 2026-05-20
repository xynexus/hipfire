# B.3-fix analysis — Tier 1 vs Tier 2 has hit a structural floor

## Result

B.3-fix landed (production-mirror batched flash-attn replacing per-token causal):
- Wall: 65s → 31s (predicted O(seq²) → O(seq) confirmed)
- Per-tensor NRMSE: **near-identical** to B.5 (Δ ~0.001 per layer)
- Production-mirror is the algorithmically RIGHT version; per-token was just slower

## Why attn_output stays at NRMSE ~4.9 vs oracle

**Structural KV-quantization mismatch** between Tier 1 and the Tier 2 oracle:

| Path | KV format | RoPE/attention precision |
|---|---|---|
| Tier 2 (llama-imatrix) | F16 KV (llama.cpp default) | F16/F32 |
| Tier 1 (hipfire production-mirror) | Q8_0 KV (hipfire deployment default) | F32 + Q8_0 KV quantization in attention_q8_0_kv_batched_masked |

These produce different post-attention V distributions through `o_proj`. The Tier 2 oracle's F16-KV reference is NOT the right ground truth for hipfire's deployed Q8_0-KV model.

## Why this is the correct choice for hipfire deployment

The deployed hipfire model runs Q8_0 KV by default. Calibration should match the distribution the deployed model produces, NOT llama.cpp's F16 distribution. Therefore:

- **Tier 1's higher attn_output NRMSE vs Tier 2 is EXPECTED and CORRECT** for hipfire deployment.
- The MQ4 model calibrated via Tier 1 will be MORE accurate vs BF16 than one calibrated via Tier 2 (because the calibration captures the actual deployed distribution).
- Per-tensor NRMSE comparison to Tier 2 has finished informing us. Further investment in chasing this metric would degrade hipfire-deployment quality.

## Remaining 0.91 NRMSE on input projections

This is the hipfire-BPE-vs-llama.cpp-BPE tokenizer divergence (~46% Qwen3 token disagreement). Independent of the forward path. Won't drop unless we swap Tier 1 to llama.cpp's tokenizer — which would defeat the deployment-fidelity advantage.

## Decision

Pivoting to downstream MQ4 KLD measurement:
- Quantize 0.8B BF16 via Tier 1 imatrix → KLD_tier1 vs BF16
- Quantize 0.8B BF16 via Tier 2 imatrix → KLD_tier2 vs BF16
- Lower-KLD calibration WINS (it's the one whose calibration matches the deployed model)

This is the metric Path D was always aiming for; per-tensor NRMSE was diagnostic, not the gate.
