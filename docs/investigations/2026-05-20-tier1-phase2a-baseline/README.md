# Tier 1 Phase 2A — Baseline NRMSE on Qwen3.5-0.8B (2026-05-20)

## Setup

- **Oracle**: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf` (1.1 MB, llama-imatrix canonical, llama.cpp GGUF naming)
- **Target**: Tier 1 v1 output from `pr-7-tier1-calibration` HEAD `e4b56cdd` on `/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/`
- **Calibration**: 32 sequences × 2048 ctx = 65,536 tokens, wikitext2-1024s slice
- **Wall**: 25.3s on MI300x (~0.78s/seq — pleasant for the iteration loop)

## Coverage

- 186 / 186 tensors match (after HF→GGUF name translation)
- All 24 layers covered (alternating linear_attn + full_attn per Qwen3.5-VL config)

## Result

**HARD FAIL** — median NRMSE = 0.9999, max = 1.0000 across every role.

| role | n | median NRMSE | max NRMSE |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9999 | 1.0000 |
| attn_k | 6 | 0.9999 | 1.0000 |
| attn_output (FullAttn o_proj) | 6 | 0.9999 | 1.0000 |
| attn_q | 6 | 0.9999 | 1.0000 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9999 | 1.0000 |
| attn_v | 6 | 0.9999 | 1.0000 |
| ffn_down (MLP) | 24 | 1.0000 | 1.0000 |
| ffn_gate (MLP) | 24 | 0.9999 | 1.0000 |
| ffn_up (MLP) | 24 | 0.9999 | 1.0000 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9999 | 1.0000 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9999 | 1.0000 |
| ssm_out (DeltaNet out_proj) | 18 | 0.9979 | 1.0000 |

## Diagnosis

The plan predicted q/k/v/in_proj/gate/up would be near-zero NRMSE
(they don't see attention output, only direct embed→linear). They're
NOT — they're at 1.0 NRMSE like the attention-dependent o_proj.

Two universal causes (apply to every captured tensor):

**1. RMSNorm skipped in Tier 1 BF16 forward (B.1 lever, expected)**

`crates/hipfire-runtime/src/bf16_forward.rs:17-22` documents the skip.
Tier 1 feeds RAW post-residual hidden state into every linear.
Tier 2 normalizes before each linear via the model's real rmsnorm.

Per-channel magnitudes diverge by O(√dim) — for hidden_dim=1024 in
the 0.8B trunk that's ~32× systematic scale mismatch on every linear's
in_sum2. After /counts normalization the ratio holds, so NRMSE pegs
at ~1.0.

**2. Tokenizer mismatch (subagent E's choice → universal stream divergence)**

`crates/hipfire-runtime/src/calibration.rs::tokenize_corpus` uses
hipfire's BPE encoder (Tokenizer::from_hf_json). Tier 2's
`llama-imatrix` uses llama.cpp's BPE.

Per `CLAUDE.md` + `benchmarks/quality-baselines/harness/tokenizer_parity.py`,
the two encoders disagree on ~46% of token positions for Qwen3 family.
Even with rmsnorm fixed, the token streams are different so per-channel
activation statistics will never byte-match.

## Strategic implications for Phase 2B

**Per-tensor NRMSE may be the wrong metric** for Tier 1 vs Tier 2
parity. The right correctness check is downstream:

1. Run `hipfire-quantize --imatrix <tier1>` → MQ4 model A
2. Run `hipfire-quantize --imatrix <tier2>` → MQ4 model B
3. Eval both against the canonical kldref. If their KLDs are within
   ≤ 0.02, Tier 1 is fit-for-purpose regardless of per-tensor NRMSE.

**Revised Phase 2B priorities:**

1. **B.1 (rmsnorm) FIRST** — fixes the magnitude-scale universal mismatch. After this, NRMSE on q/k/v/in_proj/gate/up should drop to a much lower floor representing ONLY the tokenizer-divergence noise.
2. **Pivot the metric** to downstream KLD. Per-tensor NRMSE retained as a diagnostic, not a gate.
3. **Tokenizer-parity option** (deferred to optional): swap Tier 1 to use llama.cpp's tokenizer via subprocess for STRICT byte-parity oracle comparison. Already noted as a single-function swap in `calibration.rs`. Probably not worth doing if downstream KLD gate passes.

## Files

- Comparator: `scripts/compare_imatrix.py`
- Baseline JSON: `/tmp/phase2a-baseline.json` (full per-tensor table, mirrored as `baseline.json` in this dir)
- Baseline MD: `/tmp/phase2a-baseline.md`
