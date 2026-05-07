---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
tags:
  - hipfire
  - amd
  - rdna
  - quantized
  - qwen3.5
  - moe
  - mixture-of-experts
  - agentic
  - coding
library_name: hipfire
---

# Qwen3.5-35B-A3B for hipfire

Pre-quantized **Qwen3.5-35B-A3B** (MoE, 35B total / 3B activated) for
[hipfire](https://github.com/Kaden-Schutt/hipfire), a Rust-native LLM
inference engine for AMD RDNA GPUs.

Quantized from [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
Original A3B release of the Qwen3.5 line. Hybrid attention layout:
256 experts top-8, DeltaNet (linear) + Full Attention layers in a 3:1
ratio, head_dim=256 with partial_rotary_factor=0.25, shared expert,
tied embeddings. Loaded by hipfire's `arch_id=6` MoE forward path.

## 2026-05-07 Q8-router release

This is the first hipfire release of `qwen3.5-35b-a3b.mq4` to land
[@fivetide](https://github.com/fivetide)'s
[PR #180](https://github.com/Kaden-Schutt/hipfire/pull/180) — the MoE
router (`mlp.gate.weight` and `mlp.shared_expert_gate.weight`) is now
quantized at Q8F16 instead of MQ4, costing ~10 MB additional model size.
The rationale and empirical evidence are documented at
[issue #171](https://github.com/Kaden-Schutt/hipfire/issues/171) and the
investigation log at
[docs/investigations/2026-05-06-moe-quant-cliff-survey](https://github.com/Kaden-Schutt/hipfire/tree/master/docs/investigations/2026-05-06-moe-quant-cliff-survey).

3.5-A3B was less affected by the broken-router cliff than its sibling
3.6-A3B (3.5 was clean at greedy + RP=1.05 even with 4-bit router; 3.6
was the canary that exposed the cliff on agentic prompts). 3.5-A3B is
still re-quantized for parity and to keep both A3Bs on a uniform format.

## Files

| File | Quant | Size | Min VRAM | RX 7900 XTX decode |
|------|-------|------|----------|--------------------|
| **qwen3.5-35b-a3b.mq4** ⭐ | MQ4 + Q8 router | 19 GB | 22 GB | ~148 tok/s |
| qwen3.5-35b-a3b.mq3 | MQ3 + Q8 router | 19 GB | 22 GB | TBD |

⭐ MQ4 is FWHT-rotated 4-bit with the routing tensors (`mlp.gate.weight`,
`mlp.shared_expert_gate.weight`) pinned at Q8F16. Quality-gated against
the Q8 reference on the hipfire coherence battery.

## Usage

```bash
# Install hipfire (master, includes the router-Q8 fix)
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash

# Pull the model (defaults to MQ4)
hipfire pull qwen3.5:35b-a3b

hipfire run qwen3.5:35b-a3b "Write a Rust function that parses an ISO-8601 date."
```

To pull the MQ3 variant explicitly:

```bash
hf download schuttdev/hipfire-qwen3.5-35b-a3b qwen3.5-35b-a3b.mq3 \
    --local-dir ~/.hipfire/models
```

## Configuration notes

- **Greedy + `RP=1.05`** is the default sampler and is robust on this
  model across the reference 7-prompt × 5-sampler matrix
  (see [issue #171 update](https://github.com/Kaden-Schutt/hipfire/issues/171)).
  The HF-aligned `temp=1.0 + top_k=20 + min_p=0.05` sampler is opt-in
  per request; greedy default delivers the cleanest output.
- **`thinking:auto`** — 3.5-A3B's thinking mode is healthy at MQ4.
- **DFlash speculative decoding off by default for A3B** — drafts reject
  most tokens (τ≈1.0–1.5 on non-math), so AR alone is faster unless a
  CASK sidecar is configured for the eviction-required long-context path.

## Quantization format

- **MQ4 (MagnumQuant-4)** — FWHT-rotated 4-bit with asym3 KV cache default.
  Routing tensors at Q8F16. Matches Q8 output quality at ~Q4 bandwidth on
  hipfire's WMMA/dot2 fused kernel paths.
- **MQ3 (MagnumQuant-3)** — same FWHT-rotated approach at 3-bit for the
  bulk weights. Useful when MQ4 doesn't fit on the target host.

See [docs/QUANTIZATION.md](https://github.com/Kaden-Schutt/hipfire/blob/master/docs/QUANTIZATION.md)
for details on the rotation invariance property and the quality gate.

## License

Apache 2.0, following the upstream Qwen/Qwen3.5-35B-A3B license.
