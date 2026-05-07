---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
  - hipfire
  - amd
  - rdna
  - quantized
  - qwen3.6
  - moe
  - mixture-of-experts
  - agentic
  - coding
library_name: hipfire
---

# Qwen3.6-35B-A3B for hipfire

Pre-quantized **Qwen3.6-35B-A3B** (MoE, 35B total / 3B activated) for
[hipfire](https://github.com/Kaden-Schutt/hipfire), a Rust-native LLM
inference engine for AMD RDNA GPUs.

Quantized from [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
Qwen3.6's April 2026 refresh of the A3B line, with a coding/agentic
fine-tune recipe. Architecture is unchanged from Qwen3.5-35B-A3B —
256 experts top-8, hybrid DeltaNet + Full Attention (3:1 ratio), head_dim=256
with partial_rotary_factor=0.25, shared expert, tied embeddings — so
hipfire's `arch_id=6` path loads it without any engine changes.

## ⚠️ 2026-05-07 release — Q8 router fix

This release **replaces the prior `.mq4`** with a re-quantized version that
fixes [issue #171](https://github.com/Kaden-Schutt/hipfire/issues/171) — a
structural attractor on agentic prompts when the MoE router was at 4-bit.
The contributor [@fivetide](https://github.com/fivetide)'s
[PR #180](https://github.com/Kaden-Schutt/hipfire/pull/180) promotes
`mlp.gate.weight` and `mlp.shared_expert_gate.weight` to Q8F16, costing
~10 MB additional model size. Empirical recovery on the 3.6-A3B
code-review reproducer:

| variant | unique-word ratio | verdict |
|---|---|---|
| MQ4 4-bit router (pre-fix, deprecated) | 14% | ATTRACTOR |
| **MQ4 + Q8 router (this release)** | **46%** | **CLEAN** |
| HFQ6 reference | 70% | CLEAN |

The 3.6-A3B family was the model class most exposed to the cliff (see
[issue #171](https://github.com/Kaden-Schutt/hipfire/issues/171) and the
investigation log at
[docs/investigations/2026-05-06-moe-quant-cliff-survey](https://github.com/Kaden-Schutt/hipfire/tree/master/docs/investigations/2026-05-06-moe-quant-cliff-survey));
3.5-A3B was less visibly affected but is also re-quantized for parity.

If you previously downloaded `qwen3.6-35b-a3b.mq4`, **re-pull it** to pick
up the fix. The `.hermes.triattn.bin` sidecar from the prior release is
calibrated against the broken-router weights and is currently
**deprecated** — re-calibration on the new `.mq4` is in flight.

## Files

| File | Quant | Size | Min VRAM | RX 7900 XTX decode | Status |
|------|-------|------|----------|--------------------|--------|
| **qwen3.6-35b-a3b.mq4** ⭐ | MQ4 + Q8 router | 19 GB | 22 GB | ~148 tok/s | **2026-05-07 fixed release** |
| qwen3.6-35b-a3b.mq3 | MQ3 + Q8 router | 19 GB | 22 GB | TBD | Smaller-bit variant for memory-constrained hosts |

⭐ MQ4 is FWHT-rotated 4-bit with the routing tensors (`mlp.gate.weight`,
`mlp.shared_expert_gate.weight`) pinned at Q8F16. Quality-gated against
the Q8 reference on the hipfire coherence battery.

## Usage

```bash
# Install hipfire (master, includes the router-Q8 fix)
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash

# Pull the model (defaults to MQ4)
hipfire pull qwen3.6:35b-a3b

hipfire run qwen3.6:35b-a3b "Write a Rust function that parses an ISO-8601 date."
```

To pull the MQ3 variant explicitly:

```bash
hf download schuttdev/hipfire-qwen3.6-35b-a3b qwen3.6-35b-a3b.mq3 \
    --local-dir ~/.hipfire/models
```

## Configuration notes

- **`thinking:off` recommended** — Qwen3.6-A3B is a heavy thinker and
  default thinking-mode prompts produce long reasoning chains that can
  loop on complex tasks. For production-style usage:
  ```
  hipfire config qwen3.6:35b-a3b set thinking off
  ```
- **Default `dflash_mode: auto`** — the engine keeps DFlash speculative
  decoding off for A3B unless a `cask_sidecar` is configured, because A3B
  drafts reject most tokens (τ≈1.0–1.5 on non-math), and the cycle overhead
  outweighs the AR win.
- **Greedy + `RP=1.05`** (the project default) is the safest sampler for
  this model. See [issue #171 update](https://github.com/Kaden-Schutt/hipfire/issues/171)
  for the empirical 7-prompt × 5-sampler matrix that landed on this default.

## Quantization format

- **MQ4 (MagnumQuant-4)** — FWHT-rotated 4-bit with asym3 KV cache default.
  Routing tensors at Q8F16. Matches Q8 output quality at ~Q4 bandwidth on
  hipfire's WMMA/dot2 fused kernel paths.
- **MQ3 (MagnumQuant-3)** — same FWHT-rotated approach at 3-bit for the
  bulk weights, Q8F16 for routing/embed/lm_head. Useful when MQ4 doesn't
  fit on the target host.

See [docs/QUANTIZATION.md](https://github.com/Kaden-Schutt/hipfire/blob/master/docs/QUANTIZATION.md)
for details on the rotation invariance property and the quality gate.

## License

Apache 2.0, following the upstream Qwen/Qwen3.6-35B-A3B license.
