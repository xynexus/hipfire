# TODO: qwen4_exp prefill is per-token, and MoE routing syncs to the host per layer

**Status:** open (proposed 2026-08-30). PERFORMANCE only — correctness is gate-
covered. Recorded because both costs are structural rather than oversights, and
the obvious fix for one is the wrong one.

## Prefill replays the prompt one token at a time

`Qwen4ExpBackend::prefill` loops `decode_step_into` over the prompt. That is
O(prompt) launches of the whole trunk.

It is not simply an unfinished batching pass. Two of the four things a layer does
are SEQUENTIAL BY CONSTRUCTION:

* **Gated DeltaNet** carries a recurrent state `S` from token to token. Three of
  every four layers in this model are GDN.
* **The PLE short conv** carries a dilated tap ring across positions.

So a batched prefill cannot simply run B tokens through a layer. It would have to
batch the parts that can (QSA attention over the KV cache, the MoE FFN, the
hyper-connection mixers, lm_head) while keeping GDN and PLE sequential — or adopt
a chunked-scan formulation for GDN, which is what qwen3.5's chunked SSD prefill
does and is a substantial kernel in its own right.

Worth measuring before building: with 3 of 4 layers on the sequential path, the
achievable speedup from batching only the QSA/MoE layers is bounded well below
what "batched prefill" usually implies.

## MoE routing downloads to the host every layer, every token

`moe_forward` does GPU top-k, then:

```rust
let idx: Vec<i32> = gpu.download_f32(&s.topk_idx)? ...
for (slot, &e) in idx.iter().enumerate() { /* one GEMV per selected expert */ }
```

That is a **GPU→host sync per MoE layer per token** — 48 per token on the shipped
model — and it serialises the pipeline at each one.

The obvious fix is wrong. The download exists because the host loops over the
SELECTED experts, so removing it means the expert dispatch has to happen on the
GPU: the `*_indexed*` routed-expert kernels, which take the top-k table as a
device tensor and never round-trip. That is the same work described in
`2026-08-30-qwen4exp-native-quantised-weights.md`, and it needs `INDEXED_MOE_K_TOP`
(currently 8) generalised for this model's top-10, plus the two selection kernels
that hardcode it.

So the two performance items collapse into one piece of work, and neither is on
the correctness path: the model serves, fits, and is gate-verified with both costs
present.

## What NOT to do

Do not "optimise" the sync by hoisting routing out of the layer loop. Each layer's
routing depends on that layer's input, which depends on the previous layer's
output. It is inherently sequential across layers; only the host round-trip is
removable, and only by dispatching experts on-device.
