# TODO: bf16/f16 quantize does not split 3D-stacked MoE experts

**Status:** open (pre-existing, surfaced 2026-07-20 by the qwen35 MoE OQ work).

## Symptom

The `tiny_quant` battery cells `qwen3_5_moe/kld:mq4` and `qwen3_5_moe/kld:mq3`
crash at load:

```
thread 'main' panicked at crates/hipfire-arch-qwen35/src/qwen35/loading.rs:1333:43:
tensor not found: layers.0.mlp.experts.0.gate_up_proj.weight
```

The failure is loading the **bf16 KLD reference**, not the mq4/mq3 candidate.

## Root cause

`hipfire-quantize` splits 3D-stacked routed experts
(`...mlp.experts.gate_up_proj` shape `[n_exp, N, K]`, no `.weight` suffix) into
per-expert 2D tensors `...mlp.experts.{X}.gate_up_proj.weight` (main.rs, the
`is_moe && ...experts... && shape.len()==3` split block). The loader
(`qwen35::loading::load_moe_ffn`) consumes those per-expert names.

For `--format mq4/mq6/oq4/...` the split runs and the model loads. For
`--format bf16` (and f16) the 3D expert tensor is caught by an **earlier
verbatim/passthrough branch** and emitted **stacked** (`experts.gate_up_proj`,
no per-expert index) → the per-expert loader can't find `experts.0...` → panic.

Verified: `hipfire-quantize --emit-fixture qwen3_5_moe` → `--format mq4` loads
fine (per-expert names); `--format bf16` keeps stacked names and fails to load.

## Attempted fix (insufficient)

Guarding the generic `if use_fp16 || use_bf16 { ...verbatim... continue }` block
(main.rs ~L8322) with `!is_stacked_moe_expert` did **not** fix it — the 3D
tensor is caught by a different earlier branch (the tensor has no `.weight`
suffix, so a `should_quantize()==false` verbatim path likely fires first). The
real fix needs to trace which branch catches the 3D expert under bf16 and route
it to the split block (which already emits BF16/F16 per expert, main.rs ~L8724),
or make the loader tolerate stacked experts.

## Impact / scope

- Blocks the `tiny_quant` gate for MoE archs (qwen3_5_moe) whenever MoE code is
  touched, because the battery uses a bf16 reference.
- Does NOT affect real serving: production MoE artifacts are mq4/mq6/oq4/oq8,
  all of which split correctly.
- The qwen35 MoE OQ commit (Opus Quant routed experts) was landed with
  `--no-verify` for this reason — the OQ work is GPU-validated on the real
  35B-A3B (oq4 + oq8 coherent) and is orthogonal to this bf16-ref bug.
