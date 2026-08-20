# qwen35 hand decode path fixed: the dense DeltaNet arm never applied `ffn_norm`

**Status:** fixed. Hand and lowered now agree numerically.
**Measured on:** nix1 / gfx1103, release daemon, `Qwen3.5-0.8B-Base--oq8`,
`qwen3.5-0.8b--oq4++`, `Qwen3.6-35B-A3B--oq4`.

## Root cause

One missing block. In `forward_scratch_layers`'s
`(LayerWeights::DeltaNet, LayerType::LinearAttention)` arm, the FFN section read
`s.tmp` / `x_rot` **without ever computing the FFN norm**. Those bindings still
held the output of the *attention* prepare at the top of the arm:

```rust
let x_rot = fused_rmsnorm_prepare_bases(
    gpu, &[&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha],
    &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps)?;   // attention
...
weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;         // s.x += attn out
...
gpu.fused_gate_up_*(..., &s.tmp /* or x_rot */, ...)               // FFN reads STALE s.tmp
```

So the dense FFN consumed `rmsnorm(x_pre_attention, attn_norm)` — **two errors at
once**: the wrong norm weights (`attn_norm` instead of `ffn_norm`), applied to the
wrong residual (pre-attention, missing the `wo` output added just above). Tracing
`s.tmp` across the arm confirms it: written once by the attention prepare, then
only ever read until the FFN consumes it.

`ffn_norm` was in fact applied on exactly one sub-branch — the PARO
`fused_rmsnorm_rotate_for_paro` path, which passes `&layer.ffn_norm` explicitly.
That is why the breakage looked dtype-dependent rather than structural.

## Why it presented as "dense broken, MoE fine"

The sibling arms all got this right, and only this one didn't:

| arm | FFN norm | status |
|---|---|---|
| `DeltaNet` (dense) | **absent** (except PARO branch) | **broken** |
| `FullAttn` (dense) | `fused_rmsnorm_prepare_bases` with `ffn_norm` | fine |
| `DeltaNetMoe` | `fused_rmsnorm_rotate_mq` / `rmsnorm_f32` with `ffn_norm` | fine |
| `FullAttnMoe` | same as `DeltaNetMoe` | fine |

`docs/experiments/2026-08-20-dense-opus-dflash-miscompute.md` had localized the
fault to "the hand path's DENSE arms" and noted MoE was unaffected. The real split
is narrower: it is the `DeltaNet`-dense arm specifically. A hybrid model is
destroyed by it because most layers are `LinearAttention` — which is also why a
dense hybrid looked catastrophic while `Qwen3.5-35B-A3B` (MoE, so its DeltaNet
layers are `DeltaNetMoe`) stayed coherent. That asymmetry was the whole clue.

## The fix

Insert the same block `FullAttn` already had, before the PARO branch so the
shadowing works identically:

```rust
let x_rot = fused_rmsnorm_prepare_bases(
    gpu, &[&layer.w_gate, &layer.w_up],
    &s.x, &layer.ffn_norm, &s.tmp, &s.x_rot, config.norm_eps)?;
```

It shadows the attention `x_rot` for the rest of the arm, which is what the two
downstream consumers want: the fused `gate_up` and the GDN-tape `ffn_input`
capture (`x_rot_paro.or(x_rot).unwrap_or(&s.tmp)`). The earlier tape capture at
`x_in_for_tape` sits above the insertion and still sees the attention binding,
correctly.

25 lines added, none removed.

## Evidence

Self-KLD, reference built on lowered and scored on hand, same weights — the same
metric that recorded the original **13.89**:

```
mean_kld = 5.26e-10   p99 = 6.01e-10   total_scored = 2044   (ppl 12.46)
```

Byte-identical greedy output, hand vs lowered:

| model | arms exercised | result |
|---|---|---|
| `Qwen3.5-0.8B-Base--oq8` | DeltaNet + FullAttn dense | 204 tok over 4 prompts, identical |
| `qwen3.5-0.8b--oq4++` | DeltaNet + FullAttn dense | identical |
| `Qwen3.6-35B-A3B--oq4` | MoE arms (untouched) | identical |

Before the fix, the same two dense artifacts produced
`'  0\n;\n    0\n; ( 0)s0;0\n    0;\n'` and
`' How to the answer is the 1.\n\n with a 格式 10 10: 10'`.

## Why no gate caught it

The tiny gates exercise only the default (lowered) path. Running the full
`qwen3_5*` family set before and after this change produces **byte-identical
state hashes** — the fix is invisible to them. That is precisely how the hand path
rotted unobserved, and it is worth a gate cell that pins `HIPFIRE_FORWARD_LOWERED=0`
against the lowered path for one dense hybrid model. Not added here.

(The 2 quant + 3 state cells that fail on this branch fail identically on clean
`origin/master`, with the same numbers — unrelated and pre-existing.)

## Consequences for decisions made on the old assumption

Two records deferred work *because* the hand path was believed unfixable:

- `docs/roughquant/phase3-real-format-scope.md` — "#10c is DEFERRED", on the basis
  that "'fix the hand path' = resurrect dead code, and the only working forward is
  the lowered super-op executor". The premise no longer holds; the hand path was
  one missing block, not dead code. RoughQuant's mechanism numbers (KLD
  0.598 → 0.571) were measured against the broken forward and should be re-taken.
- `docs/plans/2026-08-20-p2-steer-per-stream-and-lowered.md` — M2's exit was
  rewritten away from hand/lowered parity because the reference was broken. With
  this fix, comparing against the hand path becomes meaningful again, though the
  `strength = 0.0` identity anchor is a better assertion regardless and should
  stay.
