# Gemma 4 Phase 3 fused projections (mirrored from qwen35)

- Branch: `gemma4`
- Date: 2026-05-04
- Hardware: 7900 XTX (gfx1100)

## What changed

Mirrored qwen35's `fused_qkv_hfq4g256` and `fused_gate_up_hfq4g256` into
Gemma 4's sliding + full attention preambles and the dense MLP.

Three fused-launch points per layer (when MQ4G256-byte-compat weights):

1. `fused_rmsnorm_rotate_mq` replaces `rmsnorm_f32` immediately before
   any multi-projection — fuses the input norm with the FWHT rotation
   the fused-projection kernels expect for MQ4 weights.
2. `fused_qkv_hfq4g256` replaces 3 separate `weight_gemv` calls (q + k + v)
   in the attention preamble — only fires when `owns_kv && v_proj.is_some()`
   (no fused-qk variant exists today for the `attention_k_eq_v=true` case).
3. `fused_gate_up_hfq4g256` replaces 2 separate `weight_gemv` calls
   (gate + up) in every dense MLP block.

Per-layer launches saved when all conditions hit:
- attn block: 1 norm + 3 GEMVs (4 launches) → 1 rmsnorm-rotate + 1 fused-qkv (2) = save 2
- mlp block: 1 norm + 2 GEMVs (3 launches) → 1 rmsnorm-rotate + 1 fused-gate-up (2) = save 1
- per layer: save 3 launches × 30 layers = 90 launches/step

Correctness gate: `prep_norm_for_proj` only pre-rotates when EVERY
downstream projection will use a fused (rotation-free) kernel. This
prevents double-rotation (fused-rotate output × `gemv_mq4g256_with_rotate`
internal rotate = corrupted output) when only some projections go
through fused.

## Coherence

`coherence-gemma4.sh` — 6/6 strict pass, **bit-identical** to Phase 2.5
baseline (`coherence-phase25-output-20260504.md`) on all 6 tests
including 26B-A4B. The fused kernels read the same byte-layout
weights and the rotation+norm are mathematically equivalent.

## Bench (creative prompt, max_tokens=300)

Clean apples-to-apples (force per-projection via `HIPFIRE_GEMMA4_FUSED_PROJ=0`):

| Path | Decode | Prefill | TTFT |
|---|---|---|---|
| Phase 2.5 (no fused proj) | 72.2 | 81.9 | 598 ms |
| Phase 3 (fused proj, default) | 77.6 | 86.2 | 569 ms |

Phase 2.5 → Phase 3: **+7.4% decode, +5.3% prefill, –4.9% TTFT**.

## Cumulative Phase 0 → Phase 3 on 26B-A4B

| Metric | Phase 0 | Phase 3 | Delta |
|---|---|---|---|
| Decode tok/s | 60.8 | 77.6 | **+27.5%** |
| Prefill tok/s | 65.2 | 86.2 | **+32%** |
| TTFT (ms) | 752 | 569 | **–24%** |

## Safety hatch

`HIPFIRE_GEMMA4_FUSED_PROJ=0` forces both `fused_qkv` and `fused_gate_up`
back to per-projection `weight_gemv` (and disables the fused-rmsnorm-
rotate). Same env knob covers all four sites for a single A/B switch.

## Next levers (still on the table)

1. **Batched prefill** — Gemma 4's `forward_prefill_batch` is still a
   stub. Largest remaining win, but a Phase 4-scale refactor.
2. **rmsnorm_f32 of post_attention_layernorm** — not fused (it's
   in-place on tmp before the residual add, no projection follows).
3. **rmsnorm_batched fusion** — q_norm/k_norm/v_norm are 3 separate
   batched-norm calls; could potentially be merged.
4. **fused_qk-only kernel** — would let `attention_k_eq_v=true` (31B,
   26B-A4B) use a 2-row fused projection on full layers (currently
   falls through to per-projection on those).
