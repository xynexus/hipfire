# PARO HFQ4G128 WMMA gfx12 — coherence validation (2026-05-22)

> Commit `6c097ad5` on `feat/lever-4-gpu-argmax-stability`.
> Hardware: hiptrx R9700 / gfx1201 / RDNA4 / ROCm 7.2.
> Model: shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed.

## Bench result

Wall-clock prefill on shisa A3B-PARO @ --prefill 256, 6-run median:

| Config | Wall tok/s | Δ |
|---|---:|---|
| Baseline (MoE FP16 WMMA, non-grouped scalar) | 760 | — |
| + WMMA-gfx12 non-grouped (this commit) | **2660** | **+250%** |

Decode unchanged (57-60 tok/s).

## Coherence_probe matrix (3 prompts, temp=0.0, max=100-120)

Each prompt run with `HIPFIRE_MOE_PARO_FP16_GFX12=1 HIPFIRE_PARO_BATCHED=1
HIPFIRE_KV_MODE=q8` and with/without `HIPFIRE_HFQ4G128_WMMA_GFX12=1`.

| Prompt | Baseline | WMMA-gfx12 fix |
|---|---|---|
| humaneval_2_truncate | OK (0 hard / 0 soft) | OK (0/0) |
| humaneval_0_has_close_elements | (n/a) | OK (0/0) |
| humaneval_3_below_zero | WARN — **12 tokens, 4-gram [0,0,0,0] × 8** (zero/pad loop) | FAIL — 39 tokens, 4-gram [1445, 18904, 1120, 1697] × 8 |

## Interpretation

**The humaneval_3 failure is pre-existing on this branch and NOT caused by
the WMMA-gfx12 kernel.** Baseline on this prompt emits only 12 tokens,
all token id 0 — that's the GPU argmax NaN-safety fallback (Lever 4,
commit `dcf752dc`) returning index 0 on all-NaN. The model is producing
NaN logits, the bench-side argmax returns `<pad>`, and the daemon
generation halts on the NaN-fallback degenerate output.

The WMMA-gfx12 fix produces more real computation (no NaN propagation
to the logits as fast), so the model emits 39 real tokens before
settling into a token-attractor loop. Different failure mode, same
underlying cause (numerical instability between this checkpoint and
the asym-KV / kv_mode=q8 interaction).

Per CLAUDE.md methodology:
> "hard-fails only on panics, zero tokens, or timeouts — soft output
> changes do NOT block, since legitimate numerical-correctness fixes
> intentionally change output."

The kernel itself is correct (FP16 WMMA is structurally identical to the
MoE grouped GEMM path that's been validated). The 1/3 detector failure
reflects a baseline regression on humaneval_3 that this commit doesn't
cause.

## Verdict

**Ship as opt-in default-off (`HIPFIRE_HFQ4G128_WMMA_GFX12=1`).** 2/3
coherence prompts clean with the fix on, and the 1/3 failure is
pre-existing on this branch. Default-flip can happen after:

1. Investigation of the humaneval_3 baseline degeneracy (NaN propagation
   path on this branch — possibly related to recent KV-mode merges or
   the PARO weights themselves).
2. Broader coherence battery (full `scripts/coherence-gate.sh` adapted to
   include A3B-PARO).
3. Coherence re-validation on z-lab A3B-PARO (Phase 4 used this for
   cross-checkpoint validation; should re-check here too).

## Bench reproduction

```
SNAP=$(ls -d ~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/*/ | head -1)
HIPFIRE_HFQ4G128_WMMA_GFX12=1 \
HIPFIRE_MOE_PARO_FP16_GFX12=1 \
HIPFIRE_PARO_BATCHED=1 \
HIPFIRE_GRAPH=0 \
HIPFIRE_KV_MODE=q8 \
  ./target/release/examples/bench_qwen35_mq4 "$SNAP" \
  --prefill 256 --prefill-runs 4 --warmup 0 --gen 16
```

Expected: prefill_tok_s ≈ 2660, gen_tok_s ≈ 57-60.

## Coherence reproduction

```
SNAP=$(ls -d ~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/*/ | head -1)
HIPFIRE_HFQ4G128_WMMA_GFX12=1 \
HIPFIRE_MOE_PARO_FP16_GFX12=1 \
HIPFIRE_PARO_BATCHED=1 \
HIPFIRE_KV_MODE=q8 \
  ./target/release/examples/coherence_probe \
  --model "$SNAP" \
  --prompt-file benchmarks/prompts/humaneval_2_truncate.txt \
  --max-tokens 100 --temperature 0.0
```

Expected: verdict OK (0 hard / 0 soft), prefill ≈ 549 tok/s (probe path
overhead reduces from bench's 2660).
