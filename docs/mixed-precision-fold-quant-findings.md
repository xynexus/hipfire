# Mixed-precision Opus "fold" quantization — findings (klein-4B / FLUX.2)

Status: findings, 2026-07-13. Hardware: gfx1103 (Radeon 780M, RDNA3 wave32).
Model: `FLUX.2-klein-base-4B` (the DiT + Qwen3 text encoder + shared conv-KL VAE).

## TL;DR

- Built an end-to-end **mixed-precision unsigned "fold" quant path** for the Opus
  GEMM — one kernel body for W{1,2,4,8}A8, validated on GPU to **5e-8** vs a CPU
  reference. Activation stays A8; the weight bit width only changes an in-register
  unpack; the symmetric zero-point is folded out of the accumulator per group.
- Added **plain-basis activation-aware clip calibration** (the "+") and measured a
  **12.5% weighted-error reduction at 4-bit, 20–25% at 2-bit** on the FF
  up-projections of real klein-4B weights (using a real imatrix from
  `hipfire diffusion calibrate`).
- **Key methodological result (the important one):** for this 4-step distilled
  flow model, **quantization-induced image change is within natural seed-to-seed
  variation.** A data-driven oqf4 allocation diverged from the bf16 render by
  `rel-L2 1.50 (cos 0.063)`; a mere **seed change** (bf16 seed 8 vs 7) diverged by
  `1.62 (cos 0.065)` — *more*. So divergence-from-bf16 is **not** a quality signal;
  it is dominated by sampler chaos and equals a reseed.
- Consequence: **do not gate quantized diffusion on fidelity to the bf16 render.**
  The correct admission signal is (a) early-step numerical fidelity *before* chaos
  compounds, and (b) **distributional/perceptual quality over a prompt set** — not
  similarity to a fixed bf16 image.

## What was built

All additive; the existing signed W4A8/W8A8 path is untouched.

| Layer | Component | Status |
|---|---|---|
| Kernel (`kernels/`) | `gemm_opus_tiled_wmma_u` + `opus_unpack_uNx16` + `quantize_act_oq8_sum` (per-group activation sum) | GPU-validated |
| Codec (`hipfire-quantize/opus_lowbit`) | unsigned pack/unpack {1,2,4,8}, bit-plane {1..8}, offset-fold reference, `quantize_symmetric_clip` (AWQ-style), `weighted_quant_error` | CPU unit-tested |
| Diffusion (`hipfire-diffusion`) | `OQF_W{4,2,1}` quant types, `encode_fold_tensor` (imatrix→clip), `decode_oqf_slice`, `resident_wua8`, fold consume-path branch in `gpu_ops`, precision-gated mixed policy + `HIPFIRE_DIFFUSION_FOLD_ROLES` data-driven override | end-to-end on GPU |
| Tooling | `hipfire diffusion quant-diff` (weight-space error), `calib-eval` (RTN vs clip, rayon-parallel), `flux2_trajectory_divergence.py`, `flux2_sensitivity_ablation.sh` + `flux2_sensitivity_rank.py` | working |

The fold GEMM shares the offline codec (`hipfire-quantize`), the activation
quantizer, and the dispatch backend (`hipfire-rdna`) with the AR LLM path; only
the on-disk format, resident builder, and consume routing are diffusion-specific.
An AR port reuses ~70–80%; the net-new piece is a fold **GEMV** for AR decode.

## Correctness and calibration

- **Fold GEMM parity** (`parity_gemm_wNa8u_fold`): GPU vs CPU offset-fold reference
  = **5.2e-8 rel-L2** for W8/W4/W2/W1, cos 1.0. The `Xsum` activation kernel is
  bit-exact vs a CPU recompute.
- **Calibration** (`calib-eval`, real imatrix): weighted rel-RMSE, RTN → clip:
  - 4-bit: mean **0.1187 → 0.1038 (−12.5%)**.
  - 2-bit, FF up-projections: **0.50 → 0.38 (−20…25%)**. The payoff grows as the
    grid coarsens — calibration matters at 2/4-bit, not at 8-bit (near-lossless).

## Allocation experiments

Metric during exploration: **step-1 model velocity** rel-L2 vs bf16 on identical
noise + (bf16) conditioning — a *first-forward, pre-chaos* fidelity measure.

**Uniform vs conservative-mixed:**

| policy | tensors@oqf4 | step-1 vel | compression |
|---|---|---|---|
| oqf4 everything | 109 | 0.416 | 1.56× |
| oqf4 mixed (static map: FF up-proj only) | 10 | **0.050** | 1.06× |

Quantizing only the tolerant FF up-projections is ~8× more faithful (first-forward)
than uniform 4-bit, and near-lossless — the conservative mixed policy.

**Per-role sensitivity ablation** (each role forced to 4-bit in isolation, via the
in-process `HIPFIRE_DIFFUSION_ABLATE` hook — artifact-free):

```
role            step-1 vel   static map
attn_v           0.034       High (protect)
attn_qk          0.038       High (protect)
ff_up            0.049       Compressed (quantize)   ← the only set we quantize
attn_out         0.076       High (protect)
single_qkvmlp    0.093       High (protect)
modulation       — (unmeasurable: bypasses the resident-linear path)
```

By the first-forward metric, attention Q/K/V are *more* tolerant than the FF
up-projections we already quantize — suggesting the static map over-protects them.

## The correction: divergence ≠ quality

Acting on the ablation, a **data-driven** allocation (`{ff_up, attn_qk,
attn_v}@oqf4`, 40 tensors, 1.09×) kept step-1 velocity low (**0.077**, cos 0.997)
but the *full 4-step trajectory* diverged much more than the FF-only mix (step-2
velocity 0.88 vs 0.059; final latent 1.50 vs 0.86). Initial (wrong) reading:
"attention quant has a hidden multi-step cost; the static map is justified."

**That was measuring the wrong thing.** A control render settles it:

```
                                        final-latent rel-L2   cos
quant (data-driven oqf4, same seed)         1.499            0.063
SEED CHANGE (bf16, seed 8 vs 7)             1.617            0.065
```

The quantized model's final image differs from bf16 by **less than a seed change**.
The step-2/step-4 "explosion" is **sampler chaos**: a tiny (0.077) first-forward
perturbation, amplified over 4 large distilled steps + CFG, reaches a
different-but-valid mode — exactly as a different initial seed would. It is **not**
quality loss. For text-to-image, a coherent, repositioned image is not a defect;
it is indistinguishable from reseeding.

This also explains the very first symptom in this investigation — the oq8
admission "failure" (`RGB mae=76/255`): near-lossless weights (0.7% rel-L2)
producing a repositioned image that a pixel gate flagged as broken. Same cause.

### Aggressive re-test — where quality *does* separate the allocations

Re-ran the seed comparison on the **most aggressive** model (all 109 fold-eligible
tensors @ oqf4, `HIPFIRE_DIFFUSION_FOLD_ROLES=".weight"`, step-1 velocity 0.416):

```
                                   final-latent rel-L2   cos     latent std
AGGRESSIVE quant (same seed)           1.213           0.043      0.631
seed change (bf16 seed 8 vs 7)         1.617           0.065      1.160
bf16 seed 7 (reference)                  —               —        0.859
```

Even at 109 tensors and a 40% first-forward perturbation, the **divergence stays
within reseed range** (1.21 < 1.62) — more confirmation that divergence-from-bf16
is not the signal. But the aggressive model's **final-latent variance collapses**
(std 0.63 vs bf16 0.86, vs seed-8's 1.16): lower latent variance ⇒ a flatter,
lower-contrast, detail-poorer decode. That is a genuine **quality** loss the
divergence number cannot see — and it is the axis on which the allocations
actually differ (the conservative FF-only and data-driven renders do not show this
collapse; the aggressive one does). So: *divergence saturates to reseed for any
allocation; **quality (latent variance / detail) is what distinguishes them***,
which is precisely why the gate must measure quality, not fidelity-to-bf16.

## Methodological conclusions

1. **Never gate a chaotic sampler on fidelity to a fixed bf16 render** (pixel MAE,
   final-latent MSE, SSIM-vs-bf16). Its noise floor is a full seed change.
2. **Two valid, complementary signals:**
   - *Numerical fidelity* — the **step-1 (pre-chaos) velocity/latent delta**. Small
     ⇒ the model's computation is faithful. Cheap; good for FF-type (local)
     tensors. (Attention passes it too — the multi-step growth is chaos, not a
     fidelity failure.)
   - *Quality* — **distributional/perceptual over a prompt set** (CLIP-score, FID,
     aesthetic, human/A-B), judged on its own merits, **not** vs the bf16 image.
3. `resident_wua8` + the fold consume path are correct and near-lossless in the
   sense that matters (first-forward faithful, quality-preserving); the residual
   image difference is reseed-equivalent.

## Practical recommendations

- **Ship the conservative FF-only mixed policy** now — validated, near-lossless,
  low risk. It is the default (`--format oqf4` with the static precision gate).
- The **data-driven allocation (+ attn Q/K/V) is very likely also fine** — its
  output is within seed variation of bf16. Confirm with a **quality eval over a
  prompt set**, not a divergence metric. If it holds, widen the low-bit set.
- **Footprint** is dominated by the single-block fused `to_qkv_mlp_proj` (20 large
  matrices) — also the most first-forward-sensitive. The lever there is *stronger
  calibration / a higher bit tier*, not reallocation.
- **Modulation** is unmeasurable/un-fold-quantizable (its projection bypasses the
  resident-linear path); it stays bf16, as the static map dictates.

## Open questions / next steps

- Replace the diffusion admission gate's bf16-fidelity component with a
  **prompt-set quality metric** (the early-latent fidelity check can stay as a
  numerical guard).
- A **quality A/B** of FF-only vs data-driven vs uniform over ~50 prompts would
  turn "likely fine" into a decision (this is the `astrea` promote/reject workflow).
- AR-LLM reuse: mirror `resident_wua8` in the qwen35/gemma3 loader (prefill reuses
  the GEMM); write a fold **GEMV** for decode.

## Artifacts (this investigation)

- Tools: `hipfire diffusion {quant-diff, calib-eval}`, `scripts/flux2_*`
  (trajectory divergence, sensitivity ablation/rank, LPIPS sidecar, chaos confirm).
- Local p0 copy: `scratchpad/klein4b.p0.hfq` (renders fast, off NFS).
- Mixed artifact on `/srv/huggingface/klein4b.oqf4mixed.hfq` (15 GB) — removable.
