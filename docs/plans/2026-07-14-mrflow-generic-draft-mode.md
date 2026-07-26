# MrFlow → generic "draft" mode (plan)

Status: P1 landed (2026-07-15). Branch: `chaingun`.

## P1 status (2026-07-15)

P1 is implemented and the **structural** goal is met: the generic latent-space
Stage-2 refine replaces the pixel-re-encode default and SeFi runs end-to-end
through `--mrflow` instead of hanging.

- `#1` capture Stage-1 full latent — `generate_batch_capturing_latent`. Done.
- `#2` generic latent-space refine — `DiffusionPipeline::generate_draft_refine`
  (`pipeline_generate.rs`): bilinear latent upscale → per-stream refine noise →
  the model's own denoiser (SeFi dual-stream vs standard) → texture slice →
  decode. Done. CLI `generate_mrflow_txt2img` routes here by default; the
  pixel-space `--mrflow-sr` overlay is retained for non-SeFi and guarded (clear
  error) for SeFi.
- `#3` SeFi dual-refine schedule — `DiffusionSchedule::sefi_dual_refine` +
  `SeFiDualSchedule::add_refine_noise` (`scheduler.rs`). Done.

**Validation caveat — resolution tuning, not a mechanism bug.** The
`--mrflow krea2-turbo-8plus1` SeFi render (`A cat playing on grass`, 1024²,
seed 42) completes and is structurally a cat, but is **not yet a clean preview**.
Root cause (isolated via the `HIPFIRE_DRAFT_DECODE_UPSCALED` diagnostic and
native-resolution baselines):

- **SeFi turbo is 1024-native and resolution-sensitive.** Native **1024²/8-step
  is excellent** (sharp full-body cat, no grid/weave; minor saturation + edge
  fringing only). Native **512²/8-step is degraded** (purple cat, acid-green bg,
  faint regular grid) — low-res out-of-distribution for a 1024-trained turbo.
- The `krea2-turbo-8plus1` preset drafts Stage-1 at **512** (upscale 2×), so
  Stage-1 is already the degraded/gridded output. The 2× latent upscale then
  **amplifies the grid into a strong weave** (switching nearest → bilinear
  softened it but could not remove a grid already present pre-upscale).
- The refine denoise is **exonerated**: the corruption is present in the
  captured Stage-1 latent before any refine.

So P1's mechanism works; the clean-1024²-preview criterion is blocked on **preset
tuning**, not a bug — SeFi should draft near its native res (small upscale
factor, ~1.25–1.5×), a P2 metadata-derived-defaults concern. Re-validate the
generic path independently on a model that renders cleanly at low res
(klein / Krea-2).

**Confirmed (2026-07-15).** Re-running the same render with
`--mrflow-upscale 1.3` (Stage-1 at **784²**, close to native 1024) produces a
**coherent, correctly-coloured photographic cat** — weave, purple cast and grid
all gone, composition faithful to the native-1024 render. This validates the
generic latent-space refine end-to-end for SeFi and pins the earlier corruption
entirely on the 2× (512²) low-res draft, not the mechanism. A minor
double-exposure/ghosting remains (faint offset silhouette) — a refine-strength
tuning item (try lower `refine_sigma` or a 2-step dual refine), not a
correctness bug. **Action for P2:** derive the SeFi draft upscale from metadata
so it defaults near native res (≤~1.5×) instead of the Krea-2 preset's 2×.

---


Turn MrFlow from a per-model staged-sampling feature (named presets, pixel
re-encode) into a **model-agnostic draft/preview mode** that works for every
diffusion arch hipfire supports — klein, SeFi (144-ch semantic/texture split),
Krea-2, Qwen-Image, and future latents — by construction.

## Why the current MrFlow is not generic

MrFlow today (see `generate_mrflow_txt2img` in
`crates/hipfire-cli/src/commands/diffusion.rs`) runs:

1. Stage-1: fast low-res denoise → decode to RGB.
2. Stage-2: **pixel-space upscale → re-encode the image to a latent → refine**
   via `generate_img2img_batch_inner`.

Two couplings break genericity:

- **Pixel re-encode assumes pixels → latent recovers the model's full latent.**
  False for any model with non-pixel-derived latent channels. SeFi's latent is
  `16 semantic + 128 texture`; the VAE encode only produces the 128 texture
  channels, so the semantic stream is lost and the refine gets a
  144-vs-128 mismatch (observed: Stage-2 hangs).
- **`generate_img2img_batch_inner` calls the *standard* denoise**
  (`denoise_latents_with_runtime_context`), with **no SeFi branch** — so even a
  correctly-shaped latent would run the wrong (non-split) denoiser.
- Per-model presets (`krea2-*`, `zit-*`) hard-code step/sigma/CFG per checkpoint
  instead of deriving them.

## The generic design: latent-space staged refine

```
Stage 1:  fast low-res denoise with the model's OWN denoiser  → full latent (any C)
Stage 2:  upscale that latent in latent space (channel-agnostic)
          → add refine noise → refine with the model's OWN denoiser → decode
```

Model-agnostic because it **never re-encodes from pixels** and **never assumes a
channel structure**:
- `resize_latent_batch_nearest` (`lib.rs`) already upscales any channel count.
- The refine dispatches to whatever denoiser the model uses — `denoise_sefi_
  latents` for SeFi (semantic/texture carried untouched), `denoise_latents`
  otherwise — mirroring the `plan.sefi_dual_schedule.is_some()` split that
  `generate_batch` already does.
- No semantic *encoder* is needed (there isn't one) — the semantic channels are
  **carried** from Stage-1, not reconstructed.

Pixel-space SR becomes an **optional overlay**, not the default: when a real SR
model (`--mrflow-sr`, RealESRGAN/RRDBNet) is loaded *and* the model's latent is
fully pixel-derivable, Stage-2 may re-encode the SR image to inject true
high-frequency texture. For models like SeFi it stays latent-only (or re-encodes
only the texture sub-latent and keeps the carried semantic channels).

## Draft-mode UX (the "eventually")

- A single entry — `--draft` (or a budget knob) — instead of named presets.
- **Auto-configured from model metadata**: distilled/turbo → few steps, low
  refine sigma, no CFG; base → more steps, higher sigma, CFG. Read from the
  pipeline metadata (`sefi`, `is_distilled`, default steps/guidance) that the
  importer already records.
- Works for every arch by construction; presets remain as optional overrides.

## Phasing

- **P1 (foundational, fixes SeFi):**
  1. Expose Stage-1's **full** latent from the batch path (today it slices to
     texture and discards the rest before returning only images).
  2. **Generic latent-space Stage-2 refine**: upscale the full Stage-1 latent,
     add refine noise, route to the model's own denoiser (SeFi vs standard),
     decode. Replaces pixel-re-encode-via-img2img as the default Stage-2.
  3. **SeFi dual-refine schedule** — the one genuinely new bit: a refine that
     respects the semantic/texture sigma offset (`delta_t`) at the refine start
     sigma, analogous to `refine_direct_sigma` but dual-stream.
- **P2 (draft UX):** metadata-derived defaults + a `--draft` entry; keep the
  named presets and `--mrflow-*` overrides working.
- **P3 (SR overlay):** re-introduce pixel-space SR as an opt-in overlay for
  pixel-derivable latents (klein/Krea-2), with the SeFi texture-only variant.

## Validation

- SeFi turbo `--draft` produces a coherent 1024² image (the current hang case).
- klein-base + SeFi + Krea-2 all run the same generic path with no per-model
  code branches beyond the denoiser dispatch.
- Draft output is a faithful *preview* of the full-quality render (same seed →
  same composition, lower detail), not a different image.
- `coherence-gate-dflash.sh` is unaffected (diffusion-only changes).

## Notes / risks

- The **SeFi dual-refine schedule** is the design-risk item; needs validation
  renders to tune the refine sigma split.
- Interim: until P1 lands, `--mrflow` on a SeFi model should **guard with a clear
  error** rather than hang (Stage-2 pixel re-encode is not SeFi-safe).
- Non-determinism (see `docs/flux2-vae-dit-parity-findings.md`) is orthogonal and
  benign here — draft mode is a preview, so reseed-level variation is acceptable.
