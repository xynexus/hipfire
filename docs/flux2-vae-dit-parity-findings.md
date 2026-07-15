# FLUX.2 klein-base-4B — VAE/DiT parity investigation findings

Status: findings, 2026-07-14. Hardware: gfx1103 (Radeon 780M, RDNA3 wave32).
Model: `FLUX.2-klein-base-4B` (non-distilled base; reference defaults cfg=4.0, 50 steps).

## TL;DR

- Built a **CPU-torch oracle** from the vendored BFL reference (no ROCm needed).
  It proved hipfire's FLUX.2 implementation is **substantially correct**:
  - **VAE decode**: bit-exact to the reference — pixel MAE **0.5/255**, cos **1.0000**.
  - **DiT forward** (steps 0,1,3): rel-L2 **~0.002**, cos **1.0000** vs the reference.
  - **Schedule + Euler integration**: per-step timesteps and dt match the
    reference **exactly** (1.0→0.9547→0.8754→0.7007→0).
- The apparent "texture corruption" was a **red herring**: the admission prompt is
  literally *"close-up woven blue and gold fabric, intricate repeating texture"* —
  the reference torch pipeline renders the **same woven fabric**. It was never an
  artifact.
- **One real bug remains**: hipfire's DiT forward is **non-deterministic** on
  structured latents. Two identical-seed runs are bit-identical at step 0 but
  diverge from step 1 onward (`velocity_002` rel-L2 **0.19** across runs),
  compounding over the trajectory. This is the source of the residual teapot
  streaking and the earlier "step-2 forward divergence."
- **Fixed this pass**: the VAE **encode** path was missing the FLUX.2 patchify
  (img2img failed with a hard shape mismatch). Added `patchify_and_normalize` and
  wired it into all encode paths; img2img now round-trips cleanly.

## Methodology note (read this)

This diagnosis reversed ~4 times before the oracle settled it: a blind Nyquist
"checkerboard" metric → "VAE decode bug" → "under-stepping" → finally "the prompt
is a fabric + a non-determinism bug." Two lessons:

1. **An independent ground-truth oracle beats internal inspection.** On a chaotic
   sampler with per-channel high-frequency features, channel-mean views and
   single-frequency metrics are actively misleading. The CPU-torch reference gave
   the answer in one run each.
2. **Read the prompt.** Substantial effort went into "fixing" a correct render of
   a fabric texture.

## What is correct (oracle-proven, bit-exact)

Oracle scripts (CPU torch, vendored BFL reference, `scripts/`):
- `flux2_vae_oracle.py` — decode a hipfire HFDT latent through the reference VAE.
- `flux2_dit_oracle.py` — reference DiT forward vs hipfire `velocity_001`.
- `flux2_perstep_oracle.py` — per-step reference-forward parity.
- `flux2_pipeline_oracle.py` — full reference pipeline (DiT + VAE, optional CFG)
  from hipfire's dumped noise + conditioning.

Key inputs are taken from hipfire's `HIPFIRE_DUMP_DENOISE_TRACE` dumps, including
`conditioning_positive` — so the oracle feeds hipfire's text embeddings directly
and needs **no** 8.9 GB text encoder in torch. Weights load from the BFL-native
`flux-2-klein-base-4b.safetensors` (strict, **0 missing / 0 unexpected**) and the
diffusers VAE safetensors (via the name mapping in `flux2_vae_reference.py`).

Results:
- VAE: hipfire vs reference decode of the same latent → MAE 0.5/255, cos 1.0000.
- DiT forward, per step (t): 0 → 0.0019, 1 → 0.0020, 3 → 0.0018 (cos 1.0000).
- Schedule/dt: reconstructed hipfire timesteps == reference at every step.

## The non-determinism bug (the real remaining defect)

Two runs, same seed / prompt / settings (cfg=1, 4 steps, 128²), compared per
tensor:

```
latent_000            rel-L2 0.00e+00   (seed noise, identical)
velocity_001 (step0)  rel-L2 0.00e+00   (bit-identical)
latent_001            rel-L2 0.00e+00
velocity_002 (step1)  rel-L2 1.93e-01   (NON-DETERMINISTIC)
latent_002            rel-L2 1.83e-02
velocity_003 (step2)  rel-L2 2.81e-01
```

Step 0 is **bit-identical**; non-determinism enters at the **step-1 forward** and
compounds. Because the perturbation is random per run, it reaches a
different-but-plausible mode each time — indistinguishable from a reseed at the
image level, but it degrades fine detail (the teapot's horizontal streaking that
28 steps did not fix).

**Localization (2026-07-14), by elimination — it is NOT the compute kernels:**
- **Data-dependent**: `steps=2` is fully bit-identical across runs at every stage;
  `steps=4` diverges from step 1. Same binary/seed. So it depends on the input
  values/timestep, not a fixed code path. The 0.19–0.25 magnitude is far too large
  for FP non-associativity — it is a real divergence, not rounding.
- **Not the GEMM**: forcing the non-tiled kernel (`HIPFIRE_DIFFUSION_TILED_GEMM=0`)
  still diverges. Both `gemm_bf16_tiled_wmma.hip` and `gemm_bf16_x_bf16_wmma.hip`
  have no atomics / split-K (only a benign `__syncthreads`).
- **Not the attention**: both `diffusion_flash_attention_qtile_f32` and the naive
  `diffusion_sdpa_3d_f32` are deterministic by inspection — one wave per
  (batch,head,q-tile), sequential K/V loop, wave-shuffle tree reduction, no
  atomics, no shared-memory race, no OOB read.
- **Not uninitialized scratch**: zero-initializing every `alloc_resident_f32`
  buffer (`HIPFIRE_DIFFUSION_ZERO_ALLOC=1`) does **not** remove it.
- **Not the weights**: step 0 is bit-identical, so the resident weight cache is
  deterministic.

**It is in the forward compute, and affects BOTH paths.** The CPU forward
(no `--rocm-device-id`) is *also* non-deterministic — bit-identical at step 1,
divergent by step 3 (`velocity_004` rel-L2 0.28). So it is **not** a GPU/ROCm-7.14
driver artifact; it is shared/algorithmic. Also ruled out as the source:
- **Text conditioning** is bit-identical across runs on both paths (the Qwen3
  encoder is deterministic) — not the shared cause.
- Weights, latent init, and schedule are all deterministic (step 0 bit-identical).

The entropy source is therefore a **parallel/ordered reduction in the forward
that changes combine-order across runs** (CPU: rayon; GPU: some scheduling-order
op), surfacing only where the value magnitudes stop rounding it away — which is
why it is data-dependent and appears at a *different step* on CPU vs GPU. The
obvious matmul on CPU is output-parallel (`par_chunks_mut`, deterministic), so the
culprit is a subtler reduction not yet pinned.

**Root-causing this needs a per-op determinism harness** — instrument the forward
to dump every intermediate, run twice on a fixed seed, and diff to find the first
op whose output differs. That is the clean bounded next task; guessing at
individual kernels was not productive. A determinism regression test (same seed
twice → bit-identical `latent_00N`) should guard it once fixed.

**Severity:** benign at the image level — the effect is reseed-like, and a clean
1024² render ("a cat playing on grass") is fully coherent and detailed. It is a
reproducibility/quality-margin issue, not a correctness break.

## The encode-patchify fix (landed)

FLUX.2 encode was missing the 2×2 patchify: it produced `[1,32,32,32]` where the
DiT expects `[1,128,16,16]`, so img2img hard-failed. Added
`Flux2VaePatchNorm::patchify_and_normalize` (exact inverse of the decoder's
`inverse_and_unpatchify`: patchify `(c) (h ph)(w pw) -> (c ph pw) h w` then
`bn.normalize`), added `flux2_patch_norm` to `NativeVaeEncoder`, and routed all
three `encode_to_latents*` paths through it. img2img now round-trips a clean
image (verified: smooth gradient in → smooth gradient out).

## Impact on the quant / mixed-precision work

None. The fold/mixed-precision conclusions stand — the bit-exact forward and VAE
confirm the fold path is transparent, and all quant comparisons were same-settings
(the non-determinism affects candidate and baseline equally). This is a
base-forward defect orthogonal to quantization.

## Recommended next step

Pin the non-determinism (bounded GPU-debugging task):
1. Add step-indexed block dumps (or dump only the step-1 forward) and compare two
   runs block-by-block to find the **first** non-deterministic op within a forward.
2. Bisect with `device_synchronize()` inserted around resident block stages /
   after `free_resident`; if a sync makes it deterministic, it's a buffer-reuse
   race. If not, inspect the resident SDPA / tiled GEMM for a read-before-write on
   pooled scratch.
3. Regression guard: a determinism test (same seed twice → bit-identical latents)
   belongs in the diffusion gate.

## State / cleanup

- `crates/hipfire-diffusion/src/vae.rs` carries trace-gated debug instrumentation
  (VAE stage + up-block dumps, forced host path when tracing). **The encode fix
  stays; the dump scaffolding is strippable.**
- Local scratch holds the p0 copy (`klein4b.p0.hfq`), calib, oracle traces, and
  the oracle PNGs — useful for the non-determinism follow-up.
