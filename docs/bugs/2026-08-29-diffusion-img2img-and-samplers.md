# Diffusion: img2img noise is wrong on both scheduler families, and "Euler a" is not ancestral

Status: found 2026-08-29, master `0c9e3d252`, nix1. **All three confirmed 3/3. The
two noise defects are FIXED and pinned by tests; `"Euler a"` remains open and needs
a product decision.** No rendered A/B was run — the defects are established from the
noise algebra and the scheduler plumbing, and the fixes are verified by unit test,
not by image comparison.

Together these mean **img2img and hires-fix do not currently preserve the init
image** on either scheduler family, by two independent mechanisms.

---

## 1. [FIXED] Flow-match img2img uses the epsilon (additive) noise form (high)

`crates/hipfire-diffusion/src/pipeline_generate.rs:770` calls
`add_noise_to_latents`, whose additive branch (`scheduler.rs:854`) computes
`x = x0 + sigma * n`. Flow-matching models need the **interpolation** form
`x = (1 - sigma) * x0 + sigma * n`.

A `FlowMatchEuler` schedule built by `flow_match_euler_with_image_seq_len` has an
empty `train_timesteps`, so `add_noise_to_latents` falls to exactly that additive
branch.

At 12 steps / strength 0.6, sigma at `start_step` is 0.637: variance 1.41
(additive) vs 0.537 (correct). The x0 term keeps weight 1, so ideal-velocity
integration returns `(1 + sigma) * x0 ≈ 1.64 * x0` — the init structure is present
but **amplified**, an over-bright/over-contrast render. The DiT then sees an
out-of-distribution input, so the real output degrades further and unpredictably.

Affected surface is wider than one call site: `hipfire diffusion img2img`,
`txt2img --hr-scale` (request built at `crates/hipfire-cli/src/commands/diffusion.rs:1403-1404`),
the server SDAPI img2img routes (`sdapi.rs:588-589`, `:827-828`), the CLI MrFlow
draft weave (`diffusion.rs:1288`, strength 1.0), and the masked/inpaint path
(`denoise.rs:839`, reached from the denoise loop at `:606` — note `denoise.rs:821`
is `#[cfg(test)]` and does not count).

The latent-space draft-refine path at `pipeline_generate.rs:291` is correct — it
uses `add_flow_match_refine_noise`, which is the step-0 case of the right formula.

**FIXED** exactly that way — the dispatch went into `add_noise_to_latents` before
the `train_timesteps` branch, so img2img, hires-fix and the masked path are all
fixed at once rather than one call site at a time. Pinned by
`flow_match_noising_interpolates_where_euler_adds`, which A/Bs one schedule cloned
with only `solver` flipped, so the two branches differ by construction.

---

## 2. [FIXED] img2img reuses sigma-scaled latents as unit noise (high)

`crates/hipfire-diffusion/src/pipeline_generate.rs:763` passes `plan`'s latents as
the noise argument. Those were already multiplied by `initial_noise_sigma()` for
use as a txt2img **starting point**, so the noise term is inflated by exactly
`initial_noise_sigma()` = **14.6146555** on an SDXL-class Euler schedule.

The finder's arithmetic named the wrong branch. For any config-derived schedule —
Euler included — `train_timesteps` is populated (`scheduler.rs:608`), so the
**DDPM** branch at `:842-849` runs: `latent * alpha_t + noise * sigma_t`. At
`start_step = 10` (t = 473) the intended result is `0.560*x0 + 0.829*n`; the actual
is `0.560*x0 + 12.11*n`. The headline is unchanged: SNR at trajectory start is
~14.6x too low, so `denoising_strength` no longer means anything and the init image
is effectively discarded.

Not limited to an explicit `--scheduler Euler`: `--scheduler` defaults to
`Automatic`, which `resolve_request_scheduler` (`config.rs:334-344`) resolves to
the model's own `scheduler_config` — and SDXL ships `EulerDiscreteScheduler`. A
bare `hipfire diffusion img2img` with no flags hits it. Nor is it limited to
`strength < 1`: at 1.0, `start_step` is 0 and the init is still noised with the
scaled buffer.

**FIXED** with the local correction: the buffer is divided by
`initial_noise_sigma()` at the noise site. The plan-level restructure was NOT taken
— `plan.latents` also feeds the txt2img start (`:433`, `:446`) and the
inpainting-fill (`:754`), and moving the scale would change all three without a way
to verify the render. No-op for flow-match, whose `input_scaling` is `None`. Pinned
by `initial_noise_scaling_inverts_and_is_a_noop_for_flow_match`.

---

## 3. [OPEN — needs a decision] "Euler a" silently runs deterministic Euler (medium)

`crates/hipfire-diffusion/src/scheduler.rs:129` — `SchedulerSolver::from_config`
branches only on `!= "DPMSolverMultistepScheduler"`, so
`EulerAncestralDiscreteScheduler` returns plain `Euler`. No ancestral
`sigma_up`/`sigma_down` step and no per-step noise draw exists anywhere.

The two resolved `SchedulerConfig`s differ in **exactly one field** (`class_name`),
and all three consumers of it treat the two identically: `from_scheduler_class`
(`:259-260`), `from_config` (`:129`), `inference_train_timesteps` (`:1516`).
`DiffusionSchedule` carries no `class_name` field at all (`:602-612`), so the
ancestral distinction is destroyed at schedule construction and the outputs are
**provably bit-identical** for the same seed. The GPU arm is equally non-ancestral
(`ops_dispatch.rs:129-158` → `euler_step_hip_on_gpu` takes only sigma/next_sigma).

There is a second route that never calls `from_config` at all:
`from_config_with_image_seq_len` early-returns `Self::linear(steps)` (`:535-548`)
when beta params are absent, and `linear` hardcodes `solver: Euler` (`:501`). A fix
confined to `from_config` would miss it.

This is advertised, not incidental: `GET /sdapi/v1/samplers` lists "Euler a"
(`crates/hipfire-server/src/routes/sdapi.rs:40x`), `docs/CLI.md:908` and `:973`
enumerate it, and the response reports the ancestral sampler back to the caller.

**Fix:** add a `SchedulerSolver::EulerAncestral` arm implementing the
sigma_up/sigma_down step with a seeded per-step noise draw — or, matching the
`distilled_guidance_scale` precedent, return `InvalidMetadata` so the request fails
loudly instead of quietly rendering something else.
