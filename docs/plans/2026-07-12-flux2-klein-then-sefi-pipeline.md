# FLUX.2 [klein] → SeFi-Image DiT pipeline (plan)

Status: implementation stopped at the frozen P4 rejection (2026-07-13).
Implementation began from `12892469d1d68c331be59ec71768bfcd280219dc` and was
fast-forwarded/revalidated at `origin/chaingun`
`3ff7a351dd9832b65470e09f3c48fa92bd50f5e6` before handoff.
Implementation starts from a clean worktree at the then-current
`origin/chaingun`; do not layer this port over an unrelated dirty checkout.

Bring the **FLUX.2 [klein]** family and then **SeFi-Image** text-to-image models
into `hipfire-diffusion`. Both are flow-matching **Flux.2 MMDiT** image models
(double-stream + single-stream adaLN, 4-axis RoPE, Qwen3 text encoder, one
shared conv-KL VAE). This is the crate's **second DiT family** after Krea-2.

The strategy is **vanilla first**: land plain FLUX.2 [klein]-base-4B end-to-end,
then add SeFi-Image-2B-turbo as a **modulation + scheduler delta** on the same
DiT / VAE / encoder machinery. Vanilla klein-base-4B is the ideal first target —
it is the closest full sibling to SeFi-2B, is Apache-2.0, shares the VAE and
schedule with SeFi, and exercises the *full* path (including CFG, which the
distilled SeFi drops).

Reference code (vendored):
- Vanilla FLUX.2: `third_party/flux2/src/flux2/` (BFL minimal inference).
- SeFi-Image: `third_party/SeFi-Image/sefi/` (semantic-first wrapper over Flux.2).

Source models (on `/srv/huggingface`):
- `black-forest-labs/FLUX.2-klein-base-4B` (vanilla milestone; complete 23 GiB
  snapshot at HF revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`).
- `SeFi-Image/SeFi-Image-2B-turbo` (target; diffusers layout, snapshot revision
  `fa04be3b555fc5385e822a12f75e271d763f4d59`).

## 0. Why this is smaller than a from-scratch build

Krea-2's single-stream adaLN MMDiT forward is **confirmed working** on hipfire's
HIP path (recent `feat/perf/fix(diffusion)` commits: Turbo scheduling aligned,
register-tiled WMMA GEMM + W8A8 hot path). That answers the one hard correctness
question — an adaLN MMDiT forward is numerically correct on our kernels. The
Flux.2 forward is then an **assembly of proven primitives**:

| Flux.2 piece | Status in hipfire |
|---|---|
| Single-stream adaLN block (fused qkv+mlp, gated, QK-RMSNorm, SiLU-GLU) | Krea-2 single-stream **proven**; different fusion/naming only |
| Double-stream joint-attention block (per-stream 6-way mod, SiLU-GLU) | QwenImage family is **double-stream**; Flux.2 is a variant |
| Full SDPA over `[txt, img]` (txt2img has no ref tokens) | resident SDPA **done** |
| 4-axis RoPE `[32,32,32,32]` θ2000 | Krea-2 is 3-axis; trivial extension |
| Flow-match Euler + mu-shift schedule | added for Krea-2; **reuse** |
| conv-KL VAE decode (GroupNorm/Resnet/attn-mid, 8× up) | SD-lineage decode **exists**; z=32 config delta |
| Qwen3 hidden-state extraction (`[9,18,27]` concat) | Krea-2 needs the same mechanism (12 layers); **shared** |

New work is the **Flux.2 family assembly**, the **importer/arch-id**, a **VAE
config/channel delta**, and — for SeFi only — the **dual-timestep embedding** and
**split semantic/texture denoise loop**.

## 1. What the models are

### 1.1 Common Flux.2 DiT (`third_party/flux2/src/flux2/model.py`)

Single-file forward, BFL-native layout. Per-variant params:

| Param | klein-4B (vanilla) | klein-9B | dev (32B) | **SeFi-2B-turbo** |
|---|---|---|---|---|
| `hidden_size` | 3072 | 4096 | 6144 | **2560** |
| `num_heads` | 24 | 32 | 48 | **20** |
| `head_dim` | 128 | 128 | 128 | 128 |
| `depth` (double) | 5 | 8 | 8 | **4** |
| `depth_single_blocks` | 20 | 24 | 48 | **16** |
| `context_in_dim` | 7680 | 12288 | 15360 | **6144** |
| `axes_dim` / `theta` | [32,32,32,32] / 2000 | same | same | same |
| `in_channels` | 128 | 128 | 128 | **144** (16 sem + 128 tex) |
| `use_guidance_embed` | False | False | True | False |
| text encoder | **Qwen3-4B**, `[9,18,27]` | Qwen3-8B | Mistral-Small-24B, `[10,20,30]` | **Qwen3-VL-2B** (text tower), `[9,18,27]` |
| defaults | CFG g=4.0, 50 steps | g=1.0, 4 (distilled) | g=4.0, 50 | g=1.0, 4/8/10 (distilled) |

Forward (txt2img, `Flux2.forward`):
1. `vec = time_in(timestep_embedding(t, 256))` (+ `guidance_in` only if dev).
   `timestep_embedding` uses `time_factor=1000`, `max_period=10000`.
2. Modulations from `vec`: `double_stream_modulation_img/txt` (6-way each),
   `single_stream_modulation` (3-way). Each `Modulation = lin(silu(vec))`, chunked.
3. `img = img_in(x)` (linear, `in_channels→hidden`, no bias);
   `txt = txt_in(ctx)` (linear, `context_in_dim→hidden`, no bias).
4. RoPE: `pe = EmbedND(x_ids)` / `EmbedND(ctx_ids)`; 4 axes concatenated.
5. `depth` **double blocks** on `(img, txt)` with joint attention over
   `cat[txt,img]`, QK-RMSNorm, `(1+scale)*LN(x)+shift`, gated residuals,
   SiLU-GLU MLP (`mlp_ratio 3.0`, `mlp_mult_factor 2`).
6. `img = cat[txt, img]`, then `depth_single_blocks` **single blocks**: fused
   `linear1 → [qkv | mlp]`, QK-RMSNorm, full SDPA, `linear2(cat[attn, silu_glu(mlp)])`,
   gated by `mod_gate`.
7. Strip txt tokens; `final_layer` = `LayerNorm(no-affine) → (1+scale)·x+shift →
   linear` (adaLN from `vec`), producing `out_channels` velocity.

**KV-cache / reference-token paths (`forward_kv_extract` / `forward_kv_cached`,
`denoise_cached`) are image-editing only and are OUT OF SCOPE for this plan.**
For txt2img, `num_ref_tokens=0` makes `causal_attn_fn` a plain full SDPA.

Note two on-disk layouts:
- **BFL native** (`flux-2-klein-*.safetensors`): fused `linear1`/`linear2`,
  `img_attn.qkv`/`proj`, `time_in`, `txt_in`, `final_layer`.
- **diffusers** (SeFi checkpoint, `backbone.` prefix): `single_transformer_blocks.
  *.attn.to_qkv_mlp_proj`/`to_out`; `transformer_blocks.*.attn.to_q/to_k/to_v` +
  `add_q/k/v_proj` + `to_out.0`/`to_add_out`; `x_embedder`/`context_embedder`/
  `norm_out`/`proj_out`; `double_stream_modulation_img/txt`, `single_stream_modulation`.
The importer normalizes **both** into one canonical HFQ layout so vanilla and
SeFi share the runtime forward.

### 1.2 Shared VAE (`third_party/flux2/src/flux2/autoencoder.py`)

One `AutoEncoder` shipped as FLUX.2-dev `ae.safetensors`, used by **every** Flux.2
variant. SD-lineage 2D conv KL: `ch 128`, `ch_mult [1,2,4,4]` (8× down),
`num_res_blocks 2`, GroupNorm(32)+swish, ResnetBlocks, self-attn mid block,
`z_channels 32`.

Critical: the **patchify + BatchNorm normalization is part of the VAE**:
- `encode`: `mean → rearrange to (c·pi·pj) (patch [2,2]) → bn.normalize` ⇒ **128-ch
  patchified, bn-normalized latent** (this is the DiT's `in/out` texture space).
- `decode`: `bn.inv_normalize (z·√(var+1e-4)+mean) → unpatchify → decoder convs`.

SeFi's diffusers `AutoencoderKLFlux2` is the same net; its
`texture_latent_codec.py` just replicates the bn de-normalization from
diffusers-exposed `bn.running_mean/var` (eps 1e-4). hipfire implements the bn +
patch [2,2] pack/unpack once, in the decode path.

### 1.3 Schedule (`third_party/flux2/src/flux2/sampling.py`)

`get_schedule(num_steps, image_seq_len)`:
`t = linspace(1,0,steps+1)`, then `generalized_time_snr_shift(t, mu, sigma=1) =
exp(mu)/(exp(mu)+(1/t−1)^sigma)`, with `mu = compute_empirical_mu(seq_len,steps)`
(piecewise-linear in seq_len/steps; the `a1,b1/a2,b2` constants). This is the
`use_dynamic_shifting` path of diffusers `FlowMatchEulerDiscreteScheduler` —
identical math to what Krea-2 already added. `denoise`: `img += (t_prev−t_curr)·pred`.

### 1.4 SeFi delta (`third_party/SeFi-Image/sefi/`)

On top of vanilla Flux.2:
- **Dual-timestep embed** (`flux2_sefi_transformer.py:SEFIDualTimestepEmbeddings`):
  two `TimestepEmbedding`s (semantic, texture) each → `hidden/2`, concatenated →
  `vec`. Replaces the single `time_in`. `dual_time_embed.{semantic,texture}_embedder`.
- **Split semantic/texture denoise** (`runner.py:generate_batch`): the 144-ch
  latent = **16 semantic + 128 texture**. Each stream runs its **own sigma
  schedule offset by `delta_t=0.1`** (`u_tex = u_sem − delta_t`), so semantic
  denoises slightly ahead. Per-stream Euler: `lat_sem += (σ_sem_next−σ_sem_cur)·
  vel_sem`, likewise texture. Optional `timestep_shift_alpha` unit-interval warp.
- **Text encoder**: Qwen3-**VL**-2B text tower (visual deleted), same `[9,18,27]`
  concat. Simpler than Krea-2 (no text_fusion / layerwise / refiner stack).
- **Decode**: only the **128 texture** channels go through the VAE; the 16
  semantic channels are auxiliary and dropped at decode.
- Distilled: guidance 1.0, steps ∈ {4,8,10}, **no CFG / negative prompt**.

## 2. Op-by-op diff vs existing code

Every Flux.2 op has a near-exact analogue already in
`crates/hipfire-diffusion/src/transformer.rs` (the QwenImage double-stream and
Krea-2 single-stream forwards). Symbol names are authoritative; line refs are
orientation only and must be refreshed after the implementation branch is
rebased.

| Flux.2 (`third_party/flux2/model.py`) | Closest existing hipfire code | Delta to implement |
|---|---|---|
| `Flux2.forward` (double loop → concat → single loop → `final_layer`) | `NativeTransformerDenoiser::forward_qwen_with_runtime_context` (`transformer.rs:2770`), `forward_krea_with_runtime_context` (`:2839`) | New `forward_flux2_*`: run **double** blocks keeping img/txt separate (Qwen-shape), then `concat[txt,img]` + run **single** blocks (Krea-shape), strip txt, `final_layer`. One denoiser combines both stacks. |
| `DoubleStreamBlock` | `NativeTransformerBlock::forward_qwen_with_runtime_context` (`:2364`) — LN-no-affine eps1e-6 (`:2395`), `attend_image_text` (`:1636`), `modulate_3d`, `gated_residual_3d` | ~90% match. Deltas: (a) 6-way mod comes from a **shared top-level** `Modulation` (img/txt), not per-block `img_mod/txt_mod`; (b) **QK-RMSNorm** on q/k (Krea already has this via `attend_krea_self_gated`); (c) **SiLU-GLU** MLP not GEGLU. |
| `SingleStreamBlock` | `NativeTransformerBlock::forward_krea_with_runtime_context` (`:2219`) — adaLN via `krea_scale_shift`, `rms_norm_3d`, gated residual | Closest sibling. Deltas: (a) **fused `linear1`** emits qkv **and** mlp-up together (split then run SDPA on qkv, SiLU-GLU on mlp); (b) **3-way** mod `[shift,scale,gate]` (Krea is 6-way pre/post); (c) **no per-attn sigmoid gate** — Flux gates the *whole block output* by `mod_gate` (Krea's `attend_krea_self_gated` applies a sigmoid `to_gate` to the attn output — do **not** reuse that gating); (d) `linear2(cat[attn, silu_glu(mlp)])`. |
| `Modulation = lin(silu(vec))`, chunk 3/6, `(shift,scale,gate)` | `qwen_image_modulation_with_runtime_context` (`:673`) = silu→linear→`split_modulation_chunks` (`:3726`, order shift/scale/gate) | Reuse the silu→linear→chunk path. New `NativeTransformerBlockModulation::Flux2` reads the **3 top-level** tensors (`double_stream_modulation_img/txt.lin`, `single_stream_modulation.lin`) once; blocks receive precomputed chunks as args (like Krea's `time_modulation`). Chunk order already matches. |
| `LastLayer` (silu→linear(2·hidden)→LN-no-affine→`(1+scale)·x+shift`) | `output_norm_with_runtime_context` Qwen branch (`:422`–`:465`) | **Near-exact.** ⚠️ **Order trap:** Flux `shift,scale = mod.chunk(2)` (shift first); the Qwen branch reads **scale first** (`:461`–`:463`). Add a Flux2 branch with shift-first split. |
| `time_in` = `timestep_embedding(t,256,factor1000)` → MLPEmbedder(**silu**) | `NativeTransformerTimestepEmbedding::forward_with_runtime_context` (`:518`) — linear1→(silu for Qwen / gelu for Krea)→linear2 | Reuse Qwen (silu) path, in_dim 256. **No `time_mod_proj`** (that's Krea-only). `vec` feeds the 3 Modulations + `final_layer` directly. |
| `EmbedND`/`rope`/`apply_rope` (interleaved 2×2 rotation) | `write_qwen_rope_token` (`:3227`) + `apply_qwen_rotary_embedding` (`:3250`, interleaved pairs) | **Rotation matches exactly.** Deltas: (a) **4 axes** — `qwen_rope_axes_from_transformer_config` (`:3106`) hard-codes `[usize;3]` and rejects `len!=3`; generalize to N. `write_qwen_rope_token` already loops `axes` generically. (b) **Position ids**: Flux img=`[0,h,w,0]`, txt=`[0,0,0,arange(seq)]` (4th axis carries the **text token index**) per `sampling.py:prc_img/prc_txt` — NOT Krea's text-identity (`:3200`). Build the 4-axis grid accordingly. |
| `QKNorm` = RMSNorm(head_dim, **eps 1e-6**) | `maybe_rms_norm_attention_heads_3d` (`:3347`) / `rms_norm_attention_heads_3d` (`:3408`) | Reuse. Note eps: Flux `RMSNorm` uses **1e-6**; Krea block RMSNorms use 1e-5 — pass eps explicitly. |
| `causal_attn_fn` with `num_ref_tokens=0` | `scaled_dot_product_attention_with_runtime_context` (used at `:1661`) | Collapses to plain full SDPA over `[txt,img]`. Reuse; **skip** all ref/KV-cache/causal branches (image-editing only). |
| SiLU-GLU MLP (`SiLUActivation`: `chunk2 → silu(x1)·x2`) | `TransformerFeedForwardActivation::{GeGlu,SwiGlu}` (`:1758`); GeGlu path = fused `[2·inner,hidden]` proj → chunk → act | Add `SiLuGlu` variant reusing the **GeGlu fused-proj** codepath, swapping GELU→SiLU. ⚠️ **Chunk-order trap:** diffusers GEGLU gates the **2nd** chunk; Flux `SiLUActivation` gates the **1st** (`silu(x1)*x2`). Verify which half is gated in the existing GeGlu forward and branch. |
| VAE decode (`autoencoder.py`) | `vae.rs` SD conv-KL decode | z=32 config; add `bn.inv_normalize` (`z·√(var+1e-4)+mean`) + patch[2,2] **unpatchify** (128→32ch) before the conv decoder. |
| Text encoder hidden-state extraction `[9,18,27]` | `NativeQwen3TextEncoder` (`:4007`) + `Qwen3EncoderLayer` (`:3839`) already capture selected mid-layer hidden states for Krea-2 | Reuse; select `[9,18,27]`, concat → 3·hidden. Vanilla klein-4B uses plain Qwen3-4B; SeFi uses Qwen3-VL-2B text tower (both land here). |

Also reused unchanged: `gpu_ops` residents (`linear_optional_bias_resident`,
`scaled_dot_product_attention_resident`, `timestep_embedding_hip_on_gpu`), the
register-tiled WMMA GEMM + W8A8 hot path from Krea-2, `scheduler.rs` flow-match
Euler + dynamic mu-shift, and the resident block-stack driver (`forward_krea_
with_runtime_context` `:2899`–`:2958`) — the Flux2 denoiser follows the same
upload-once / run-stack / download-once pattern.

**Structural note:** unlike Qwen (per-block modulation weights), Flux.2 computes
`vec` and the three modulations **once** at the top of the forward and shares
them across every block — so Flux2 block forwards should take precomputed mod
chunks as arguments (mirroring how `forward_krea_with_runtime_context` passes a
shared `time_modulation`), not recompute per block.

## 3. Arch ids

- `ARCH_ID_FLUX2 = 23` in `crates/hipfire-arch-api/src/lib.rs`, plus
  `docs/architecture-ids.md`. IDs 20 and 21 are already reserved by tooling-only
  DFlash/MTP sidecars and 22 is `ARCH_ID_DSPARK_DRAFT`; do not reuse them. Recheck
  the registry immediately before landing P0 in case another family takes 23.
- Optionally a distinct `ARCH_ID_SEFI` if routing needs to distinguish the
  dual-time / split-denoise runtime; otherwise reuse `ARCH_ID_FLUX2` with a
  metadata flag (`sefi: true`, `semantic_channels`, `delta_t`). **Decision:**
  start with one `ARCH_ID_FLUX2` + metadata flag; split only if the runtime seam
  gets awkward.
- `diffusion_arch_id_for_metadata`: map `Flux2Transformer2DModel` /
  `Flux2KleinPipeline` / `SEFIInferencePipeline` → `ARCH_ID_FLUX2`.

## 4. Canonical artifact names

Per `AGENTS.md` naming:
- `FLUX.2-klein-base-4B.bf16.hfq` (+ `.oq8`/`.oq4++` quant variants later).
- `SeFi-Image-2B-turbo.sefi.bf16.hfq` (feature dot-group `.sefi` for the
  dual-time / split-denoise role; quant token after, e.g. `.sefi--oq8.hfq` or
  `.sefi--oq4.25.hfq`).

## 5. Phasing

**Vanilla FLUX.2 [klein]-base-4B first.**

- **P-1 — Clean synchronized baseline.** Fetch `origin`, create a clean worktree
  from the current `origin/chaingun`, confirm the model/reference inventories,
  and refresh the orientation-only line references in this document. Preserve
  unrelated work in the existing checkout. Record the exact code commit and HF
  snapshot revisions in the first experiment result.
- **P0 — Import + metadata + arch-id.** Recognize FLUX.2 (native single-file
  *and* diffusers layouts), normalize block/qkv naming into the canonical HFQ
  layout, stamp `ARCH_ID_FLUX2`, capture Qwen3-4B encoder ref + shared AE. Unit
  tests mirror the Krea-2 import tests. Fidelity check: `in_channels 128`,
  `context_in_dim 7680`, `axes_dim [32,32,32,32]`, `use_guidance_embed false`.
  Pin the complete Klein snapshot revision above. Normalize the vanilla
  `text_encoder/model.*` keys and SeFi's `Qwen3-VL-2B-Instruct/model.*` /
  `model.language_model.*` keys into one diffusion-native Qwen3 text-tower
  layout; do not route either through the Qwen3.5-VL vision crate.
- **P1 — Flux.2 DiT forward.** `TransformerDenoiserFamily::Flux2`. Concretely, in
  `transformer.rs`:
  - Generalize `qwen_rope_axes_from_transformer_config` (`:3106`) + the grid
    builder to N axes; build the 4-axis `[t,h,w,l]` position ids
    (img `[0,h,w,0]`, txt `[0,0,0,arange]`). Reuse `apply_qwen_rotary_embedding`
    unchanged.
  - Add `NativeTransformerBlockModulation::Flux2` reading the 3 top-level
    `*_stream_modulation.lin` tensors; compute `vec` + chunks once.
  - Add `NativeTransformerBlock::forward_flux2_double` (adapt `forward_qwen`,
    `:2364`: QK-RMSNorm + SiLU-GLU + shared mod) and `forward_flux2_single`
    (adapt `forward_krea`, `:2219`: fused `linear1`, 3-way mod, block-output
    gate, no per-attn sigmoid).
  - Add `TransformerFeedForwardActivation::SiLuGlu` (reuse GeGlu fused-proj path,
    `:1782`; watch the gated-chunk order).
  - Add `NativeTransformerDenoiser::forward_flux2_*` combining both stacks +
    `output_norm` shift-first branch (`:422`).

  Unit-test each helper against a **tiny Flux.2 fixture** (depth=1, single=1,
  synthetic ctx) — no 8 GB encoder needed, mirroring the existing tiny
  QwenImage/Krea fixtures. Numeric parity vs. the vendored `model.py` on fixed
  CPU tensors (per-block, then full forward).
- **P2 — Qwen3-4B hidden-state extraction.** Add a "run selected layers, emit
  hidden states `[9,18,27]` concatenated" mode to the Qwen3 runner; chat-template
  wrap (`add_generation_prompt=True`, `enable_thinking=False`, `max_length 512`).
  Pin whether indices address the embedding output or
  post-layer states with a reference tensor test; do not rely on an assumed
  zero-/one-based convention.
- **P3 — VAE decode + schedule + CFG → end-to-end.** conv-KL decode at z=32 with
  bn inv_normalize + patch [2,2] unpack; mu-shift flow schedule; `denoise` Euler;
  CFG two-pass (`denoise_cfg`, g=4.0). Produce an image from a prompt.
- **P4 — Quant (oq).** Start from the admitted BF16 artifact, then evaluate
  `.oq8`, calibrated `.oq4++`, and only then decimal mixed-Opus candidates.
  Image-generation Opus activations are **W4A8 only**: the legacy W4A4/W4A16
  rungs are not admission candidates because current Krea evidence shows
  unacceptable image-quality loss. Add a Flux2-specific `Ingest` importance /
  `PrecisionClass` policy rather than inheriting the generic MMDiT prior:
  protect input/output/modulation tensors, both ends of the double-stream and
  single-stream stacks, attention residual writers and Q/K/V in descending
  order, and keep FF expansion tensors as the compressible bulk. Mixed-precision
  allocation must compute boundary distance independently for both stacks.

**Then SeFi-Image-2B-turbo delta.**

- **P5 — Dual-timestep embed + split denoise.** `dual_time_embed` (two embedders
  → concat `vec`); 16/128 channel split; dual sigma schedules offset by
  `delta_t`; per-stream Euler; texture-only decode. `semantic_channels`/`delta_t`
  from metadata. Distilled defaults (g=1.0, steps ∈ {4,8,10}, no CFG).
- **P6 — Qwen3-VL-2B text tower.** Reuse/extract the diffusion-native
  `NativeQwen3TextEncoder` / `Qwen3EncoderLayer` seam; do **not** reuse
  `hipfire-arch-qwen35-vl`, which implements the distinct Qwen3.5-VL + SigLIP-2
  architecture. Load only `model.language_model.*` and omit visual weights.
  Match the SeFi reference exactly: processor chat template with
  `add_generation_prompt=true`, `enable_thinking=false`, max length 1024,
  truncation and `padding=max_length`, then concatenate hidden states
  `[9,18,27]` without Krea's `text_fusion` or prefix drop.
- **P7 — SeFi quant + eval.** Repeat the BF16 → OQ8 → calibrated OQ4++ / decimal
  mixed-Opus ladder using the SeFi workload. Stamp calibration signatures,
  compare every candidate to the frozen BF16 baseline, and promote only on a
  passing `hipfire eval` admission battery.

## 6. Validation

- Each phase has an explicit exit gate and stops on failure. Record metric names
  and thresholds before evaluating the first full candidate; do not loosen them
  after observing results.
- **P0 gate:** both native and Diffusers imports round-trip to the same canonical
  tensor roles; configs, tokenizer, complete transformer/text/VAE weights, arch
  id and source revisions are present. No legacy-name fallback is accepted.
- **P1 gate:** fixed-input per-op, per-block and tiny full-forward CPU tensors
  match `third_party/flux2`; compare values and shapes, require finite outputs,
  and report max-absolute/max-relative error. A successful compile is not parity.
- **P2/P6 gate:** token ids, attention masks and all three selected hidden-state
  tensors match the appropriate Transformers reference before concatenation;
  the concatenated conditioning tensor then matches independently.
- **P3 gate:** schedule timesteps/sigmas, every denoise-step latent, VAE
  de-normalized/unpatchified latent and decoded pixels match the pinned Klein
  reference for a fixed prompt/seed. A coherent image is secondary evidence, not
  the correctness gate.
- **P5 gate:** dual semantic/texture timesteps, per-stream sigmas, velocity
  slices, every Euler-updated latent and texture-only decode match
  `third_party/SeFi-Image` for 4, 8 and 10 steps. Failure ends the phase before
  any quant work.
- **P4/P7 gate:** establish the BF16 eval baseline first, then compare each quant
  candidate using the same frozen prompts, seeds, dimensions, step counts and
  thresholds. Finite/coherent output alone is not admission evidence.
- The native `hipfire eval --battery diffusion` admission battery freezes three
  committed object/scene/texture prompts at seeds `7/23/101`, `64x64`, four
  Euler steps, CFG `4.0` for Klein and `1.0` for SeFi. Every candidate image is
  compared to the matching BF16 RGB output and must satisfy both MAE
  `<= 1 / 255` and maximum channel error `<= 4 / 255` for every prompt. The
  device/dimensions/steps can be overridden only for diagnostic runs; promotion
  uses these defaults and records prompt hashes plus both batch timings.
- `./tests/coherence-gate-dflash.sh` after kernel/dispatch-touching changes.
- `./tests/no-gpu-ci.sh` for importer, metadata, CLI and workflow-only changes.
- `hipfire eval` battery for admission before promoting quant variants; keep
  shell gates as enforcement wrappers where they still compare a baseline.
- GPU work coordinated via `hipfire lock {acquire,release,status}`.
- Run `graphify update .` after code changes.

## 7. Parity traps (found while diffing `model.py` ↔ `transformer.rs`)

Concrete numeric-mismatch risks to pin with per-op parity tests:

1. **`final_layer` chunk order** — Flux `shift,scale = mod.chunk(2)` (shift
   first); the existing Qwen `output_norm` branch reads **scale first**
   (`transformer.rs:461`). A Flux2 branch must split shift-first.
2. **SiLU-GLU gated half** — Flux `SiLUActivation` gates the **1st** chunk
   (`silu(x1)*x2`); diffusers GEGLU gates the **2nd**. Confirm the existing
   GeGlu forward's convention before reusing it.
3. **Single-block gating** — Flux gates the **whole block output** by `mod_gate`
   with a plain SDPA; Krea's `attend_krea_self_gated` applies a **sigmoid
   `to_gate` to the attn output**. Use plain SDPA + block-output gate; do not
   inherit Krea's sigmoid gate.
4. **RoPE position ids** — Flux txt tokens rotate on the 4th axis
   (`arange(seq)`); Krea uses text-identity (`:3200`). Wrong ids → subtly wrong
   text conditioning, not an obvious crash.
5. **RMSNorm eps** — Flux QKNorm/RMSNorm use **1e-6**; Krea block norms use 1e-5.
   Pass eps explicitly per family.
6. **Layout normalization** — native (`flux-2-*.safetensors`, fused
   `linear1`/`img_attn.qkv`) vs diffusers (`backbone.*.to_q/add_q/…`,
   `to_qkv_mlp_proj`). Normalize both into one HFQ layout in the importer; pin
   with a round-trip test.
7. **mu-shift constants** — confirm `compute_empirical_mu` (`sampling.py`)
   matches diffusers' `FlowMatchEulerDiscreteScheduler` dynamic-shift for the
   klein `image_seq_len` before trusting the schedule.

Other constraints:
- **gfx1103 LDS hazard** — keep the DiT forward on register-tiled / resident
  (no-LDS) kernels, consistent with the Krea-2 hot path.
- klein-base-4B fits nix1 (64 GB UMA) / halo easily; BFL claims consumer-GPU /
  gfx1103-class viability for klein-4B.

## 8. Implementation record (2026-07-12)

The synchronized implementation has changed the risk assessment in four useful
ways:

1. The Flux.2 DiT is no longer an uncertain assembly task. Tiny BFL parity now
   covers double-stream, single-stream and full-forward paths, and the real
   5+20-block Klein stack stays resident on gfx1103. The fixed one-step 16x16
   smoke dropped from 320.3 s on the initial hybrid path to 166.4 s resident.
2. The tokenizer limit must be a pipeline contract, not inherited blindly from
   Qwen config. The upstream tokenizer advertises a very large generic model
   limit; padding Klein to that value exhausted host RAM. Import and runtime now
   enforce 512 for Klein and 1024 for SeFi.
3. SeFi is confirmed to be the intended thin delta: the complete local artifact
   runs its 144-channel denoiser, independent semantic/texture Euler update and
   texture-only VAE decode. Its dual timestep embedding matches the actual
   checkpoint reference to `5.72e-6` max absolute error.
4. Full-image success remains supporting evidence only. The actual-weight
   P2/P6 selected-state oracles and the P3/P5 step traces now pass; P4/P7 still
   require BF16-first admission evidence. Those gates remain unchanged despite
   successful real-model PNG smokes.
5. The first real BFL denoise trace exposed a layout difference that the tiny
   native fixture could not: BFL stores final adaLN rows as `[shift, scale]`,
   while Diffusers and SeFi store `[scale, shift]`. Canonical HFQ now splits the
   tensor into explicit `norm_out.shift.weight` and `norm_out.scale.weight`
   roles. Ambiguous pre-split Flux.2 artifacts are rejected rather than treated
   as a legacy fallback. Keeping the shared timestep/modulation and final
   adaLN/projection boundaries in f32 then brought the actual one-step velocity
   to max-absolute `0.00842`, NRMSE `0.00241`, and the updated latent to NRMSE
   `0.00911` against the frozen `0.5 / 0.02` limits. The full serialized
   `hipfire-diffusion` suite subsequently passed `293` tests with no failures
   (`5` explicitly ignored), including all local actual-checkpoint gates.
6. The original calibration command forced the CPU-reference runtime and was
   not viable for the 4B workload. Calibration now uses the requested ROCm
   device, observes name-keyed inputs at both decoded and resident linears, and
   downloads resident activations only while the calibration collector is
   armed. This plumbing compiles cleanly; a real Klein calibration artifact and
   non-zero LDLQ packing count are still required before P4 can complete.
7. The first corrected-weight OQ8 pack failed the frozen implementation screen:
   RGB MAE was `1.422 / 255` and maximum channel error was `11 / 255`, above the
   frozen `1 / 255` and `4 / 255` limits, and cold generation took `607.3 s`.
   The pack had quantized all 435 matrix/conv tensors, including the Qwen text
   tower and VAE, instead of applying the Flux2 DiT policy. The gate was not
   weakened and OQ4++ admission did not proceed. General Opus packing now keeps
   text/VAE weights at source precision, quantizes only transformer weights,
   applies the Flux2 precision class, and streams one tensor at a time. The
   component-scoped replacement improved cold generation to `411.9 s` but also
   failed (`1.470 / 255` MAE, `7 / 255` maximum): protected roles were still
   OQ8, which provides no protection inside the OQ8 rung. The next replacement
   therefore keeps `High`/`Pinned`/`SourcePrecision` roles in BF16 and applies
   OQ8 only to `Compressed` DiT bulk. In an OQ4 candidate those protected roles
   become OQ8 while the compressible FF bulk becomes OQ4. That third pack
   quantized only 10 tensors (819 copied), was 15.42 GB / `1.04x`, and passed
   the 16x16 screen at RGB MAE `0.453 / 255`, maximum `1 / 255`, in `152.7 s`.
   This was a fidelity smoke only; the fused single-stream QKV+MLP projection
   prevents useful tensor-granular compression because its protected and
   compressible subranges cannot receive different formats.
8. `hipfire eval` now has a native `diffusion` battery rather than relying on a
   shell-only PNG check. It loads a BF16 baseline and candidate, generates the
   three committed prompt/seed cases as matched batches, compares decoded RGB,
   records prompt hashes/settings/timings, and returns an internal promote or
   reject admission verdict. Its 106-test library suite and the explicit
   diffusion admission verdict test pass. Diagnostic overrides are forbidden
   with `--fail-on-admission`, and step-level progress is reported so a slow
   first forward is distinguishable from VAE/postprocessing cost.
9. The OQ8 candidate is rejected at the P4 admission boundary. A full three-case
   run was stopped after more than 90 minutes when case 2 had not completed its
   first step; a one-case diagnostic then completed both matched four-step
   64x64 generations and produced definitive failure on seed 7: RGB MAE
   `76.006 / 255`, maximum `255 / 255` against the unchanged `1 / 4` limits.
   Evidence (including candidate/baseline hashes and prompt hash) is retained in
   `benchmarks/results/flux2-klein-oq8-admission-diagnostic-2026-07-12/`.
   Because every case must pass, this single failure rejects OQ8. Per the frozen
   ladder, calibrated OQ4++ and P7 do not proceed. Meaningful future quant work
   needs sub-tensor precision for the fused single-stream projection plus a
   materially faster/persistent admission runtime; it is not an adjustment to
   the existing thresholds.
10. Final verification on 2026-07-13 passed `293` diffusion tests with zero
    failures (`5` ignored), `106` hipfire-eval tests, `38` hipfire-evidence
    tests, `108` Python tests, Rust checks, Ruff, mypy, fixture round-trips,
    env/config/model-support freshness, artifact naming and arch-spec purity.
    `coherence-gate-dflash.sh` had zero hard errors: DFlash prose and both
    DDTree cases were OK; DFlash code retained one non-blocking repetition soft
    warning. `no-gpu-ci.sh` reaches only the known pre-existing CLI-doc drift
    (`docs/CLI.md` plus 13 manpages); the task-owned env docs are regenerated
    and current. The post-fast-forward `graphify update .` completed with
    22,685 nodes / 55,680 edges.

Current phase state:

| Phase | State | Current evidence |
|---|---|---|
| P-1 / P0 | complete | synchronized commit and pinned revisions; native/Diffusers/SeFi canonical imports |
| P1 | complete | per-op/block/tiny parity; resident 5+20 stack; actual BFL one-step velocity parity |
| P2 | complete | tokenizer/mask plus actual Qwen3 layers 9/18/27 and concatenation |
| P3 | complete | BFL schedule, velocity/updated-latent gate and actual VAE parity |
| P4 | rejected / stop | protected OQ8 passed the 16x16 screen but failed formal 64x64 admission at MAE 76.006/max 255; OQ4++ not run after failure |
| P5 / P6 | complete | exact 4/8/10 split-loop trace, actual dual embed/text tower, corrected four-step SeFi smoke |
| P7 | not run | frozen sequencing stops after the P4 rejection; SeFi BF16 correctness remains complete but its quant ladder is not admitted |

Quant screening is frozen before inspecting the first OQ8 image: use the same
Klein prompt, seed, 16x16 dimensions, one Euler step and CFG=1 as the resident
BF16 smoke; require RGB pixel MAE <= 1/255 and maximum channel error <= 4/255.
This is a deterministic implementation screen, not the P4 admission battery.
Promotion still requires the BF16-first multi-prompt `hipfire eval` gate above.
