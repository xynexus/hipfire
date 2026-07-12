# FLUX.2 [klein] → SeFi-Image DiT pipeline (plan)

Status: proposed (2026-07-12). Branch: `chaingun`.

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
- `black-forest-labs/FLUX.2-klein-base-4B` (vanilla milestone; downloading).
- `SeFi-Image/SeFi-Image-2B-turbo` (target; diffusers layout).

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
Krea-2 single-stream forwards). Line refs are as of this writing.

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

- `ARCH_ID_FLUX2 = 20` (next free; 17/18 = Krea2/QwenImage, 19 = embeddinggemma)
  in `crates/hipfire-arch-api/src/lib.rs`, plus `docs/architecture-ids.md`.
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
  dual-time / split-denoise role; quant token after, e.g. `.sefi.oq8.hfq`).

## 5. Phasing

**Vanilla FLUX.2 [klein]-base-4B first.**

- **P0 — Import + metadata + arch-id.** Recognize FLUX.2 (native single-file
  *and* diffusers layouts), normalize block/qkv naming into the canonical HFQ
  layout, stamp `ARCH_ID_FLUX2`, capture Qwen3-4B encoder ref + shared AE. Unit
  tests mirror the Krea-2 import tests. Fidelity check: `in_channels 128`,
  `context_in_dim 7680`, `axes_dim [32,32,32,32]`, `use_guidance_embed false`.
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
  wrap (`add_generation_prompt=False`, `max_length 512`, `enable_thinking=False`).
- **P3 — VAE decode + schedule + CFG → end-to-end.** conv-KL decode at z=32 with
  bn inv_normalize + patch [2,2] unpack; mu-shift flow schedule; `denoise` Euler;
  CFG two-pass (`denoise_cfg`, g=4.0). Produce an image from a prompt.
- **P4 — Quant (oq).** DiT is all-linear ⇒ the oq weight ladder + progressive
  per-step precision apply directly (as with Krea-2). `.oq8` then `.oq4++`.

**Then SeFi-Image-2B-turbo delta.**

- **P5 — Dual-timestep embed + split denoise.** `dual_time_embed` (two embedders
  → concat `vec`); 16/128 channel split; dual sigma schedules offset by
  `delta_t`; per-stream Euler; texture-only decode. `semantic_channels`/`delta_t`
  from metadata. Distilled defaults (g=1.0, steps ∈ {4,8,10}, no CFG).
- **P6 — Qwen3-VL-2B text tower.** Reuse `hipfire-arch-qwen35-vl`; text-tower-only
  load (drop visual); same `[9,18,27]` extraction as P2.
- **P7 — SeFi quant + eval.** oq variants; admission evidence.

## 6. Validation

- Per-block CPU parity vs. `third_party/flux2` (P1) and `third_party/SeFi-Image`
  (P5) reference forwards on fixed seeds/tensors.
- `./tests/coherence-gate-dflash.sh` after kernel/dispatch-touching changes.
- End-to-end image sanity: a fixed prompt/seed renders a coherent (non-noise)
  image — the Krea-2 noise-bug regression is the guardrail here.
- `hipfire eval` battery for admission before promoting quant variants.
- GPU work coordinated via `hipfire lock {acquire,release,status}`.

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
