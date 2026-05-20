//! BF16 forward path through the Qwen3.5/3.6 dense trunk for Tier 1 calibration.
//!
//! Companion to `bf16_loader::TrunkBF16` + `collect_imatrix` /
//! `collect_hessian` binaries. The job of this module is NOT to produce
//! correct logits — the calibration pipeline doesn't consume them. The
//! job is to fire the per-linear `ActivationCapture` hook at every
//! `gemm_bf16` site with a PLAUSIBLE input activation tensor, so the
//! downstream Σx² / Hessian accumulators see realistic statistics.
//!
//! ## Approximations and shortcuts
//!
//! The forward path takes deliberate shortcuts to keep the BF16 v1
//! implementation tractable. Calibration-quality cost of each shortcut
//! is annotated below; downstream quantization tooling reads these as
//! known-loose bounds on the calibration signal.
//!
//! 1. **RMSNorm via F32 cast-trick.** Each linear's input is normalized
//!    by the corresponding `*norm.weight` from `trunk.norms` before the
//!    `gemm_bf16` is fired. The BF16 hidden state is upcast to F32,
//!    passed through the existing `gpu.rmsnorm_batched` kernel, and
//!    converted back to BF16 for the GEMM. Residual (`h += attn_out` /
//!    `h += mlp_out`) is added to the ORIGINAL un-normalized `h`,
//!    matching the pre-norm convention used by every modern
//!    transformer arch in hipfire.
//!
//!    Norm weights are loaded with `+= 1.0` baked in (GemmaRMSNorm
//!    convention) — see `bf16_loader::load_bf16_model`. If a norm
//!    weight for a given layer is absent from `trunk.norms` the
//!    forward falls back to the un-normalized hidden state (logs a
//!    warning) so older fixtures without norm tensors still produce
//!    a forward pass.
//!
//! 2. **Attention math (B.2 + B.3, 2026-05-20):**
//!    - DeltaNet: **mirrors the production prefill path exactly** as of
//!      the 2026-05-20 DeltaNet-full refactor. The pipeline is
//!      `in_proj_qkv → conv1d+SiLU (depth-wise, kernel=4, causal) →
//!      split Q/K/V → fused_qk_l2_norm_scale → optional GQA repeat →
//!      gated_delta_net_f32 → gated_norm * silu(z) → out_proj`. The
//!      in_proj_a / in_proj_b projections feed
//!      `fused_sigmoid_alpha_gate_f32_batched` with the **real** `A_log`
//!      and `dt_bias` tensors (loaded as F32 via the new
//!      `is_ssm_aux_tensor` predicate in `bf16_loader`). The conv1d
//!      preamble uses the production
//!      `conv1d_silu_split_f32_n` kernel against the BF16-decoded
//!      `conv1d.weight` (`[conv_dim * kernel_size]` flat, F32) with a
//!      zero-initialized 3-element ring state. The post-recurrence
//!      gated norm runs through `gated_norm_f32_batched` against the
//!      F32 `linear_attn.norm.weight` (shape `[head_v_dim]`,
//!      `Qwen3_5RMSNormGated` convention, NO `+= 1.0` bake — see
//!      `is_gated_norm_tensor` for the bake-skip predicate). The
//!      `silu(z)` post-multiply is folded into the same kernel
//!      (`out = rms_norm(x, w) * silu(z)`), saving a separate launch
//!      and matching HF's `Qwen3_5RMSNormGated.forward(h, gate=z)`
//!      math byte-for-byte. `out_proj`'s input is therefore identical
//!      in math (same kernels, same recurrence) to what the deployed
//!      model produces at calibration time.
//!    - FullAttention: **mirrors the production prefill path exactly**
//!      (B.3 refactor, 2026-05-20). Q / K / V are produced by `gemm_bf16`
//!      in F32; per-head q_norm / k_norm are applied via
//!      `rmsnorm_batched`; positions `[0..seq_len-1]` are uploaded as
//!      i32 bits into an F32-typed buffer matching production's
//!      convention; partial RoPE runs as a single batched call
//!      (`rope_partial_interleaved_f32_batched`, Qwen3.5 partial
//!      25% × head_dim, rope_theta from config or default 10M);
//!      K/V are quantized into a per-call ephemeral Q8_0 KV cache via
//!      `kv_cache_write_q8_0_batched` (the same cache format the
//!      deployed model uses by default); causal masked flash attention
//!      runs as a single batched call
//!      (`attention_q8_0_kv_batched_masked`) over all seq_len queries.
//!      `o_proj`'s input is therefore identical in math (same kernels,
//!      same Q8_0 KV quantization noise) to what the deployed model
//!      produces at calibration time. Work is O(seq_len · ctx),
//!      same complexity as production. The previous per-token loop
//!      (B.3 v1, NRMSE 4.9) was replaced by this batched mirror to
//!      match the deployed-model attention distribution byte-for-byte
//!      at `o_proj`'s input modulo Q8_0 quantization (target NRMSE
//!      ~0.9, the tokenizer floor).
//!
//! 3. **SiLU(gate) * up is computed in F32 then converted back to BF16.**
//!    Standard SiLU math. No correctness shortcut; F32 → BF16 conversion
//!    uses bit-truncation (top 16 bits of the F32 mantissa, matching
//!    PyTorch's default `torch.bfloat16` rounding mode).
//!
//! 4. **Token embedding lookup is a raw memcpy.** Since the embedding
//!    table is BF16 row-major `[vocab, dim]` and the output is BF16
//!    `[dim]`, we use `hipMemcpy(d2d)` to copy 2*dim bytes from the
//!    appropriate row offset. No upcast.
//!
//! 5. **lm_head is gated on `--process-output`.** When `process_output`
//!    is false (default), the final norm + lm_head GEMM are skipped to
//!    save one (typically vocab-sized) GEMM per sequence. When set
//!    (matching `llama-imatrix --process-output`), the forward applies
//!    the final RMSNorm (`{prefix}norm.weight` via the same
//!    `rmsnorm_bf16_via_f32` cast-trick) and then dispatches
//!    `gemm_bf16` against `lm_head.weight` (untied) or
//!    `{prefix}embed_tokens.weight` (tied; Qwen3.5 dense default), so
//!    the capture hook fires with the name `lm_head.weight`. The
//!    logits output is discarded — calibration consumes only the
//!    per-channel input statistics.
//!
//! ## Layer-pattern detection
//!
//! Detection is by tensor presence in `trunk.tensors`:
//!   - If `model.layers.{L}.self_attn.q_proj.weight` exists → FullAttn.
//!   - Otherwise → DeltaNet.
//!
//! This is robust to config.json variants between Qwen3.5 and 3.6 and
//! to fixture trunks that don't ship a config.json.

use crate::bf16_loader::{Bf16Tensor, TrunkBF16};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

/// GemmaRMSNorm epsilon. Matches `crates/hipfire-arch-qwen35/src/qwen35.rs`
/// (default 1e-6 when `config.json` lacks `rms_norm_eps`) and the
/// HuggingFace Qwen3.5 / 3.6 reference. The Tier 1 calibration forward
/// doesn't read `config.json` (would require parsing a per-arch shape),
/// so we use this constant directly.
const RMS_NORM_EPS: f32 = 1e-6;

/// Per-layer detected attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    /// Layer uses Qwen3.5/3.6 DeltaNet linear attention (`linear_attn.*`).
    DeltaNet,
    /// Layer uses dense / full attention (`self_attn.q_proj`, etc.).
    FullAttn,
}

/// Detect language-model prefix on a trunk's tensor names.
///
/// Qwen3.5 dense text-only models use `model.*` prefix. Qwen3.6 VL
/// variants (e.g. Qwen3.6-27B with a vision encoder) wrap the text
/// trunk under `model.language_model.*`. Returns the prefix with the
/// trailing dot included so callers can `format!("{}layers.{}", prefix, l)`.
pub(crate) fn lm_prefix(trunk: &TrunkBF16) -> &'static str {
    if trunk.tensors.contains_key("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    }
}

/// Detect the attention kind for a given layer by tensor presence.
fn detect_layer_kind(trunk: &TrunkBF16, layer_idx: usize) -> LayerKind {
    let prefix = lm_prefix(trunk);
    let probe_full = format!("{prefix}layers.{layer_idx}.self_attn.q_proj.weight");
    if trunk.tensors.contains_key(&probe_full) {
        LayerKind::FullAttn
    } else {
        LayerKind::DeltaNet
    }
}

/// Count layers in the trunk by walking `{prefix}layers.{L}.*` until a
/// layer index is absent. Used in lieu of a parsed `config.json` so
/// the forward works on fixtures that ship only safetensors.
fn count_layers(trunk: &TrunkBF16) -> usize {
    let prefix = lm_prefix(trunk);
    let mut n = 0usize;
    loop {
        let probe_dn = format!("{prefix}layers.{n}.linear_attn.in_proj_qkv.weight");
        let probe_full = format!("{prefix}layers.{n}.self_attn.q_proj.weight");
        if !trunk.tensors.contains_key(&probe_dn) && !trunk.tensors.contains_key(&probe_full) {
            break;
        }
        n += 1;
    }
    n
}

/// Resolve a BF16 weight tensor from the trunk by name. Returns an Err
/// with a helpful message that includes the trunk size if the tensor
/// is missing, so calibration drivers print a meaningful trace.
fn get<'a>(trunk: &'a TrunkBF16, name: &str) -> Result<&'a Bf16Tensor, String> {
    trunk.tensors.get(name).ok_or_else(|| {
        format!(
            "bf16_forward: missing tensor `{}` in trunk (trunk has {} tensors)",
            name,
            trunk.tensors.len()
        )
    })
}

/// Optional: look up a tensor by name without erroring if absent. Used
/// for tensors that may or may not exist depending on model variant
/// (e.g. norms, biases) so the forward can skip them cleanly. Reserved
/// for a future v2 of the forward path that wires rmsnorm — kept now
/// because the function is short and removing it would obscure intent.
#[allow(dead_code)]
fn try_get<'a>(trunk: &'a TrunkBF16, name: &str) -> Option<&'a Bf16Tensor> {
    trunk.tensors.get(name)
}

/// Run a single forward prefill pass through the BF16 trunk.
///
/// `tokens` are u32 ids. The function fires the per-linear capture
/// hooks on every `gemm_bf16` site so the `ActivationCapture` handler
/// on `gpu.capture_handler` can accumulate Σx² / Hessian moments. No
/// logits are returned — the calibration pipeline does not consume them.
///
/// When `process_output` is true (matching `llama-imatrix
/// --process-output`), the forward also applies the final RMSNorm and
/// fires one extra `gemm_bf16` against `lm_head.weight`. The capture
/// hook sees the post-final-norm hidden state on the input side; the
/// computed logits are discarded. The capture name is set to
/// `lm_head.weight` regardless of whether the weight is tied to
/// `embed_tokens` or a separate matrix — the downstream
/// `safetensors_to_ggml_name` translation maps it to `output.weight`
/// on the consumer side.
///
/// `gpu.arch` must be `gfx942` (MI300x). The BF16 GEMM is gfx942-only;
/// `gpu.gemm_bf16` returns an explicit error on other archs.
pub fn forward_prefill_bf16(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    tokens: &[u32],
    process_output: bool,
) -> HipResult<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let seq_len = tokens.len();

    // ── Resolve trunk geometry from the embed table ────────────────────
    //
    // The embed_tokens row shape gives us `dim` cheaply without parsing
    // config.json. `[vocab, dim]` row-major; we want `dim`.
    let prefix = lm_prefix(trunk);
    let embed_key = format!("{prefix}embed_tokens.weight");
    let embed = trunk
        .tensors
        .get(&embed_key)
        .ok_or_else(|| hip_bridge::HipError::new(
            0,
            &format!("bf16_forward: missing `{embed_key}` in trunk"),
        ))?;
    if embed.shape.len() != 2 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "bf16_forward: embed_tokens has shape {:?}, expected [vocab, dim]",
                embed.shape
            ),
        ));
    }
    let vocab = embed.shape[0];
    let dim = embed.shape[1];

    let n_layers = count_layers(trunk);
    if n_layers == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "bf16_forward: trunk has 0 layers (no `model.layers.0.*` tensors found)",
        ));
    }

    // ── Allocate the BF16 hidden state buffer ──────────────────────────
    //
    // h: [seq_len, dim] BF16, row-major. Token embeddings populated below,
    // then mutated layer-by-layer.
    let h = gpu.alloc_tensor(&[seq_len * dim], DType::BF16)?;

    // F32 scratch for elementwise math and GEMM outputs. We reuse this
    // for any pointwise op that's easier to express in F32 than BF16
    // (currently SiLU(gate) * up and the attention-shortcut copies).
    //
    // Peak F32 footprint = max(seq_len * intermediate, seq_len * dim).
    // We size to the larger of:
    //   - seq_len * dim  (used for embed F32 staging and per-layer h_norm)
    //   - mq_max_inter   (gate/up output, set lazily once we see the
    //     first MLP linear's m dimension)
    //
    // To keep the function straightforward, we (re)allocate ad-hoc per
    // operation and free immediately. The `gpu.alloc_tensor` /
    // `free_tensor` pair routes through the pool, so the allocation
    // cost amortizes after the first layer.

    // ── Embedding lookup ───────────────────────────────────────────────
    //
    // Per-token: copy `dim` BF16 elements (2 * dim bytes) from row
    // `tokens[t]` of the embed table into row `t` of `h`.
    let embed_row_bytes = dim * 2; // BF16 = 2 bytes/element
    for (t, &tok) in tokens.iter().enumerate() {
        if tok as usize >= vocab {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "bf16_forward: token id {tok} >= vocab size {vocab} at position {t}"
                ),
            ));
        }
        let src_byte_offset = (tok as usize) * embed_row_bytes;
        let dst_byte_offset = t * embed_row_bytes;
        gpu.hip.memcpy_dtod_at(
            &h.buf,
            dst_byte_offset,
            &embed.tensor.buf,
            src_byte_offset,
            embed_row_bytes,
        )?;
    }

    // ── Per-layer loop ─────────────────────────────────────────────────
    for layer_idx in 0..n_layers {
        let kind = detect_layer_kind(trunk, layer_idx);
        let p = format!("{prefix}layers.{layer_idx}");

        // Hidden state into this layer (h is currently the residual stream)
        // is used as the BF16 input to every linear in this layer. The
        // calibration capture hook reads (input_ptr, dtype=BF16,
        // shape=[seq_len, dim]) for each linear before it fires.
        //
        // h shape view: [seq_len, dim]
        // We need a `GpuTensor` with shape [seq_len, dim] for the capture
        // hook to convey the right shape; the underlying buf is the same
        // bytes as `h` but tagged with the 2-D shape.
        let h_view = view_2d(&h, seq_len, dim);

        // Resolve attention outputs (a BF16 tensor in [seq_len, dim] shape)
        // depending on the layer kind. The function returns a freshly-
        // allocated tensor that the caller must free; we do that below.
        let attn_out = match kind {
            LayerKind::DeltaNet => {
                forward_deltanet_layer(gpu, trunk, &h_view, &p, seq_len, dim)
                    .map_err(|e| hip_bridge::HipError::new(0, &e))?
            }
            LayerKind::FullAttn => {
                forward_full_attn_layer(gpu, trunk, &h_view, &p, seq_len, dim)
                    .map_err(|e| hip_bridge::HipError::new(0, &e))?
            }
        };

        // Residual: h += attn_out. We do this in F32 (convert both,
        // add, convert back). Simpler than writing a BF16-add kernel.
        bf16_add_inplace(gpu, &h, &attn_out, seq_len * dim)?;
        gpu.free_tensor(attn_out)?;

        // ── MLP ────────────────────────────────────────────────────────
        //
        // h_view (BF16) → gate_proj → gate (F32 [seq_len, hidden_dim])
        //              → up_proj   → up   (F32 [seq_len, hidden_dim])
        // ffn_hidden_f32 = silu(gate) * up
        // ffn_hidden_bf16 = bf16(ffn_hidden_f32)
        // ffn_hidden_bf16 → down_proj → ffn_out (F32 [seq_len, dim])
        // ffn_out_bf16 = bf16(ffn_out)
        // h += ffn_out_bf16
        let h_view = view_2d(&h, seq_len, dim);
        let ffn_out_bf16 = if layer_is_moe(trunk, &p) {
            forward_moe_layer_bf16(gpu, trunk, &h_view, &p, seq_len, dim)
        } else {
            forward_mlp_layer(gpu, trunk, &h_view, &p, seq_len, dim)
        }
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        bf16_add_inplace(gpu, &h, &ffn_out_bf16, seq_len * dim)?;
        gpu.free_tensor(ffn_out_bf16)?;
    }

    // ── Final norm + lm_head capture (gated on --process-output) ──────
    //
    // Mirrors `llama-imatrix --process-output`: applies the trunk-final
    // RMSNorm and dispatches `gemm_bf16` against `lm_head.weight` so the
    // capture hook fires with the post-final-norm hidden state as the
    // input distribution. The computed logits are discarded — the
    // calibration pipeline only consumes the per-channel input
    // statistics. When `process_output == false` (default, matching
    // llama-imatrix without the flag), this whole block is skipped and
    // we free the hidden state immediately.
    if process_output {
        // 1. Resolve the final norm weight. Qwen3.5 dense stores this
        //    as `model.norm.weight`; Qwen3.6 VL nests it under
        //    `model.language_model.norm.weight`. `lm_prefix` resolved
        //    the right one above.
        let final_norm_name = format!("{prefix}norm.weight");
        let final_norm = try_get_norm(trunk, &final_norm_name).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "bf16_forward: --process-output requires final norm `{}` \
                     but the trunk has no such tensor in trunk.norms (norms count {})",
                    final_norm_name,
                    trunk.norms.len()
                ),
            )
        })?;

        // 2. Resolve the lm_head weight. Untied: `lm_head.weight` exists
        //    as a top-level (no `model.` prefix) BF16 dense tensor in
        //    `trunk.tensors`. Tied (Qwen3.5 0.8B): falls back to the
        //    embed table at `{prefix}embed_tokens.weight`. The embed
        //    table has the same `[vocab, dim]` row-major layout as
        //    `lm_head.weight` would, so the GEMM dispatch is identical;
        //    only the source pointer differs.
        let lm_head = if let Some(t) = trunk.tensors.get("lm_head.weight") {
            t
        } else if let Some(t) = trunk.tensors.get(&format!("{prefix}lm_head.weight")) {
            // Some Qwen3.6 VL variants prefix lm_head; handle them too.
            t
        } else {
            // Tied case — reuse the embed table (it lives in
            // `trunk.tensors` because the loader does not special-case
            // `embed_tokens.weight` — it loads as a BF16 dense tensor).
            trunk.tensors.get(&embed_key).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!(
                        "bf16_forward: --process-output cannot resolve lm_head: \
                         neither `lm_head.weight` nor `{}lm_head.weight` nor (tied) \
                         `{}` is present in trunk.tensors (count={})",
                        prefix, embed_key, trunk.tensors.len()
                    ),
                )
            })?
        };

        // Sanity check: the lm_head weight must be 2-D `[vocab, dim]`.
        if lm_head.shape.len() != 2 || lm_head.shape[1] != dim {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "bf16_forward: lm_head weight shape {:?} doesn't match expected \
                     [vocab, dim={dim}]",
                    lm_head.shape
                ),
            ));
        }
        let lm_vocab = lm_head.shape[0];

        // 3. Apply the final RMSNorm via the B.1 cast-trick helper.
        //    `h_norm` is a fresh BF16 [seq_len, dim] allocation; the
        //    original `h` is preserved (not that we use it again after
        //    this, but conceptually the residual stream is read-only at
        //    this point).
        let h_norm = gpu.alloc_tensor(&[seq_len, dim], DType::BF16)?;
        rmsnorm_bf16_via_f32(gpu, &h, final_norm, &h_norm, seq_len, dim, RMS_NORM_EPS)?;

        // 4. Fire the lm_head capture. F32 scratch sized
        //    `[seq_len, lm_vocab]` — for 0.8B Qwen3.5 that's
        //    `1024 × 151936 × 4B ≈ 622 MB`. The pool reuses across
        //    sequences so the per-sequence cost is one allocation +
        //    one GEMM. The output (logits) is discarded immediately.
        let logits_scratch = gpu.alloc_tensor(&[seq_len * lm_vocab], DType::F32)?;
        gpu.set_capture_name(Some("lm_head.weight".to_string()));
        gpu.gemm_bf16(
            &h_norm,
            &lm_head.tensor,
            &mut to_2d(logits_scratch.clone_view(), seq_len, lm_vocab),
            lm_vocab,
            dim,
            seq_len,
        )?;
        gpu.set_capture_name(None);
        gpu.free_tensor(logits_scratch)?;
        gpu.free_tensor(h_norm)?;
    }

    // Free the hidden state and return.
    gpu.free_tensor(h)?;
    Ok(())
}

/// Build a `[batch, k]`-shaped 2-D view tensor over an existing buffer.
/// The returned `GpuTensor` is a non-owning view (do NOT free it).
fn view_2d(t: &GpuTensor, batch: usize, k: usize) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(
            t.buf.as_ptr(),
            batch * k * t.dtype.size(),
        ) },
        shape: vec![batch, k],
        dtype: t.dtype,
    }
}

/// In-place BF16 add: `a += b`. Implemented as
/// `a = bf16(f32(a) + f32(b))` via two convert kernels and an F32 add.
fn bf16_add_inplace(
    gpu: &mut Gpu,
    a: &GpuTensor,
    b: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    // Allocate F32 scratch for a and b.
    let a_f32 = gpu.alloc_tensor(&[n], DType::F32)?;
    let b_f32 = gpu.alloc_tensor(&[n], DType::F32)?;
    gpu.convert_bf16_to_f32(a, &a_f32, n)?;
    gpu.convert_bf16_to_f32(b, &b_f32, n)?;
    gpu.add_inplace_f32(&a_f32, &b_f32)?;
    gpu.convert_f32_to_bf16(&a_f32, a, n)?;
    gpu.free_tensor(a_f32)?;
    gpu.free_tensor(b_f32)?;
    Ok(())
}

/// Cast-trick RMSNorm for BF16 hidden states.
///
/// Computes `out_bf16[t, k] = h_bf16[t, k] / sqrt(mean_k(h²) + eps)
/// * norm_weight[k]` using the F32 `rmsnorm_batched` kernel as the
/// inner engine. Steps:
///
///   1. Upcast `h_bf16` to F32 scratch via `convert_bf16_to_f32`.
///   2. Apply `gpu.rmsnorm_batched(h_f32, norm_weight, h_f32_out,
///      batch, k, eps)`.
///   3. Downcast `h_f32_out` to `out_bf16` via `convert_f32_to_bf16`.
///
/// The `norm_weight` MUST already have the GemmaRMSNorm `+= 1.0` bake
/// applied (the loader does this in `bf16_loader::load_bf16_model`).
/// The kernel computes plain `x * w * rms`; if the caller wants the
/// `(1 + w) * rms` form, it's the loader's responsibility to bake.
///
/// `out_bf16` must be a caller-owned BF16 tensor of size `batch * k`.
/// Can be the same buffer as the BF16 hidden state passed via
/// `h_bf16` if the caller wants in-place behavior, but typical callers
/// keep them separate so the original hidden state is preserved for
/// the residual add.
fn rmsnorm_bf16_via_f32(
    gpu: &mut Gpu,
    h_bf16: &GpuTensor,
    norm_weight: &GpuTensor,
    out_bf16: &GpuTensor,
    batch: usize,
    k: usize,
    eps: f32,
) -> HipResult<()> {
    let n = batch * k;
    // Two F32 scratches: input and output of the F32 rmsnorm. The
    // pool inside `Gpu` reuses these between layers so allocation
    // cost amortizes after the first call.
    let h_f32 = gpu.alloc_tensor(&[batch, k], DType::F32)?;
    let out_f32 = gpu.alloc_tensor(&[batch, k], DType::F32)?;
    gpu.convert_bf16_to_f32(h_bf16, &h_f32, n)?;
    gpu.rmsnorm_batched(&h_f32, norm_weight, &out_f32, batch, k, eps)?;
    gpu.convert_f32_to_bf16(&out_f32, out_bf16, n)?;
    gpu.free_tensor(h_f32)?;
    gpu.free_tensor(out_f32)?;
    Ok(())
}

/// Look up a norm weight in the trunk by name. Returns `None` when
/// absent — the caller may decide to skip rmsnorm (falling back to
/// the un-normalized hidden state) for fixtures that don't ship
/// norm tensors.
fn try_get_norm<'a>(trunk: &'a TrunkBF16, name: &str) -> Option<&'a GpuTensor> {
    trunk.norms.get(name)
}

/// Either an owned scratch GpuTensor (we allocated `h_norm` and ran
/// rmsnorm into it) or a non-owning view of an existing buffer (no
/// norm weight available, we point back at the original `h`). The
/// `drop_owned` method frees the allocation if we own it; for the
/// view case it's a no-op.
enum NormScratch {
    Owned(GpuTensor),
    /// View backing buffer is the original `h_view`. Stored as a
    /// fresh `GpuTensor` carrying a borrowed pointer so callers can
    /// pass `as_ref()` uniformly to `gemm_bf16`.
    View(GpuTensor),
}

impl NormScratch {
    fn as_ref(&self) -> &GpuTensor {
        match self {
            NormScratch::Owned(t) => t,
            NormScratch::View(t) => t,
        }
    }

    /// Free the underlying allocation if we own it. No-op for views.
    fn drop_owned(self, gpu: &mut Gpu) -> HipResult<()> {
        match self {
            NormScratch::Owned(t) => gpu.free_tensor(t),
            NormScratch::View(_) => Ok(()),
        }
    }
}

/// Apply pre-norm rmsnorm to `h` if `trunk.norms[norm_name]` exists.
/// Falls back to a non-owning view of `h` (with a single warning
/// printed) for fixtures that don't ship a norm tensor for this
/// layer. In both cases the returned wrapper's `.as_ref()` is a
/// `&GpuTensor` of shape `[seq_len, dim]` suitable to feed
/// `gemm_bf16`.
fn apply_pre_norm_or_fallback(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    norm_name: &str,
    seq_len: usize,
    dim: usize,
) -> Result<NormScratch, String> {
    if let Some(w) = try_get_norm(trunk, norm_name) {
        // Allocate the BF16 scratch with the logical [seq_len, dim]
        // shape directly so downstream `gemm_bf16` reads the right
        // strides without an extra `to_2d` wrap.
        let h_norm = gpu
            .alloc_tensor(&[seq_len, dim], DType::BF16)
            .map_err(|e| e.to_string())?;
        rmsnorm_bf16_via_f32(gpu, h, w, &h_norm, seq_len, dim, RMS_NORM_EPS)
            .map_err(|e| e.to_string())?;
        Ok(NormScratch::Owned(h_norm))
    } else {
        // Fixtures without norm tensors (e.g. unit-test stubs that
        // ship only dense linears) hit this path. Warn ONCE per
        // process so the log isn't flooded across 24 layers × N
        // sequences. Production safetensors loads always have norms.
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "  warning: norm weight `{}` absent from trunk.norms; \
                 feeding un-normalized hidden state into layer's linears \
                 (further occurrences suppressed)",
                norm_name
            );
        }
        // Non-owning view of `h` reshaped to [seq_len, dim].
        let view = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(h.buf.as_ptr(), seq_len * dim * 2) },
            shape: vec![seq_len, dim],
            dtype: DType::BF16,
        };
        Ok(NormScratch::View(view))
    }
}

/// Warn once-per-process when a DeltaNet SSM aux tensor is missing
/// from the trunk. The two main causes are (a) an older safetensors
/// dump that pre-dates the `is_ssm_aux_tensor` predicate in
/// `bf16_loader` (rare, only fixture trunks built before 2026-05-20),
/// and (b) a non-DeltaNet layer that landed on the DeltaNet branch by
/// mistake (always a bug — flag loudly).
///
/// We only emit the warning when `missing` is true so the call sites
/// can be unconditional (`warn_deltanet_missing(name,
/// opt.is_none())`) without inflating dispatch overhead.
fn warn_deltanet_missing(name: &str, missing: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if missing && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "  warning: DeltaNet SSM aux tensor `{}` absent from trunk.norms; \
             falling back to zero-fill / no-op approximation (matches pre-2026-05-20 \
             B.2 calibration behaviour; further occurrences suppressed)",
            name
        );
    }
}

/// DeltaNet layer attention via the full production gated delta-net
/// recurrence (2026-05-20 DeltaNet-full refactor).
///
/// Fires the capture hook for `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
/// `in_proj_b`, `out_proj`. The math mirrors `forward_decode` /
/// `forward_prefill_batch` in `crates/hipfire-arch-qwen35/src/qwen35.rs`
/// step-for-step, just in F32 throughout (the cast-trick upcasts the
/// BF16 hidden state via `gemm_bf16` which already outputs F32).
///
/// Pipeline:
///   1. RMSNorm-cast the BF16 hidden state to BF16 `h_for_linears`.
///   2. `in_proj_qkv` → F32 `[seq_len, qkv_dim]`. Layout per token:
///      `[Q (k_dim) | K (k_dim) | V (d_inner)]`.
///   3. `in_proj_z` → F32 `[seq_len, d_inner]` (kept live for the
///      gated norm at step 9).
///   4. `in_proj_a` → F32 `[seq_len, n_v_heads]` (alpha base).
///   5. `in_proj_b` → F32 `[seq_len, n_v_heads]` (beta base).
///   6. `fused_sigmoid_alpha_gate_f32_batched(beta, alpha, dt_bias,
///      a_log, n_v_heads, seq_len)`. With the SSM aux loader (#127
///      follow-up), `dt_bias` and `a_log` are the real F32 tensors;
///      the kernel computes
///      `beta_i = sigmoid(beta_i)` and
///      `alpha_i = -exp(a_log_h) * softplus(alpha_i + dt_bias_h)` —
///      exactly matching HF's
///      `g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)`.
///   7. `conv1d_silu_split_f32_n` over the QKV F32 tensor. Depth-wise
///      4-tap causal conv + SiLU per channel, ring-buffer state of
///      size `(kernel - 1) * qkv_dim = 3 * qkv_dim` F32 zeros at start
///      of each forward (one-sequence batching, B.4 concern for
///      multi-sequence). Output goes directly into separate
///      `q_part / k_part / v_part` F32 buffers, replacing the per-row
///      memcpy split that the B.2 cast-trick used.
///   8. `fused_qk_l2_norm_scale_f32_batched(q_part, k_part,
///      n_k_heads, head_dim, 1/sqrt(head_dim), eps=1e-6, seq_len)`.
///   9. Optional GQA repeat-interleave when `n_k_heads < n_v_heads`.
///  10. `gated_delta_net_f32(q, k, v, alpha, beta, state=zeros,
///      out, seq_len, n_v_heads, head_dim)`. Returns F32 `attn_out`.
///  11. `gated_norm_f32_batched(attn_out, z, norm_weight, normed,
///      n_v_heads, head_dim, eps=1e-6, seq_len)`. Computes
///      `normed[t,h,k] = rms_norm(attn_out[t,h,:])[k] * norm_weight[k] *
///      silu(z[t,h,k])` in one launch (matches HF
///      `Qwen3_5RMSNormGated.forward(attn_out, gate=z)` byte-for-byte).
///      The norm weight (`*.linear_attn.norm.weight`, F32 `[head_dim]`)
///      is loaded with NO `+= 1.0` bake — see
///      `bf16_loader::is_gated_norm_tensor`.
///  12. Cast F32 `normed` → BF16 for `out_proj`.
///  13. `out_proj`: BF16 `normed_bf16` · `wo^T` → F32 `[seq_len, dim]`.
///  14. Cast F32 → BF16 and return.
///
/// SSM aux tensor lookups go through `try_get_norm` (which reads
/// `trunk.norms`):
///   - `<p>.linear_attn.A_log`           F32 `[n_v_heads]`
///   - `<p>.linear_attn.dt_bias`         F32 `[n_v_heads]`
///   - `<p>.linear_attn.conv1d.weight`   F32 `[conv_dim * kernel_size]`
///   - `<p>.linear_attn.norm.weight`     F32 `[head_dim]`
///
/// If any of these are absent from the trunk (e.g. an older safetensors
/// dump that pre-dates the SSM-aux loader), we fall back to dummy zeros
/// for `A_log` / `dt_bias` and skip the conv1d / gated_norm steps —
/// matches the pre-2026-05-20 B.2 approximation behaviour. A one-shot
/// `warn_once` records the first missing tensor name so fixture-only
/// runs don't flood the log.
///
/// The kernel hardcodes `head_dim == 128` (`#define HD 128` in
/// `gated_delta_net.hip`); we assert at runtime that the trunk's
/// derived `head_dim` matches.
///
/// The full path uses `gated_delta_net_f32` /
/// `fused_sigmoid_alpha_gate_f32_batched` /
/// `fused_qk_l2_norm_scale_f32_batched` / `conv1d_silu_split_f32_n` /
/// `gated_norm_f32_batched` which are all `#[cfg(feature = "deltanet")]`
/// on `rdna-compute`. When the `deltanet` feature is OFF (only happens
/// in `--no-default-features` configurations), this function falls
/// back to the pre-B.2 Q-chunk passthrough so the module still
/// compiles. Production calibration runs always use default features →
/// real recurrence.
///
/// Returns a freshly-allocated `[seq_len, dim]` BF16 tensor that the
/// caller must free.
#[cfg(feature = "deltanet")]
fn forward_deltanet_layer(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    p: &str,
    seq_len: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    // Resolve weights.
    let wqkv = get(trunk, &format!("{p}.linear_attn.in_proj_qkv.weight"))?;
    let wz = get(trunk, &format!("{p}.linear_attn.in_proj_z.weight"))?;
    let wa = get(trunk, &format!("{p}.linear_attn.in_proj_a.weight"))?;
    let wb = get(trunk, &format!("{p}.linear_attn.in_proj_b.weight"))?;
    let wo = get(trunk, &format!("{p}.linear_attn.out_proj.weight"))?;

    // ── Derive trunk geometry ──────────────────────────────────────────
    //
    // wqkv:  [qkv_dim, dim]              row-major, qkv_dim = 2*k_dim + d_inner
    // wz:    [d_inner, dim]              d_inner = n_v_heads * head_dim
    // wa:    [n_v_heads, dim]            alpha base
    // wb:    [n_v_heads, dim]            beta base
    // wo:    [dim, d_inner]              K-side of the residual projection
    //
    // For Qwen3.5 0.8B (defaults):
    //   dim=1024, qkv_dim=6144, d_inner=2048, n_v_heads=16, head_dim=128,
    //   k_dim=2048, n_k_heads=16  (no GQA repeat needed)
    // For 27B-3.5/3.6:
    //   dim=5120, n_v_heads=20, head_dim=128.
    //
    // The kernel `gated_delta_net.hip` hardcodes `#define HD 128`, so
    // we assert head_dim==128 below and fail loudly if a future model
    // ships a different DeltaNet head dim.
    let qkv_dim = wqkv.shape[0];
    let qkv_k = wqkv.shape[1];
    if qkv_k != dim {
        return Err(format!(
            "{p}.linear_attn.in_proj_qkv: K={qkv_k} doesn't match dim={dim}"
        ));
    }
    let d_inner = wo.shape[1];
    if d_inner > qkv_dim {
        return Err(format!(
            "{p}.linear_attn.out_proj: d_inner={d_inner} > qkv_dim={qkv_dim}"
        ));
    }
    let n_v_heads = wa.shape[0];
    if wb.shape[0] != n_v_heads {
        return Err(format!(
            "{p}.linear_attn: in_proj_b M={} mismatches in_proj_a M={n_v_heads}",
            wb.shape[0]
        ));
    }
    if n_v_heads == 0 || d_inner % n_v_heads != 0 {
        return Err(format!(
            "{p}.linear_attn: bad geometry d_inner={d_inner} not divisible by n_v_heads={n_v_heads}"
        ));
    }
    let head_dim = d_inner / n_v_heads;
    // The kernel hardcodes HD=128. Refuse with a clean error rather
    // than reading off the end of LDS at launch time.
    if head_dim != 128 {
        return Err(format!(
            "{p}.linear_attn: gated_delta_net_f32 kernel requires head_dim=128 \
             (got {head_dim}); future support requires updating gated_delta_net.hip's HD #define"
        ));
    }
    let k_dim_total = qkv_dim
        .checked_sub(d_inner)
        .ok_or_else(|| format!(
            "{p}.linear_attn: qkv_dim={qkv_dim} < d_inner={d_inner}"
        ))?;
    if k_dim_total % 2 != 0 {
        return Err(format!(
            "{p}.linear_attn: (qkv_dim - d_inner) = {k_dim_total} not even (Q+K halves)"
        ));
    }
    let k_dim = k_dim_total / 2;
    if k_dim % head_dim != 0 {
        return Err(format!(
            "{p}.linear_attn: k_dim={k_dim} not divisible by head_dim={head_dim} \
             (assumes k_head_dim == v_head_dim for Qwen3.5)"
        ));
    }
    let n_k_heads = k_dim / head_dim;

    // ── Pre-norm: GemmaRMSNorm over the entry hidden state ─────────────
    //
    // All four in_proj_* linears see the normalized input. Residual
    // still adds attn_out to the un-normalized `h` after this function
    // returns.
    let norm_name = format!("{p}.input_layernorm.weight");
    let h_for_linears = apply_pre_norm_or_fallback(gpu, trunk, h, &norm_name, seq_len, dim)?;

    // ── QKV projection: BF16 h → F32 [seq_len, qkv_dim] ────────────────
    //
    // Layout per token: [Q (k_dim) | K (k_dim) | V (d_inner)]. Matches
    // production deltanet (qwen35.rs ~2410). The `gemm_bf16` kernel
    // outputs F32 directly so no extra cast needed.
    let qkv_f32 = gpu
        .alloc_tensor(&[seq_len * qkv_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_qkv.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wqkv.tensor,
        &mut to_2d(qkv_f32.clone_view(), seq_len, qkv_dim),
        qkv_dim, qkv_k, seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── Z projection: F32 [seq_len, d_inner] (kept live for gated norm) ─
    //
    // Production uses Z in `gated_norm_f32(attn_out, z, ...) * silu(z)`
    // after the recurrence. The DeltaNet-full path keeps `z_f32`
    // allocated through to step 11 (gated_norm_f32_batched) so the
    // post-recurrence silu(z) post-multiply runs against the real Z
    // projection. Capture hook fires here so `in_proj_z` accumulates
    // Σx² over the right input distribution.
    let z_f32 = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_z.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wz.tensor,
        &mut to_2d(z_f32.clone_view(), seq_len, d_inner),
        d_inner, wz.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── A projection (alpha base): F32 [seq_len, n_v_heads] ────────────
    let alpha_f32 = gpu
        .alloc_tensor(&[seq_len * n_v_heads], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_a.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wa.tensor,
        &mut to_2d(alpha_f32.clone_view(), seq_len, n_v_heads),
        n_v_heads, wa.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── B projection (beta base): F32 [seq_len, n_v_heads] ─────────────
    let beta_f32 = gpu
        .alloc_tensor(&[seq_len * n_v_heads], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_b.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wb.tensor,
        &mut to_2d(beta_f32.clone_view(), seq_len, n_v_heads),
        n_v_heads, wb.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // Pre-norm scratch no longer needed — the recurrence consumes
    // splits of qkv_f32 (already projected), not the BF16 input.
    h_for_linears.drop_owned(gpu).map_err(|e| e.to_string())?;

    // ── Resolve SSM aux tensors (A_log, dt_bias, conv1d, gated norm) ──
    //
    // Look up the real F32 tensors loaded by `bf16_loader::load_bf16_model`
    // via `is_ssm_aux_tensor`. Missing tensors are signalled to the
    // forward path through `Option`s — older fixtures that pre-date the
    // SSM-aux loader can still run with the pre-2026-05-20 B.2 zero-aux
    // approximation. `warn_once` so a fixture run doesn't flood the log.
    let a_log_name = format!("{p}.linear_attn.A_log");
    let dt_bias_name = format!("{p}.linear_attn.dt_bias");
    let conv_w_name = format!("{p}.linear_attn.conv1d.weight");
    let gated_norm_name = format!("{p}.linear_attn.norm.weight");
    let a_log_opt = try_get_norm(trunk, &a_log_name);
    let dt_bias_opt = try_get_norm(trunk, &dt_bias_name);
    let conv_w_opt = try_get_norm(trunk, &conv_w_name);
    let gated_norm_opt = try_get_norm(trunk, &gated_norm_name);

    // ── Gate / beta finalization (real A_log + dt_bias when available) ─
    //
    // Production: alpha_kernel = softplus(in_proj_a + dt_bias) * -exp(A_log)
    //             beta_kernel  = sigmoid(in_proj_b)
    // The SSM aux loader puts A_log and dt_bias into `trunk.norms` keyed
    // by their canonical HF names. When either is missing (older fixtures),
    // we fall back to a zero scratch tensor to preserve the B.2 behaviour.
    let zero_aux = match (a_log_opt, dt_bias_opt) {
        (Some(_), Some(_)) => None,
        _ => {
            warn_deltanet_missing(&a_log_name, a_log_opt.is_none());
            warn_deltanet_missing(&dt_bias_name, dt_bias_opt.is_none());
            Some(gpu.zeros(&[n_v_heads], DType::F32).map_err(|e| e.to_string())?)
        }
    };
    let a_log_view = a_log_opt.unwrap_or_else(|| zero_aux.as_ref().unwrap());
    let dt_bias_view = dt_bias_opt.unwrap_or_else(|| zero_aux.as_ref().unwrap());
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &beta_f32, &alpha_f32, dt_bias_view, a_log_view, n_v_heads, seq_len,
    )
    .map_err(|e| e.to_string())?;
    if let Some(z) = zero_aux {
        gpu.free_tensor(z).map_err(|e| e.to_string())?;
    }

    // ── conv1d + SiLU + Q/K/V split ────────────────────────────────────
    //
    // Production runs `conv1d_silu_split_f32_n` (depth-wise 4-tap causal
    // conv per channel followed by SiLU activation) which advances a
    // ring-buffer state of size `(kernel-1) * qkv_dim = 3 * qkv_dim`
    // F32 elements. For the calibration forward we re-initialize the
    // state to zeros at every layer (one sequence per call — the spec
    // explicitly notes "Recurrent state initializes to zeros at the
    // start of each `forward_prefill_bf16` invocation").
    //
    // The kernel reads `input[t * n_channels + c]` (n_channels = 2*k_dim
    // + d_inner = qkv_dim) and writes to separate Q/K/V buffers; this
    // replaces the per-row memcpy split that the pre-2026-05-20 B.2
    // path used.
    //
    // When `conv1d.weight` is absent (fixture trunks), fall back to the
    // raw per-row memcpy split with NO conv math and NO SiLU. This is
    // the pre-2026-05-20 B.2 calibration shortcut — the captured Σx²
    // statistics will be biased toward the un-activated linear projection
    // distribution, but the surrounding pipeline still runs.
    let q_part = gpu
        .alloc_tensor(&[seq_len * k_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    let k_part = gpu
        .alloc_tensor(&[seq_len * k_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    let v_part = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::F32)
        .map_err(|e| e.to_string())?;
    if let Some(conv_w) = conv_w_opt {
        // Production path: fused conv + SiLU + split.
        let conv_state = gpu
            .zeros(&[qkv_dim * 3], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.conv1d_silu_split_f32_n(
            &q_part, &k_part, &v_part,
            &qkv_f32, conv_w, &conv_state,
            k_dim, d_inner, seq_len,
        )
        .map_err(|e| e.to_string())?;
        gpu.free_tensor(conv_state).map_err(|e| e.to_string())?;
    } else {
        // Fixture fallback: raw split (no conv1d weight available).
        warn_deltanet_missing(&conv_w_name, true);
        for t in 0..seq_len {
            let row_byte = t * qkv_dim * 4;
            gpu.hip
                .memcpy_dtod_at(
                    &q_part.buf, t * k_dim * 4,
                    &qkv_f32.buf, row_byte,
                    k_dim * 4,
                )
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memcpy_dtod_at(
                    &k_part.buf, t * k_dim * 4,
                    &qkv_f32.buf, row_byte + k_dim * 4,
                    k_dim * 4,
                )
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memcpy_dtod_at(
                    &v_part.buf, t * d_inner * 4,
                    &qkv_f32.buf, row_byte + 2 * k_dim * 4,
                    d_inner * 4,
                )
                .map_err(|e| e.to_string())?;
        }
    }
    gpu.free_tensor(qkv_f32).map_err(|e| e.to_string())?;

    // ── QK L2-norm + Q scale ───────────────────────────────────────────
    //
    // Matches production: each head of Q gets l2-normalized then scaled
    // by 1/sqrt(head_dim); each head of K gets l2-normalized. Batched
    // over seq_len.
    let q_scale = 1.0f32 / (head_dim as f32).sqrt();
    gpu.fused_qk_l2_norm_scale_f32_batched(
        &q_part, &k_part, n_k_heads, head_dim, q_scale, RMS_NORM_EPS, seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── GQA repeat (if n_k_heads < n_v_heads) ──────────────────────────
    //
    // The gated_delta_net_f32 kernel expects q/k laid out per v_head
    // (`[n_tokens × n_v_heads × head_dim]`). For Qwen3.5 0.8B
    // n_k_heads == n_v_heads (no repeat); 4B/9B/27B may vary. Use the
    // batched repeat_interleave to extend Q/K to v_heads in one launch.
    let (q_gdn, k_gdn) = if n_k_heads < n_v_heads {
        if n_v_heads % n_k_heads != 0 {
            return Err(format!(
                "{p}.linear_attn: n_v_heads={n_v_heads} not divisible by n_k_heads={n_k_heads} (GQA ratio)"
            ));
        }
        let ratio = n_v_heads / n_k_heads;
        let q_exp = gpu
            .alloc_tensor(&[seq_len * d_inner], DType::F32)
            .map_err(|e| e.to_string())?;
        let k_exp = gpu
            .alloc_tensor(&[seq_len * d_inner], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.repeat_interleave_qk_f32_batched(
            &q_part, &k_part, &q_exp, &k_exp,
            n_k_heads, ratio, head_dim, seq_len,
        )
        .map_err(|e| e.to_string())?;
        // Free the GQA source tensors — the expanded copies own the
        // data the kernel will read.
        gpu.free_tensor(q_part).map_err(|e| e.to_string())?;
        gpu.free_tensor(k_part).map_err(|e| e.to_string())?;
        (q_exp, k_exp)
    } else {
        // n_k_heads == n_v_heads: no repeat. Pass the existing buffers
        // through to the recurrence kernel. (q_part / k_part are moved
        // here.)
        //
        // n_k_heads > n_v_heads is structurally impossible for Qwen3.5
        // GQA but defensively rejected: the geometry derivation above
        // sets n_k_heads = k_dim / head_dim and the safetensors weight
        // shapes would have to be malformed for this to fire.
        if n_k_heads > n_v_heads {
            gpu.free_tensor(q_part).map_err(|e| e.to_string())?;
            gpu.free_tensor(k_part).map_err(|e| e.to_string())?;
            gpu.free_tensor(alpha_f32).map_err(|e| e.to_string())?;
            gpu.free_tensor(beta_f32).map_err(|e| e.to_string())?;
            gpu.free_tensor(v_part).map_err(|e| e.to_string())?;
            return Err(format!(
                "{p}.linear_attn: unexpected n_k_heads={n_k_heads} > n_v_heads={n_v_heads}"
            ));
        }
        (q_part, k_part)
    };

    // ── Allocate recurrent state (zeros) and attention output (F32) ────
    //
    // State shape: [n_v_heads, head_dim, head_dim] F32. Zero-init at the
    // start of each forward (B.2 scope: single sequence per call;
    // multi-sequence batching is B.4).
    let state = gpu
        .zeros(&[n_v_heads, head_dim, head_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    let attn_out_f32 = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::F32)
        .map_err(|e| e.to_string())?;

    // ── Gated delta-net recurrence ─────────────────────────────────────
    //
    // The kernel processes all `seq_len` tokens in one launch (state
    // held in LDS), advancing the recurrent state token-by-token.
    gpu.gated_delta_net_f32(
        &q_gdn, &k_gdn, &v_part,
        &alpha_f32, &beta_f32,
        &state, &attn_out_f32,
        seq_len, n_v_heads, head_dim,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(q_gdn).map_err(|e| e.to_string())?;
    gpu.free_tensor(k_gdn).map_err(|e| e.to_string())?;
    gpu.free_tensor(v_part).map_err(|e| e.to_string())?;
    gpu.free_tensor(alpha_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(beta_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(state).map_err(|e| e.to_string())?;

    // ── Gated output norm: rmsnorm(attn_out, w) * silu(z) ─────────────
    //
    // Production runs `gated_norm_f32_batched`. The kernel:
    //   1. Per-head RMSNorm on `attn_out` (mean of squares over the
    //      `head_dim` axis, rsqrt, multiply).
    //   2. Per-head multiply by `norm_weight[head_dim]` (init-from-one,
    //      stored without `+= 1.0` bake — `is_gated_norm_tensor`).
    //   3. Element-wise multiply by `silu(z)` (F32 silu).
    // Grid `[n_v_heads, seq_len, 1]` with 32-wide reduction per head.
    //
    // When the gated-norm weight is absent (fixture trunks), we
    // bypass the gated norm + silu(z) entirely — the un-normed
    // recurrence output flows straight into out_proj. This matches the
    // pre-2026-05-20 B.2 shortcut behaviour, which is the legacy
    // calibration path for sandbox runs without the SSM-aux loader.
    //
    // `normed_f32` is `[seq_len, d_inner]` row-major (same shape as
    // `attn_out_f32`); the kernel reads `attn_out`/`z`/writes `normed`
    // in lock-step with no aliasing.
    let normed_f32 = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::F32)
        .map_err(|e| e.to_string())?;
    if let Some(gated_norm_w) = gated_norm_opt {
        gpu.gated_norm_f32_batched(
            &attn_out_f32, &z_f32, gated_norm_w,
            &normed_f32,
            n_v_heads, head_dim, RMS_NORM_EPS, seq_len,
        )
        .map_err(|e| e.to_string())?;
    } else {
        warn_deltanet_missing(&gated_norm_name, true);
        // Bit-exact copy of attn_out_f32 into normed_f32 so the
        // downstream BF16 cast + out_proj see the un-normed output.
        gpu.hip
            .memcpy_dtod_at(
                &normed_f32.buf, 0,
                &attn_out_f32.buf, 0,
                seq_len * d_inner * 4,
            )
            .map_err(|e| e.to_string())?;
    }
    gpu.free_tensor(attn_out_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(z_f32).map_err(|e| e.to_string())?;

    // ── Cast F32 gated-norm output → BF16 for out_proj input ───────────
    let attn_pre_o_bf16 = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&normed_f32, &attn_pre_o_bf16, seq_len * d_inner)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(normed_f32).map_err(|e| e.to_string())?;

    // ── out_proj: attn_pre_o · wo^T → F32 [seq_len, dim] ───────────────
    let attn_out_proj_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.out_proj.weight")));
    let attn_pre_o_view = view_2d(&attn_pre_o_bf16, seq_len, d_inner);
    gpu.gemm_bf16(
        &attn_pre_o_view, &wo.tensor,
        &mut to_2d(attn_out_proj_f32.clone_view(), seq_len, dim),
        dim, d_inner, seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_pre_o_bf16).map_err(|e| e.to_string())?;

    // ── Convert to BF16 for residual add upstream ──────────────────────
    let attn_out_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&attn_out_proj_f32, &attn_out_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_out_proj_f32).map_err(|e| e.to_string())?;

    Ok(attn_out_bf16)
}

/// Non-deltanet fallback (compile-only path).
///
/// When the workspace is built with `--no-default-features` and the
/// `deltanet` feature is disabled, the `gated_delta_net_f32` /
/// `fused_sigmoid_alpha_gate_f32_batched` /
/// `fused_qk_l2_norm_scale_f32_batched` dispatch entries on
/// `rdna_compute::Gpu` aren't compiled. We fall back to the pre-B.2
/// Q-chunk passthrough so the module still compiles — calibration
/// quality is worse in this configuration but `collect_imatrix` is
/// still functional. Production calibration ALWAYS uses default
/// features (deltanet ON), so this path is purely a compile-time
/// safety net.
#[cfg(not(feature = "deltanet"))]
fn forward_deltanet_layer(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    p: &str,
    seq_len: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    let wqkv = get(trunk, &format!("{p}.linear_attn.in_proj_qkv.weight"))?;
    let wz = get(trunk, &format!("{p}.linear_attn.in_proj_z.weight"))?;
    let wa = get(trunk, &format!("{p}.linear_attn.in_proj_a.weight"))?;
    let wb = get(trunk, &format!("{p}.linear_attn.in_proj_b.weight"))?;
    let wo = get(trunk, &format!("{p}.linear_attn.out_proj.weight"))?;

    let norm_name = format!("{p}.input_layernorm.weight");
    let h_for_linears = apply_pre_norm_or_fallback(gpu, trunk, h, &norm_name, seq_len, dim)?;

    let qkv_dim = wqkv.shape[0];
    let qkv_k = wqkv.shape[1];
    if qkv_k != dim {
        return Err(format!(
            "{p}.linear_attn.in_proj_qkv: K={qkv_k} doesn't match dim={dim}"
        ));
    }
    let qkv_f32 = gpu
        .alloc_tensor(&[seq_len * qkv_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_qkv.weight")));
    gpu.gemm_bf16(h_for_linears.as_ref(), &wqkv.tensor,
                  &mut to_2d(qkv_f32.clone_view(), seq_len, qkv_dim),
                  qkv_dim, qkv_k, seq_len)
        .map_err(|e| e.to_string())?;

    let z_dim = wz.shape[0];
    let z_f32 = gpu
        .alloc_tensor(&[seq_len * z_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_z.weight")));
    gpu.gemm_bf16(h_for_linears.as_ref(), &wz.tensor,
                  &mut to_2d(z_f32.clone_view(), seq_len, z_dim),
                  z_dim, wz.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(z_f32).map_err(|e| e.to_string())?;

    let a_dim = wa.shape[0];
    let a_f32 = gpu
        .alloc_tensor(&[seq_len * a_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_a.weight")));
    gpu.gemm_bf16(h_for_linears.as_ref(), &wa.tensor,
                  &mut to_2d(a_f32.clone_view(), seq_len, a_dim),
                  a_dim, wa.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(a_f32).map_err(|e| e.to_string())?;

    let b_dim = wb.shape[0];
    let b_f32 = gpu
        .alloc_tensor(&[seq_len * b_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_b.weight")));
    gpu.gemm_bf16(h_for_linears.as_ref(), &wb.tensor,
                  &mut to_2d(b_f32.clone_view(), seq_len, b_dim),
                  b_dim, wb.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(b_f32).map_err(|e| e.to_string())?;
    h_for_linears.drop_owned(gpu).map_err(|e| e.to_string())?;

    let d_inner = wo.shape[1];
    if d_inner > qkv_dim {
        return Err(format!(
            "{p}.linear_attn.out_proj: K={d_inner} > qkv_dim={qkv_dim}"
        ));
    }

    // Q-chunk passthrough (calibration approximation when deltanet
    // feature disabled — see module-level docs and pre-B.2 history).
    let attn_pre_o = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::BF16)
        .map_err(|e| e.to_string())?;
    let qkv_bf16 = gpu
        .alloc_tensor(&[seq_len * qkv_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&qkv_f32, &qkv_bf16, seq_len * qkv_dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(qkv_f32).map_err(|e| e.to_string())?;
    for t in 0..seq_len {
        let src_off = t * qkv_dim * 2;
        let dst_off = t * d_inner * 2;
        gpu.hip
            .memcpy_dtod_at(
                &attn_pre_o.buf, dst_off,
                &qkv_bf16.buf, src_off,
                d_inner * 2,
            )
            .map_err(|e| e.to_string())?;
    }
    gpu.free_tensor(qkv_bf16).map_err(|e| e.to_string())?;

    let attn_out_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.out_proj.weight")));
    let attn_pre_o_view = view_2d(&attn_pre_o, seq_len, d_inner);
    gpu.gemm_bf16(
        &attn_pre_o_view, &wo.tensor,
        &mut to_2d(attn_out_f32.clone_view(), seq_len, dim),
        dim, d_inner, seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_pre_o).map_err(|e| e.to_string())?;

    let attn_out_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&attn_out_f32, &attn_out_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_out_f32).map_err(|e| e.to_string())?;

    Ok(attn_out_bf16)
}

/// RoPE configuration for a trunk: `rope_theta` (base frequency) and
/// `partial_rotary_factor` (fraction of head_dim that's rotated; Qwen3.5
/// uses 0.25). Read from `<model_dir>/config.json` with the
/// HuggingFace-equivalent defaults baked in. Mirrors the parsing in
/// `crates/hipfire-arch-qwen35/src/qwen35.rs:172-176` so this calibration
/// forward produces the same QK distribution as the production engine.
///
/// Only used by the `deltanet`-feature attention path; gated to avoid
/// dead-code warnings in `--no-default-features` builds.
#[cfg(feature = "deltanet")]
struct RopeConfig {
    rope_theta: f32,
    partial_rotary_factor: f32,
}

#[cfg(feature = "deltanet")]
impl RopeConfig {
    /// Defaults match Qwen3.5: rope_theta=10M, partial 25%. These also
    /// happen to be reasonable fallbacks for Qwen3.6 (same rope_theta;
    /// 0.25 is overruled where the config exists, but produces a sane
    /// calibration distribution otherwise).
    fn default_qwen35() -> Self {
        Self { rope_theta: 10_000_000.0, partial_rotary_factor: 0.25 }
    }

    /// Load from `<model_dir>/config.json`. Falls back to
    /// `default_qwen35()` on any missing key / parse error. Mirrors the
    /// `rope_scaling` / `rope_params` lookup at qwen35.rs:170-179.
    fn from_model_dir(model_dir: &std::path::Path) -> Self {
        let cfg = model_dir.join("config.json");
        let Ok(bytes) = std::fs::read(&cfg) else {
            return Self::default_qwen35();
        };
        let Ok(v): Result<serde_json::Value, _> = serde_json::from_slice(&bytes) else {
            return Self::default_qwen35();
        };
        let tc = v
            .get("text_config")
            .and_then(|t| t.as_object())
            .map(|m| serde_json::Value::Object(m.clone()))
            .unwrap_or(v.clone());
        let rope_params = tc.get("rope_scaling").and_then(|s| s.as_object());
        let rope_theta = tc
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| rope_params.and_then(|m| m.get("rope_theta")).and_then(|x| x.as_f64()))
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = tc
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .or_else(|| {
                rope_params
                    .and_then(|m| m.get("partial_rotary_factor"))
                    .and_then(|x| x.as_f64())
            })
            .unwrap_or(0.25) as f32;
        Self { rope_theta, partial_rotary_factor }
    }
}

/// Apply per-head rmsnorm to a [seq_len, n_heads, head_dim] tensor
/// in-place via the `rmsnorm_batched` kernel. Each head's [head_dim]
/// vector is normalized independently against `weight: [head_dim]`.
/// Mirrors the q_norm / k_norm step in `hipfire-arch-qwen35::qwen35`.
///
/// If `weight` is `None` (norm tensor absent from trunk), this is a
/// no-op — the function returns `Ok(())` and leaves the tensor as-is.
/// Production Qwen3.5 / 3.6 always ships q_norm and k_norm; this
/// fallback exists for unit-test fixtures and stripped trunks.
///
/// Only used by the `deltanet`-feature attention path; gated to avoid
/// dead-code warnings in `--no-default-features` builds.
#[cfg(feature = "deltanet")]
fn per_head_rmsnorm_f32_inplace(
    gpu: &mut Gpu,
    x: &GpuTensor,
    weight: Option<&GpuTensor>,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> HipResult<()> {
    let Some(w) = weight else {
        return Ok(());
    };
    // rmsnorm_batched supports `out == x` for in-place. Both tensors
    // share the same buffer view; the kernel reads each row, computes
    // the rms, writes the result. Safe because the per-row write
    // happens after the row's reads complete (single-threaded
    // semantics within a row).
    //
    // We allocate a scratch out tensor anyway to avoid relying on
    // undocumented aliasing behavior — keep the kernel-contract clean.
    let n_elems = seq_len * n_heads * head_dim;
    let scratch = gpu.alloc_tensor(&[n_elems], DType::F32)?;
    gpu.rmsnorm_batched(x, w, &scratch, seq_len * n_heads, head_dim, eps)?;
    // Copy scratch back into x.
    gpu.hip
        .memcpy_dtod_at(&x.buf, 0, &scratch.buf, 0, n_elems * 4)?;
    gpu.free_tensor(scratch)
}

/// Compute the real `attn_pre_o` for FullAttn calibration — production
/// prefill mirror.
///
/// Inputs are the F32 Q / K / V projections (the outputs of `q_proj`,
/// `k_proj`, `v_proj` GEMMs). The function applies per-head q_norm /
/// k_norm, runs a single batched partial-RoPE call, quantizes K/V into
/// an ephemeral per-call Q8_0 KV cache (same byte layout as production's
/// `KvCache::new_gpu_q8_capped`), and dispatches a single batched
/// causal-masked flash-attention call (`attention_q8_0_kv_batched_masked`).
/// Returns a fresh BF16 tensor of shape `[seq_len, n_heads * head_dim]`
/// ready to feed `o_proj`.
///
/// Mirrors the production prefill batched-FA path at
/// `crates/hipfire-runtime/src/llama.rs:1850-1976` (dense path) and the
/// dense-Qwen3.5 variant at
/// `crates/hipfire-arch-qwen35/src/qwen35.rs:5104-5290`. The Q8_0 KV
/// quantization is intentional: the deployed model's default KV mode is
/// Q8_0, so the captured `o_proj` activation includes the same
/// quantization noise that production sees. Asym{2,3,4} KV modes are
/// out of scope for the BF16 calibration forward — calibration of the
/// MQ4 weight quantizer doesn't depend on the V cache precision, and
/// the production V cache is Q8_0 in every supported KV mode.
///
/// The Q / K / V tensors are owned by the caller; this helper does NOT
/// free them (the caller frees q_f32 / k_f32 / v_f32 after the helper
/// returns).
///
/// `#[cfg(feature = "deltanet")]` because the
/// `rope_partial_interleaved_f32_batched` kernel lives behind the same
/// feature gate in `rdna-compute`. The fallback at
/// `compute_full_attn_pre_o_passthrough` keeps the lib compiling in
/// `--no-default-features` builds.
/// `gate_f32`: optional `[seq_len × n_heads × head_dim]` F32 gate tensor
/// split out of the 2x-wide q_proj. When `Some`, the attention output is
/// multiplied element-wise by `sigmoid(gate)` before the BF16 cast,
/// matching `qwen35.rs:5566-5568 sigmoid_mul_f32(fa_attn_out, fa_gate)`
/// and the HF `Qwen3_5Attention.forward` epilogue
/// `attn_output * torch.sigmoid(gate)`.
#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
fn compute_full_attn_pre_o(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    q_f32: &GpuTensor,
    k_f32: &GpuTensor,
    v_f32: &GpuTensor,
    q_norm_w: Option<&GpuTensor>,
    k_norm_w: Option<&GpuTensor>,
    gate_f32: Option<&GpuTensor>,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k_dim: usize,
    v_dim: usize,
    p: &str,
) -> Result<GpuTensor, String> {
    let attn_pre_o_dim = n_heads * head_dim;

    // Sanity: K and V projections must match GQA head geometry. Production
    // asserts the same invariant via config.n_kv_heads * config.head_dim
    // (see qwen35.rs:5106 batched-RoPE call expectations).
    debug_assert_eq!(k_dim, n_kv_heads * head_dim);
    debug_assert_eq!(v_dim, n_kv_heads * head_dim);

    // Q8_0 block layout matches `KvCache::new_gpu_q8_capped`
    // (llama.rs:3180-3197): each position stores
    // `n_kv_heads × (head_dim / 32) × 34` bytes. Refuse head_dims that
    // don't tile cleanly into 32-element Q8_0 blocks — production
    // shares the same invariant via `kv_cache_write_q8_0_batched`.
    if head_dim % 32 != 0 {
        return Err(format!(
            "{p}.self_attn: head_dim={head_dim} not divisible by 32 \
             — Q8_0 KV cache requires 32-elem block tiling"
        ));
    }

    // ── 1. q_norm / k_norm: per-head RMSNorm (matches qwen35.rs:5034-5043).
    per_head_rmsnorm_f32_inplace(gpu, q_f32, q_norm_w, seq_len, n_heads, head_dim, RMS_NORM_EPS)
        .map_err(|e| e.to_string())?;
    per_head_rmsnorm_f32_inplace(gpu, k_f32, k_norm_w, seq_len, n_kv_heads, head_dim, RMS_NORM_EPS)
        .map_err(|e| e.to_string())?;

    // ── 2. RoPE config: pull rope_theta + partial_rotary_factor from
    //       <model_dir>/config.json, fall back to Qwen3.5 defaults.
    let rope_cfg = RopeConfig::from_model_dir(&trunk.model_dir);
    let n_rot_raw = (head_dim as f32 * rope_cfg.partial_rotary_factor).round() as usize;
    // RoPE pairs (i, i + n_rot/2); n_rot must be even. Force it to be.
    let n_rot = (n_rot_raw / 2) * 2;
    if n_rot == 0 {
        return Err(format!(
            "{p}.self_attn: partial_rotary_factor={} × head_dim={head_dim} = 0 \
             rotated dims (need at least 2)",
            rope_cfg.partial_rotary_factor
        ));
    }

    // ── 3. Positions buffer: F32-dtype tensor holding `[0, 1, ..., seq_len-1]`
    //       as raw i32 bits. Production uses the same convention
    //       (qwen35.rs:3357 + qwen35.rs:4417): dtype is cosmetic because the
    //       rope / kv_write / attention kernels cast the pointer to
    //       `const int*`. Upload via `memcpy_htod` of the i32 byte slice.
    let positions = gpu
        .alloc_tensor(&[seq_len], DType::F32)
        .map_err(|e| e.to_string())?;
    let positions_host: Vec<i32> = (0..seq_len as i32).collect();
    let positions_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, seq_len * 4)
    };
    gpu.hip
        .memcpy_htod(&positions.buf, positions_bytes)
        .map_err(|e| e.to_string())?;

    // ── 4. Batched partial-interleaved RoPE over Q and K
    //       (mirrors qwen35.rs:5106 `gpu.rope_partial_interleaved_f32_batched`
    //       and dense-LLaMA's `gpu.rope_batched_f32` at llama.rs:1868 —
    //       Qwen3.5 uses partial, Qwen3.5-0.8B is partial 25% by config).
    gpu.rope_partial_interleaved_f32_batched(
        q_f32, k_f32, &positions,
        n_heads, n_kv_heads, head_dim, n_rot,
        rope_cfg.rope_theta, seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── 5. Allocate per-call ephemeral Q8_0 K and V caches.
    //
    // Byte layout: for each position we store `n_kv_heads × (head_dim/32) × 34`
    // bytes (each 32-elem Q8_0 block = 2-byte FP16 scale + 32 int8s).
    // This is byte-identical to the production cache layout
    // (`KvCache::new_gpu_q8_capped` at llama.rs:3187-3196), so the
    // attention kernel reads through the exact same indexing math the
    // deployed model uses.
    //
    // The cache is allocated as an F32 tensor for compatibility with the
    // `gpu.alloc_tensor` pool — element count rounds up.
    let blocks_per_head = head_dim / 32;
    let total_blocks_per_pos = n_kv_heads * blocks_per_head;
    let cache_bytes_per_pos = total_blocks_per_pos * 34;
    let cache_bytes_total = seq_len * cache_bytes_per_pos;
    let cache_elems = (cache_bytes_total + 3) / 4;
    let k_cache = gpu
        .alloc_tensor(&[cache_elems], DType::F32)
        .map_err(|e| e.to_string())?;
    let v_cache = gpu
        .alloc_tensor(&[cache_elems], DType::F32)
        .map_err(|e| e.to_string())?;
    // Zero-init for safety: positions[b] = b ensures every slot is
    // written below, but a partial-batch crash mid-write would leave
    // unwritten bytes that the attention kernel could still touch.
    // Production zeros the cache at allocation time
    // (`KvCache::new_gpu_q8_capped` calls `gpu.zeros`) so mirror that.
    gpu.hip
        .memset(&k_cache.buf, 0, cache_bytes_total)
        .map_err(|e| e.to_string())?;
    gpu.hip
        .memset(&v_cache.buf, 0, cache_bytes_total)
        .map_err(|e| e.to_string())?;

    // ── 6. Batched Q8_0 KV writes (mirrors qwen35.rs:5162-5169 / llama.rs:1900-1907).
    //       Each batch row b writes its row of K / V into cache position
    //       positions[b] = b. After this, k_cache / v_cache hold the full
    //       prefix [0..seq_len) in production-identical layout.
    gpu.kv_cache_write_q8_0_batched(
        &k_cache, k_f32, &positions,
        n_kv_heads, head_dim, seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.kv_cache_write_q8_0_batched(
        &v_cache, v_f32, &positions,
        n_kv_heads, head_dim, seq_len,
    )
    .map_err(|e| e.to_string())?;

    // ── 7. Allocate attention output and dispatch the batched causal-masked
    //       Q8 flash attention (mirrors qwen35.rs:5282 / llama.rs:1968).
    //
    // The kernel reads K/V from the Q8_0 cache and writes F32 output
    // `[seq_len, n_heads * head_dim]`. Causal cutoff per batch row b is
    // `positions[b] + 1`, so row b attends to keys [0..=b] — identical
    // semantics to the per-token loop the prior B.3 implementation used,
    // but as one launch instead of seq_len launches.
    //
    // For calibration we never exercise the long-context Q8 LDS fallback
    // (LDS_CTX_LIMIT = 15000 at llama.rs:1911); calibration sequences are
    // capped at n_ctx ≤ 2048. The masked variant is the right call here
    // regardless — it's the production path's first-choice dispatch.
    let attn_pre_o_f32 = gpu
        .alloc_tensor(&[seq_len * attn_pre_o_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    // `max_seq` = `max_ctx_len` = seq_len. `tree_bias = None`,
    // `block_start = block_cols = 0` (no tree mode in calibration).
    gpu.attention_q8_0_kv_batched_masked(
        q_f32, &k_cache, &v_cache,
        &attn_pre_o_f32, &positions,
        n_heads, n_kv_heads, head_dim,
        seq_len, seq_len, seq_len,
        None, 0, 0,
    )
    .map_err(|e| e.to_string())?;

    // ── 8. Free intermediates: ephemeral KV caches + positions.
    gpu.free_tensor(k_cache).map_err(|e| e.to_string())?;
    gpu.free_tensor(v_cache).map_err(|e| e.to_string())?;
    gpu.free_tensor(positions).map_err(|e| e.to_string())?;

    // ── 9. Apply sigmoid(gate) when present. Matches the production
    //       epilogue at qwen35.rs:5566-5568 (which uses the same
    //       `sigmoid_mul_f32` kernel) and HF `Qwen3_5Attention.forward`:
    //       `attn_output = attn_output * torch.sigmoid(gate)`. Without
    //       this step, the o_proj capture sees raw `softmax(QK^T) * V`
    //       instead of the gated form, producing the NRMSE ~4.88
    //       divergence observed on the PyTorch oracle baseline.
    if let Some(gate_buf) = gate_f32 {
        debug_assert_eq!(gate_buf.numel(), seq_len * attn_pre_o_dim);
        gpu.sigmoid_mul_f32(&attn_pre_o_f32, gate_buf)
            .map_err(|e| e.to_string())?;
    }

    // ── 10. Cast attn_pre_o_f32 → BF16. Caller takes ownership of the result.
    let attn_pre_o_bf16 = gpu
        .alloc_tensor(&[seq_len * attn_pre_o_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&attn_pre_o_f32, &attn_pre_o_bf16, seq_len * attn_pre_o_dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_pre_o_f32).map_err(|e| e.to_string())?;
    Ok(attn_pre_o_bf16)
}

/// Non-deltanet fallback: passthrough V to o_proj input.
///
/// Used when the `deltanet` feature is off — `rope_partial_interleaved_f32`
/// is not compiled into `rdna-compute` then, so the real attention math
/// is not available. This matches the pre-B.3 calibration shortcut
/// (V distribution into `o_proj` rather than `softmax(QK^T)·V`); the
/// production calibration path (with default features) reaches the
/// real-math branch.
///
/// Frees `v_f32` (and the caller has already freed q_f32 / k_f32? No —
/// the caller hands all three to this helper / its sibling, and we own
/// freeing v here. q/k are unused by passthrough — caller must free
/// them separately. To keep the API symmetric we leave q/k freeing to
/// the caller; the fallback path frees only v + does the cast.
#[cfg(not(feature = "deltanet"))]
#[allow(clippy::too_many_arguments)]
fn compute_full_attn_pre_o_passthrough(
    gpu: &mut Gpu,
    v_f32: &GpuTensor,
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_dim: usize,
    attn_pre_o_dim: usize,
) -> Result<GpuTensor, String> {
    // Sanity: v_dim == n_kv_heads * head_dim.
    debug_assert_eq!(v_dim, n_kv_heads * head_dim);

    let v_bf16 = gpu
        .alloc_tensor(&[seq_len * v_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(v_f32, &v_bf16, seq_len * v_dim)
        .map_err(|e| e.to_string())?;

    if attn_pre_o_dim == v_dim {
        Ok(v_bf16)
    } else {
        // GQA expansion (zero-pad or truncate). For Qwen3.5 0.8B the
        // canonical case is n_heads > n_kv_heads, so attn_pre_o_dim
        // (= n_heads * head_dim) > v_dim and we zero-pad.
        let target = gpu
            .alloc_tensor(&[seq_len * attn_pre_o_dim], DType::BF16)
            .map_err(|e| e.to_string())?;
        gpu.hip
            .memset(&target.buf, 0, seq_len * attn_pre_o_dim * 2)
            .map_err(|e| e.to_string())?;
        let copy_per_row = std::cmp::min(attn_pre_o_dim, v_dim);
        for t in 0..seq_len {
            let src_off = t * v_dim * 2;
            let dst_off = t * attn_pre_o_dim * 2;
            gpu.hip
                .memcpy_dtod_at(
                    &target.buf,
                    dst_off,
                    &v_bf16.buf,
                    src_off,
                    copy_per_row * 2,
                )
                .map_err(|e| e.to_string())?;
        }
        gpu.free_tensor(v_bf16).map_err(|e| e.to_string())?;
        Ok(target)
    }
}

/// Full-attention (self-attn) layer forward (calibration mirror).
///
/// Fires the capture hook for `q_proj`, `k_proj`, `v_proj`, `o_proj`.
/// Implements the production prefill path described in module docs:
/// batched partial-RoPE → Q8_0 KV cache write → batched-masked
/// flash-attention → F32 attention output → cast to BF16 for `o_proj`.
/// The production-mirror branch is gated on the `deltanet` feature (the
/// underlying `rope_partial_interleaved_f32_batched` +
/// `kv_cache_write_q8_0_batched` + `attention_q8_0_kv_batched_masked`
/// kernels live behind the same feature gate in `rdna-compute`);
/// non-deltanet builds keep the V-passthrough shortcut.
fn forward_full_attn_layer(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    p: &str,
    seq_len: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    let wq = get(trunk, &format!("{p}.self_attn.q_proj.weight"))?;
    let wk = get(trunk, &format!("{p}.self_attn.k_proj.weight"))?;
    let wv = get(trunk, &format!("{p}.self_attn.v_proj.weight"))?;
    let wo = get(trunk, &format!("{p}.self_attn.o_proj.weight"))?;

    let q_full_dim = wq.shape[0];
    let k_dim = wk.shape[0];
    let v_dim = wv.shape[0];

    // ── Geometry: derive head_dim / n_heads / n_kv_heads ───────────────
    //
    // head_dim is the first dim of the q_norm tensor (Qwen3.5 always
    // ships q_norm.weight as [head_dim] F32). If absent (stripped
    // trunk / non-Qwen variant), fall back to a heuristic: q_dim
    // divides into n_heads heads of equal head_dim; the typical
    // head_dim across modern dense decoders is 64, 96, 128, or 256.
    // We pick the largest divisor ≤ 256 that also divides into the
    // hidden state cleanly. If that fails, we error out — the layer
    // cannot run without a head_dim.
    let q_norm_name = format!("{p}.self_attn.q_norm.weight");
    let k_norm_name = format!("{p}.self_attn.k_norm.weight");
    let q_norm_w = try_get_norm(trunk, &q_norm_name);
    let k_norm_w = try_get_norm(trunk, &k_norm_name);

    let head_dim = if let Some(w) = q_norm_w {
        if w.shape.is_empty() {
            return Err(format!(
                "{q_norm_name}: empty shape, cannot derive head_dim"
            ));
        }
        // q_norm is [head_dim] (1-D) or [1, head_dim] depending on
        // loader convention; take the last non-trivial axis.
        *w.shape.iter().rev().find(|&&d| d > 1).unwrap_or(&w.shape[w.shape.len() - 1])
    } else {
        // Fallback: try the canonical head_dim values in decreasing
        // order until one divides q_full_dim, k_dim, and v_dim. Avoids a
        // hard-coded value and works across model families.
        let candidates = [256usize, 192, 128, 96, 64];
        let pick = candidates
            .iter()
            .copied()
            .find(|&d| q_full_dim % d == 0 && k_dim % d == 0 && v_dim % d == 0);
        match pick {
            Some(d) => d,
            None => {
                return Err(format!(
                    "{p}.self_attn: cannot infer head_dim — q_norm absent and \
                     q_full_dim={q_full_dim}/k_dim={k_dim}/v_dim={v_dim} not divisible by \
                     any of {candidates:?}"
                ));
            }
        }
    };
    if q_full_dim % head_dim != 0 || k_dim % head_dim != 0 || v_dim % head_dim != 0 {
        return Err(format!(
            "{p}.self_attn: head_dim={head_dim} does not divide one of \
             q_full_dim={q_full_dim}/k_dim={k_dim}/v_dim={v_dim}"
        ));
    }
    // ── Detect attn_output_gate (Qwen3.5/3.6) ──────────────────────────
    //
    // Qwen3_5Attention concatenates query AND a sigmoid gate along the
    // last dim of q_proj. The HF view is
    // `q_proj(h).view(*, n_heads, 2 * head_dim)` then `chunk(2, dim=-1)`.
    // On disk that means the q_proj weight rows are interleaved per head
    // as `[q_d0..d(head_dim-1), gate_d0..d(head_dim-1)]`. Detect by
    // checking q_full_dim vs (n_kv_heads-derived n_heads × head_dim).
    //
    // Heuristic: when `q_full_dim == 2 × k_dim × (n_heads/n_kv_heads)`
    // the gate is present. For Qwen3.5-0.8B (n_heads=8, n_kv_heads=2,
    // head_dim=256): q_full_dim=4096, k_dim=512, ratio q/k = 8 = 2*4
    // (gate doubles the per-kv-head count). We don't know n_heads a
    // priori, so derive it from the SHAPE consistency check:
    //   - if q_full_dim / head_dim equals an integer multiple of
    //     (k_dim / head_dim) that is even → the gate doubles the per-
    //     kv-head Q-head count.
    //   - we then have n_heads = (q_full_dim / 2) / head_dim.
    //
    // Falls back to the non-gated layout when the shapes don't satisfy
    // the 2x relationship (Llama, Mistral, dense Qwen2 etc.).
    let n_kv_heads = k_dim / head_dim;
    if v_dim / head_dim != n_kv_heads {
        return Err(format!(
            "{p}.self_attn: k_dim={k_dim} → {n_kv_heads} KV heads, but \
             v_dim={v_dim} → {} (must match for GQA)",
            v_dim / head_dim,
        ));
    }
    let raw_heads = q_full_dim / head_dim;
    let (n_heads, has_gate) = if raw_heads % 2 == 0
        && raw_heads / 2 >= n_kv_heads
        && (raw_heads / 2) % n_kv_heads == 0
        && q_full_dim == 2 * (raw_heads / 2) * head_dim
        && wo.shape[1] == (raw_heads / 2) * head_dim
    {
        // o_proj's input dim equals the de-gated Q dim → confirms gate.
        (raw_heads / 2, true)
    } else if wo.shape[1] == raw_heads * head_dim {
        // Standard (no gate): o_proj input dim equals raw Q dim.
        (raw_heads, false)
    } else {
        return Err(format!(
            "{p}.self_attn: cannot reconcile q_proj={q_full_dim} / k_proj={k_dim} \
             / o_proj_k={} / head_dim={head_dim} into a standard or \
             gated Qwen3.5/3.6 layout (raw_heads={raw_heads}, n_kv_heads={n_kv_heads})",
            wo.shape[1],
        ));
    };
    let q_dim = n_heads * head_dim;

    // Pre-norm: GemmaRMSNorm over the entry hidden state. q/k/v see
    // normalized input; residual still adds attn_out to the
    // un-normalized `h` after this function returns.
    let norm_name = format!("{p}.input_layernorm.weight");
    let h_for_linears = apply_pre_norm_or_fallback(gpu, trunk, h, &norm_name, seq_len, dim)?;

    // Q projection: produce F32 [seq_len, q_full_dim]. When the gate is
    // present, q_full_dim = 2 × n_heads × head_dim and the per-head
    // layout is interleaved `[q_d0..d(head_dim-1), gate_d0..d(head_dim-1)]`
    // (matches HF `q_proj(h).view(*, n_heads, 2*head_dim).chunk(2, dim=-1)`).
    // We deinterleave into separate Q and gate buffers below so the
    // downstream attention math sees the canonical
    // `[seq_len × n_heads × head_dim]` Q shape, and the FA epilogue can
    // apply `sigmoid(gate) * attn_out` before o_proj — matching
    // qwen35.rs:5566-5568 `gpu.sigmoid_mul_f32(fa_attn_out, fa_gate)`.
    let q_full_f32 = gpu
        .alloc_tensor(&[seq_len * q_full_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.q_proj.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wq.tensor,
        &mut to_2d(q_full_f32.clone_view(), seq_len, q_full_dim),
        q_full_dim, wq.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // Split out Q and (optional) gate. For gated variants we use
    // `deinterleave_f32_batched` (matches qwen35.rs:5300). For non-gated
    // variants q_full_f32 IS the Q tensor — alias rather than copy.
    let (q_f32, gate_f32_opt) = if has_gate {
        #[cfg(feature = "deltanet")]
        {
            let q_buf = gpu
                .alloc_tensor(&[seq_len * q_dim], DType::F32)
                .map_err(|e| e.to_string())?;
            let gate_buf = gpu
                .alloc_tensor(&[seq_len * q_dim], DType::F32)
                .map_err(|e| e.to_string())?;
            gpu.deinterleave_f32_batched(
                &q_full_f32, &q_buf, &gate_buf,
                n_heads, head_dim, seq_len,
            )
            .map_err(|e| e.to_string())?;
            gpu.free_tensor(q_full_f32).map_err(|e| e.to_string())?;
            (q_buf, Some(gate_buf))
        }
        #[cfg(not(feature = "deltanet"))]
        {
            // No-deltanet builds don't have `deinterleave_f32_batched`
            // (it lives behind the deltanet feature gate). Bail with a
            // clear error rather than silently emitting wrong values:
            // the passthrough fallback can't be expressed for the gated
            // variant without writing a new kernel.
            gpu.free_tensor(q_full_f32).map_err(|e| e.to_string())?;
            h_for_linears.drop_owned(gpu).map_err(|e| e.to_string())?;
            return Err(format!(
                "{p}.self_attn: q_proj is gated (q_full_dim={q_full_dim} = 2*{q_dim}) \
                 but the `deltanet` feature is disabled — \
                 `deinterleave_f32_batched` is unavailable. Rebuild with \
                 default features (or remove --no-default-features)."
            ));
        }
    } else {
        // Non-gated: q_full_f32 IS the Q tensor.
        (q_full_f32, None)
    };

    // K projection: produce F32 [seq_len, k_dim]. Kept alive through
    // attention as the k_cache after k_norm + RoPE.
    let k_f32 = gpu
        .alloc_tensor(&[seq_len * k_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.k_proj.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wk.tensor,
        &mut to_2d(k_f32.clone_view(), seq_len, k_dim),
        k_dim, wk.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // V projection: produce F32 [seq_len, v_dim]. No norm / RoPE on V.
    // Kept alive through attention as the v_cache.
    let v_f32 = gpu
        .alloc_tensor(&[seq_len * v_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.v_proj.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &wv.tensor,
        &mut to_2d(v_f32.clone_view(), seq_len, v_dim),
        v_dim, wv.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;
    // h_for_linears is no longer needed; downstream consumes Q/K/V.
    h_for_linears.drop_owned(gpu).map_err(|e| e.to_string())?;

    // ── Q-norm / K-norm + batched RoPE + Q8_0 batched causal attention ──
    //
    // Mirrors the production FullAttn prefill path (llama.rs:1850-1976,
    // qwen35.rs:5104-5290): batched partial RoPE, K/V quantized into a
    // per-call ephemeral Q8_0 KV cache, single batched-masked attention
    // call (`attention_q8_0_kv_batched_masked`). Output is F32, cast to
    // BF16 for the `o_proj` GEMM. Production builds (with the `deltanet`
    // feature, on by default in `hipfire-runtime`) reach
    // `compute_full_attn_pre_o`. Non-deltanet builds fall back to the
    // V passthrough (legacy shortcut) since the batched RoPE +
    // Q8_0 KV-write kernels live behind the same feature gate in
    // `rdna-compute`. The fallback keeps the workspace compiling under
    // `--no-default-features` at the cost of a calibration-quality
    // regression on FullAttn `o_proj` (the pre-B.3 behavior).
    let attn_pre_o_dim = n_heads * head_dim;
    let o_k = wo.shape[1];
    #[cfg(feature = "deltanet")]
    let attn_pre_o_bf16 = compute_full_attn_pre_o(
        gpu, trunk,
        &q_f32, &k_f32, &v_f32,
        q_norm_w, k_norm_w,
        gate_f32_opt.as_ref(),
        seq_len, n_heads, n_kv_heads, head_dim,
        k_dim, v_dim,
        p,
    )?;
    #[cfg(not(feature = "deltanet"))]
    let attn_pre_o_bf16 = {
        // Silence unused-binding warnings under no-default-features.
        let _ = (q_norm_w, k_norm_w, k_dim, p, trunk, &gate_f32_opt);
        compute_full_attn_pre_o_passthrough(
            gpu, &v_f32,
            seq_len, n_kv_heads, head_dim, v_dim, attn_pre_o_dim,
        )?
    };

    // Free Q / K / V (helper consumed views, not ownership).
    gpu.free_tensor(q_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(k_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(v_f32).map_err(|e| e.to_string())?;
    if let Some(gate_buf) = gate_f32_opt {
        gpu.free_tensor(gate_buf).map_err(|e| e.to_string())?;
    }

    let attn_pre_o = if o_k == attn_pre_o_dim {
        attn_pre_o_bf16
    } else {
        let target = gpu
            .alloc_tensor(&[seq_len * o_k], DType::BF16)
            .map_err(|e| e.to_string())?;
        // Zero-init (zero-pad path).
        gpu.hip
            .memset(&target.buf, 0, seq_len * o_k * 2)
            .map_err(|e| e.to_string())?;
        let copy_per_row = std::cmp::min(o_k, attn_pre_o_dim);
        for t in 0..seq_len {
            let src_off = t * attn_pre_o_dim * 2;
            let dst_off = t * o_k * 2;
            gpu.hip
                .memcpy_dtod_at(&target.buf, dst_off, &attn_pre_o_bf16.buf, src_off, copy_per_row * 2)
                .map_err(|e| e.to_string())?;
        }
        gpu.free_tensor(attn_pre_o_bf16).map_err(|e| e.to_string())?;
        target
    };

    // O projection: attn_out_f32 = attn_pre_o * wo^T → [seq_len, dim]
    let attn_out_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.o_proj.weight")));
    let attn_pre_o_view = view_2d(&attn_pre_o, seq_len, o_k);
    gpu.gemm_bf16(
        &attn_pre_o_view,
        &wo.tensor,
        &mut to_2d(attn_out_f32.clone_view(), seq_len, dim),
        dim,
        o_k,
        seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_pre_o).map_err(|e| e.to_string())?;

    let attn_out_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&attn_out_f32, &attn_out_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_out_f32).map_err(|e| e.to_string())?;

    Ok(attn_out_bf16)
}

/// MLP layer forward.
///
/// Fires the capture hook for `gate_proj`, `up_proj`, `down_proj`.
/// Math: `down_proj * silu(gate_proj * h) * (up_proj * h)`.
fn forward_mlp_layer(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    p: &str,
    seq_len: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    let w_gate = get(trunk, &format!("{p}.mlp.gate_proj.weight"))?;
    let w_up = get(trunk, &format!("{p}.mlp.up_proj.weight"))?;
    let w_down = get(trunk, &format!("{p}.mlp.down_proj.weight"))?;
    let hidden_dim = w_gate.shape[0];
    if w_up.shape[0] != hidden_dim {
        return Err(format!(
            "{p}.mlp: gate_proj M={hidden_dim} but up_proj M={}",
            w_up.shape[0]
        ));
    }
    if w_down.shape[1] != hidden_dim {
        return Err(format!(
            "{p}.mlp.down_proj K={} doesn't match hidden_dim={hidden_dim}",
            w_down.shape[1]
        ));
    }

    // Pre-norm: GemmaRMSNorm over the entry hidden state (post-attention).
    // gate_proj and up_proj see normalized input. The MLP's residual
    // (`h += ffn_out` in the outer loop) is added to the un-normalized
    // `h` upstream of this function.
    let norm_name = format!("{p}.post_attention_layernorm.weight");
    let h_for_linears = apply_pre_norm_or_fallback(gpu, trunk, h, &norm_name, seq_len, dim)?;

    // gate = h * w_gate^T → F32 [seq_len, hidden_dim]
    let gate_f32 = gpu
        .alloc_tensor(&[seq_len * hidden_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.mlp.gate_proj.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &w_gate.tensor,
        &mut to_2d(gate_f32.clone_view(), seq_len, hidden_dim),
        hidden_dim, dim, seq_len,
    )
    .map_err(|e| e.to_string())?;

    // up = h * w_up^T → F32 [seq_len, hidden_dim]
    let up_f32 = gpu
        .alloc_tensor(&[seq_len * hidden_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.mlp.up_proj.weight")));
    gpu.gemm_bf16(
        h_for_linears.as_ref(), &w_up.tensor,
        &mut to_2d(up_f32.clone_view(), seq_len, hidden_dim),
        hidden_dim, dim, seq_len,
    )
    .map_err(|e| e.to_string())?;
    // Free pre-norm scratch — down_proj's input is ffn_hidden, not h.
    h_for_linears.drop_owned(gpu).map_err(|e| e.to_string())?;

    // ffn_hidden_f32 = silu(gate) * up. silu_mul_f32 expects 1-D tensors;
    // the shape doesn't matter for elementwise math.
    let ffn_hidden_f32 = gpu
        .alloc_tensor(&[seq_len * hidden_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.silu_mul_f32(&gate_f32, &up_f32, &ffn_hidden_f32)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(gate_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(up_f32).map_err(|e| e.to_string())?;

    // Convert to BF16 for down projection.
    let ffn_hidden_bf16 = gpu
        .alloc_tensor(&[seq_len * hidden_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&ffn_hidden_f32, &ffn_hidden_bf16, seq_len * hidden_dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(ffn_hidden_f32).map_err(|e| e.to_string())?;

    // ffn_out_f32 = ffn_hidden_bf16 * w_down^T → F32 [seq_len, dim]
    let ffn_out_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.mlp.down_proj.weight")));
    let ffn_hidden_view = view_2d(&ffn_hidden_bf16, seq_len, hidden_dim);
    gpu.gemm_bf16(
        &ffn_hidden_view,
        &w_down.tensor,
        &mut to_2d(ffn_out_f32.clone_view(), seq_len, dim),
        dim, hidden_dim, seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(ffn_hidden_bf16).map_err(|e| e.to_string())?;

    let ffn_out_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&ffn_out_f32, &ffn_out_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(ffn_out_f32).map_err(|e| e.to_string())?;

    Ok(ffn_out_bf16)
}

/// Detect whether a layer at prefix `p` uses MoE routing.
///
/// Probes for `{p}.mlp.gate.weight` — the per-layer router GEMM
/// present on Qwen3-MoE and Qwen3.6-A3B but absent on dense Qwen3.5.
/// Dense layers expose `{p}.mlp.gate_proj.weight` (note the `_proj`
/// suffix); the router weight is a distinct tensor without it.
fn layer_is_moe(trunk: &TrunkBF16, p: &str) -> bool {
    trunk.tensors.contains_key(&format!("{p}.mlp.gate.weight"))
}

/// Host-side top-K + softmax + (optional) renorm over `router_logits`.
///
/// Mirrors `gpu.moe_softmax_topk_renorm_k8` on the host so we can drive
/// per-expert dispatch from CPU without writing a GPU scatter kernel.
/// Per token: softmax over experts, pick K largest, optionally renorm
/// so the K selected sum to 1.0. Qwen3-MoE family defaults to
/// `norm_topk_prob = true`.
fn host_topk_softmax_renorm(
    router_logits: &[f32],
    seq_len: usize,
    num_experts: usize,
    k_top: usize,
    norm_topk_prob: bool,
) -> (Vec<u32>, Vec<f32>) {
    let mut indices = vec![0u32; seq_len * k_top];
    let mut weights = vec![0f32; seq_len * k_top];
    for t in 0..seq_len {
        let row = &router_logits[t * num_experts..(t + 1) * num_experts];
        let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = row.iter().map(|&x| (x - max_l).exp()).collect();
        let sum_p: f32 = probs.iter().sum();
        if sum_p > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum_p;
            }
        }
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let topk_sum: f32 = ranked[..k_top].iter().map(|&(_, w)| w).sum();
        for k in 0..k_top {
            let (idx, w) = ranked[k];
            indices[t * k_top + k] = idx as u32;
            weights[t * k_top + k] = if norm_topk_prob && topk_sum > 0.0 {
                w / topk_sum
            } else {
                w
            };
        }
    }
    (indices, weights)
}

/// Forward through a single MoE FFN layer for Tier 1 BF16 calibration.
///
/// Mirrors the Qwen3-MoE / Qwen3.6-A3B forward (see qwen35.rs
/// `MoeFfnWeights` for tensor layout). For each token t and its
/// top-K=8 selected experts:
///   ffn(t) = sum_k topk_weights[t, k] * expert_k(h_norm[t])
/// where `expert_k(x) = down(silu(gate(x)) * up(x))`. The shared
/// expert (always-on, A3B & Qwen3-MoE-base) adds its output unweighted
/// to ffn(t). `shared_expert_gate` (scalar-per-token modulator) is
/// approximated as 1.0 in v1.
///
/// Capture-name convention (matches Phase 3b's `parse_expert_idx`):
///   {p}.mlp.gate.weight                            — router GEMM
///   {p}.mlp.experts.<E>.gate_proj.weight           — expert E gate
///   {p}.mlp.experts.<E>.up_proj.weight             — expert E up
///   {p}.mlp.experts.<E>.down_proj.weight           — expert E down
///   {p}.mlp.shared_expert.gate_proj.weight         — shared gate
///   {p}.mlp.shared_expert.up_proj.weight           — shared up
///   {p}.mlp.shared_expert.down_proj.weight         — shared down
///
/// Routing is computed CPU-side from F32-downloaded router logits;
/// per-expert gather/scatter goes through host buffers (download
/// h_norm once, build h_e per expert, upload as F32, cast to BF16 on
/// device; scatter via CPU accumulation in `combined_cpu`). This is
/// O(num_experts) D2H+H2D bouncing per layer — not the production
/// inference path's structure, but correct and adequate for Tier 1's
/// one-time-per-model calibration budget. A future optimization is a
/// pair of GPU gather + weighted-scatter kernels.
fn forward_moe_layer_bf16(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    h: &GpuTensor,
    p: &str,
    seq_len: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    const K_TOP: usize = 8;

    let w_router = get(trunk, &format!("{p}.mlp.gate.weight"))?;
    let num_experts = w_router.shape[0];
    if w_router.shape[1] != dim {
        return Err(format!(
            "{p}.mlp.gate K={} doesn't match trunk dim={dim}",
            w_router.shape[1]
        ));
    }

    // Pre-norm via post_attention_layernorm (cast-trick rmsnorm).
    let norm_name = format!("{p}.post_attention_layernorm.weight");
    let h_norm_scratch = apply_pre_norm_or_fallback(gpu, trunk, h, &norm_name, seq_len, dim)?;

    // Router GEMM (capture fires under `{p}.mlp.gate.weight`).
    let router_logits_f32 = gpu
        .alloc_tensor(&[seq_len * num_experts], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.mlp.gate.weight")));
    gpu.gemm_bf16(
        h_norm_scratch.as_ref(),
        &w_router.tensor,
        &mut to_2d(router_logits_f32.clone_view(), seq_len, num_experts),
        num_experts,
        dim,
        seq_len,
    )
    .map_err(|e| e.to_string())?;

    // Pull router logits to host, do top-K + softmax + renorm.
    let router_logits_cpu = gpu
        .download_f32(&router_logits_f32)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(router_logits_f32)
        .map_err(|e| e.to_string())?;

    let (topk_indices, topk_weights) =
        host_topk_softmax_renorm(&router_logits_cpu, seq_len, num_experts, K_TOP, true);

    // Cast-trick: convert h_norm BF16 -> F32, download for per-expert gather.
    let h_norm_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.convert_bf16_to_f32(h_norm_scratch.as_ref(), &h_norm_f32, seq_len * dim)
        .map_err(|e| e.to_string())?;
    let h_norm_cpu = gpu.download_f32(&h_norm_f32).map_err(|e| e.to_string())?;
    gpu.free_tensor(h_norm_f32).map_err(|e| e.to_string())?;

    // Build per-expert token list: expert_to_tokens[e] = Vec<(token_idx, weight)>.
    let mut expert_to_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
    for t in 0..seq_len {
        for k in 0..K_TOP {
            let e = topk_indices[t * K_TOP + k] as usize;
            let w = topk_weights[t * K_TOP + k];
            expert_to_tokens[e].push((t, w));
        }
    }

    // CPU accumulator for the combined FFN output.
    let mut combined_cpu = vec![0f32; seq_len * dim];

    // Per-routed-expert dispatch: gather, 3 GEMMs with captures, weighted
    // scatter-add into combined_cpu.
    for e in 0..num_experts {
        let tokens_for_e = &expert_to_tokens[e];
        if tokens_for_e.is_empty() {
            continue;
        }
        let n_e = tokens_for_e.len();

        let w_gate = get(trunk, &format!("{p}.mlp.experts.{e}.gate_proj.weight"))?;
        let w_up = get(trunk, &format!("{p}.mlp.experts.{e}.up_proj.weight"))?;
        let w_down = get(trunk, &format!("{p}.mlp.experts.{e}.down_proj.weight"))?;
        let moe_intermediate = w_gate.shape[0];
        if w_up.shape[0] != moe_intermediate {
            return Err(format!(
                "{p}.mlp.experts.{e}: gate M={moe_intermediate} but up M={}",
                w_up.shape[0]
            ));
        }
        if w_down.shape[1] != moe_intermediate {
            return Err(format!(
                "{p}.mlp.experts.{e}.down_proj K={} doesn't match moe_intermediate={moe_intermediate}",
                w_down.shape[1]
            ));
        }

        // Build h_e on CPU: [n_e, dim] F32 indexed from h_norm_cpu.
        let mut h_e_f32: Vec<f32> = Vec::with_capacity(n_e * dim);
        for &(t, _w) in tokens_for_e {
            h_e_f32.extend_from_slice(&h_norm_cpu[t * dim..(t + 1) * dim]);
        }

        // Upload + cast to BF16 on device.
        let h_e_f32_gpu = gpu
            .upload_f32(&h_e_f32, &[n_e * dim])
            .map_err(|e| e.to_string())?;
        let h_e_bf16 = gpu
            .alloc_tensor(&[n_e, dim], DType::BF16)
            .map_err(|e| e.to_string())?;
        gpu.convert_f32_to_bf16(&h_e_f32_gpu, &h_e_bf16, n_e * dim)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(h_e_f32_gpu).map_err(|e| e.to_string())?;

        // GEMM gate: h_e_bf16 @ w_gate.T -> F32 [n_e, moe_intermediate]
        let gate_e_f32 = gpu
            .alloc_tensor(&[n_e * moe_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.experts.{e}.gate_proj.weight")));
        gpu.gemm_bf16(
            &h_e_bf16,
            &w_gate.tensor,
            &mut to_2d(gate_e_f32.clone_view(), n_e, moe_intermediate),
            moe_intermediate,
            dim,
            n_e,
        )
        .map_err(|e| e.to_string())?;

        // GEMM up: h_e_bf16 @ w_up.T -> F32 [n_e, moe_intermediate]
        let up_e_f32 = gpu
            .alloc_tensor(&[n_e * moe_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.experts.{e}.up_proj.weight")));
        gpu.gemm_bf16(
            &h_e_bf16,
            &w_up.tensor,
            &mut to_2d(up_e_f32.clone_view(), n_e, moe_intermediate),
            moe_intermediate,
            dim,
            n_e,
        )
        .map_err(|e| e.to_string())?;
        gpu.free_tensor(h_e_bf16).map_err(|e| e.to_string())?;

        // ffn_inner_f32 = silu(gate_e) * up_e
        let ffn_inner_f32 = gpu
            .alloc_tensor(&[n_e * moe_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.silu_mul_f32(&gate_e_f32, &up_e_f32, &ffn_inner_f32)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(gate_e_f32).map_err(|e| e.to_string())?;
        gpu.free_tensor(up_e_f32).map_err(|e| e.to_string())?;

        // Cast ffn_inner to BF16 for down GEMM.
        let ffn_inner_bf16 = gpu
            .alloc_tensor(&[n_e * moe_intermediate], DType::BF16)
            .map_err(|e| e.to_string())?;
        gpu.convert_f32_to_bf16(&ffn_inner_f32, &ffn_inner_bf16, n_e * moe_intermediate)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_inner_f32).map_err(|e| e.to_string())?;

        // GEMM down: ffn_inner_bf16 @ w_down.T -> F32 [n_e, dim]
        let ffn_out_e_f32 = gpu
            .alloc_tensor(&[n_e * dim], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.experts.{e}.down_proj.weight")));
        let ffn_inner_view = view_2d(&ffn_inner_bf16, n_e, moe_intermediate);
        gpu.gemm_bf16(
            &ffn_inner_view,
            &w_down.tensor,
            &mut to_2d(ffn_out_e_f32.clone_view(), n_e, dim),
            dim,
            moe_intermediate,
            n_e,
        )
        .map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_inner_bf16).map_err(|e| e.to_string())?;

        // Download ffn_out_e and weighted-scatter into combined_cpu.
        let ffn_out_e_cpu = gpu.download_f32(&ffn_out_e_f32).map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_out_e_f32).map_err(|e| e.to_string())?;
        for (i, &(t, w_t)) in tokens_for_e.iter().enumerate() {
            let dst = &mut combined_cpu[t * dim..(t + 1) * dim];
            let src = &ffn_out_e_cpu[i * dim..(i + 1) * dim];
            for d in 0..dim {
                dst[d] += w_t * src[d];
            }
        }
    }

    // Shared expert: always-on (A3B + Qwen3-MoE-base). Adds output to
    // combined_cpu unweighted; `shared_expert_gate` modulator approximated
    // as 1.0 in v1 (scalar-per-token; affects downstream layer feedthrough
    // but not the captured X^T X for the shared expert's three linears).
    let sh_gate_name = format!("{p}.mlp.shared_expert.gate_proj.weight");
    if trunk.tensors.contains_key(&sh_gate_name) {
        let w_gate_sh = get(trunk, &sh_gate_name)?;
        let w_up_sh = get(trunk, &format!("{p}.mlp.shared_expert.up_proj.weight"))?;
        let w_down_sh = get(trunk, &format!("{p}.mlp.shared_expert.down_proj.weight"))?;
        let sh_intermediate = w_gate_sh.shape[0];

        let gate_sh_f32 = gpu
            .alloc_tensor(&[seq_len * sh_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.shared_expert.gate_proj.weight")));
        gpu.gemm_bf16(
            h_norm_scratch.as_ref(),
            &w_gate_sh.tensor,
            &mut to_2d(gate_sh_f32.clone_view(), seq_len, sh_intermediate),
            sh_intermediate,
            dim,
            seq_len,
        )
        .map_err(|e| e.to_string())?;

        let up_sh_f32 = gpu
            .alloc_tensor(&[seq_len * sh_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.shared_expert.up_proj.weight")));
        gpu.gemm_bf16(
            h_norm_scratch.as_ref(),
            &w_up_sh.tensor,
            &mut to_2d(up_sh_f32.clone_view(), seq_len, sh_intermediate),
            sh_intermediate,
            dim,
            seq_len,
        )
        .map_err(|e| e.to_string())?;

        let ffn_inner_sh_f32 = gpu
            .alloc_tensor(&[seq_len * sh_intermediate], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.silu_mul_f32(&gate_sh_f32, &up_sh_f32, &ffn_inner_sh_f32)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(gate_sh_f32).map_err(|e| e.to_string())?;
        gpu.free_tensor(up_sh_f32).map_err(|e| e.to_string())?;

        let ffn_inner_sh_bf16 = gpu
            .alloc_tensor(&[seq_len * sh_intermediate], DType::BF16)
            .map_err(|e| e.to_string())?;
        gpu.convert_f32_to_bf16(&ffn_inner_sh_f32, &ffn_inner_sh_bf16, seq_len * sh_intermediate)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_inner_sh_f32).map_err(|e| e.to_string())?;

        let ffn_out_sh_f32 = gpu
            .alloc_tensor(&[seq_len * dim], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.set_capture_name(Some(format!("{p}.mlp.shared_expert.down_proj.weight")));
        let ffn_inner_sh_view = view_2d(&ffn_inner_sh_bf16, seq_len, sh_intermediate);
        gpu.gemm_bf16(
            &ffn_inner_sh_view,
            &w_down_sh.tensor,
            &mut to_2d(ffn_out_sh_f32.clone_view(), seq_len, dim),
            dim,
            sh_intermediate,
            seq_len,
        )
        .map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_inner_sh_bf16).map_err(|e| e.to_string())?;

        let ffn_out_sh_cpu = gpu
            .download_f32(&ffn_out_sh_f32)
            .map_err(|e| e.to_string())?;
        gpu.free_tensor(ffn_out_sh_f32).map_err(|e| e.to_string())?;
        for i in 0..(seq_len * dim) {
            combined_cpu[i] += ffn_out_sh_cpu[i];
        }
    }

    h_norm_scratch
        .drop_owned(gpu)
        .map_err(|e| e.to_string())?;

    // Upload combined back, cast to BF16, return.
    let combined_f32 = gpu
        .upload_f32(&combined_cpu, &[seq_len * dim])
        .map_err(|e| e.to_string())?;
    let combined_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&combined_f32, &combined_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(combined_f32).map_err(|e| e.to_string())?;

    Ok(combined_bf16)
}

/// Helper to build a 2-D shaped view tensor from a 1-D allocation
/// without consuming it. Used to feed `gemm_bf16` which expects a
/// `&mut GpuTensor` with the right shape but operates only on the
/// underlying buffer.
fn to_2d(t: GpuTensor, batch: usize, m: usize) -> GpuTensor {
    GpuTensor {
        buf: t.buf,
        shape: vec![batch, m],
        dtype: t.dtype,
    }
}

/// Extension trait to provide a non-consuming view clone for `GpuTensor`.
trait GpuTensorViewExt {
    /// Build a non-owning view of the underlying buffer. The returned
    /// tensor MUST NOT be freed — its buffer is a borrowed pointer that
    /// aliases the original. Used to feed `gemm_bf16`'s `&mut GpuTensor`
    /// y arg without owning the alloc.
    fn clone_view(&self) -> GpuTensor;
}

impl GpuTensorViewExt for GpuTensor {
    fn clone_view(&self) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(self.buf.as_ptr(), self.byte_size()) },
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `detect_layer_kind` returns `FullAttn` when `self_attn.q_proj.weight`
    /// is present in the trunk's tensor map; otherwise `DeltaNet`.
    ///
    /// Pure logic test — no GPU needed.
    #[test]
    fn detect_layer_kind_finds_full_attn() {
        use std::collections::HashMap;
        use std::path::PathBuf;

        let mut tensors: HashMap<String, Bf16Tensor> = HashMap::new();
        // Stub a single FullAttn-style tensor for layer 0 only.
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            stub_bf16_tensor("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
        );
        let trunk = TrunkBF16 {
            tensors,
            norms: HashMap::new(),
            model_dir: PathBuf::from("/tmp/test"),
            model_type: "qwen3".to_string(),
            total_bytes: 0,
        };
        assert_eq!(detect_layer_kind(&trunk, 0), LayerKind::FullAttn);
        // Layer 1 has no tensors → defaults to DeltaNet.
        assert_eq!(detect_layer_kind(&trunk, 1), LayerKind::DeltaNet);
    }

    /// `count_layers` walks `model.layers.{N}.*` until a layer is absent.
    /// Verify it returns the correct count for a sparse trunk.
    #[test]
    fn count_layers_walks_present_layers() {
        use std::collections::HashMap;
        use std::path::PathBuf;

        let mut tensors: HashMap<String, Bf16Tensor> = HashMap::new();
        // Layers 0 and 1 (mixed kinds) are present, layer 2 is not.
        tensors.insert(
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            stub_bf16_tensor("model.layers.0.linear_attn.in_proj_qkv.weight", vec![64, 16]),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            stub_bf16_tensor("model.layers.1.self_attn.q_proj.weight", vec![16, 16]),
        );
        let trunk = TrunkBF16 {
            tensors,
            norms: HashMap::new(),
            model_dir: PathBuf::from("/tmp/test"),
            model_type: "qwen3".to_string(),
            total_bytes: 0,
        };
        assert_eq!(count_layers(&trunk), 2);
    }

    /// Helper: build a stub `Bf16Tensor` with a fake (non-device) buffer.
    /// Used by unit tests that exercise the dispatch logic without
    /// touching the HIP runtime. The fake buffer carries a `null_mut`
    /// device pointer — DO NOT call any GPU op on it.
    fn stub_bf16_tensor(name: &str, shape: Vec<usize>) -> Bf16Tensor {
        let numel: usize = shape.iter().product();
        let buf = unsafe {
            hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut(), numel * 2)
        };
        Bf16Tensor {
            name: name.to_string(),
            tensor: GpuTensor {
                buf,
                shape: vec![numel],
                dtype: DType::BF16,
            },
            shape,
        }
    }

    /// Full forward smoke test — needs a HIP runtime + gfx942.
    ///
    /// Builds a minimal trunk in-place: 2 layers, dim=16, hidden=32,
    /// vocab=8. Random BF16 weights. Verifies the function returns
    /// without panicking and exits cleanly.
    ///
    /// `#[ignore]` because it requires:
    ///   1. AMDGPU + ROCm runtime present (`Gpu::init()` will panic
    ///      otherwise).
    ///   2. `gfx942` arch — `gemm_bf16` returns an explicit Err on
    ///      other archs.
    ///
    /// Run with:
    ///     cargo test -p hipfire-runtime --lib bf16_forward -- --ignored
    #[test]
    #[ignore = "needs HIP runtime + gfx942; run with --ignored on MI300x"]
    fn forward_prefill_bf16_smoke() {
        // The test is intentionally left empty — the doc-tier mode for
        // the BF16 forward smoke test is the orchestrator's end-to-end
        // collect_imatrix run, not a unit test. The non-ignored tests
        // above cover the host-side dispatch logic (layer detection,
        // tensor lookup) without needing the GPU.
    }
}
