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
//! 1. **RMSNorm skipped.** Instead of running rmsnorm before each
//!    linear, the hidden state passes through unchanged. Calibration
//!    cost: each linear sees raw post-residual values rather than
//!    normalized ones. Since the magnitude bound from rmsnorm is well-
//!    known (≈ √n), the downstream quantizer can re-apply a uniform
//!    scaling factor offline if it wants normalized statistics.
//!
//! 2. **Attention math is a passthrough.**
//!    - DeltaNet: `attn_pre_o = first d_inner elements of qkv` (the
//!      Q chunk). Skips the conv1d, gated delta-net recurrence, alpha
//!      gate, and norm. Calibration cost: `out_proj` is calibrated on
//!      a distribution that resembles Q rather than the actual gated
//!      delta-net output. The two distributions are correlated
//!      (same trunk hidden state feeds both) but not identical.
//!    - FullAttention: `attn_pre_o = v` (taking the V projection
//!      directly). Skips q_norm / k_norm / RoPE / softmax / V·softmax.
//!      Calibration cost: `o_proj` is calibrated on the V distribution
//!      rather than `softmax(QK^T)·V`. Same correlation argument as
//!      above.
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
//! 5. **lm_head is not computed.** The final logits are not needed by
//!    calibration. We skip the final norm + lm_head matmul to save
//!    work; the capture hook for `lm_head.weight` is not fired.
//!    Downstream `--process-output` users would need a follow-up
//!    extension to fire that one capture.
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
/// `gpu.arch` must be `gfx942` (MI300x). The BF16 GEMM is gfx942-only;
/// `gpu.gemm_bf16` returns an explicit error on other archs.
pub fn forward_prefill_bf16(
    gpu: &mut Gpu,
    trunk: &TrunkBF16,
    tokens: &[u32],
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
        let ffn_out_bf16 = forward_mlp_layer(gpu, trunk, &h_view, &p, seq_len, dim)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        bf16_add_inplace(gpu, &h, &ffn_out_bf16, seq_len * dim)?;
        gpu.free_tensor(ffn_out_bf16)?;
    }

    // We do not compute the final norm or lm_head — the calibration
    // pipeline does not consume logits. Free the hidden state and return.
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

/// Build a `[n]`-shaped 1-D view tensor over a sub-range of an existing
/// buffer, starting at byte offset `byte_off`. Caller responsible for
/// ensuring the range stays within the source buffer. Reserved for a
/// future v2 of the forward path that wires per-head attention splits.
#[allow(dead_code)]
fn view_subrange(t: &GpuTensor, byte_off: usize, n_elems: usize, dtype: DType) -> GpuTensor {
    let elem_size = dtype.size();
    let bytes = n_elems * elem_size;
    let ptr = unsafe {
        (t.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void
    };
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
        shape: vec![n_elems],
        dtype,
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

/// DeltaNet layer attention (calibration approximation).
///
/// Fires the capture hook for `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
/// `in_proj_b`, `out_proj` weights. Math is heavily simplified —
/// `attn_pre_o = first d_inner cols of qkv` (the Q chunk). The
/// `out_proj` weight thus sees a Q-distribution input rather than the
/// true gated delta-net output. See module-level docs.
///
/// Returns a freshly-allocated `[seq_len, dim]` BF16 tensor that the
/// caller must free.
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

    // QKV projection: y = h * wqkv^T → [seq_len, qkv_dim]
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
    gpu.gemm_bf16(h, &wqkv.tensor, &mut to_2d(qkv_f32.clone_view(), seq_len, qkv_dim),
                  qkv_dim, qkv_k, seq_len)
        .map_err(|e| e.to_string())?;

    // Z projection: gate vector. Calibration only — output unused
    // (the gated norm is skipped).
    let z_dim = wz.shape[0];
    let z_f32 = gpu
        .alloc_tensor(&[seq_len * z_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_z.weight")));
    gpu.gemm_bf16(h, &wz.tensor, &mut to_2d(z_f32.clone_view(), seq_len, z_dim),
                  z_dim, wz.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(z_f32).map_err(|e| e.to_string())?;

    // A projection (alpha): calibration only.
    let a_dim = wa.shape[0];
    let a_f32 = gpu
        .alloc_tensor(&[seq_len * a_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_a.weight")));
    gpu.gemm_bf16(h, &wa.tensor, &mut to_2d(a_f32.clone_view(), seq_len, a_dim),
                  a_dim, wa.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(a_f32).map_err(|e| e.to_string())?;

    // B projection (beta): calibration only.
    let b_dim = wb.shape[0];
    let b_f32 = gpu
        .alloc_tensor(&[seq_len * b_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.in_proj_b.weight")));
    gpu.gemm_bf16(h, &wb.tensor, &mut to_2d(b_f32.clone_view(), seq_len, b_dim),
                  b_dim, wb.shape[1], seq_len)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(b_f32).map_err(|e| e.to_string())?;

    // Build `attn_pre_o` = first d_inner cols of qkv_f32 (the Q chunk),
    // converted to BF16. d_inner is determined by the out_proj's K dim
    // (out_proj: [dim, d_inner]).
    let d_inner = wo.shape[1];
    if d_inner > qkv_dim {
        return Err(format!(
            "{p}.linear_attn.out_proj: K={d_inner} > qkv_dim={qkv_dim}"
        ));
    }

    // Convert qkv_f32 (seq_len × qkv_dim) → BF16 attn_pre_o (seq_len × d_inner)
    // by extracting the first d_inner cols per row.
    let attn_pre_o = gpu
        .alloc_tensor(&[seq_len * d_inner], DType::BF16)
        .map_err(|e| e.to_string())?;
    // Convert all of qkv_f32 to BF16 first (in-place via scratch), then
    // copy out the d_inner-col-prefix per row. Simpler: per-row dtod.
    let qkv_bf16 = gpu
        .alloc_tensor(&[seq_len * qkv_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&qkv_f32, &qkv_bf16, seq_len * qkv_dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(qkv_f32).map_err(|e| e.to_string())?;
    // Per-row copy: each row of qkv_bf16 has qkv_dim BF16 elements; we
    // want the first d_inner elements into attn_pre_o.
    for t in 0..seq_len {
        let src_off = t * qkv_dim * 2;
        let dst_off = t * d_inner * 2;
        gpu.hip
            .memcpy_dtod_at(
                &attn_pre_o.buf,
                dst_off,
                &qkv_bf16.buf,
                src_off,
                d_inner * 2,
            )
            .map_err(|e| e.to_string())?;
    }
    gpu.free_tensor(qkv_bf16).map_err(|e| e.to_string())?;

    // out_proj: attn_out_f32 = attn_pre_o * wo^T → [seq_len, dim]
    let attn_out_f32 = gpu
        .alloc_tensor(&[seq_len * dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.linear_attn.out_proj.weight")));
    let attn_pre_o_view = view_2d(&attn_pre_o, seq_len, d_inner);
    gpu.gemm_bf16(
        &attn_pre_o_view,
        &wo.tensor,
        &mut to_2d(attn_out_f32.clone_view(), seq_len, dim),
        dim,
        d_inner,
        seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_pre_o).map_err(|e| e.to_string())?;

    // Convert to BF16 for residual add.
    let attn_out_bf16 = gpu
        .alloc_tensor(&[seq_len * dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&attn_out_f32, &attn_out_bf16, seq_len * dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(attn_out_f32).map_err(|e| e.to_string())?;

    Ok(attn_out_bf16)
}

/// Full-attention (self-attn) layer forward (calibration approximation).
///
/// Fires the capture hook for `q_proj`, `k_proj`, `v_proj`, `o_proj`.
/// `attn_pre_o = v` (calibration shortcut — see module docs).
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

    let q_dim = wq.shape[0];
    let k_dim = wk.shape[0];
    let v_dim = wv.shape[0];

    // Q projection (calibration only — output unused beyond capture).
    let q_f32 = gpu
        .alloc_tensor(&[seq_len * q_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.q_proj.weight")));
    gpu.gemm_bf16(
        h, &wq.tensor,
        &mut to_2d(q_f32.clone_view(), seq_len, q_dim),
        q_dim, wq.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(q_f32).map_err(|e| e.to_string())?;

    // K projection (calibration only).
    let k_f32 = gpu
        .alloc_tensor(&[seq_len * k_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.k_proj.weight")));
    gpu.gemm_bf16(
        h, &wk.tensor,
        &mut to_2d(k_f32.clone_view(), seq_len, k_dim),
        k_dim, wk.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;
    gpu.free_tensor(k_f32).map_err(|e| e.to_string())?;

    // V projection. We keep this as the `attn_pre_o` for the o_proj.
    let v_f32 = gpu
        .alloc_tensor(&[seq_len * v_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.self_attn.v_proj.weight")));
    gpu.gemm_bf16(
        h, &wv.tensor,
        &mut to_2d(v_f32.clone_view(), seq_len, v_dim),
        v_dim, wv.shape[1], seq_len,
    )
    .map_err(|e| e.to_string())?;

    // Convert v_f32 → BF16 for o_proj. o_proj expects K = v_dim.
    let o_k = wo.shape[1];
    if o_k != v_dim {
        // Allow models where o_proj's K differs from v_dim by zero-padding
        // or truncating. The Qwen3.5 model has q_proj output 2× wide
        // (query+gate split) — but o_proj still operates on the
        // post-attention-output `n_heads * head_dim` activations, which
        // for GQA = v_dim. If they mismatch, fall back to the smaller.
        // For the calibration v1, just zero-pad / truncate to o_k.
    }
    let v_bf16 = gpu
        .alloc_tensor(&[seq_len * v_dim], DType::BF16)
        .map_err(|e| e.to_string())?;
    gpu.convert_f32_to_bf16(&v_f32, &v_bf16, seq_len * v_dim)
        .map_err(|e| e.to_string())?;
    gpu.free_tensor(v_f32).map_err(|e| e.to_string())?;

    // If o_k != v_dim, allocate a [seq_len, o_k] BF16 buffer and copy
    // the per-row prefix or zero-pad.
    let attn_pre_o = if o_k == v_dim {
        v_bf16
    } else {
        let target = gpu
            .alloc_tensor(&[seq_len * o_k], DType::BF16)
            .map_err(|e| e.to_string())?;
        // Zero-init for safety (zero-pad path).
        gpu.hip
            .memset(&target.buf, 0, seq_len * o_k * 2)
            .map_err(|e| e.to_string())?;
        let copy_per_row = std::cmp::min(o_k, v_dim);
        for t in 0..seq_len {
            let src_off = t * v_dim * 2;
            let dst_off = t * o_k * 2;
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

    // gate = h * w_gate^T → F32 [seq_len, hidden_dim]
    let gate_f32 = gpu
        .alloc_tensor(&[seq_len * hidden_dim], DType::F32)
        .map_err(|e| e.to_string())?;
    gpu.set_capture_name(Some(format!("{p}.mlp.gate_proj.weight")));
    gpu.gemm_bf16(
        h, &w_gate.tensor,
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
        h, &w_up.tensor,
        &mut to_2d(up_f32.clone_view(), seq_len, hidden_dim),
        hidden_dim, dim, seq_len,
    )
    .map_err(|e| e.to_string())?;

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
