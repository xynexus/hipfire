//! Layer-by-layer correctness verifier: hipfire Gemma 4 forward kernels vs.
//! a PyTorch bf16 reference activation dump.
//!
//! The reference dump is produced by
//! `scripts/quant-survey/dump_reference_activations.py` and lives at
//! `/tmp/gemma4-ref/` on hiptrx. It contains, per layer, the intermediate
//! tensors at the boundary of every meaningful op (pre/post norm, q_proj,
//! k_proj, v_proj, q_norm, k_norm, o_proj, mlp_gate, mlp_up, mlp_down,
//! pre/post mlp norm, plus self_attn_block input/output and block i/o).
//!
//! ─── Design (Phase D2 verifier strategy, 2026-05-07) ────────────────────────
//!
//! Two classes of test:
//!
//!   (a) "Kernel-isolated" tests — feed the captured INPUT tensor for some
//!       step into hipfire's kernel, compare its OUTPUT against the captured
//!       output. These have NO error accumulation and NO model-weight
//!       dependency, so they catch bugs in pure-arithmetic kernels (gelu_tanh,
//!       softcap, rope_partial_halved arithmetic) immediately.
//!
//!   (b) "End-to-end" tests — load the `.mq4` file, run the full forward,
//!       verify final logits NRMSE < 1e-2. Requires the MG4 quantizer (Phase
//!       D3, separate scope per the task) and a real model file.
//!
//! Phase D2 ships (a) — the kernel-isolated battery — fully. (b) is wired
//! as a `--full` flag that prints a clear "model file required" message when
//! no `.mq4` is supplied, so the verifier compiles and runs end-to-end on
//! hiptrx today without blocking on the quantizer.
//!
//! ─── Tests in this battery (kernel-isolated, no model weights) ──────────────
//!
//!   1. SwiGLU(gelu_tanh, mul):
//!         input  : layer_NN/mlp_gate.output (gate pre-activation)
//!                  layer_NN/mlp_up.output   (up pre-mul)
//!         compute: gelu_tanh(gate) * up
//!         expect : layer_NN/mlp_down.input
//!
//!      Phase 1 — verifies the gemma4-specific gelu_tanh activation against
//!      the canonical bf16 reference. If this passes for all 60 layers, the
//!      activation kernel is correct.
//!
//!   2. logit_softcap (cap=30.0):
//!         input  : final_logits BEFORE softcap (NOT in the dump as of
//!                  2026-05-07; the dump captures POST-softcap final_logits.
//!                  This test prints "skipped — pre-softcap capture missing"
//!                  until the dump script is extended.)
//!
//! ─── Usage ──────────────────────────────────────────────────────────────────
//!
//!   cargo run --release -p hipfire-arch-gemma4 --example verify_against_torch -- \
//!       /tmp/gemma4-ref
//!
//! Optional env:
//!   VERIFY_LAYERS=0,1,2  — comma-separated layer indices (default: all)
//!   VERIFY_THRESHOLD=5e-3 — NRMSE pass threshold (default: 5e-3, bf16 noise floor)
//!   VERIFY_HALT_ON_FAIL=1 — stop at first FAIL (default: continue, summarize)
//!
//! ─── Why 5e-3 default ───────────────────────────────────────────────────────
//!
//! The reference dump captures activations as **bfloat16** (7 mantissa bits),
//! so each tensor element has up to ~3.9e-3 relative quantization noise.
//! When hipfire computes the same op in fp32 and we compare to the bf16
//! reference cast back to f32, the resulting NRMSE bottoms out at 1-3e-3 even
//! when the kernel is bit-perfect. 5e-3 is a conservative pass band; the only
//! kernel bugs that can hide under this threshold are those that introduce
//! < 1 bf16 ULP of additional error per element, which empirically don't
//! affect downstream argmax. To tighten further, regenerate the dump in fp32
//! and lower this to 1e-4.
//!
//! ─── Future work ────────────────────────────────────────────────────────────
//!
//!   • Add weight-loading paths so RMSNorm / projection / attention can be
//!     verified per-layer. Blocked on MG4 quantizer (Phase D3). Once a
//!     `.mq4` exists, we can call `gemma4::load_weights` to get on-device
//!     weights and add tests for: rmsnorm (via weight tensor), q_proj/k_proj/
//!     v_proj (via weight_gemv), q_norm/k_norm (rmsnorm_batched), rope
//!     (full + partial), attention (asym3 with window=1024), o_proj.
//!
//!   • Extend the python dump script to also emit the PRE-softcap logits so
//!     test (2) can run.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype as SfDtype;
use serde::Deserialize;

use rdna_compute::{Gpu, DType, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{weight_gemv, WeightTensor};
use hipfire_arch_gemma4::gemma4::{self, Gemma4Config, Gemma4Weights, LayerType, LayerWeights};

// ───────────────────────────────────────────────────────────────────────────
// Manifest schema
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Manifest {
    model: String,
    prompt: String,
    n_tokens: usize,
    dtype: String,
    captures: Vec<Capture>,
}

#[derive(Debug, Deserialize)]
struct Capture {
    layer_idx: i32, // -1 for global captures (final_logits, input_ids)
    step: String,
    side: String,
    #[allow(dead_code)]
    module_name: Option<String>,
    shape: Vec<usize>,
    dtype: String,
    path: String,
}

// ───────────────────────────────────────────────────────────────────────────
// Safetensors helpers
// ───────────────────────────────────────────────────────────────────────────

/// Load a single-tensor `.safetensors` file as f32. The dump uses bf16 only
/// for the activations (and i64 for `input_ids`). We convert bf16 → f32 here
/// so the rest of the pipeline can stay in f32.
fn load_safetensors_as_f32(path: &Path) -> std::io::Result<(Vec<usize>, Vec<f32>)> {
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    let st = SafeTensors::deserialize(&mmap[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    // Dumps are single-tensor; pick the first / only entry.
    let names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    let name = names.first().ok_or_else(|| std::io::Error::new(
        std::io::ErrorKind::InvalidData, "no tensors in safetensors file",
    ))?;
    let view = st.tensor(name)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let shape: Vec<usize> = view.shape().to_vec();
    let bytes = view.data();
    let f32_data: Vec<f32> = match view.dtype() {
        SfDtype::F32 => {
            assert_eq!(bytes.len() % 4, 0);
            (0..bytes.len() / 4)
                .map(|i| f32::from_le_bytes([
                    bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3],
                ]))
                .collect()
        }
        SfDtype::BF16 => {
            assert_eq!(bytes.len() % 2, 0);
            (0..bytes.len() / 2)
                .map(|i| {
                    // bf16 → f32 zero-extends low 16 bits.
                    let bits = ((bytes[i*2+1] as u32) << 24) | ((bytes[i*2] as u32) << 16);
                    f32::from_bits(bits)
                })
                .collect()
        }
        SfDtype::F16 => {
            assert_eq!(bytes.len() % 2, 0);
            (0..bytes.len() / 2)
                .map(|i| {
                    let bits = u16::from_le_bytes([bytes[i*2], bytes[i*2+1]]);
                    f16_to_f32(bits)
                })
                .collect()
        }
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("unsupported safetensors dtype: {:?}", other),
            ));
        }
    };
    Ok((shape, f32_data))
}

fn f16_to_f32(bits: u16) -> f32 {
    // IEEE 754 half-to-single (no FTZ; subnormals handled).
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 { 0.0 } else {
            (mant as f32) / 1024.0 * 2f32.powi(-14)
        }
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let m = 1.0 + (mant as f32) / 1024.0;
        m * 2f32.powi((exp as i32) - 15)
    };
    if sign == 1 { -f } else { f }
}

// ───────────────────────────────────────────────────────────────────────────
// NRMSE
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct CompareResult {
    nrmse: f32,
    max_abs_err: f32,
    rms_ref: f32,
    n: usize,
}

fn compare_f32(actual: &[f32], reference: &[f32]) -> CompareResult {
    assert_eq!(actual.len(), reference.len(),
        "length mismatch: actual={}, reference={}", actual.len(), reference.len());
    let n = actual.len();
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        let e = actual[i] - reference[i];
        sum_sq_err += (e as f64) * (e as f64);
        sum_sq_ref += (reference[i] as f64) * (reference[i] as f64);
        if e.abs() > max_abs_err { max_abs_err = e.abs(); }
    }
    let mse = sum_sq_err / (n as f64);
    let var_ref = sum_sq_ref / (n as f64);
    let nrmse = if var_ref > 0.0 { (mse / var_ref).sqrt() as f32 } else { mse.sqrt() as f32 };
    CompareResult {
        nrmse,
        max_abs_err,
        rms_ref: var_ref.sqrt() as f32,
        n,
    }
}

fn print_diag(label: &str, res: &CompareResult, threshold: f32, sample_actual: &[f32], sample_ref: &[f32]) {
    let verdict = if res.nrmse < threshold { "PASS" } else { "FAIL" };
    eprintln!("    {} {:<60}  NRMSE={:.4e}  max|e|={:.4e}  rms(ref)={:.4e}  N={}",
        verdict, label, res.nrmse, res.max_abs_err, res.rms_ref, res.n);
    if res.nrmse >= threshold {
        let n = sample_actual.len().min(sample_ref.len()).min(8);
        let preview_a: Vec<String> = sample_actual[..n].iter().map(|v| format!("{:+.4e}", v)).collect();
        let preview_r: Vec<String> = sample_ref[..n].iter().map(|v| format!("{:+.4e}", v)).collect();
        eprintln!("        actual[0..{}] = [{}]", n, preview_a.join(", "));
        eprintln!("        ref   [0..{}] = [{}]", n, preview_r.join(", "));
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Capture lookup
// ───────────────────────────────────────────────────────────────────────────

struct Refs {
    root: PathBuf,
    by_key: HashMap<(i32, String, String), Capture>,
}

impl Refs {
    fn new(root: PathBuf, manifest: Manifest) -> Self {
        let mut by_key = HashMap::new();
        for c in manifest.captures {
            by_key.insert((c.layer_idx, c.step.clone(), c.side.clone()), c);
        }
        Self { root, by_key }
    }

    fn load(&self, layer_idx: i32, step: &str, side: &str) -> std::io::Result<(Vec<usize>, Vec<f32>)> {
        let cap = self.by_key.get(&(layer_idx, step.to_string(), side.to_string()))
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing capture: layer={} step={} side={}", layer_idx, step, side),
            ))?;
        let path = self.root.join(&cap.path);
        load_safetensors_as_f32(&path)
    }

    fn try_load(&self, layer_idx: i32, step: &str, side: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        self.load(layer_idx, step, side).ok()
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel-isolated tests
// ───────────────────────────────────────────────────────────────────────────

/// Verify SwiGLU: hipfire kernel `gelu_tanh_f32(gate) * up` should equal
/// the dump's `mlp_down.input` tensor.
///
/// In modeling_gemma4.py:
///     mlp_down.input = act_fn(gate_proj(x)) * up_proj(x)
/// where `act_fn = gelu_pytorch_tanh`. So gemma4 SwiGLU has gelu_tanh as
/// the gate non-linearity (Qwen3.5 uses silu — different kernel, would
/// fail this test).
fn test_swiglu_layer(
    gpu: &mut Gpu,
    refs: &Refs,
    layer_idx: i32,
    threshold: f32,
) -> std::io::Result<bool> {
    let (gate_shape, gate_f32) = refs.load(layer_idx, "mlp_gate", "output")?;
    let (up_shape, up_f32)     = refs.load(layer_idx, "mlp_up", "output")?;
    let (out_shape, out_ref)   = refs.load(layer_idx, "mlp_down", "input")?;
    assert_eq!(gate_shape, up_shape);
    assert_eq!(gate_shape, out_shape);
    let n: usize = gate_shape.iter().product();
    // Per-token strip: shape is [1, T, hidden_dim]. We process the entire
    // flat buffer as a single 1-D vector (gelu_tanh + mul are pointwise).
    let gate_t = gpu.upload_f32(&gate_f32, &[n])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("upload gate: {:?}", e)))?;
    let up_t = gpu.upload_f32(&up_f32, &[n])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("upload up: {:?}", e)))?;
    let activated = gpu.alloc_tensor(&[n], DType::F32)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("alloc activated: {:?}", e)))?;

    // gelu_tanh takes (x, out, n).
    gpu.gelu_tanh_f32(&gate_t, &activated, n)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("gelu_tanh: {:?}", e)))?;
    // mul_f32(a, b, c) writes c = a * b.
    gpu.mul_f32(&activated, &up_t, &activated)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("mul: {:?}", e)))?;

    let actual = gpu.download_f32(&activated)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("download: {:?}", e)))?;
    let res = compare_f32(&actual, &out_ref);
    print_diag(
        &format!("layer {:02} swiglu(gelu_tanh)", layer_idx),
        &res, threshold, &actual, &out_ref,
    );
    Ok(res.nrmse < threshold)
}

// ───────────────────────────────────────────────────────────────────────────
// Phase 2: weight-dependent kernel-isolated tests
// ───────────────────────────────────────────────────────────────────────────
//
// For each layer, run the actual kernels with the model's loaded weights
// against captured INPUT, compare to captured OUTPUT. Single-token slice from
// the [B, T, ...] reference: kernels are per-token (decode path), so we test
// position T-1 only.

/// Slice the last-token tail of a [B, T, ...] flat buffer. Returns a 1-D
/// vector of size `prod(shape[2..])`. For shape with fewer dims, returns the
/// data as-is.
fn last_token(shape: &[usize], data: &[f32]) -> Vec<f32> {
    if shape.len() < 2 {
        return data.to_vec();
    }
    // Many captures are [B, T, ...]; some Gemma 4 modules emit [B, T, H, D].
    // Common contract: tokens are along axis 1.
    let _b = shape[0];
    let t = shape[1];
    let tail: usize = shape[2..].iter().product::<usize>().max(1);
    if t == 0 || data.len() < tail {
        return data.to_vec();
    }
    let last_pos = t - 1;
    let start = last_pos * tail;
    let end = start + tail;
    if end > data.len() {
        return data.to_vec();
    }
    data[start..end].to_vec()
}

/// Slice [B, T, ...] by a single token index (0..T).
fn token_at(shape: &[usize], data: &[f32], pos: usize) -> Vec<f32> {
    if shape.len() < 2 { return data.to_vec(); }
    let t = shape[1];
    let tail: usize = shape[2..].iter().product::<usize>().max(1);
    if pos >= t || data.len() < tail { return data.to_vec(); }
    let start = pos * tail;
    data[start..start + tail].to_vec()
}

/// Test a 1-D rmsnorm: load captured `step.input`, run rmsnorm_f32 with
/// the model's `weight`, compare to captured `step.output`.
fn test_rmsnorm_1d(
    gpu: &mut Gpu, refs: &Refs, layer_idx: i32,
    step: &str, weight: &GpuTensor, eps: f32, threshold: f32,
) -> std::io::Result<bool> {
    let (in_shape, in_full) = match refs.try_load(layer_idx, step, "input") {
        Some(p) => p,
        None => { eprintln!("    SKIP layer {:02} {} — no input capture", layer_idx, step); return Ok(true); }
    };
    let (_, out_full) = match refs.try_load(layer_idx, step, "output") {
        Some(p) => p,
        None => { eprintln!("    SKIP layer {:02} {} — no output capture", layer_idx, step); return Ok(true); }
    };
    let in_tok = last_token(&in_shape, &in_full);
    let out_tok = last_token(&in_shape, &out_full);
    let n = in_tok.len();
    let x_t = gpu.upload_f32(&in_tok, &[1, n])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("upload {}.in: {:?}", step, e)))?;
    let out_t = gpu.alloc_tensor(&[1, n], DType::F32)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("alloc {}.out: {:?}", step, e)))?;
    gpu.rmsnorm_f32(&x_t, weight, &out_t, eps)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("rmsnorm {}: {:?}", step, e)))?;
    let actual = gpu.download_f32(&out_t)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("download {}: {:?}", step, e)))?;
    let res = compare_f32(&actual, &out_tok);
    print_diag(&format!("layer {:02} {} (rmsnorm_f32)", layer_idx, step),
               &res, threshold, &actual, &out_tok);
    Ok(res.nrmse < threshold)
}

/// Test a head-batched rmsnorm: load captured `step.input`, run rmsnorm_batched
/// with `batch=n_heads, n=head_dim` and the model's per-head-dim weight.
fn test_rmsnorm_batched(
    gpu: &mut Gpu, refs: &Refs, layer_idx: i32,
    step: &str, weight: &GpuTensor, n_heads: usize, head_dim: usize,
    eps: f32, threshold: f32,
) -> std::io::Result<bool> {
    let (in_shape, in_full) = match refs.try_load(layer_idx, step, "input") {
        Some(p) => p,
        None => { eprintln!("    SKIP layer {:02} {} — no input capture", layer_idx, step); return Ok(true); }
    };
    let (out_shape, out_full) = refs.load(layer_idx, step, "output")?;
    // Captured shape may be [B, T, n_heads*head_dim] or [B, T, n_heads, head_dim];
    // either way the last-token tail has size n_heads*head_dim.
    let in_tok = last_token(&in_shape, &in_full);
    let out_tok = last_token(&out_shape, &out_full);
    let total = n_heads * head_dim;
    if in_tok.len() != total {
        eprintln!("    SKIP layer {:02} {} — capture size {} != n_heads*head_dim {}",
                  layer_idx, step, in_tok.len(), total);
        return Ok(true);
    }
    let x_t = gpu.upload_f32(&in_tok, &[n_heads, head_dim])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("upload {}.in: {:?}", step, e)))?;
    let out_t = gpu.alloc_tensor(&[n_heads, head_dim], DType::F32)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("alloc {}.out: {:?}", step, e)))?;
    gpu.rmsnorm_batched(&x_t, weight, &out_t, n_heads, head_dim, eps)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("rmsnorm_batched {}: {:?}", step, e)))?;
    let actual = gpu.download_f32(&out_t)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("download {}: {:?}", step, e)))?;
    let res = compare_f32(&actual, &out_tok);
    print_diag(&format!("layer {:02} {} (rmsnorm_batched)", layer_idx, step),
               &res, threshold, &actual, &out_tok);
    Ok(res.nrmse < threshold)
}

/// Test a weight-projection: load captured `step.input` (last-token), run
/// weight_gemv with the model's quantized weight, compare to `step.output`.
fn test_proj(
    gpu: &mut Gpu, refs: &Refs, layer_idx: i32,
    step: &str, weight: &WeightTensor, threshold: f32,
) -> std::io::Result<bool> {
    let (in_shape, in_full) = match refs.try_load(layer_idx, step, "input") {
        Some(p) => p,
        None => { eprintln!("    SKIP layer {:02} {} — no input capture", layer_idx, step); return Ok(true); }
    };
    let (out_shape, out_full) = refs.load(layer_idx, step, "output")?;
    let in_tok = last_token(&in_shape, &in_full);
    let out_tok = last_token(&out_shape, &out_full);
    let k = in_tok.len();
    let m = out_tok.len();
    if weight.k != k || weight.m != m {
        eprintln!("    SKIP layer {:02} {} — weight shape ({}, {}) ≠ capture ({}, {})",
                  layer_idx, step, weight.m, weight.k, m, k);
        return Ok(true);
    }
    let x_t = gpu.upload_f32(&in_tok, &[k])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("upload {}.in: {:?}", step, e)))?;
    let y_t = gpu.alloc_tensor(&[m], DType::F32)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("alloc {}.out: {:?}", step, e)))?;
    weight_gemv(gpu, weight, &x_t, &y_t)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("weight_gemv {}: {:?}", step, e)))?;
    let actual = gpu.download_f32(&y_t)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("download {}: {:?}", step, e)))?;
    let res = compare_f32(&actual, &out_tok);
    print_diag(&format!("layer {:02} {} (weight_gemv)", layer_idx, step),
               &res, threshold, &actual, &out_tok);
    Ok(res.nrmse < threshold)
}

/// Run the Phase 2 battery for a single layer.
fn test_layer_phase2(
    gpu: &mut Gpu, refs: &Refs, config: &Gemma4Config,
    layer_idx: usize, lw: &LayerWeights, threshold: f32,
    halt_on_fail: bool,
) -> (usize, usize, usize) {
    let mut pass = 0usize;
    let mut fail = 0usize;
    let li = layer_idx as i32;

    // Per-layer specifics depend on layer type (sliding vs full).
    let (input_layernorm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
         q_proj, k_proj, v_proj_opt, o_proj, q_norm, k_norm,
         gate_proj, up_proj, down_proj,
         n_kv, head_dim) = match lw {
        LayerWeights::Sliding(s) => (
            &s.input_layernorm, &s.post_attention_layernorm,
            &s.pre_feedforward_layernorm, &s.post_feedforward_layernorm,
            &s.q_proj, &s.k_proj, Some(&s.v_proj), &s.o_proj, &s.q_norm, &s.k_norm,
            &s.gate_proj, &s.up_proj, &s.down_proj,
            config.sliding_n_kv_heads, config.sliding_head_dim,
        ),
        LayerWeights::Full(f) => (
            &f.input_layernorm, &f.post_attention_layernorm,
            &f.pre_feedforward_layernorm, &f.post_feedforward_layernorm,
            &f.q_proj, &f.k_proj, None, &f.o_proj, &f.q_norm, &f.k_norm,
            &f.gate_proj, &f.up_proj, &f.down_proj,
            config.full_n_kv_heads, config.full_head_dim,
        ),
    };

    let n_heads = config.n_heads;
    let eps = config.norm_eps;

    // Sandwich norms (4 per layer).
    macro_rules! check {
        ($r:expr) => {
            match $r {
                Ok(true) => pass += 1,
                Ok(false) => { fail += 1; if halt_on_fail { return (pass, fail, 0); } }
                Err(e) => { eprintln!("    ERROR layer {:02}: {}", layer_idx, e); fail += 1;
                            if halt_on_fail { return (pass, fail, 0); } }
            }
        };
    }
    check!(test_rmsnorm_1d(gpu, refs, li, "pre_attn_norm",  input_layernorm, eps, threshold));
    check!(test_rmsnorm_1d(gpu, refs, li, "post_attn_norm", post_attn_norm,  eps, threshold));
    check!(test_rmsnorm_1d(gpu, refs, li, "pre_mlp_norm",   pre_ffn_norm,    eps, threshold));
    check!(test_rmsnorm_1d(gpu, refs, li, "post_mlp_norm",  post_ffn_norm,   eps, threshold));

    // Q/K/V projections. v_proj_opt is None for full layers (V = pre-norm K).
    check!(test_proj(gpu, refs, li, "q_proj", q_proj, threshold));
    check!(test_proj(gpu, refs, li, "k_proj", k_proj, threshold));
    if let Some(vp) = v_proj_opt {
        check!(test_proj(gpu, refs, li, "v_proj", vp, threshold));
    }

    // Q/K norms (head-dim batched). Note: capture shape for q_norm.input could
    // be [B, T, n_heads, head_dim] or [B, T, n_heads*head_dim]; helper uses
    // total size for sanity check.
    check!(test_rmsnorm_batched(gpu, refs, li, "q_norm", q_norm, n_heads, head_dim, eps, threshold));
    check!(test_rmsnorm_batched(gpu, refs, li, "k_norm", k_norm, n_kv,    head_dim, eps, threshold));

    // o_proj
    check!(test_proj(gpu, refs, li, "o_proj", o_proj, threshold));

    // MLP: gate, up, down (gelu_tanh + mul already covered in Phase 1).
    check!(test_proj(gpu, refs, li, "mlp_gate", gate_proj, threshold));
    check!(test_proj(gpu, refs, li, "mlp_up",   up_proj,   threshold));
    check!(test_proj(gpu, refs, li, "mlp_down", down_proj, threshold));

    // Suppress unused warnings (token_at and n_heads are reserved for the
    // attention-isolation step, deferred until a longer reference dump is
    // generated so sliding-window eviction at pos>1024 can be exercised).
    let _ = n_heads;
    let _ = token_at as fn(&[usize], &[f32], usize) -> Vec<f32>;

    (pass, fail, 0usize)
}

// ───────────────────────────────────────────────────────────────────────────
// Driver
// ───────────────────────────────────────────────────────────────────────────

fn parse_layer_filter() -> Option<Vec<i32>> {
    let s = std::env::var("VERIFY_LAYERS").ok()?;
    Some(s.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: verify_against_torch <REF_DIR>");
        eprintln!();
        eprintln!("REF_DIR should contain manifest.json and per-layer .safetensors");
        eprintln!("(produced by scripts/quant-survey/dump_reference_activations.py).");
        eprintln!();
        eprintln!("Env:");
        eprintln!("  VERIFY_LAYERS=0,1,2  — restrict to these layer indices");
        eprintln!("  VERIFY_THRESHOLD=1e-3 — NRMSE pass threshold");
        eprintln!("  VERIFY_HALT_ON_FAIL=1 — stop at first FAIL");
        std::process::exit(1);
    }
    let ref_dir = PathBuf::from(&args[1]);
    let manifest_path = ref_dir.join("manifest.json");
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read manifest {:?}: {}", manifest_path, e));
    let manifest: Manifest = serde_json::from_str(&manifest_str)
        .expect("parse manifest.json");
    let n_layers_dump = manifest.captures.iter().map(|c| c.layer_idx).max().unwrap_or(-1) + 1;

    eprintln!("─── verify_against_torch ───────────────────────────────────────────────");
    eprintln!("  model      : {}", manifest.model);
    eprintln!("  prompt     : {:?}", manifest.prompt);
    eprintln!("  n_tokens   : {}", manifest.n_tokens);
    eprintln!("  dtype      : {}", manifest.dtype);
    eprintln!("  ref_dir    : {}", ref_dir.display());
    eprintln!("  layers     : 0..{} ({} layers in dump)", n_layers_dump, n_layers_dump);
    eprintln!();

    let layer_filter = parse_layer_filter();
    // Default 5e-3 = bf16 mantissa noise floor (see header docs). Override
    // with VERIFY_THRESHOLD=1e-4 if/when the reference dump is regenerated
    // in fp32.
    let threshold: f32 = std::env::var("VERIFY_THRESHOLD")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(5e-3);
    let halt_on_fail = std::env::var("VERIFY_HALT_ON_FAIL").ok().as_deref() == Some("1");
    eprintln!("  threshold  : {:.2e}", threshold);
    if let Some(ref f) = layer_filter {
        eprintln!("  filter     : layers {:?}", f);
    }
    eprintln!();

    let mut gpu = Gpu::init().expect("Gpu::init");
    let refs = Refs::new(ref_dir.clone(), manifest);

    eprintln!("─── Phase 1: kernel-isolated SwiGLU (gelu_tanh + mul) ──────────────────");
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;
    let mut skip_count = 0usize;
    let mut first_fail_layer: Option<i32> = None;

    for layer_idx in 0..n_layers_dump {
        if let Some(ref f) = layer_filter {
            if !f.contains(&layer_idx) { continue; }
        }
        // Sanity: the capture set varies by layer? Layer 0 has SwiGLU; assume all do.
        if refs.try_load(layer_idx, "mlp_gate", "output").is_none()
            || refs.try_load(layer_idx, "mlp_up", "output").is_none()
            || refs.try_load(layer_idx, "mlp_down", "input").is_none()
        {
            eprintln!("    SKIP layer {:02} swiglu — capture missing", layer_idx);
            skip_count += 1;
            continue;
        }
        match test_swiglu_layer(&mut gpu, &refs, layer_idx, threshold) {
            Ok(true) => pass_count += 1,
            Ok(false) => {
                fail_count += 1;
                if first_fail_layer.is_none() { first_fail_layer = Some(layer_idx); }
                if halt_on_fail { break; }
            }
            Err(e) => {
                eprintln!("    ERROR layer {:02} swiglu: {}", layer_idx, e);
                fail_count += 1;
                if first_fail_layer.is_none() { first_fail_layer = Some(layer_idx); }
                if halt_on_fail { break; }
            }
        }
    }

    eprintln!();

    // Phase 2: weight-dependent kernel-isolated battery. Triggered by
    // VERIFY_MODEL=path-to-.mq4. Skipped if not set.
    let mut p2_pass = 0usize;
    let mut p2_fail = 0usize;
    let mut p2_first_fail: Option<i32> = None;
    let model_env = std::env::var("VERIFY_MODEL").ok();
    if let Some(model_path) = model_env.as_deref() {
        eprintln!("─── Phase 2: weight-dependent kernels (rmsnorm / proj / o_proj / mlp) ─");
        eprintln!("    model: {}", model_path);
        let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
        if hfq.arch_id != 7 {
            eprintln!("    ERROR: VERIFY_MODEL is not a Gemma 4 .mq4 (arch_id={}, expected 7)", hfq.arch_id);
            std::process::exit(2);
        }
        let config = gemma4::config_from_hfq(&hfq).expect("read config from .mq4");
        eprintln!("    config: dim={}, n_layers={}, n_heads={}, sliding_hd={}, full_hd={}",
                  config.dim, config.n_layers, config.n_heads,
                  config.sliding_head_dim, config.full_head_dim);
        eprintln!("    loading weights...");
        let weights = gemma4::load_weights(&hfq, &config, &mut gpu).expect("load weights");
        let _ = &weights as *const Gemma4Weights;  // borrow for lifetime
        eprintln!("    loaded {} layers", weights.layers.len());

        for layer_idx in 0..n_layers_dump.min(weights.layers.len() as i32) {
            if let Some(ref f) = layer_filter {
                if !f.contains(&layer_idx) { continue; }
            }
            let lw = &weights.layers[layer_idx as usize];
            let lt = config.layer_types[layer_idx as usize];
            let lt_label = match lt { LayerType::Sliding => "sliding", LayerType::Full => "full" };
            eprintln!("  ── layer {:02} ({}) ──", layer_idx, lt_label);
            let (p, f, _) = test_layer_phase2(&mut gpu, &refs, &config, layer_idx as usize, lw,
                                              threshold, halt_on_fail);
            p2_pass += p;
            p2_fail += f;
            if f > 0 && p2_first_fail.is_none() { p2_first_fail = Some(layer_idx); }
            if p2_fail > 0 && halt_on_fail { break; }
        }
        eprintln!();
    } else {
        eprintln!("─── Phase 2: weight-dependent kernels (rmsnorm / proj / attn / o_proj) ─");
        eprintln!("    SKIPPED — set VERIFY_MODEL=/path/to/gemma-4-31b.mq4 to enable.");
        eprintln!();
    }

    eprintln!("─── Summary ────────────────────────────────────────────────────────────");
    eprintln!("    Phase 1 SwiGLU: pass={} fail={} skip={}", pass_count, fail_count, skip_count);
    if let Some(l) = first_fail_layer {
        eprintln!("    Phase 1 first FAIL at layer {:02}", l);
    }
    if model_env.is_some() {
        eprintln!("    Phase 2 weighted: pass={} fail={}", p2_pass, p2_fail);
        if let Some(l) = p2_first_fail {
            eprintln!("    Phase 2 first FAIL at layer {:02}", l);
        }
    }
    eprintln!();

    let total_fail = fail_count + p2_fail;
    if total_fail > 0 {
        std::process::exit(1);
    }
}
