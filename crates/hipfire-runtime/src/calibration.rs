// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 calibration collector (lib-ified core).
//
//! The reusable, model-agnostic calibration collector: an [`ActivationCapture`]
//! that accumulates a per-tensor GPTQ Hessian (`Σ x·xᵀ`) and imatrix diagonal
//! (`Σ x²`) on-GPU via the `calib_*_reduce_f32` kernels, and drains to HFQ
//! tensors (`<name>.hessian` [K,K] + `<name>.imatrix` [K]) plus an
//! internal-consistency metric (`diag(Σxxᵀ)` must equal `Σx²`).
//!
//! This is generic (hipfire-rdna + the HFQ writer only) so it sits in
//! hipfire-runtime without a cycle on the arch crates. Callers (the
//! `collect_artifacts` CLI, the daemon `Collect` op) own the forward loop +
//! the model-specific taps (MoE router histogram, KLDREF) and arm this via
//! `gpu.active_capture = Some(Arc::new(CalibCollector::default()))`.

use crate::hfq::HfqMemTensor;
use hipfire_rdna::{ActivationCapture, DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Rows buffered per tensor before flushing the outer-product. A single
/// `calib_hessian_outer_f32` over `[FLUSH_BATCH, K]` is ~FLUSH_BATCH× more
/// efficient than per-token (N=1) launches (the tiled GEMM is built for N≥16),
/// so this is the dominant calibration-throughput lever.
const FLUSH_BATCH: usize = 256;

/// Calibration-only HFQM quant_type for compact Hessians:
/// exact F32 diagonal followed by BF16 lower strict triangle.
const QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32: u8 = 130;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HessianStorage {
    DenseF32,
    Bf16TrilDiagF32,
}

fn hessian_storage_from_env() -> HessianStorage {
    match std::env::var("HIPFIRE_CALIB_HESSIAN_STORAGE")
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("f32" | "dense-f32" | "full-f32" | "legacy") => HessianStorage::DenseF32,
        _ => HessianStorage::Bf16TrilDiagF32,
    }
}

fn compact_hessian_bytes(k: usize) -> u64 {
    (k * 4 + k * (k - 1)) as u64
}

/// Per-tensor on-GPU accumulators + a small activation row buffer.
struct Acc {
    diag: GpuTensor,      // [K]   Σx²  (imatrix)
    h: Option<GpuTensor>, // [K,K] Σxxᵀ (Hessian); `None` = imatrix-only tensor
    /// Host f64 reference accumulator (`Some` only under `HIPFIRE_CALIB_F64_AUDIT`).
    /// The GPU outer-product accumulates `Σxxᵀ` in f32; RDNA has no f64 matrix
    /// units and only ~1:16 scalar f64, so a faithful f64 reference is computed
    /// CPU-side from the same staged rows. `drain` then reports the max relative
    /// f32-vs-f64 divergence — measure-first before deciding whether f32
    /// accumulation needs replacing for large token counts.
    h_f64: Option<Vec<f64>>,
    buf: GpuTensor,  // [FLUSH_BATCH, K] staged activation rows
    buf_rows: usize, // rows currently staged in `buf`
    k: usize,
    n_tokens: u64,
}

impl Acc {
    /// Reduce the staged rows into the accumulators (one batched launch each),
    /// then reset the buffer. No-op when empty. Imatrix-only tensors (`h` is
    /// `None`) skip the [K,K] outer-product — this is how MoE routed experts
    /// are captured: a full per-expert Hessian (256 experts × ~48 layers ×
    /// [K,K]) is ~196 GB and does not fit, but the imatrix (Σx², a K-vector)
    /// is ~100 MB and is the importance signal AWQ-style quant needs.
    fn flush(&mut self, gpu: &mut Gpu) {
        if self.buf_rows == 0 {
            return;
        }
        gpu.calib_sumsq_reduce_f32(&self.buf, &self.diag, self.buf_rows, self.k)
            .unwrap();
        if let Some(h) = &self.h {
            gpu.calib_hessian_outer_f32(&self.buf, h, self.buf_rows, self.k)
                .unwrap();
        }
        // Audit: accumulate the same rows in f64 on the CPU (no GPU f64 path).
        if let Some(h_f64) = &mut self.h_f64 {
            let k = self.k;
            let rows = gpu
                .download_f32(&self.buf)
                .expect("download buf (f64 audit)");
            for r in 0..self.buf_rows {
                let x = &rows[r * k..r * k + k];
                for i in 0..k {
                    let xi = x[i] as f64;
                    let hrow = &mut h_f64[i * k..i * k + k];
                    for j in 0..k {
                        hrow[j] += xi * x[j] as f64;
                    }
                }
            }
        }
        self.buf_rows = 0;
    }
}

/// Unified Hessian + imatrix collector. Arm via `gpu.active_capture`.
///
/// By default every captured tensor accumulates a full [K,K] Hessian. Tensors
/// whose canonical name contains any of `imatrix_only_substr` accumulate only
/// the imatrix (Σx²); used for MoE routed experts whose full Hessians do not
/// fit in memory (see [`Acc::flush`]).
#[derive(Default)]
pub struct CalibCollector {
    accs: Mutex<HashMap<String, Acc>>,
    imatrix_only_substr: Vec<String>,
    /// When set (`HIPFIRE_CALIB_F64_AUDIT=1`), also accumulate each Hessian in
    /// f64 on the CPU and report the f32-vs-f64 divergence in `drain`. Opt-in,
    /// slow (CPU outer-products) — a measurement tool, not the default path.
    f64_audit: bool,
}

/// `HIPFIRE_CALIB_F64_AUDIT=1` → run the CPU f64 reference accumulation.
fn f64_audit_enabled() -> bool {
    std::env::var("HIPFIRE_CALIB_F64_AUDIT").ok().as_deref() == Some("1")
}

impl CalibCollector {
    pub fn new() -> Self {
        Self {
            accs: Mutex::new(HashMap::new()),
            imatrix_only_substr: Vec::new(),
            f64_audit: f64_audit_enabled(),
        }
    }

    /// Collector that stores imatrix-only (no [K,K] Hessian) for any tensor
    /// whose name contains one of `substr` (e.g. `".experts."` for MoE).
    pub fn with_imatrix_only(substr: Vec<String>) -> Self {
        Self {
            accs: Mutex::new(HashMap::new()),
            imatrix_only_substr: substr,
            f64_audit: f64_audit_enabled(),
        }
    }

    fn wants_hessian(&self, name: &str) -> bool {
        !self.imatrix_only_substr.iter().any(|s| name.contains(s))
    }

    /// Number of distinct tensors captured so far.
    pub fn len(&self) -> usize {
        self.accs.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Per-tensor descriptors (no GPU work): `name`, whether it has a full
    /// Hessian, `k`, and `n_tokens`. The caller uses these to compute counts +
    /// `name -> n_tokens` provenance for the metadata BEFORE the streaming write
    /// (the HFQM index/metadata must be written ahead of the payloads).
    pub fn tensor_descriptors(&self) -> Vec<CalibTensorDesc> {
        let accs = self.accs.lock().unwrap();
        let mut names: Vec<&String> = accs.keys().collect();
        names.sort();
        names
            .iter()
            .map(|name| {
                let acc = &accs[*name];
                CalibTensorDesc {
                    name: (*name).clone(),
                    has_hessian: acc.h.is_some(),
                    k: acc.k,
                    n_tokens: acc.n_tokens,
                }
            })
            .collect()
    }

    /// Release all GPU accumulators owned by this collector. Grouped
    /// calibration runs call this after streaming a part file so the next group
    /// can reuse the memory instead of waiting for process teardown.
    pub fn free_gpu(&self, gpu: &mut Gpu) {
        let mut accs = self.accs.lock().unwrap();
        for (_, acc) in accs.drain() {
            let _ = gpu.free_tensor(acc.diag);
            if let Some(h) = acc.h {
                let _ = gpu.free_tensor(h);
            }
            let _ = gpu.free_tensor(acc.buf);
        }
    }

    /// GuidedQuant capture: accumulate the per-token **Fisher-weighted** Hessian
    /// `H̄ = Σ_n w[n]·xₙxₙᵀ` (and its diagonal) for `tensor_name`. `x` is the
    /// linear's input activation `[n,k]` (a real contiguous block, not the shared
    /// scratch the `ActivationCapture::capture` tap takes); `w` `[n]` is the
    /// per-token weight the caller forms from that linear's output-grad `∂ℓ/∂z`
    /// (see `calib_row_meansq_f32`). Unbuffered — one weighted outer-product +
    /// weighted sumsq per call, fine offline. `w≡1` makes this identical to the
    /// plain unweighted capture.
    pub fn capture_weighted(
        &self,
        gpu: &mut Gpu,
        tensor_name: &str,
        x: &GpuTensor,
        w: &GpuTensor,
        n: usize,
        k: usize,
    ) {
        let mut accs = self.accs.lock().unwrap();
        if !accs.contains_key(tensor_name) {
            let diag = gpu.zeros(&[k], DType::F32).unwrap();
            let h = if self.wants_hessian(tensor_name) {
                Some(gpu.zeros(&[k, k], DType::F32).unwrap())
            } else {
                None
            };
            // No row buffering on this path; a minimal placeholder keeps `Acc`
            // uniform (`flush` is a no-op while `buf_rows == 0`).
            let buf = gpu.zeros(&[1, k], DType::F32).unwrap();
            accs.insert(
                tensor_name.to_string(),
                Acc {
                    diag,
                    h,
                    h_f64: None,
                    buf,
                    buf_rows: 0,
                    k,
                    n_tokens: 0,
                },
            );
        }
        let acc = accs.get_mut(tensor_name).unwrap();
        gpu.calib_sumsq_weighted_f32(x, w, &acc.diag, n, k).unwrap();
        if let Some(h) = &acc.h {
            gpu.calib_hessian_outer_weighted_f32(x, w, h, n, k).unwrap();
        }
        acc.n_tokens += n as u64;
    }

    /// Stream the accumulated tensors into an HFQM `.calib.hfq` at `path`,
    /// **one tensor at a time** (download → normalize `/ n_tokens` → write →
    /// drop), so peak host memory is a single Hessian rather than all of them
    /// (a 9B is ~32 GB if materialized at once). `extra` holds any small
    /// already-in-RAM tensors (e.g. KLDREF) the caller wants in the same
    /// package. The metadata + index are written first (payload sizes are
    /// deterministic from `k`), then the payloads stream. Returns the max
    /// relative `diag(H)`-vs-`Σx²` consistency error. Also runs the optional
    /// f64 audit (`HIPFIRE_CALIB_F64_AUDIT`) during the per-Hessian download.
    pub fn write_streaming(
        &self,
        gpu: &mut Gpu,
        path: &std::path::Path,
        arch_id: u32,
        metadata_json: &str,
        extra: &[HfqMemTensor],
    ) -> std::io::Result<f32> {
        use crate::hfq::{write_hfqm_package_streaming, HfqStreamEntry};
        use std::cell::{Cell, RefCell};

        let mut accs = self.accs.lock().unwrap();
        let hessian_storage = hessian_storage_from_env();
        // Fold any staged activation rows before reading the accumulators.
        for acc in accs.values_mut() {
            acc.flush(gpu);
        }
        let mut names: Vec<String> = accs.keys().cloned().collect();
        names.sort();

        // Build the index entries (payload sizes from `k`) + a parallel plan of
        // how to produce each payload, in the SAME order.
        enum Plan {
            Hessian(String),
            Imatrix(String),
            Extra(usize),
        }
        let mut entries: Vec<HfqStreamEntry> = Vec::new();
        let mut plan: Vec<Plan> = Vec::new();
        for name in &names {
            let acc = &accs[name];
            if acc.h.is_some() {
                let (quant_type, data_len) = match hessian_storage {
                    HessianStorage::DenseF32 => (2, (acc.k * acc.k * 4) as u64),
                    HessianStorage::Bf16TrilDiagF32 => (
                        QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32,
                        compact_hessian_bytes(acc.k),
                    ),
                };
                entries.push(HfqStreamEntry {
                    name: format!("{name}.hessian"),
                    quant_type,
                    shape: vec![acc.k as u32, acc.k as u32],
                    group_size: 0,
                    data_len,
                });
                plan.push(Plan::Hessian(name.clone()));
            }
            entries.push(HfqStreamEntry {
                name: format!("{name}.imatrix"),
                quant_type: 2,
                shape: vec![acc.k as u32],
                group_size: 0,
                data_len: (acc.k * 4) as u64,
            });
            plan.push(Plan::Imatrix(name.clone()));
        }
        for (j, t) in extra.iter().enumerate() {
            entries.push(HfqStreamEntry {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_len: t.data.len() as u64,
            });
            plan.push(Plan::Extra(j));
        }

        let max_consistency = Cell::new(0.0f32);
        let audit_max = Cell::new(0.0f64);
        let audit_n = Cell::new(0usize);
        let audit_worst = RefCell::new(String::new());
        let io_err = |e: hipfire_rdna::HipError| std::io::Error::other(e.to_string());

        write_hfqm_package_streaming(path, arch_id, metadata_json, &entries, |i, w| {
            match &plan[i] {
                Plan::Hessian(name) => {
                    let acc = &accs[name];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let h = gpu.download_f32(acc.h.as_ref().unwrap()).map_err(io_err)?;
                    let diag = gpu.download_f32(&acc.diag).map_err(io_err)?;
                    let mut mc = max_consistency.get();
                    for c in 0..acc.k {
                        let rel = (h[c * acc.k + c] - diag[c]).abs() / diag[c].abs().max(1.0);
                        mc = mc.max(rel);
                    }
                    max_consistency.set(mc);
                    if let Some(h_ref) = &acc.h_f64 {
                        let mut tmax = 0.0f64;
                        for idx in 0..acc.k * acc.k {
                            let r = h_ref[idx];
                            tmax = tmax.max((h[idx] as f64 - r).abs() / r.abs().max(1.0));
                        }
                        audit_n.set(audit_n.get() + 1);
                        if tmax > audit_max.get() {
                            audit_max.set(tmax);
                            *audit_worst.borrow_mut() = name.clone();
                        }
                    }
                    match hessian_storage {
                        HessianStorage::DenseF32 => write_f32_scaled(w, &h, inv),
                        HessianStorage::Bf16TrilDiagF32 => {
                            write_hessian_bf16_tril_diag_f32(w, &h, &diag, acc.k, inv)
                        }
                    }
                }
                Plan::Imatrix(name) => {
                    let acc = &accs[name];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let diag = gpu.download_f32(&acc.diag).map_err(io_err)?;
                    write_f32_scaled(w, &diag, inv)
                }
                Plan::Extra(j) => w.write_all(&extra[*j].data),
            }
        })?;

        if audit_n.get() > 0 {
            eprintln!(
                "F64 AUDIT: max f32-vs-f64 Σxxᵀ rel-diff = {:.3e} over {} Hessians (worst: {})",
                audit_max.get(),
                audit_n.get(),
                audit_worst.borrow()
            );
        }
        Ok(max_consistency.get())
    }
}

/// Upload a small host-resident activation matrix and feed it through the
/// currently armed calibration collector. Embedding projection heads use this
/// seam because their inputs are pooled host vectors rather than GPU GEMM
/// scratch tensors.
pub fn capture_host_activations(
    gpu: &mut Gpu,
    tensor_name: &str,
    activations: &[f32],
    rows: usize,
    width: usize,
) -> Result<(), String> {
    validate_host_activations(tensor_name, activations, rows, width)?;
    let collector = gpu
        .active_capture
        .clone()
        .ok_or_else(|| "calib: no active collector for host activation".to_string())?;
    let input = gpu
        .upload_f32(activations, &[rows, width])
        .map_err(|e| format!("calib: upload host activation {tensor_name}: {e}"))?;
    collector.capture(gpu, tensor_name, &input, rows, width);
    gpu.device_synchronize()
        .map_err(|e| format!("calib: synchronize host activation {tensor_name}: {e}"))?;
    gpu.free_tensor(input)
        .map_err(|e| format!("calib: free host activation {tensor_name}: {e}"))?;
    Ok(())
}

fn validate_host_activations(
    tensor_name: &str,
    activations: &[f32],
    rows: usize,
    width: usize,
) -> Result<(), String> {
    if tensor_name.is_empty() {
        return Err("calib: host activation tensor name is empty".to_string());
    }
    if rows == 0 || width == 0 {
        return Err("calib: host activation shape must be non-zero".to_string());
    }
    let expected = rows
        .checked_mul(width)
        .ok_or_else(|| "calib: host activation shape overflow".to_string())?;
    if activations.len() != expected {
        return Err(format!(
            "calib: host activation {tensor_name} has {} values, expected {expected} ({rows}x{width})",
            activations.len()
        ));
    }
    Ok(())
}

/// Tokenize a text corpus into independent embedding samples. Blank lines
/// delimit paragraphs; when the corpus has no blank lines, each non-empty line
/// is a sample. Samples are never truncated or concatenated: collection stops
/// before the first sample that would exceed `max_tokens`.
pub fn tokenize_embedding_samples<F>(text: &str, max_tokens: usize, mut encode: F) -> Vec<Vec<u32>>
where
    F: FnMut(&str) -> Vec<u32>,
{
    if max_tokens == 0 {
        return Vec::new();
    }

    let has_blank_line = text.lines().any(|line| line.trim().is_empty());
    let mut texts = Vec::new();
    if has_blank_line {
        let mut paragraph = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                if !paragraph.is_empty() {
                    texts.push(paragraph.join("\n"));
                    paragraph.clear();
                }
            } else {
                paragraph.push(line);
            }
        }
        if !paragraph.is_empty() {
            texts.push(paragraph.join("\n"));
        }
    } else {
        texts.extend(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string),
        );
    }

    let mut samples = Vec::new();
    let mut total_tokens = 0usize;
    for text in texts {
        let tokens = encode(&text);
        if tokens.is_empty() {
            continue;
        }
        if total_tokens + tokens.len() > max_tokens {
            break;
        }
        total_tokens += tokens.len();
        samples.push(tokens);
    }
    samples
}

/// Sample-oriented wrapper around [`collect_grouped`]. Each layer group reruns
/// every independent embedding sample, preserving per-sample pooling semantics
/// while retaining the grouped Hessian memory bound.
#[allow(clippy::too_many_arguments)]
pub fn collect_embedding_grouped<C, F>(
    gpu: &mut Gpu,
    arch_id: u32,
    num_layers: usize,
    group_size: usize,
    output: &std::path::Path,
    samples: &[Vec<u32>],
    static_meta: &[(&str, serde_json::Value)],
    capture_names_for: C,
    mut forward_sample: F,
) -> Result<CalibSummary, String>
where
    C: FnMut(usize, usize) -> HashMap<usize, String>,
    F: FnMut(&mut Gpu, usize, usize, &[u32]) -> Result<(), String>,
{
    if samples.is_empty() {
        return Err("calib: embedding sample set is empty".to_string());
    }
    if samples.iter().any(Vec::is_empty) {
        return Err("calib: embedding samples must be non-empty".to_string());
    }
    let total_tokens: usize = samples.iter().map(Vec::len).sum();
    let max_sample_length = samples.iter().map(Vec::len).max().unwrap_or(0);
    let mut metadata = static_meta.to_vec();
    metadata.push(("sample_count", serde_json::json!(samples.len())));
    metadata.push(("total_tokens", serde_json::json!(total_tokens)));
    metadata.push(("max_sample_length", serde_json::json!(max_sample_length)));

    collect_grouped(
        gpu,
        arch_id,
        num_layers,
        group_size,
        Vec::new(),
        output,
        &metadata,
        capture_names_for,
        |gpu, group_idx| {
            for (sample_idx, sample) in samples.iter().enumerate() {
                forward_sample(gpu, group_idx, sample_idx, sample)?;
            }
            Ok(CalibForward::default())
        },
    )
}

fn qwen3_embedding_layer_tensor_names(layer_idx: usize) -> [String; 7] {
    let prefix = format!("model.layers.{layer_idx}");
    [
        format!("{prefix}.self_attn.q_proj"),
        format!("{prefix}.self_attn.k_proj"),
        format!("{prefix}.self_attn.v_proj"),
        format!("{prefix}.self_attn.o_proj"),
        format!("{prefix}.mlp.gate_proj"),
        format!("{prefix}.mlp.up_proj"),
        format!("{prefix}.mlp.down_proj"),
    ]
}

fn qwen3_embedding_capture_names_for_layers(
    weights: &crate::llama::LlamaWeights,
    start_layer: usize,
    end_layer: usize,
) -> HashMap<usize, String> {
    let mut names = HashMap::new();
    for (layer_idx, layer) in weights
        .layers
        .iter()
        .enumerate()
        .skip(start_layer)
        .take(end_layer.saturating_sub(start_layer))
    {
        let linears = [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
        ];
        for (weight, name) in linears
            .into_iter()
            .zip(qwen3_embedding_layer_tensor_names(layer_idx))
        {
            names.insert(weight.buf.buf.as_ptr() as usize, name);
        }
    }
    names
}

fn validate_qwen3_embedding_samples(
    samples: &[Vec<u32>],
    max_sequence_length: usize,
) -> Result<(), String> {
    if samples.is_empty() {
        return Err("qwen3 embedding calibration: sample set is empty".to_string());
    }
    if max_sequence_length == 0 {
        return Err("qwen3 embedding calibration: maximum sequence length is zero".to_string());
    }
    if let Some((sample_idx, sample)) = samples
        .iter()
        .enumerate()
        .find(|(_, sample)| sample.is_empty() || sample.len() > max_sequence_length)
    {
        if sample.is_empty() {
            return Err(format!(
                "qwen3 embedding calibration: sample {sample_idx} is empty"
            ));
        }
        return Err(format!(
            "qwen3 embedding calibration: sample {sample_idx} length {} exceeds maximum {max_sequence_length}",
            sample.len()
        ));
    }
    Ok(())
}

/// Collect activation statistics for a SentenceTransformers Qwen3 encoder.
///
/// Qwen3 embedding models use the normal causal Qwen3 transformer and pool the
/// last real token. Each corpus sample therefore runs as an independent causal
/// prefill starting at position zero. OQ8+ consumes only the imatrix diagonal,
/// so this path deliberately skips full KxK Hessians for every encoder linear;
/// that keeps the collector bounded while preserving the exact activations
/// seen by AWQ.
pub fn collect_qwen3_embedding_artifacts(
    gpu: &mut Gpu,
    weights: &crate::llama::LlamaWeights,
    config: &crate::llama::LlamaConfig,
    samples: &[Vec<u32>],
    output: &std::path::Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    use crate::llama::ModelArch;

    const RELEASE_MAX_SEQUENCE: usize = 2048;
    if config.arch != ModelArch::Qwen3 {
        return Err("qwen3 embedding calibration requires model_type=qwen3".to_string());
    }
    if !config.has_qk_norm {
        return Err("qwen3 embedding calibration requires q_norm and k_norm tensors".to_string());
    }
    let maximum = config.max_seq_len.min(RELEASE_MAX_SEQUENCE);
    validate_qwen3_embedding_samples(samples, maximum)?;

    let total_tokens: usize = samples.iter().map(Vec::len).sum();
    let max_sample_length = samples.iter().map(Vec::len).max().unwrap_or(0);
    let mut metadata = provenance.to_vec();
    metadata.extend([
        ("task", serde_json::json!("embedding")),
        ("causal", serde_json::json!(true)),
        ("pooling_mode", serde_json::json!("last_token")),
        ("imatrix_only", serde_json::json!(true)),
        ("sample_count", serde_json::json!(samples.len())),
        ("total_tokens", serde_json::json!(total_tokens)),
        ("max_sample_length", serde_json::json!(max_sample_length)),
    ]);

    let mut kv_cache = crate::llama::KvCache::new_gpu(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_sample_length,
    )
    .map_err(|error| format!("qwen3 embedding calibration KV cache: {error:?}"))?;

    // Imatrix storage is small enough to capture all layers in one pass. The
    // env override is useful for diagnosing a problematic layer without
    // changing the artifact contract.
    let layers_per_pass = std::env::var("HIPFIRE_QWEN3_EMBEDDING_CALIB_LAYERS_PER_PASS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(config.n_layers);
    let result = collect_grouped(
        gpu,
        1,
        config.n_layers,
        layers_per_pass,
        vec!["model.layers.".to_string()],
        output,
        &metadata,
        |start, end| qwen3_embedding_capture_names_for_layers(weights, start, end),
        |gpu, _group_idx| {
            for (sample_idx, tokens) in samples.iter().enumerate() {
                crate::llama::prefill_forward(gpu, weights, config, tokens, &mut kv_cache)
                    .map_err(|error| {
                        format!(
                            "qwen3 embedding calibration forward for sample {sample_idx}: {error:?}"
                        )
                    })?;
            }
            Ok(CalibForward::default())
        },
    );
    kv_cache.free_gpu(gpu);
    result
}

/// Memory-bounded variant of [`collect`] for dense arches whose full Hessians
/// for ALL layers do not fit at once. Captures the layers in groups of
/// `group_size` — each group re-runs the arch forward but registers only that
/// group's tensors, streams a part file, and frees the GPU accumulators before
/// the next group — then concatenates the parts (plus any `extra_tensors` the
/// forward returned, e.g. KLDREF) into the final `.calib.hfq` at `output`.
///
/// Seams: `capture_names_for(start, end)` builds the capture map for one layer
/// range; `forward(gpu, group_idx)` runs the model over the calibration tokens
/// for that group (it may return [`CalibForward`] extras — typically only on
/// `group_idx == 0`, since the extras are written once into the combined file).
#[allow(clippy::too_many_arguments)]
pub fn collect_grouped<C, F>(
    gpu: &mut Gpu,
    arch_id: u32,
    num_layers: usize,
    group_size: usize,
    imatrix_only: Vec<String>,
    output: &std::path::Path,
    static_meta: &[(&str, serde_json::Value)],
    mut capture_names_for: C,
    mut forward: F,
) -> Result<CalibSummary, String>
where
    C: FnMut(usize, usize) -> HashMap<usize, String>,
    F: FnMut(&mut Gpu, usize) -> Result<CalibForward, String>,
{
    let group = group_size.max(1);
    let mut part_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut all_descriptors: Vec<CalibTensorDesc> = Vec::new();
    let mut extra_tensors: Vec<HfqMemTensor> = Vec::new();
    let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
    let mut max_consistency = 0.0f32;

    for (group_idx, start) in (0..num_layers).step_by(group).enumerate() {
        let end = (start + group).min(num_layers);
        let collector = std::sync::Arc::new(if imatrix_only.is_empty() {
            CalibCollector::new()
        } else {
            CalibCollector::with_imatrix_only(imatrix_only.clone())
        });
        gpu.capture_names = capture_names_for(start, end);
        gpu.active_capture = Some(collector.clone());

        let out = forward(gpu, group_idx);
        gpu.active_capture = None;
        gpu.capture_names = HashMap::new();
        let mut out = out?;

        let descriptors = collector.tensor_descriptors();
        if descriptors.is_empty() {
            return Err(format!(
                "calib: no tensors captured for layers {start}..{end}"
            ));
        }
        let part = calib_part_path(output, group_idx);
        let part_meta = serde_json::json!({
            "artifact_kind": "calibration-part",
            "layer_start": start,
            "layer_end": end,
        })
        .to_string();
        let consistency = collector
            .write_streaming(gpu, &part, arch_id, &part_meta, &[])
            .map_err(|e| format!("calib write part {}: {e}", part.display()))?;
        collector.free_gpu(gpu);
        max_consistency = max_consistency.max(consistency);
        all_descriptors.extend(descriptors);
        part_paths.push(part);
        extra_tensors.append(&mut out.extra_tensors);
        extra_meta.append(&mut out.extra_meta);
    }

    let n_hessian = all_descriptors.iter().filter(|d| d.has_hessian).count();
    let n_imatrix = all_descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for d in &all_descriptors {
        per_tensor_tokens.insert(d.name.clone(), serde_json::json!(d.n_tokens));
    }
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "layers_per_pass": group,
        "n_hessian": n_hessian,
        "n_imatrix": n_imatrix,
        "per_tensor_tokens": serde_json::Value::Object(per_tensor_tokens),
        "artifacts": ["hessian", "imatrix"],
    });
    if let Some(obj) = meta.as_object_mut() {
        for (k, v) in static_meta {
            obj.insert((*k).to_string(), v.clone());
        }
        for (k, v) in &extra_meta {
            obj.insert(k.clone(), v.clone());
        }
    }
    let metadata_json = serde_json::to_string(&meta).unwrap();
    combine_calib_parts(output, arch_id, &metadata_json, &part_paths, &extra_tensors)
        .map_err(|e| format!("calib combine {}: {e}", output.display()))?;
    for part in part_paths {
        let _ = std::fs::remove_file(part);
    }

    Ok(CalibSummary {
        n_hessian,
        n_imatrix,
        max_consistency,
    })
}

/// Temp part path for a grouped calibration pass: `.<name>.part-NNN.hfq` beside
/// `output`.
fn calib_part_path(output: &std::path::Path, group_idx: usize) -> std::path::PathBuf {
    let file_name = output
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("calib.hfq");
    output.with_file_name(format!(".{file_name}.part-{group_idx:03}.hfq"))
}

/// Concatenate the per-group part packages (+ in-RAM `extra` tensors) into the
/// final combined `.calib.hfq`, streaming each part's blobs through without
/// materializing them all at once.
fn combine_calib_parts(
    output: &std::path::Path,
    arch_id: u32,
    metadata_json: &str,
    part_paths: &[std::path::PathBuf],
    extra: &[HfqMemTensor],
) -> std::io::Result<()> {
    use crate::hfq::{write_hfqm_package_streaming, HfqPackage, HfqStreamEntry};
    enum Plan {
        Part { package_idx: usize, name: String },
        Extra { extra_idx: usize },
    }
    let mut packages = Vec::with_capacity(part_paths.len());
    let mut entries = Vec::new();
    let mut plan = Vec::new();
    for part in part_paths {
        let package = HfqPackage::open(part)?;
        let package_idx = packages.len();
        for e in package.entries() {
            entries.push(HfqStreamEntry {
                name: e.name.clone(),
                quant_type: e.quant_type,
                shape: e.shape.clone(),
                group_size: e.group_size,
                data_len: e.data_size as u64,
            });
            plan.push(Plan::Part {
                package_idx,
                name: e.name.clone(),
            });
        }
        packages.push(package);
    }
    for (extra_idx, t) in extra.iter().enumerate() {
        entries.push(HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape: t.shape.clone(),
            group_size: t.group_size,
            data_len: t.data.len() as u64,
        });
        plan.push(Plan::Extra { extra_idx });
    }
    write_hfqm_package_streaming(
        output,
        arch_id,
        metadata_json,
        &entries,
        |i, w| match &plan[i] {
            Plan::Part { package_idx, name } => {
                let data = packages[*package_idx].blob_data(name).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("part tensor not found: {name}"),
                    )
                })?;
                w.write_all(data)
            }
            Plan::Extra { extra_idx } => w.write_all(&extra[*extra_idx].data),
        },
    )
}

/// Per-tensor descriptor from [`CalibCollector::tensor_descriptors`].
pub struct CalibTensorDesc {
    pub name: String,
    pub has_hessian: bool,
    pub k: usize,
    pub n_tokens: u64,
}

/// Result of a calibration-collection pass — the three fields every arch's
/// public collector reports back to the `collect_artifacts` CLI / daemon.
pub struct CalibSummary {
    pub n_hessian: usize,
    pub n_imatrix: usize,
    pub max_consistency: f32,
}

/// Daemon-side calibration seam: collect a `.calib.hfq` from an ALREADY-RESIDENT
/// model — no second load. Parallels [`crate::kld_eval::ChunkScoredForward`] (the
/// `kld_eval` op's seam), but unlike KLD — which rides the blanket `SimpleAr`
/// impl — calibration needs each arch's RAW weights + capturing forward
/// (`gpu_forward_calib`-style), which `SimpleAr` does not expose. So there is no
/// blanket impl: each arch implements this by delegating to its existing
/// `collect_calibration_artifacts`. Bundled backends (`ZayaModel`,
/// `Gemma3Backend`) impl it directly; loose-slot arches (qwen35, lfm2moe) impl it
/// on a thin `&weights`/`&config` adapter, mirroring `qwen35::Qwen35KldForward`.
///
/// `tokenizer` is supplied because some collectors (gemma3 text-only) need it
/// inside the forward; arches that don't simply ignore it. `kldref` bakes the
/// per-position top-k lm-head reference into the sidecar (top-k fixed at 64, the
/// `collect_artifacts` example's value). The data plane stays in-process — only
/// the resulting [`CalibSummary`] crosses back to the daemon's JSONL.
pub trait CalibratableBackend {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        tokenizer: &crate::tokenizer::Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String>;
}

/// Outputs of an arch's capturing forward that the driver folds into the
/// streamed package. `extra_tensors` are small, already-in-RAM tensors (KLDREF
/// reference, MoE router histogram) appended to the `.calib.hfq`; `extra_meta`
/// are metadata fields known only AFTER the forward (e.g. KLDREF position/top-k
/// counts, the `artifacts` list when KLDREF is present). Both default empty for
/// arches that capture nothing beyond the Hessian/imatrix.
#[derive(Default)]
pub struct CalibForward {
    pub extra_tensors: Vec<HfqMemTensor>,
    pub extra_meta: Vec<(String, serde_json::Value)>,
}

/// General single-pass calibration-collection driver — the orchestration every
/// arch shares. Arms the [`CalibCollector`] as `gpu.active_capture`, runs the
/// arch's capturing `forward`, reads the descriptors, builds the standard
/// metadata, streams the HFQM `.calib.hfq`, and releases the GPU accumulators.
///
/// The arch supplies only its seams:
/// - `capture_names`: weight-buffer-addr → canonical tensor name (sans
///   `.weight`, so the quantizer joins `<name>.{hessian,imatrix}` to the source
///   weight). MoE routed experts are registered INDIVIDUALLY
///   (`…experts.{e}.gate_up_proj`) so the per-tensor consumer path matches them
///   uniformly across arches.
/// - `imatrix_only`: name substrings whose tensors skip the full [K,K] Hessian
///   and keep only the imatrix (e.g. `[".experts."]` — per-expert Hessians do
///   not fit).
/// - `static_meta`: provenance + arch-constant metadata fields.
/// - `forward`: runs the model over the calibration tokens (the
///   `gpu.maybe_capture_activation` taps fire inside it) and returns any
///   [`CalibForward`] extras it produced.
pub fn collect<F>(
    gpu: &mut Gpu,
    arch_id: u32,
    capture_names: HashMap<usize, String>,
    imatrix_only: Vec<String>,
    output: &std::path::Path,
    static_meta: &[(&str, serde_json::Value)],
    forward: F,
) -> Result<CalibSummary, String>
where
    F: FnOnce(&mut Gpu) -> Result<CalibForward, String>,
{
    let collector = arm(gpu, capture_names, imatrix_only);

    // Run the arch forward, then ALWAYS disarm the capture before propagating
    // any error (a half-armed `gpu` would mis-capture a later forward).
    let forward_out = forward(gpu);
    disarm(gpu);
    let forward_out = forward_out?;

    finish(gpu, &collector, arch_id, output, static_meta, &forward_out)
}

/// Arm `gpu` with a fresh [`CalibCollector`] and the caller's
/// weight-buffer-addr → tensor-name map. The returned handle must be passed to
/// [`finish`] after [`disarm`].
///
/// This is [`collect`]'s first half, exposed for drivers whose capturing
/// forward is not expressible as a single `FnOnce` closure — e.g. the DFlash
/// spec-decode loop, where the drafter's real inputs only exist inside a
/// long-running target/draft decode loop with borrowed state. Prefer
/// [`collect`] when a closure works.
pub fn arm(
    gpu: &mut Gpu,
    capture_names: HashMap<usize, String>,
    imatrix_only: Vec<String>,
) -> std::sync::Arc<CalibCollector> {
    let collector = std::sync::Arc::new(if imatrix_only.is_empty() {
        CalibCollector::new()
    } else {
        CalibCollector::with_imatrix_only(imatrix_only)
    });
    gpu.capture_names = capture_names;
    gpu.active_capture = Some(collector.clone());
    collector
}

/// Disarm the capture. Idempotent; always call before dropping the collector
/// (a half-armed `gpu` would mis-capture a later forward).
pub fn disarm(gpu: &mut Gpu) {
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
}

/// [`collect`]'s second half: build the standard metadata from the collector's
/// descriptors, stream the HFQM `.calib.hfq`, and release the GPU accumulators.
pub fn finish(
    gpu: &mut Gpu,
    collector: &CalibCollector,
    arch_id: u32,
    output: &std::path::Path,
    static_meta: &[(&str, serde_json::Value)],
    forward_out: &CalibForward,
) -> Result<CalibSummary, String> {
    let descriptors = collector.tensor_descriptors();
    if descriptors.is_empty() {
        return Err("calib: no tensors captured (check capture_names wiring)".to_string());
    }
    let n_hessian = descriptors.iter().filter(|d| d.has_hessian).count();
    let n_imatrix = descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for d in &descriptors {
        per_tensor_tokens.insert(d.name.clone(), serde_json::json!(d.n_tokens));
    }
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "n_hessian": n_hessian,
        "n_imatrix": n_imatrix,
        "per_tensor_tokens": serde_json::Value::Object(per_tensor_tokens),
        "artifacts": ["hessian", "imatrix"],
    });
    if let Some(obj) = meta.as_object_mut() {
        // static_meta first, then forward extras (so a dynamic field — e.g. the
        // KLDREF-augmented `artifacts` list — overrides the static default).
        for (k, v) in static_meta {
            obj.insert((*k).to_string(), v.clone());
        }
        for (k, v) in &forward_out.extra_meta {
            obj.insert(k.clone(), v.clone());
        }
    }
    let metadata_json = serde_json::to_string(&meta).unwrap();
    let max_consistency = collector
        .write_streaming(
            gpu,
            output,
            arch_id,
            &metadata_json,
            &forward_out.extra_tensors,
        )
        .map_err(|e| format!("calib write {}: {e}", output.display()))?;
    collector.free_gpu(gpu);

    Ok(CalibSummary {
        n_hessian,
        n_imatrix,
        max_consistency,
    })
}

/// Stream `v * scale` as little-endian f32 to `w` in bounded chunks (so a
/// multi-hundred-MB Hessian never materializes a second full byte buffer).
fn write_f32_scaled(w: &mut dyn std::io::Write, v: &[f32], scale: f32) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(16384);
    for &x in v {
        buf.extend_from_slice(&(x * scale).to_le_bytes());
        if buf.len() >= 16384 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        w.write_all(&buf)?;
    }
    Ok(())
}

fn write_hessian_bf16_tril_diag_f32(
    w: &mut dyn std::io::Write,
    h: &[f32],
    diag: &[f32],
    k: usize,
    scale: f32,
) -> std::io::Result<()> {
    assert_eq!(h.len(), k * k);
    assert_eq!(diag.len(), k);
    let mut buf: Vec<u8> = Vec::with_capacity(16384);
    for &x in diag {
        buf.extend_from_slice(&(x * scale).to_le_bytes());
        if buf.len() >= 16384 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    for i in 1..k {
        for j in 0..i {
            let bits = hipfire_primitives::conv::f32_to_bf16_bits(h[i * k + j] * scale);
            buf.extend_from_slice(&bits.to_le_bytes());
            if buf.len() >= 16384 {
                w.write_all(&buf)?;
                buf.clear();
            }
        }
    }
    if !buf.is_empty() {
        w.write_all(&buf)?;
    }
    Ok(())
}

impl ActivationCapture for CalibCollector {
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize) {
        // n/k come from the gemm — `input` is a shared scratch buffer whose shape
        // (max(dim,hidden)) does NOT reflect the linear's input width.
        let mut accs = self.accs.lock().unwrap();
        if !accs.contains_key(tensor_name) {
            let diag = gpu.zeros(&[k], DType::F32).unwrap();
            let h = if self.wants_hessian(tensor_name) {
                Some(gpu.zeros(&[k, k], DType::F32).unwrap())
            } else {
                None
            };
            let buf = gpu.zeros(&[FLUSH_BATCH, k], DType::F32).unwrap();
            let h_f64 = if self.f64_audit && h.is_some() {
                Some(vec![0.0f64; k * k])
            } else {
                None
            };
            accs.insert(
                tensor_name.to_string(),
                Acc {
                    diag,
                    h,
                    h_f64,
                    buf,
                    buf_rows: 0,
                    k,
                    n_tokens: 0,
                },
            );
        }
        let acc = accs.get_mut(tensor_name).unwrap();
        // Stage each activation row into the flush buffer; the actual reductions
        // run a single batched launch per FLUSH_BATCH rows (Acc::flush). `input`
        // is a shared scratch buffer of width `row_stride` ≥ k, so copy the first
        // k columns of each of the n rows.
        let row_stride = input.numel() / n.max(1);
        for r in 0..n {
            if acc.buf_rows == FLUSH_BATCH {
                acc.flush(gpu);
            }
            gpu.memcpy_dtod_at_auto(
                &acc.buf.buf,
                acc.buf_rows * k * 4,
                &input.buf,
                r * row_stride * 4,
                k * 4,
            )
            .unwrap();
            acc.buf_rows += 1;
        }
        acc.n_tokens += n as u64;
    }
}

/// log(Σ exp(logits)) — numerically stable. For the KLDREF reference (callers
/// that tap lm-head logits).
pub fn logsumexp(logits: &[f32]) -> f32 {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln()
}

/// Top-`k` (index, logit) descending — for the KLDREF reference.
pub fn topk_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.truncate(k);
    idx.into_iter().map(|i| (i, logits[i as usize])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_hessian_size_matches_diag_plus_lower_triangle() {
        assert_eq!(compact_hessian_bytes(1), 4);
        assert_eq!(compact_hessian_bytes(2), 10);
        assert_eq!(compact_hessian_bytes(3), 18);
        assert_eq!(compact_hessian_bytes(4096), 16_789_504);
    }

    #[test]
    fn compact_hessian_writer_keeps_diag_f32_and_lower_bf16() {
        let h = [1.0f32, 0.5, -0.25, 0.5, 2.0, 0.75, -0.25, 0.75, 4.0];
        let diag = [1.0f32, 2.0, 4.0];
        let mut out = Vec::new();
        write_hessian_bf16_tril_diag_f32(&mut out, &h, &diag, 3, 1.0).unwrap();
        assert_eq!(out.len(), compact_hessian_bytes(3) as usize);

        let read_f32 = |off: usize| f32::from_le_bytes(out[off..off + 4].try_into().unwrap());
        let read_bf16 = |off: usize| {
            hipfire_primitives::conv::bf16_bits_to_f32(u16::from_le_bytes(
                out[off..off + 2].try_into().unwrap(),
            ))
        };
        assert_eq!(read_f32(0), 1.0);
        assert_eq!(read_f32(4), 2.0);
        assert_eq!(read_f32(8), 4.0);
        assert_eq!(read_bf16(12), 0.5);
        assert_eq!(read_bf16(14), -0.25);
        assert_eq!(read_bf16(16), 0.75);
    }

    #[test]
    fn embedding_sample_split_preserves_boundaries_and_budget() {
        let text = " first paragraph \ncontinues here\n\n\nsecond\n\nthird";
        let samples = tokenize_embedding_samples(text, 5, |sample| {
            sample
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| i as u32)
                .collect()
        });
        assert_eq!(samples.iter().map(Vec::len).collect::<Vec<_>>(), [4, 1]);

        let lines = tokenize_embedding_samples("one two\nthree\nfour five", 3, |sample| {
            sample
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| i as u32)
                .collect()
        });
        assert_eq!(lines.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1]);
    }

    #[test]
    fn embedding_sample_split_drops_empty_encodings() {
        let samples = tokenize_embedding_samples("skip\nkeep", 2, |sample| {
            if sample == "skip" {
                Vec::new()
            } else {
                vec![7]
            }
        });
        assert_eq!(samples, vec![vec![7]]);
    }

    #[test]
    fn qwen3_embedding_capture_names_cover_all_encoder_linears() {
        let names = qwen3_embedding_layer_tensor_names(3);
        assert_eq!(names.len(), 7);
        assert_eq!(names[0], "model.layers.3.self_attn.q_proj");
        assert_eq!(names[1], "model.layers.3.self_attn.k_proj");
        assert_eq!(names[2], "model.layers.3.self_attn.v_proj");
        assert_eq!(names[3], "model.layers.3.self_attn.o_proj");
        assert_eq!(names[4], "model.layers.3.mlp.gate_proj");
        assert_eq!(names[5], "model.layers.3.mlp.up_proj");
        assert_eq!(names[6], "model.layers.3.mlp.down_proj");
    }

    #[test]
    fn qwen3_embedding_calibration_rejects_bad_sample_geometry() {
        assert!(validate_qwen3_embedding_samples(&[], 2048).is_err());
        assert!(validate_qwen3_embedding_samples(&[Vec::new()], 2048).is_err());
        let error = validate_qwen3_embedding_samples(&[vec![1, 2, 3]], 2).unwrap_err();
        assert!(error.contains("sample 0 length 3 exceeds maximum 2"));
        assert!(validate_qwen3_embedding_samples(&[vec![1], vec![2, 3]], 2).is_ok());
    }

    #[test]
    fn host_activation_validation_rejects_bad_shapes_before_gpu_use() {
        assert!(validate_host_activations("dense.0", &[], 0, 3).is_err());
        assert!(validate_host_activations("dense.0", &[1.0, 2.0], 1, 3).is_err());
        assert!(validate_host_activations("", &[1.0], 1, 1).is_err());
    }

    #[test]
    #[ignore = "requires working ROCm calibration kernels"]
    fn host_activation_capture_writes_expected_imatrix_and_hessian() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => return,
        };
        let collector = std::sync::Arc::new(CalibCollector::new());
        gpu.active_capture = Some(collector.clone());
        capture_host_activations(&mut gpu, "dense.0", &[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        gpu.active_capture = None;

        let path = std::env::temp_dir().join(format!(
            "hipfire-host-calib-{}-{}.calib.hfq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        collector
            .write_streaming(&mut gpu, &path, 19, "{}", &[])
            .unwrap();
        collector.free_gpu(&mut gpu);

        let hfq = crate::hfq::HfqFile::open(&path).unwrap();
        let (_, imatrix) = hfq.tensor_data_vec("dense.0.imatrix").unwrap();
        let read_f32 = |data: &[u8], offset: usize| {
            f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
        };
        assert!((read_f32(&imatrix, 0) - 5.0).abs() < 1e-6);
        assert!((read_f32(&imatrix, 4) - 10.0).abs() < 1e-6);

        let (hessian_info, hessian) = hfq.tensor_data_vec("dense.0.hessian").unwrap();
        match hessian_info.quant_type {
            2 => {
                assert!((read_f32(&hessian, 0) - 5.0).abs() < 1e-6);
                assert!((read_f32(&hessian, 4) - 7.0).abs() < 1e-6);
                assert!((read_f32(&hessian, 12) - 10.0).abs() < 1e-6);
            }
            QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32 => {
                assert!((read_f32(&hessian, 0) - 5.0).abs() < 1e-6);
                assert!((read_f32(&hessian, 4) - 10.0).abs() < 1e-6);
                let lower = hipfire_primitives::conv::bf16_bits_to_f32(u16::from_le_bytes(
                    hessian[8..10].try_into().unwrap(),
                ));
                assert!((lower - 7.0).abs() < 1e-3);
            }
            quant_type => panic!("unexpected Hessian quant type {quant_type}"),
        }
        drop(hfq);
        let _ = std::fs::remove_file(path);
    }
}
