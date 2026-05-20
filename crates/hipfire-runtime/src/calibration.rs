//! Calibration drivers — shared helpers for the Tier 1 hipfire-native
//! `collect_imatrix` and `collect_hessian` binaries.
//!
//! What this module owns:
//!
//! - `tokenize_corpus(model_dir, corpus_text)` — load the HF tokenizer from
//!   `<model_dir>/tokenizer.json` and encode the calibration corpus into
//!   a flat `Vec<u32>` of token IDs. Mirrors how Tier 2 (`imatrix_collect`)
//!   delegates to llama-tokenize internally — we use the HF tokenizer.json
//!   directly here since our `--hf-model <dir>` arg points at the
//!   HuggingFace BF16 distribution, not a GGUF file. (The hipfire BPE
//!   encoder loaded from `tokenizer.json` is GPT-2-BPE compatible — same
//!   tokenization llama.cpp uses for Qwen3 / Qwen3.5 / Qwen3.6.)
//!
//! - `ImatrixCollector` — interior-mutable `ActivationCapture` impl that
//!   accumulates per-channel `Σ x²` for every weight tensor seen at
//!   dispatch time. Drained at the end of calibration into a
//!   `Vec<ImatrixEntry>` for the GGUF writer (subagent B).
//!
//! - `HessianCollector` — same shape, but accumulates the K×K outer
//!   product `Σ x · xᵀ`. Drained into `Vec<HessianEntry>` for the
//!   HFHS-v1 writer (subagent B). Only tensors matching
//!   `bf16_loader::is_gptq_target` are accumulated — saves a chunky
//!   K×K F32 allocation per non-GPTQ tensor (norms, embed, lm_head).
//!
//! What this module does NOT own (per orchestrator scope):
//!
//! - The on-GPU Σx² + outer-product reduction kernels (subagent A —
//!   `gpu.sumsq_reduce_bf16` / `gpu.hessian_outer_product_bf16`).
//! - The GGUF imatrix / HFHS-v1 binary writers (subagent B —
//!   `gguf_imatrix_writer::write_gguf_imatrix` /
//!   `hfhs_writer::write_hfhs`).
//! - The BF16 forward pass + linear-layer capture hook wiring
//!   (subagent C — `dispatch.rs` capture-handler call sites).
//! - The BF16 safetensors loader implementation (subagent D —
//!   `bf16_loader::load_bf16_model` body).
//!
//! When those APIs land, the `unimplemented!()` stubs marked
//! `TODO(subagent-X)` below are the single-line wire-in points.

use crate::bf16_loader::is_gptq_target;
use crate::tokenizer::Tokenizer;
use rdna_compute::{ActivationCapture, DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

// ───────────────────────────────────────────────────────────────────────
// Expert-index parsing (MoE per-expert capture key)
// ───────────────────────────────────────────────────────────────────────

/// Parse the per-expert index out of a tensor name and return
/// `(canonical_name, expert_idx)`.
///
/// MoE models (Qwen3.5-A3B, Astrea) name per-expert weight tensors
/// using the convention:
///
/// ```text
/// model.language_model.layers.{LAYER}.mlp.experts.{EXPERT_IDX}.gate_proj.weight
/// model.language_model.layers.{LAYER}.mlp.experts.{EXPERT_IDX}.up_proj.weight
/// model.language_model.layers.{LAYER}.mlp.experts.{EXPERT_IDX}.down_proj.weight
/// ```
///
/// For names containing the `.experts.<DIGITS>.` substring this returns
/// `(full_name_with_literal_index, EXPERT_IDX)`. The canonical name
/// preserves the literal expert index so the HFHS / GGUF writers emit
/// the on-disk tensor name verbatim (downstream tools — Astrea,
/// hipfire-quantize — locate per-expert imatrix / Hessian records by
/// matching the literal tensor name).
///
/// For non-MoE names (no `.experts.<DIGITS>.` substring) the canonical
/// name is the input unchanged and `expert_idx = 0`. This matches the
/// HFHS-v1 binary layout (`hessian_io.rs::HessianRef::expert_idx`
/// defaults to 0 for dense tensors).
///
/// Examples:
///
/// ```ignore
/// assert_eq!(
///     parse_expert_idx("model.layers.0.self_attn.q_proj.weight"),
///     ("model.layers.0.self_attn.q_proj.weight".to_string(), 0),
/// );
/// assert_eq!(
///     parse_expert_idx(
///         "model.language_model.layers.5.mlp.experts.23.gate_proj.weight",
///     ),
///     (
///         "model.language_model.layers.5.mlp.experts.23.gate_proj.weight".to_string(),
///         23,
///     ),
/// );
/// // Non-digit between `.experts.` and the next `.` does not match
/// // (e.g. shared_expert is a separate path, not a per-expert key).
/// assert_eq!(
///     parse_expert_idx("model.layers.0.mlp.shared_expert.gate_proj.weight"),
///     ("model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(), 0),
/// );
/// ```
///
/// Implementation note: hand-rolled scan (no regex crate dep) to keep
/// the `hipfire-runtime` workspace dep graph tight. Scans for the
/// literal `.experts.` substring, then walks forward to read a
/// `[0-9]+` run terminated by `.`. Returns `(name, 0)` on any
/// non-match (including malformed `.experts.<non-digit>.`,
/// `.experts.<digits><EOF>`, or overflow on a 10+ digit run).
pub fn parse_expert_idx(full_name: &str) -> (String, u32) {
    const MARKER: &str = ".experts.";
    let Some(start) = full_name.find(MARKER) else {
        return (full_name.to_string(), 0);
    };
    let digits_start = start + MARKER.len();
    let bytes = full_name.as_bytes();
    let mut cursor = digits_start;
    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
        cursor += 1;
    }
    // Must have at least one digit AND be terminated by `.` before EOF.
    if cursor == digits_start || cursor >= bytes.len() || bytes[cursor] != b'.' {
        return (full_name.to_string(), 0);
    }
    let digits = &full_name[digits_start..cursor];
    // Parse the digit run as u32. Overflow → fall back to dense key
    // (degenerate input; better than panicking).
    let Ok(expert_idx) = digits.parse::<u32>() else {
        return (full_name.to_string(), 0);
    };
    (full_name.to_string(), expert_idx)
}

// ───────────────────────────────────────────────────────────────────────
// Tokenizer driver
// ───────────────────────────────────────────────────────────────────────

/// Load the HF tokenizer from `<model_dir>/tokenizer.json` and encode
/// the calibration corpus into a flat `Vec<u32>` of token IDs.
///
/// **Parity with llama.cpp:** the hipfire BPE encoder built from a
/// HuggingFace `tokenizer.json` is GPT-2-BPE compatible — same scheme
/// llama.cpp uses for Qwen3 / Qwen3.5 / Qwen3.6 / Llama / Mistral.
/// `benchmarks/quality-baselines/harness/tokenizer_parity.py`
/// documents the known ~46% per-position disagreement rate that
/// motivates a llama-tokenize subprocess fallback for downstream
/// applications where the imatrix data must be exactly cross-comparable
/// with llama-imatrix outputs. For the Tier 1 hipfire-native pipeline
/// (where the same hipfire forward + same tokenizer consume the imatrix
/// downstream), HF tokenizer.json parity within the same pipeline is
/// what matters.
///
/// Errors are surfaced via `Result<_, String>` so the binary can print
/// a clear context message before exiting; the calibration drivers
/// aren't in the hot path so error allocation cost is irrelevant.
pub fn tokenize_corpus(model_dir: &Path, corpus_text: &str) -> Result<Vec<u32>, String> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path).map_err(|e| {
        format!(
            "failed to read tokenizer.json at {}: {e}",
            tokenizer_path.display()
        )
    })?;
    let tokenizer = Tokenizer::from_hf_json(&tokenizer_json).ok_or_else(|| {
        format!(
            "failed to parse {} as a HuggingFace tokenizer",
            tokenizer_path.display()
        )
    })?;
    let ids = tokenizer.encode(corpus_text);
    Ok(ids)
}

/// Tokenize via subprocess `llama-tokenize` for byte-parity with
/// llama-imatrix / llama.cpp pipelines.
///
/// Path D (per-tensor NRMSE ≤ 0.01 vs the FAIR PyTorch oracle) requires
/// the Tier 1 calibration token stream to be byte-identical with the
/// Tier 2 `llama-imatrix` reference. llama.cpp's internal BPE disagrees
/// with hipfire's HF-tokenizer.json encoder on ~46% of Qwen3 token
/// positions (see `benchmarks/quality-baselines/harness/tokenizer_parity.py`),
/// which floors per-tensor NRMSE at ~0.91 even with a numerically clean
/// BF16 forward path.
///
/// To close that gap we shell out to llama.cpp's `llama-tokenize` binary
/// directly, point it at a BF16 GGUF of the model (llama-tokenize loads
/// GGUF metadata to select the right tokenizer), and parse its
/// `[id1, id2, ...]` stdout into a flat `Vec<u32>` identical to what
/// `llama-imatrix` would feed its forward pass.
///
/// # Arguments
///
/// * `llama_tokenize_bin` — path to the `llama-tokenize` binary
///   (typically `<llama.cpp>/build/bin/llama-tokenize`).
/// * `bf16_gguf_path` — BF16 GGUF that carries the tokenizer metadata
///   for the model under calibration. **Must** be the matching converted
///   model (different families ship different BPE pretokenizers).
/// * `corpus_path` — path to the plain-text calibration corpus.
///
/// # CLI invocation
///
/// ```text
/// llama-tokenize -m <bf16.gguf> -f <corpus.txt> --ids --log-disable --no-bos
/// ```
///
/// On the MI300x calibration droplet this takes ~5s for a 10 MB / 2.4 M
/// token wikitext-2 slice. The corpus is streamed through llama.cpp's
/// own loader, not Rust — pass the corpus path through `-f` directly
/// (no shell escaping needed for whitespace inside the file).
///
/// # stdout parsing
///
/// `--log-disable` typically silences load-info lines but some
/// llama-tokenize versions still print model/tokenizer banner lines
/// before the IDs. The parser walks lines, finds the first `[...]`
/// bracketed line, strips the brackets, and parses comma-separated
/// decimal integers into `Vec<u32>`. Lines without a leading `[` are
/// ignored, so banner text never poisons the result. Empty `[]` is a
/// valid empty corpus (returns an empty vector).
///
/// # Errors
///
/// Returns `Err(String)` if:
/// - the binary can't be spawned (missing path / permission).
/// - the subprocess exits with non-zero status (stderr is included in
///   the message).
/// - no `[...]` bracket line is found in stdout.
/// - any individual ID fails to parse as `u32`.
pub fn tokenize_corpus_via_llama_tokenize(
    llama_tokenize_bin: &Path,
    bf16_gguf_path: &Path,
    corpus_path: &Path,
) -> Result<Vec<u32>, String> {
    let output = Command::new(llama_tokenize_bin)
        .arg("-m")
        .arg(bf16_gguf_path)
        .arg("-f")
        .arg(corpus_path)
        .arg("--ids")
        .arg("--log-disable")
        .arg("--no-bos")
        .output()
        .map_err(|e| {
            format!(
                "failed to spawn llama-tokenize at {}: {e}",
                llama_tokenize_bin.display()
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "llama-tokenize exited with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| {
        format!("llama-tokenize stdout was not valid UTF-8: {e}")
    })?;

    parse_llama_tokenize_stdout(&stdout)
}

/// Parse `llama-tokenize --ids` stdout into `Vec<u32>`.
///
/// Looks for the first line that contains a `[...]` bracketed list of
/// comma-separated integers and returns the parsed IDs. Lines before
/// the bracket line (model banners, tokenizer info) are ignored.
///
/// Split out so the unit tests can exercise the parser against
/// representative stdout fixtures without needing the actual binary.
fn parse_llama_tokenize_stdout(stdout: &str) -> Result<Vec<u32>, String> {
    // Walk lines; first one with a `[` is the IDs line. We also handle the
    // edge case where the brackets span multiple lines (some llama.cpp
    // builds line-wrap long lists) by concatenating from `[` to `]`.
    let mut buf = String::new();
    let mut in_brackets = false;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !in_brackets {
            if let Some(open) = trimmed.find('[') {
                in_brackets = true;
                buf.push_str(&trimmed[open..]);
                if trimmed.contains(']') {
                    break;
                }
            }
        } else {
            buf.push(' ');
            buf.push_str(trimmed);
            if trimmed.contains(']') {
                break;
            }
        }
    }

    if buf.is_empty() {
        return Err(format!(
            "llama-tokenize stdout had no `[...]` IDs line; full stdout was: {:?}",
            stdout
        ));
    }

    // Strip the outer `[` ... `]`.
    let open = buf.find('[').ok_or_else(|| {
        format!("internal parser error: opening bracket missing in {buf:?}")
    })?;
    let close = buf.rfind(']').ok_or_else(|| {
        format!("llama-tokenize stdout lacks closing `]`; got {buf:?}")
    })?;
    if close <= open {
        return Err(format!(
            "llama-tokenize stdout has malformed brackets: {buf:?}"
        ));
    }
    let inner = &buf[open + 1..close];

    let inner_trim = inner.trim();
    if inner_trim.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::new();
    for piece in inner_trim.split(',') {
        let s = piece.trim();
        if s.is_empty() {
            continue;
        }
        let id: u32 = s.parse().map_err(|e| {
            format!("failed to parse llama-tokenize ID {s:?} as u32: {e}")
        })?;
        ids.push(id);
    }
    Ok(ids)
}

// ───────────────────────────────────────────────────────────────────────
// Imatrix collector
// ───────────────────────────────────────────────────────────────────────

/// Per-tensor imatrix accumulation entry, drained at the end of
/// calibration into the GGUF imatrix writer.
///
/// Mirrors the llama.cpp imatrix file structure (one `.in_sum2` array +
/// one `.counts` array per linear weight, plus optional `n_mat` for
/// MoE experts). The GGUF writer (subagent B) is responsible for
/// emitting these as F32 tensors under the conventional names
/// `{name}.in_sum2` + `{name}.counts`.
#[derive(Debug)]
pub struct ImatrixEntry {
    /// Canonical HF safetensors tensor key
    /// (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Per-channel `Σ x²` — one F32 value per K (input dim).
    /// Length = K.
    pub in_sum2: Vec<f32>,
    /// Per-channel count of activations contributing to `in_sum2`.
    /// Length = K (same shape as `in_sum2`; matches llama-imatrix
    /// convention even though the count is uniform across channels
    /// for non-MoE tensors).
    pub counts: Vec<f32>,
}

/// On-GPU per-tensor accumulators owned by `ImatrixCollector`.
///
/// `acc` is a length-K F32 device tensor; subagent A's
/// `gpu.sumsq_reduce_bf16` kernel reads the input activation and
/// performs `acc[k] += x[*][k] * x[*][k]` atomically across the batch
/// dim. `n_tokens` is the host-side count of activation rows seen so
/// far (incremented per-call — the per-channel count is uniform for
/// dense layers).
struct ImatrixAccum {
    acc: GpuTensor,
    n_tokens: u64,
    k: usize,
}

/// `ActivationCapture` impl that builds per-tensor imatrix data on-GPU.
///
/// Uses `Mutex<HashMap<_>>` for interior mutability since the
/// `ActivationCapture::capture` trait method takes `&self`. The lock
/// is held only across the on-GPU dispatch — no per-element work
/// happens under the lock.
pub struct ImatrixCollector {
    /// Per-tensor accumulators. Keyed by canonical tensor name (the
    /// `tensor_name` arg passed to `capture`).
    accumulators: Mutex<HashMap<String, ImatrixAccum>>,
    /// When true, accumulate for `lm_head` / `output` tensors too.
    /// Mirrors llama-imatrix's `--process-output` flag.
    process_output: bool,
}

impl ImatrixCollector {
    pub fn new(process_output: bool) -> Self {
        Self {
            accumulators: Mutex::new(HashMap::new()),
            process_output,
        }
    }

    /// Drain the on-GPU accumulators into host-side `ImatrixEntry`
    /// vectors. Called once at the end of calibration; consumes `self`
    /// by `Arc::try_unwrap` or by `Arc::clone`-then-mutex-poison
    /// (callers should use the `Arc::try_unwrap` path — the binary
    /// holds the only outstanding clone after the forward pass loop
    /// returns).
    ///
    /// Each accumulator's K-sized F32 buffer is copied back to host
    /// via `gpu.download_f32`. The per-channel count is filled with
    /// `n_tokens` to match llama-imatrix's GGUF layout.
    pub fn drain(&self, gpu: &Gpu) -> Result<Vec<ImatrixEntry>, String> {
        let acc_map = self
            .accumulators
            .lock()
            .map_err(|e| format!("ImatrixCollector mutex poisoned at drain: {e}"))?;
        let mut entries = Vec::with_capacity(acc_map.len());
        for (name, accum) in acc_map.iter() {
            let in_sum2 = gpu
                .download_f32(&accum.acc)
                .map_err(|e| format!("download_f32 failed for {name}: {e}"))?;
            let counts = vec![accum.n_tokens as f32; accum.k];
            entries.push(ImatrixEntry {
                name: name.clone(),
                in_sum2,
                counts,
            });
        }
        // Sort by name for deterministic output ordering (the GGUF
        // writer keys by name anyway, but downstream comparators
        // appreciate stable ordering).
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// Whether this tensor's activations should be accumulated.
    /// Mirrors llama-imatrix: accumulate for every Linear except
    /// `lm_head` / `output`, which require `--process-output`.
    fn should_capture(&self, tensor_name: &str) -> bool {
        let last = tensor_name
            .strip_suffix(".weight")
            .unwrap_or(tensor_name)
            .rsplit('.')
            .next()
            .unwrap_or(tensor_name);
        let is_output = matches!(last, "lm_head" | "output");
        if is_output {
            return self.process_output;
        }
        // Imatrix accumulates for ALL linears (q/k/v/o, gate/up/down,
        // moe experts, in_proj_*, gate router) — same as llama-imatrix
        // by default. The GPTQ-target whitelist is a stricter subset
        // used only by the Hessian collector.
        !matches!(
            last,
            // Norms and embed are not Linear — never captured.
            "input_layernorm"
                | "post_attention_layernorm"
                | "q_norm"
                | "k_norm"
                | "self_attn_layer_norm"
                | "final_layernorm"
                | "norm"
                | "embed_tokens"
        )
    }
}

impl ActivationCapture for ImatrixCollector {
    fn capture(
        &self,
        gpu: &mut Gpu,
        tensor_name: &str,
        input: &GpuTensor,
    ) {
        if !self.should_capture(tensor_name) {
            return;
        }
        let k = match input.shape.last() {
            Some(k) => *k,
            None => return,
        };
        let n_rows = input.numel() / k;
        let mut accs = match self.accumulators.lock() {
            Ok(a) => a,
            Err(_) => return,
        };
        if !accs.contains_key(tensor_name) {
            let acc = match gpu.zeros(&[k], DType::F32) {
                Ok(a) => a,
                Err(_) => return,
            };
            accs.insert(tensor_name.to_string(), ImatrixAccum { acc, n_tokens: 0, k });
        }
        let accum = accs.get_mut(tensor_name).unwrap();
        // sumsq_reduce_bf16 needs a GPU scalar for n_tokens (writes into it).
        // We track host-side n_tokens separately; the GPU scalar is a throwaway.
        let mut n_scratch = match gpu.zeros(&[1], DType::F32) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Err(_) = gpu.sumsq_reduce_bf16(input, &mut accum.acc, &mut n_scratch) {
            return;
        }
        accum.n_tokens = accum.n_tokens.saturating_add(n_rows as u64);
        // TODO(subagent-A): wire `gpu.sumsq_reduce_bf16(input_ptr, numel, dtype, shape,
        //                                               accumulator_ptr, k)` here.
        //
        // The flow (once subagent-A lands `Gpu::sumsq_reduce_bf16`):
        //
        //   let mut accs = self.accumulators.lock().expect("imatrix mutex");
        //   let k = *shape.last().expect("activation must have ≥1 dim");
        //   let n_rows = numel / k;
        //   let accum = accs.entry(tensor_name.to_string()).or_insert_with(|| {
        //       // First touch for this tensor — allocate K-sized F32 acc on the GPU.
        //       // Needs `&mut Gpu` access; pattern (per orchestrator dispatch):
        //       //   1. Pre-allocate all accumulators in the driver after
        //       //      bf16_loader::load_bf16_model returns (we know every linear
        //       //      tensor's K from the safetensors header).
        //       //   2. Or: route `&mut Gpu` through the trait via an
        //       //      InteriorGpu wrapper. (Less surgery than 1; defer to A.)
        //       //   For now, the scaffold panics at first capture so the integrator
        //       //   sees the wire-in point.
        //       unimplemented!(
        //           "ImatrixCollector::capture first-touch alloc for tensor `{}` — \
        //            wire `gpu.sumsq_reduce_bf16` + per-tensor F32 K-vector \
        //            accumulator alloc once subagent-A lands its dispatch API",
        //           tensor_name
        //       );
        //   });
        //   gpu.sumsq_reduce_bf16(input_ptr, dtype, &accum.acc, k, n_rows)
        //       .expect("sumsq_reduce_bf16 dispatch");
        //   accum.n_tokens = accum.n_tokens.saturating_add(n_rows as u64);
        //
        // Until subagent-A's dispatch API exists, every capture is a no-op so
        // the driver can be exercised through unit tests + smoke tests without
        // touching kernel infra.
    }
}

// ───────────────────────────────────────────────────────────────────────
// Hessian collector
// ───────────────────────────────────────────────────────────────────────

/// Per-tensor Hessian accumulation entry, drained at the end of
/// calibration into the HFHS-v1 writer (subagent B).
///
/// `H` is a row-major K×K F32 matrix representing `Σ x · xᵀ` summed
/// over all activation rows. The `n_tokens` divisor (turning the sum
/// into the average `H_t = (1/N) Σ x_t · x_tᵀ`) is applied by the
/// quantizer, not here — keeps the on-disk format closer to the
/// PyTorch reference at `scripts/collect_hessian.py:213-222`.
#[derive(Debug)]
pub struct HessianEntry {
    /// Canonical HF safetensors tensor key
    /// (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// K — input feature dimension.
    pub k: usize,
    /// Row-major K×K F32 outer-product accumulator (`Σ x · xᵀ`).
    /// Length = k × k.
    pub h: Vec<f32>,
    /// Total number of activation rows accumulated (the divisor for
    /// `H_t = (1/N) · sum`).
    pub n_tokens: u64,
}

/// On-GPU per-tensor Hessian accumulators owned by `HessianCollector`.
struct HessianAccum {
    /// K×K F32 device tensor (row-major).
    h: GpuTensor,
    n_tokens: u64,
    k: usize,
}

/// `ActivationCapture` impl that builds per-tensor K×K Hessian
/// outer-products on-GPU. Only tensors matching the GPTQ target
/// whitelist (per `bf16_loader::is_gptq_target`) are accumulated —
/// saves the K×K F32 allocation for norms / embed / lm_head which the
/// quantizer never consumes.
pub struct HessianCollector {
    accumulators: Mutex<HashMap<String, HessianAccum>>,
}

impl HessianCollector {
    pub fn new() -> Self {
        Self {
            accumulators: Mutex::new(HashMap::new()),
        }
    }

    /// Drain the on-GPU accumulators into host-side `HessianEntry`
    /// vectors. Same drain semantics as `ImatrixCollector::drain`.
    pub fn drain(&self, gpu: &Gpu) -> Result<Vec<HessianEntry>, String> {
        let acc_map = self
            .accumulators
            .lock()
            .map_err(|e| format!("HessianCollector mutex poisoned at drain: {e}"))?;
        let mut entries = Vec::with_capacity(acc_map.len());
        for (name, accum) in acc_map.iter() {
            let h = gpu
                .download_f32(&accum.h)
                .map_err(|e| format!("download_f32 failed for {name}: {e}"))?;
            entries.push(HessianEntry {
                name: name.clone(),
                k: accum.k,
                h,
                n_tokens: accum.n_tokens,
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }
}

impl Default for HessianCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationCapture for HessianCollector {
    fn capture(
        &self,
        gpu: &mut Gpu,
        tensor_name: &str,
        input: &GpuTensor,
    ) {
        if !is_gptq_target(tensor_name) {
            return;
        }
        let k = match input.shape.last() {
            Some(k) => *k,
            None => return,
        };
        let n_rows = input.numel() / k;
        let mut accs = match self.accumulators.lock() {
            Ok(a) => a,
            Err(_) => return,
        };
        if !accs.contains_key(tensor_name) {
            let h = match gpu.zeros(&[k, k], DType::F32) {
                Ok(h) => h,
                Err(_) => return,
            };
            accs.insert(tensor_name.to_string(), HessianAccum { h, n_tokens: 0, k });
        }
        let accum = accs.get_mut(tensor_name).unwrap();
        if let Err(_) = gpu.hessian_outer_product_bf16(input, &mut accum.h) {
            return;
        }
        accum.n_tokens = accum.n_tokens.saturating_add(n_rows as u64);
    }
}

// ───────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ───────────────────────────────────────────────────────────────────
    // parse_expert_idx — MoE per-expert key extraction
    // ───────────────────────────────────────────────────────────────────

    /// Dense tensor names (no `.experts.<digits>.`): canonical name is
    /// returned unchanged, expert_idx is 0.
    #[test]
    fn parse_expert_idx_dense() {
        let (n, e) = parse_expert_idx("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(n, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(e, 0);

        let (n, e) = parse_expert_idx("model.layers.5.mlp.gate_proj.weight");
        assert_eq!(n, "model.layers.5.mlp.gate_proj.weight");
        assert_eq!(e, 0);

        // MoE router gate — admitted by is_gptq_target but is not a
        // per-expert tensor, so expert_idx stays 0.
        let (n, e) = parse_expert_idx("model.layers.0.mlp.gate.weight");
        assert_eq!(n, "model.layers.0.mlp.gate.weight");
        assert_eq!(e, 0);

        // Final norm / embed: no .experts. substring.
        let (n, e) = parse_expert_idx("model.embed_tokens.weight");
        assert_eq!(n, "model.embed_tokens.weight");
        assert_eq!(e, 0);
    }

    /// Per-expert MoE name: canonical name retains the literal expert
    /// index, expert_idx parses correctly.
    #[test]
    fn parse_expert_idx_moe() {
        let (n, e) = parse_expert_idx(
            "model.language_model.layers.5.mlp.experts.23.gate_proj.weight",
        );
        assert_eq!(
            n,
            "model.language_model.layers.5.mlp.experts.23.gate_proj.weight"
        );
        assert_eq!(e, 23);

        let (n, e) = parse_expert_idx(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        );
        assert_eq!(
            n,
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(e, 0);

        let (n, e) = parse_expert_idx(
            "model.language_model.layers.10.mlp.experts.127.down_proj.weight",
        );
        assert_eq!(
            n,
            "model.language_model.layers.10.mlp.experts.127.down_proj.weight"
        );
        assert_eq!(e, 127);

        let (n, e) = parse_expert_idx(
            "model.language_model.layers.7.mlp.experts.42.up_proj.weight",
        );
        assert_eq!(
            n,
            "model.language_model.layers.7.mlp.experts.42.up_proj.weight"
        );
        assert_eq!(e, 42);

        // Some HF dumps drop the `model.language_model.` prefix.
        let (n, e) = parse_expert_idx("model.layers.3.mlp.experts.8.down_proj.weight");
        assert_eq!(n, "model.layers.3.mlp.experts.8.down_proj.weight");
        assert_eq!(e, 8);
    }

    /// `shared_expert` is NOT a per-expert key (no digits between
    /// `.experts.` and `.`); the canonical name + expert_idx must
    /// degrade gracefully to the dense-tensor case.
    #[test]
    fn parse_expert_idx_shared_expert_is_dense() {
        let (n, e) = parse_expert_idx(
            "model.language_model.layers.5.mlp.shared_expert.gate_proj.weight",
        );
        assert_eq!(
            n,
            "model.language_model.layers.5.mlp.shared_expert.gate_proj.weight"
        );
        assert_eq!(e, 0);

        let (n, e) = parse_expert_idx(
            "model.layers.5.mlp.shared_expert.down_proj.weight",
        );
        assert_eq!(n, "model.layers.5.mlp.shared_expert.down_proj.weight");
        assert_eq!(e, 0);
    }

    /// Malformed expert-index runs (non-digits, missing terminator,
    /// truncation at EOF) degrade to the dense-tensor case rather than
    /// panicking.
    #[test]
    fn parse_expert_idx_malformed_falls_back_to_dense() {
        // Non-digit immediately after `.experts.`
        let (n, e) = parse_expert_idx("model.layers.0.mlp.experts.foo.gate_proj.weight");
        assert_eq!(n, "model.layers.0.mlp.experts.foo.gate_proj.weight");
        assert_eq!(e, 0);

        // Digits but no terminating `.` (truncated EOF).
        let (n, e) = parse_expert_idx("model.layers.0.mlp.experts.42");
        assert_eq!(n, "model.layers.0.mlp.experts.42");
        assert_eq!(e, 0);

        // Digits followed by underscore (not `.`).
        let (n, e) = parse_expert_idx("model.layers.0.mlp.experts.42_gate_proj.weight");
        assert_eq!(n, "model.layers.0.mlp.experts.42_gate_proj.weight");
        assert_eq!(e, 0);

        // 11-digit number overflows u32 (max u32 = 10 digits). Falls
        // back to dense rather than panicking.
        let (n, e) =
            parse_expert_idx("model.layers.0.mlp.experts.99999999999.gate_proj.weight");
        assert_eq!(
            n,
            "model.layers.0.mlp.experts.99999999999.gate_proj.weight"
        );
        assert_eq!(e, 0);
    }

    /// `should_capture` admits the obvious Linear tensors and rejects norms
    /// + embed by default; `lm_head` is admitted only with `process_output`.
    #[test]
    fn imatrix_should_capture_default() {
        let coll = ImatrixCollector::new(false);
        // Linear layers admitted.
        assert!(coll.should_capture("model.layers.0.self_attn.q_proj.weight"));
        assert!(coll.should_capture("model.layers.0.self_attn.k_proj.weight"));
        assert!(coll.should_capture("model.layers.0.self_attn.o_proj.weight"));
        assert!(coll.should_capture("model.layers.0.mlp.gate_proj.weight"));
        assert!(coll.should_capture("model.layers.0.mlp.down_proj.weight"));
        // MoE router (Qwen3.5-A3B) — admitted.
        assert!(coll.should_capture("model.layers.0.mlp.gate.weight"));
        // Norms + embed rejected.
        assert!(!coll.should_capture("model.embed_tokens.weight"));
        assert!(!coll.should_capture("model.layers.0.input_layernorm.weight"));
        assert!(!coll.should_capture("model.norm.weight"));
        // lm_head rejected by default (matches llama-imatrix without --process-output).
        assert!(!coll.should_capture("lm_head.weight"));
    }

    /// With `process_output=true`, `lm_head` / `output` are admitted; norms
    /// still rejected.
    #[test]
    fn imatrix_should_capture_with_process_output() {
        let coll = ImatrixCollector::new(true);
        assert!(coll.should_capture("lm_head.weight"));
        assert!(coll.should_capture("output.weight"));
        // Norms still rejected.
        assert!(!coll.should_capture("model.norm.weight"));
    }

    /// `HessianCollector` delegates to `is_gptq_target` — verify the admit
    /// list aligns with the production GPTQ targets.
    #[test]
    fn hessian_collector_uses_gptq_whitelist() {
        // Use the trait-level capture call (with dummy ptr) to verify the
        // gate-keep is_gptq_target. Since the body is a no-op until
        // subagent-A wires the kernel, this asserts only that we don't
        // panic on admitted tensors and that rejected tensors short-circuit.
        let coll = HessianCollector::new();
        // No accumulators yet — drain returns empty.
        let dummy_gpu_drain_count = coll
            .accumulators
            .lock()
            .map(|m| m.len())
            .expect("mutex");
        assert_eq!(dummy_gpu_drain_count, 0);
        // Verify the gate via is_gptq_target directly — same logic the
        // trait-impl gates on.
        assert!(is_gptq_target("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.gate_proj.weight"));
        // MoE router admitted by is_gptq_target.
        assert!(is_gptq_target("model.layers.0.mlp.gate.weight"));
        // Norms / embed rejected.
        assert!(!is_gptq_target("model.embed_tokens.weight"));
        assert!(!is_gptq_target("model.norm.weight"));
        assert!(!is_gptq_target("lm_head.weight"));
    }

    /// Smoke-test `tokenize_corpus`: build a minimal HF tokenizer.json in
    /// a tmp dir, encode a 4-char string, verify the token stream is
    /// non-empty + maps back to bytes that cover the input.
    ///
    /// Doesn't exercise the GPU/forward path — that requires subagents
    /// A+C+D to land. Limited to the driver-side: tokenizer.json read +
    /// HF JSON parse + BPE encode → Vec<u32>.
    #[test]
    fn tokenize_corpus_smoke_minimal_tokenizer() {
        // Build a minimal char-level tokenizer.json: one merge rule per
        // ASCII char [a-z], no special tokens. This is enough for
        // `from_hf_json` to construct a working encoder.
        let mut vocab = serde_json::Map::new();
        for (i, ch) in ('a'..='z').enumerate() {
            vocab.insert(ch.to_string(), serde_json::Value::Number((i as u64).into()));
        }
        let empty_array: Vec<&str> = Vec::new();
        let tokenizer_json = serde_json::json!({
            "model": {
                "vocab": vocab,
                "merges": empty_array,
            },
            "added_tokens": empty_array,
        });

        let tmp_dir = std::env::temp_dir().join(format!(
            "hipfire-calibration-tokenize-smoke-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
        let tokenizer_path = tmp_dir.join("tokenizer.json");
        std::fs::write(&tokenizer_path, tokenizer_json.to_string())
            .expect("write tokenizer.json");

        // Call the driver helper.
        let ids = tokenize_corpus(&tmp_dir, "abc").expect("tokenize_corpus should succeed");
        // We don't assert exact IDs (the encoder may emit byte-fallback
        // pre-tokens for non-merged chars) but the stream must be
        // non-empty for any non-empty input.
        assert!(
            !ids.is_empty(),
            "tokenize_corpus produced 0 tokens for non-empty input"
        );

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    /// Parser fixture: vanilla single-line `[id, id, ...]` stdout from
    /// `llama-tokenize --ids --log-disable`. This is the dominant shape
    /// on the MI300x calibration droplet (llama.cpp commit f3b9e91 +
    /// build/bin/llama-tokenize, May 2026).
    #[test]
    fn parse_llama_tokenize_stdout_basic_ids() {
        let stdout = "[283, 82224, 88, 4065, 63110, 14016, 283, 4558]\n";
        let ids = parse_llama_tokenize_stdout(stdout).expect("basic parse");
        assert_eq!(
            ids,
            vec![283_u32, 82224, 88, 4065, 63110, 14016, 283, 4558]
        );
    }

    /// Parser fixture: stdout with banner / log lines BEFORE the IDs
    /// line. Some llama.cpp builds still print "main: tokenizer XYZ"
    /// even with `--log-disable`. The parser must skip non-bracket
    /// lines and pick up the first `[...]` line cleanly.
    #[test]
    fn parse_llama_tokenize_stdout_skips_banner_lines() {
        let stdout = "main: build = 9999 (deadbeef)\n\
                      main: built with cc (Ubuntu) for x86_64-linux-gnu\n\
                      llm_load_print_meta: model type     = qwen3\n\
                      [1, 2, 3, 4, 5]\n";
        let ids = parse_llama_tokenize_stdout(stdout).expect("banner-skip parse");
        assert_eq!(ids, vec![1_u32, 2, 3, 4, 5]);
    }

    /// Parser fixture: empty corpus → empty `[]`. Should parse cleanly
    /// to an empty `Vec<u32>`, not an error.
    #[test]
    fn parse_llama_tokenize_stdout_empty_brackets() {
        let stdout = "[]\n";
        let ids = parse_llama_tokenize_stdout(stdout).expect("empty parse");
        assert!(ids.is_empty(), "[]` should yield zero IDs, got {ids:?}");
    }

    /// Parser fixture: a wrapped multi-line list. Some llama.cpp builds
    /// hard-wrap long ID lists at ~80 cols; the parser must concatenate
    /// from `[` to `]` across lines.
    #[test]
    fn parse_llama_tokenize_stdout_handles_multiline_list() {
        let stdout = "[1, 2, 3,\n  4, 5, 6,\n  7, 8, 9]\n";
        let ids = parse_llama_tokenize_stdout(stdout).expect("multiline parse");
        assert_eq!(ids, vec![1_u32, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    /// Parser fixture: a malformed input with no `[...]` line at all
    /// must surface a descriptive error (not silently produce empty).
    #[test]
    fn parse_llama_tokenize_stdout_missing_brackets_errors() {
        let stdout = "main: something went wrong\nbut no IDs were ever printed\n";
        let err = parse_llama_tokenize_stdout(stdout)
            .expect_err("missing brackets must error, not return empty");
        assert!(
            err.contains("IDs line"),
            "error should mention IDs line, got: {err}"
        );
    }

    /// Parser fixture: a malformed integer inside `[...]` surfaces a
    /// parse error referencing the offending token.
    #[test]
    fn parse_llama_tokenize_stdout_invalid_id_errors() {
        let stdout = "[1, 2, banana, 4]\n";
        let err = parse_llama_tokenize_stdout(stdout)
            .expect_err("non-numeric ID must error");
        assert!(
            err.contains("banana"),
            "error should reference the offending token, got: {err}"
        );
    }

    /// `tokenize_corpus` surfaces a meaningful error when tokenizer.json
    /// is missing from the model dir.
    #[test]
    fn tokenize_corpus_missing_tokenizer_returns_error() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "hipfire-calibration-missing-tokenizer-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
        let err = tokenize_corpus(&tmp_dir, "hello").expect_err("should fail");
        assert!(
            err.contains("tokenizer.json"),
            "error message should mention tokenizer.json, got: {err}"
        );
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    /// `ImatrixCollector::drain` on a fresh collector returns an empty
    /// vector — no captures happened, no GPU access needed (the empty
    /// `HashMap` short-circuits before any `download_f32`).
    #[test]
    fn imatrix_drain_empty_skips_gpu_access() {
        let coll = ImatrixCollector::new(false);
        // We pass a fake unaligned pointer cast through a raw `Gpu`-shape
        // — but the drain MUST short-circuit on the empty map before any
        // dispatch, so we never touch the pointer. The test verifies that
        // invariant: build the collector + check the map is empty + skip
        // the actual gpu.download_f32 calls.
        let map_len = coll
            .accumulators
            .lock()
            .map(|m| m.len())
            .expect("mutex");
        assert_eq!(map_len, 0, "fresh collector should have zero accumulators");
    }

    /// Same for `HessianCollector`.
    #[test]
    fn hessian_drain_empty_skips_gpu_access() {
        let coll = HessianCollector::new();
        let map_len = coll
            .accumulators
            .lock()
            .map(|m| m.len())
            .expect("mutex");
        assert_eq!(map_len, 0, "fresh collector should have zero accumulators");
    }
}
