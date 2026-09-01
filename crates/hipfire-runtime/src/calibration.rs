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

pub mod boundary;
pub mod contracts;
pub mod expert_capture;
pub mod layer_stream;
pub mod residual_probe;
pub mod schedule;
pub mod source;
pub mod stream;

/// Rows buffered per tensor before flushing the outer-product. A single
/// `calib_hessian_outer_f32` over `[FLUSH_BATCH, K]` is ~FLUSH_BATCH× more
/// efficient than per-token (N=1) launches (the tiled GEMM is built for N≥16),
/// so this is the dominant calibration-throughput lever.
const FLUSH_BATCH: usize = 256;

/// Calibration-only HFQM quant_type for compact Hessians:
/// exact F32 diagonal followed by BF16 lower strict triangle.
const QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32: u8 = 130;

/// Sequence length calibration splits to when `HIPFIRE_CALIB_SEQ_LEN` is unset.
///
/// 2048 because that is what KLD references are built at — calibrating under a
/// longer context than is served is a mismatch, and the attention cost of the
/// unbounded alternative is O(n²) in the token budget.
pub const DEFAULT_CALIB_SEQ_LEN: usize = 2048;

/// Split a calibration token stream into independent sequences, per
/// `HIPFIRE_CALIB_SEQ_LEN`. This is the chunking POLICY, owned by the shared
/// seam so no arch can accidentally calibrate under one unbounded context.
///
/// Every arch's resident calibration used to consume the whole token budget as
/// a single growing context — zaya as one long sequence, and nemotron/minimax/
/// lfm2moe as a per-token decode loop with `pos` running to the end and state
/// reset only once at the start. Both shapes make attention cost accumulate as
/// O(n²) in the budget, and both calibrate under a context far longer than the
/// one being served (KLD references are built at `n_ctx=2048`). Splitting fixes
/// the cost and the mismatch at once: a Hessian is a sum of per-row outer
/// products, so it does not care which context a row came from. Measured on
/// zaya1-8b at 32768 tokens: 10746 s → 1065 s, quality unchanged
/// (`docs/quant-formats/moe-expert-hessians.md`, Q6).
///
/// **Default 2048 as of 2026-09-01.** Unset used to yield a single unbounded
/// sequence, which meant the historical behaviour was the O(n²) one and every
/// caller had to know to set the variable to avoid it. Two reasons the default
/// is the right place to fix that rather than the call sites:
///
/// * the cost is quadratic in the budget, so the larger the calibration run the
///   worse the unset default behaves — precisely backwards;
/// * KLD references are built at `n_ctx=2048`, so 2048 also makes calibration
///   match the context actually served, instead of calibrating under a longer
///   one.
///
/// Set `HIPFIRE_CALIB_SEQ_LEN` to override; a value below 2 restores the single
/// unbounded sequence, which is now the thing you have to ask for.
///
/// The ARCH still owns what "independent" means for it — resetting KV, SSM or
/// recurrent state between sequences, and restarting positions at 0.
/// Takes the arch's natural sequences — one flat stream for the corpus-driven
/// arches, or a sample set for those that already have one — and re-splits each
/// to at most `HIPFIRE_CALIB_SEQ_LEN`. Unset leaves them untouched.
pub fn calib_sequences<'a>(inputs: &[&'a [u32]]) -> Vec<&'a [u32]> {
    let seq_len = hipfire_env::CALIB_SEQ_LEN
        .parse::<usize>()
        .filter(|n| *n >= 2)
        .unwrap_or(DEFAULT_CALIB_SEQ_LEN);
    let n_in: usize = inputs.iter().map(|s| s.len()).sum();
    // A 1-token sequence has no next-token target and contributes only a
    // degenerate row, so drop a trailing remainder of one.
    let seqs: Vec<&'a [u32]> = inputs
        .iter()
        .flat_map(|s| s.chunks(seq_len).filter(|c| c.len() >= 2))
        .collect();
    if seqs.len() > inputs.len() {
        tracing::info!(
            "{n_in} tokens as {} independent sequences of <= {seq_len} \
             (was {})",
            seqs.len(),
            inputs.len()
        );
    }
    seqs
}

/// Refuse to calibrate on the frozen evaluation slice.
///
/// `benchmarks/quality-baselines/slice/` is "the frozen prompt bytes used by
/// every quant-quality eval" (its README); `benchmarks/calib/` holds the
/// calibration corpora. Pointing calibration at the eval slice trains on the
/// test set, and the damage is invisible in the numbers — it shows up as a
/// large fake improvement plus an inverted-U budget curve, because a
/// calibration that has seen exactly the evaluated tokens scores best.
/// That cost this project a retracted "-13.6% from more calibration tokens"
/// result (`docs/quant-formats/moe-expert-hessians.md`, Q7).
///
/// Set `HIPFIRE_CALIB_ALLOW_EVAL_CORPUS=1` to proceed anyway — the only honest
/// use is deliberately measuring the size of the train-on-test effect.
pub fn reject_eval_corpus(corpus: &str) -> Result<(), String> {
    let is_eval = corpus.contains("quality-baselines");
    if !is_eval || hipfire_env::CALIB_ALLOW_EVAL_CORPUS.get().is_some() {
        if is_eval {
            tracing::warn!(
                "calibrating on the EVAL slice {corpus} \
                 (HIPFIRE_CALIB_ALLOW_EVAL_CORPUS set) — results are train-on-test"
            );
        }
        return Ok(());
    }
    Err(format!(
        "calib: {corpus} is the frozen EVALUATION slice, not a calibration corpus. \
         Calibrating on it trains on the test set and silently inflates every \
         quality number. Use a corpus from `benchmarks/calib/` (e.g. \
         calib-multi-8m.txt), or set HIPFIRE_CALIB_ALLOW_EVAL_CORPUS=1 if you are \
         deliberately measuring the train-on-test effect."
    ))
}

/// The arch's imatrix-only substring list, or the `HIPFIRE_CALIB_IMATRIX_ONLY`
/// override. Every arch blocks routed experts from full `[K,K]` Hessians by
/// default (`vec![".experts."]`), because storing one per expert was assumed
/// prohibitive. It is not, for `down_proj`: its K is the MoE INTERMEDIATE
/// width, which shrinks as experts multiply, so the total stays ~2.6 GB on both
/// zaya1-8b (640 x K=2048) and qwen3.6-35b-a3b (9778 x K=512). It is `gate_up`
/// that is expensive at K=hidden — and that one pools
/// (`docs/quant-formats/moe-expert-hessians.md`).
///
/// Set to a comma-separated substring list to re-scope, or to an empty string
/// to capture a full Hessian for every registered tensor. Each entry is matched
/// with `contains`, so `.gate_up_proj` blocks expert gate/up while leaving
/// expert `down_proj` (and every dense tensor) captured.
fn effective_imatrix_only(arch_default: Vec<String>) -> Vec<String> {
    let Some(raw) = hipfire_env::CALIB_IMATRIX_ONLY.get() else {
        return arch_default;
    };
    let list: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    tracing::info!("imatrix-only override active: {list:?} (was {arch_default:?})");
    list
}

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
    /// Canonical checkpoint outputs that share this exact input activation.
    /// Gate/up and Q/K/V aliases therefore reuse one GPU accumulator while the
    /// writer still emits one record per source tensor.
    output_names: Vec<String>,
    diag: GpuTensor, // [K]   Σx²  (imatrix)
    /// `[4, K]` activation SHAPE statistics: Σx, Σ|x|, Σx⁴, max|x|.
    /// `diag` measures channel ENERGY, which is what AWQ-style weight scaling
    /// wants and which says nothing about tail shape — two channels with equal
    /// Σx² can be flat or a single spike, and 4-bit ACTIVATIONS treat those
    /// oppositely. These four are what an A4 decision can actually be made
    /// against. `None` only on the no-buffer weighted path below.
    stats: Option<GpuTensor>,
    /// `[2, K/256]` per-(token, group) crest: row 0 sums it over tokens, row 1
    /// maxes. This is what an Opus scale ACTUALLY sees — `stats` reduces over
    /// tokens and so reports a corpus-wide per-channel crest, which on
    /// Qwen3.8-27B read 226 on down_proj while that model serves fine at W4A8.
    /// `None` when K is not a multiple of the 256 group, or on the no-buffer
    /// weighted path.
    gcrest: Option<GpuTensor>,
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
    buf_capacity: usize,
    k: usize,
    n_tokens: u64,
}

/// Env-gated capture trace (`HIPFIRE_CALIB_TRACE=1`), for diagnosing
/// nondeterministic calibration artifacts.
///
/// Prints one line per reduction with a hash of the STAGED ROWS that reduction
/// is about to consume. Diffing two runs' traces finds the first flush whose
/// input diverged, which separates "the forward fed us different activations"
/// from "the reduction turned identical input into a different accumulator".
/// Off by default and costs a device download per flush, so it is a debugging
/// tool, not a monitoring one.
pub(crate) fn calib_trace_enabled() -> bool {
    std::env::var("HIPFIRE_CALIB_TRACE").as_deref() == Ok("1")
}

/// FNV-1a over the raw f32 bits, so the hash is exact rather than formatted.
fn calib_trace_hash(values: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

impl Acc {
    /// Reduce the staged rows into the accumulators (one batched launch each),
    /// then reset the buffer. No-op when empty. Imatrix-only tensors (`h` is
    /// `None`) skip the [K,K] outer-product — this is how MoE routed experts
    /// are captured: a full per-expert Hessian (256 experts × ~48 layers ×
    /// [K,K]) is ~196 GB and does not fit, but the imatrix (Σx², a K-vector)
    /// is ~100 MB and is the importance signal AWQ-style quant needs.
    fn flush_result(&mut self, gpu: &mut Gpu) -> Result<(), contracts::CalibError> {
        if self.buf_rows == 0 {
            return Ok(());
        }
        if calib_trace_enabled() {
            // Hash exactly the rows this reduction will read. A row the gather
            // skipped still sits inside `buf_rows` and still contributes, so it
            // is included here by construction.
            let staged = gpu
                .download_f32(&self.buf)
                .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
            let n = self.buf_rows * self.k;
            let nonfinite = staged[..n].iter().filter(|v| !v.is_finite()).count();
            eprintln!(
                "[calib-trace] flush names={:?} rows={} k={} staged_hash={:#018x} nonfinite={}",
                self.output_names,
                self.buf_rows,
                self.k,
                calib_trace_hash(&staged[..n]),
                nonfinite,
            );
        }
        if let Some(stats) = self.stats.as_ref() {
            gpu.calib_actstats_reduce_f32(&self.buf, stats, self.buf_rows, self.k)
                .map_err(|e| contracts::CalibError::Runtime(e.to_string()))?;
        }
        if let Some(gc) = self.gcrest.as_ref() {
            gpu.calib_group_crest_reduce_f32(&self.buf, gc, self.buf_rows, self.k, 256)
                .map_err(|e| contracts::CalibError::Runtime(e.to_string()))?;
        }
        gpu.calib_sumsq_reduce_f32(&self.buf, &self.diag, self.buf_rows, self.k)
            .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
        if let Some(h) = &self.h {
            gpu.calib_hessian_outer_f32(&self.buf, h, self.buf_rows, self.k)
                .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
        }
        // Audit: accumulate the same rows in f64 on the CPU (no GPU f64 path).
        if let Some(h_f64) = &mut self.h_f64 {
            let k = self.k;
            let rows = gpu
                .download_f32(&self.buf)
                .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
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
        Ok(())
    }

    fn flush(&mut self, gpu: &mut Gpu) {
        self.flush_result(gpu)
            .expect("calibration activation reduction");
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

    fn allocate_acc(
        &self,
        gpu: &mut Gpu,
        output_names: Vec<String>,
        policy: contracts::CapturePolicy,
        k: usize,
        buf_capacity: usize,
    ) -> Result<Acc, contracts::CalibError> {
        let diag = gpu
            .zeros(&[k], DType::F32)
            .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
        let h = if policy == contracts::CapturePolicy::HessianAndImatrix {
            match gpu.zeros(&[k, k], DType::F32) {
                Ok(h) => Some(h),
                Err(error) => {
                    let _ = gpu.free_tensor(diag);
                    return Err(contracts::CalibError::Runtime(error.to_string()));
                }
            }
        } else {
            None
        };
        let buf = match gpu.zeros(&[buf_capacity, k], DType::F32) {
            Ok(buf) => buf,
            Err(error) => {
                let _ = gpu.free_tensor(diag);
                if let Some(h) = h {
                    let _ = gpu.free_tensor(h);
                }
                return Err(contracts::CalibError::Runtime(error.to_string()));
            }
        };
        let h_f64 = if self.f64_audit && h.is_some() {
            Some(vec![0.0f64; k * k])
        } else {
            None
        };
        // HIPFIRE_CALIB_NO_ACTSTATS=1 skips the shape statistics. They are
        // [4, K] against the Hessian's [K, K], i.e. 0.08% at K=5120 — but on
        // IMATRIX-ONLY tensors (MoE routed experts, where a per-expert Hessian
        // does not fit at all) there is no [K,K] term to hide behind and this is
        // 4x the imatrix. Calibration memory is the standing risk on this path,
        // so the escape hatch exists.
        let want_stats = std::env::var("HIPFIRE_CALIB_NO_ACTSTATS").is_err();
        let stats = match if want_stats {
            gpu.zeros(&[4, k], DType::F32).map(Some)
        } else {
            Ok(None)
        } {
            Ok(t) => t,
            Err(error) => {
                let _ = gpu.free_tensor(diag);
                if let Some(h) = h {
                    let _ = gpu.free_tensor(h);
                }
                let _ = gpu.free_tensor(buf);
                return Err(contracts::CalibError::Runtime(error.to_string()));
            }
        };
        // 256 is the Opus group; a tensor whose K is not a multiple of it has
        // no group-crest to report rather than a differently-shaped one.
        let gcrest = if want_stats && k % 256 == 0 {
            match gpu.zeros(&[2, k / 256], DType::F32) {
                Ok(t) => Some(t),
                Err(error) => {
                    let _ = gpu.free_tensor(diag);
                    if let Some(h) = h {
                        let _ = gpu.free_tensor(h);
                    }
                    let _ = gpu.free_tensor(buf);
                    if let Some(t) = stats {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(contracts::CalibError::Runtime(error.to_string()));
                }
            }
        } else {
            None
        };
        Ok(Acc {
            output_names,
            diag,
            stats,
            gcrest,
            h,
            h_f64,
            buf,
            buf_rows: 0,
            buf_capacity,
            k,
            n_tokens: 0,
        })
    }

    /// Number of distinct tensors captured so far.
    pub fn len(&self) -> usize {
        self.accs
            .lock()
            .unwrap()
            .values()
            .map(|acc| acc.output_names.len())
            .sum()
    }

    /// Number of physical GPU accumulators. This may be smaller than [`len`]
    /// when multiple checkpoint outputs share an input activation.
    pub fn accumulator_len(&self) -> usize {
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
        let mut descriptors = accs
            .values()
            .flat_map(|acc| {
                acc.output_names.iter().map(|name| CalibTensorDesc {
                    name: name.clone(),
                    has_hessian: acc.h.is_some(),
                    k: acc.k,
                    n_tokens: acc.n_tokens,
                })
            })
            .collect::<Vec<_>>();
        descriptors.sort_by(|left, right| left.name.cmp(&right.name));
        descriptors
    }

    /// Release all GPU accumulators owned by this collector. Grouped
    /// calibration runs call this after streaming a part file so the next group
    /// can reuse the memory instead of waiting for process teardown.
    pub fn free_gpu(&self, gpu: &mut Gpu) {
        let mut accs = self.accs.lock().unwrap();
        for (_, acc) in accs.drain() {
            let _ = gpu.free_tensor(acc.diag);
            if let Some(stats) = acc.stats {
                let _ = gpu.free_tensor(stats);
            }
            if let Some(gc) = acc.gcrest {
                let _ = gpu.free_tensor(gc);
            }
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
                    output_names: vec![tensor_name.to_string()],
                    diag,
                    // Weighted path stages no rows, so the shape statistics have
                    // nothing to reduce over; the imatrix is accumulated directly.
                    stats: None,
                    gcrest: None,
                    h,
                    h_f64: None,
                    buf,
                    buf_rows: 0,
                    buf_capacity: 1,
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

    /// Capture through a stable logical descriptor rather than a weight-buffer
    /// address. Multiple `output_names` share one physical accumulator and are
    /// expanded only when descriptors and HFQ records are emitted.
    pub fn capture_by_id(
        &self,
        gpu: &mut Gpu,
        registry: &contracts::CaptureRegistry,
        capture_id: contracts::CaptureId,
        input: &GpuTensor,
        n: usize,
        k: usize,
    ) -> Result<(), contracts::CalibError> {
        let descriptor = registry.get(capture_id).ok_or_else(|| {
            contracts::CalibError::InvalidCapture(format!(
                "unknown logical capture id {}",
                capture_id.0
            ))
        })?;
        if descriptor.policy == contracts::CapturePolicy::Skip {
            return Ok(());
        }
        if n == 0 || k == 0 || descriptor.input_width != k {
            return Err(contracts::CalibError::InvalidCapture(format!(
                "capture {} received shape [{n}, {k}], expected non-zero rows with width {}",
                capture_id.0, descriptor.input_width
            )));
        }
        // A wider-than-`k` stride means the caller handed us a scratch buffer
        // sized for its MAXIMUM row count while `n` is this call's ACTUAL row
        // count. For `n > 1` that is silent accumulator corruption: rows 1..n-1
        // are then read at `row * (numel / n)`, i.e. stale data from a previous,
        // wider slice. Reject it rather than striding over it.
        //
        // For `n == 1` the row offset `row * row_stride` is always 0, so only
        // the first `k` elements are ever touched and a wider buffer is
        // harmless. Several arches (gemma3/gemma4/cohere2) legitimately pass a
        // shared scratch tensor at `n = 1`; requiring an exact stride there
        // would reject correct callers, so only the lower bound is enforced.
        let row_stride = input.numel() / n;
        let stride_ok = if n == 1 {
            row_stride >= k
        } else {
            row_stride == k
        };
        if !stride_ok {
            return Err(contracts::CalibError::InvalidCapture(format!(
                "capture {} input row stride {row_stride} != width {k} \
                 (input has {} elements for {n} rows; narrow it to n*k first)",
                capture_id.0,
                input.numel()
            )));
        }

        let key = format!("@capture:{:016x}", capture_id.0);
        let mut accs = self.accs.lock().unwrap();
        if !accs.contains_key(&key) {
            accs.insert(
                key.clone(),
                self.allocate_acc(
                    gpu,
                    descriptor.output_names.clone(),
                    descriptor.policy,
                    k,
                    FLUSH_BATCH,
                )?,
            );
        }
        let acc = accs.get_mut(&key).unwrap();
        if acc.k != k || acc.output_names != descriptor.output_names {
            return Err(contracts::CalibError::InvalidCapture(format!(
                "capture {} descriptor changed after accumulation started",
                capture_id.0
            )));
        }
        for row in 0..n {
            if acc.buf_rows == acc.buf_capacity {
                acc.flush(gpu);
            }
            gpu.memcpy_dtod_at_auto(
                &acc.buf.buf,
                acc.buf_rows * k * 4,
                &input.buf,
                row * row_stride * 4,
                k * 4,
            )
            .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
            acc.buf_rows += 1;
        }
        acc.n_tokens += n as u64;
        Ok(())
    }

    /// Execute a quota-capped grouped-expert capture plan against the existing
    /// grouped-MoE permutation. This stages only admitted routes; teacher
    /// execution continues to use the unmodified routing tensors.
    pub fn capture_grouped_plan(
        &self,
        gpu: &mut Gpu,
        registry: &contracts::CaptureRegistry,
        source: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        plan: &expert_capture::GroupedExpertCapturePlan,
    ) -> Result<(), contracts::CalibError> {
        if source.dtype != DType::F32 || sorted_slot_index.dtype != DType::Raw {
            return Err(contracts::CalibError::InvalidCapture(
                "grouped expert capture requires an F32 source and Raw i32 sorted indices".into(),
            ));
        }
        let projection_role = match plan.role {
            contracts::ExpertCaptureRole::GateUpInput => contracts::ProjectionRole::GateUpInput,
            contracts::ExpertCaptureRole::DownInput => contracts::ProjectionRole::DownInput,
        };
        let mut accs = self.accs.lock().unwrap();
        for action in &plan.actions {
            if action.layer != plan.layer || action.role != plan.role || action.rows == 0 {
                return Err(contracts::CalibError::InvalidCapture(
                    "grouped expert capture action does not match its plan".into(),
                ));
            }
            let capture_id =
                contracts::CaptureId::new(plan.layer, projection_role, Some(action.expert));
            let descriptor = registry.get(capture_id).ok_or_else(|| {
                contracts::CalibError::InvalidCapture(format!(
                    "missing grouped expert capture descriptor {}",
                    capture_id.0
                ))
            })?;
            if descriptor.layer != plan.layer
                || descriptor.role != projection_role
                || descriptor.expert != Some(action.expert)
            {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} does not match the routed expert capture identity",
                    capture_id.0
                )));
            }
            if descriptor.policy == contracts::CapturePolicy::Skip {
                continue;
            }
            if descriptor.policy != contracts::CapturePolicy::ImatrixOnly {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} is not the expected imatrix-only routed expert capture",
                    capture_id.0
                )));
            }
            let quota = descriptor.expert_quota.ok_or_else(|| {
                contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} has no expert quota",
                    capture_id.0
                ))
            })?;
            if quota.tile_rows != action.tile_rows
                || action.destination_row + action.rows > action.tile_rows
            {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} tile geometry does not match capture action",
                    capture_id.0
                )));
            }
            let k = descriptor.input_width;
            if source.numel() % k != 0 {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} source size {} is not row-major width {k}",
                    capture_id.0,
                    source.numel()
                )));
            }
            let key = format!("@capture:{:016x}", capture_id.0);
            if !accs.contains_key(&key) {
                accs.insert(
                    key.clone(),
                    self.allocate_acc(
                        gpu,
                        descriptor.output_names.clone(),
                        descriptor.policy,
                        k,
                        action.tile_rows,
                    )?,
                );
            }
            let acc = accs.get_mut(&key).expect("accumulator inserted above");
            if acc.k != k
                || acc.output_names != descriptor.output_names
                || acc.buf_capacity != action.tile_rows
                || acc.buf_rows != action.destination_row
            {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} capture staging state diverged from its plan",
                    capture_id.0
                )));
            }
            if calib_trace_enabled() {
                // The hypothesis under test: `calib_gather_rows_f32` returns
                // WITHOUT writing any row whose slot index is negative, and the
                // staging tile is zeroed only once at allocation and reused
                // across flushes. A negative slot inside `0..rows` would
                // therefore leave the PREVIOUS tile's values in a row that this
                // flush still reduces. If this never prints, that path is not
                // the cause and the divergence is upstream of the gather.
                let bytes = gpu
                    .download_raw(sorted_slot_index, sorted_slot_index.byte_size())
                    .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
                let slots: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let lo = action.sorted_start.min(slots.len());
                let hi = (action.sorted_start + action.rows).min(slots.len());
                let negatives = slots[lo..hi].iter().filter(|s| **s < 0).count();
                if negatives > 0 {
                    eprintln!(
                        "[calib-trace] NEGATIVE SLOTS names={:?} count={negatives}/{} \
                         sorted_start={lo} dst_row={}",
                        descriptor.output_names, action.rows, action.destination_row,
                    );
                }
            }
            gpu.calib_gather_rows_f32(
                source,
                sorted_slot_index,
                &acc.buf,
                action.sorted_start,
                action.destination_row,
                action.rows,
                k,
                action.source_row_div,
            )
            .map_err(|error| contracts::CalibError::Runtime(error.to_string()))?;
            acc.buf_rows += action.rows;
            acc.n_tokens += action.rows as u64;
            if action.flush_full_tile {
                if acc.buf_rows != acc.buf_capacity {
                    return Err(contracts::CalibError::InvalidCapture(format!(
                        "descriptor {} requested a partial normal-path flush",
                        capture_id.0
                    )));
                }
                acc.flush_result(gpu)?;
            } else if acc.buf_rows == acc.buf_capacity {
                return Err(contracts::CalibError::InvalidCapture(format!(
                    "descriptor {} filled a tile without scheduling its reduction",
                    capture_id.0
                )));
            }
        }
        Ok(())
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
        let mut outputs = accs
            .iter()
            .flat_map(|(key, acc)| {
                acc.output_names
                    .iter()
                    .cloned()
                    .map(|output_name| (output_name, key.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        outputs.sort_by(|left, right| left.0.cmp(&right.0));

        // Build the index entries (payload sizes from `k`) + a parallel plan of
        // how to produce each payload, in the SAME order.
        enum Plan {
            ActStats { key: String },
            GroupCrest { key: String },
            Hessian { key: String, output_name: String },
            Imatrix { key: String },
            Extra(usize),
        }
        let mut entries: Vec<HfqStreamEntry> = Vec::new();
        let mut plan: Vec<Plan> = Vec::new();
        for (output_name, key) in &outputs {
            let acc = &accs[key];
            if acc.h.is_some() {
                let (quant_type, data_len) = match hessian_storage {
                    HessianStorage::DenseF32 => (2, (acc.k * acc.k * 4) as u64),
                    HessianStorage::Bf16TrilDiagF32 => (
                        QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32,
                        compact_hessian_bytes(acc.k),
                    ),
                };
                entries.push(HfqStreamEntry {
                    name: format!("{output_name}.hessian"),
                    quant_type,
                    shape: vec![acc.k as u32, acc.k as u32],
                    group_size: 0,
                    data_len,
                });
                plan.push(Plan::Hessian {
                    key: key.clone(),
                    output_name: output_name.clone(),
                });
            }
            entries.push(HfqStreamEntry {
                name: format!("{output_name}.imatrix"),
                quant_type: 2,
                shape: vec![acc.k as u32],
                group_size: 0,
                data_len: (acc.k * 4) as u64,
            });
            plan.push(Plan::Imatrix { key: key.clone() });
            if acc.stats.is_some() {
                entries.push(HfqStreamEntry {
                    name: format!("{output_name}.actstats"),
                    quant_type: 2,
                    shape: vec![4, acc.k as u32],
                    group_size: 0,
                    data_len: (4 * acc.k * 4) as u64,
                });
                plan.push(Plan::ActStats { key: key.clone() });
            }
            if acc.gcrest.is_some() {
                let ngc = acc.k / 256;
                entries.push(HfqStreamEntry {
                    name: format!("{output_name}.groupcrest"),
                    quant_type: 2,
                    shape: vec![2, ngc as u32],
                    group_size: 256,
                    data_len: (2 * ngc * 4) as u64,
                });
                plan.push(Plan::GroupCrest { key: key.clone() });
            }
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
                Plan::Hessian { key, output_name } => {
                    let acc = &accs[key];
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
                            *audit_worst.borrow_mut() = output_name.clone();
                        }
                    }
                    match hessian_storage {
                        HessianStorage::DenseF32 => write_f32_scaled(w, &h, inv),
                        HessianStorage::Bf16TrilDiagF32 => {
                            write_hessian_bf16_tril_diag_f32(w, &h, &diag, acc.k, inv)
                        }
                    }
                }
                Plan::GroupCrest { key } => {
                    let acc = &accs[key];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let gc = gpu
                        .download_f32(acc.gcrest.as_ref().expect("groupcrest planned"))
                        .map_err(io_err)?;
                    // Row 0 is a SUM over tokens -> mean; row 1 is a MAX and is
                    // written raw.
                    let ngc = acc.k / 256;
                    let mut out = gc.clone();
                    for v in out[..ngc].iter_mut() {
                        *v *= inv;
                    }
                    write_f32_scaled(w, &out, 1.0)
                }
                Plan::ActStats { key } => {
                    let acc = &accs[key];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let stats = gpu
                        .download_f32(acc.stats.as_ref().expect("actstats planned"))
                        .map_err(io_err)?;
                    // Rows 0-2 are SUMS and are normalised per token so the
                    // record does not depend on corpus size. Row 3 is a MAX and
                    // must be written raw — scaling it would turn a range into a
                    // meaningless per-token average.
                    let mut out = stats.clone();
                    for v in out[..3 * acc.k].iter_mut() {
                        *v *= inv;
                    }
                    write_f32_scaled(w, &out, 1.0)
                }
                Plan::Imatrix { key } => {
                    let acc = &accs[key];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let diag = gpu.download_f32(&acc.diag).map_err(io_err)?;
                    write_f32_scaled(w, &diag, inv)
                }
                Plan::Extra(j) => w.write_all(&extra[*j].data),
            }
        })?;

        if audit_n.get() > 0 {
            tracing::debug!(
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
        let imatrix_only = effective_imatrix_only(imatrix_only.clone());
        let collector = std::sync::Arc::new(if imatrix_only.is_empty() {
            CalibCollector::new()
        } else {
            CalibCollector::with_imatrix_only(imatrix_only)
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

    let metadata =
        build_calibration_metadata(&all_descriptors, Some(group), static_meta, &extra_meta)?;
    combine_calib_parts(output, arch_id, &metadata.json, &part_paths, &extra_tensors)
        .map_err(|e| format!("calib combine {}: {e}", output.display()))?;
    for part in part_paths {
        let _ = std::fs::remove_file(part);
    }

    Ok(CalibSummary {
        n_hessian: metadata.n_hessian,
        n_imatrix: metadata.n_imatrix,
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
/// materializing them all at once. Part payloads use the index-only + `pread`
/// path on Unix: mmaping every layer part at once makes tens of GiB of
/// calibration pages participate in AMD HMM on unified-memory APUs and can
/// reduce the final sequential copy to kilobytes per second.
pub fn combine_calib_parts(
    output: &std::path::Path,
    arch_id: u32,
    metadata_json: &str,
    part_paths: &[std::path::PathBuf],
    extra: &[HfqMemTensor],
) -> std::io::Result<()> {
    use crate::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry};
    enum Plan {
        Part { package_idx: usize, name: String },
        Extra { extra_idx: usize },
    }
    let mut packages = Vec::with_capacity(part_paths.len());
    let mut entries = Vec::new();
    let mut plan = Vec::new();
    for part in part_paths {
        #[cfg(unix)]
        let package = HfqFile::open_index_only(part)?;
        #[cfg(not(unix))]
        let package = HfqFile::open(part)?;
        let package_idx = packages.len();
        for e in package.tensors() {
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
                let (info, data) =
                    packages[*package_idx]
                        .tensor_data_vec(name)
                        .ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!("part tensor not found: {name}"),
                            )
                        })?;
                if info.name != *name || data.len() != info.data_size {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "part tensor {name} resolved as {} with {}/{} bytes",
                            info.name,
                            data.len(),
                            info.data_size
                        ),
                    ));
                }
                w.write_all(&data)
            }
            Plan::Extra { extra_idx } => w.write_all(&extra[*extra_idx].data),
        },
    )
}

/// Per-tensor descriptor from [`CalibCollector::tensor_descriptors`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CalibTensorDesc {
    pub name: String,
    pub has_hessian: bool,
    pub k: usize,
    pub n_tokens: u64,
}

/// Canonical calibration metadata assembled before the streaming writer emits
/// the HFQM index. Resident and grouped collectors share this path so artifact
/// counts, per-tensor rows, provenance, and dynamic extras cannot drift.
pub struct CalibrationArtifactMetadata {
    pub n_hessian: usize,
    pub n_imatrix: usize,
    pub json: String,
}

pub fn build_calibration_metadata(
    descriptors: &[CalibTensorDesc],
    layers_per_pass: Option<usize>,
    static_meta: &[(&str, serde_json::Value)],
    extra_meta: &[(String, serde_json::Value)],
) -> Result<CalibrationArtifactMetadata, String> {
    let n_hessian = descriptors
        .iter()
        .filter(|descriptor| descriptor.has_hessian)
        .count();
    let n_imatrix = descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for descriptor in descriptors {
        if per_tensor_tokens
            .insert(
                descriptor.name.clone(),
                serde_json::json!(descriptor.n_tokens),
            )
            .is_some()
        {
            return Err(format!(
                "calib: duplicate tensor descriptor {}",
                descriptor.name
            ));
        }
    }

    let mut object = serde_json::Map::new();
    object.insert(
        "artifact_kind".into(),
        serde_json::Value::String("calibration".into()),
    );
    if let Some(layers_per_pass) = layers_per_pass {
        object.insert("layers_per_pass".into(), serde_json::json!(layers_per_pass));
    }
    object.insert("n_hessian".into(), serde_json::json!(n_hessian));
    object.insert("n_imatrix".into(), serde_json::json!(n_imatrix));
    object.insert(
        "per_tensor_tokens".into(),
        serde_json::Value::Object(per_tensor_tokens),
    );
    object.insert(
        "artifacts".into(),
        serde_json::json!(["hessian", "imatrix"]),
    );
    for (key, value) in static_meta {
        object.insert((*key).to_string(), value.clone());
    }
    for (key, value) in extra_meta {
        object.insert(key.clone(), value.clone());
    }

    let json = serde_json::to_string(&serde_json::Value::Object(object))
        .map_err(|error| format!("calib metadata serialization: {error}"))?;
    Ok(CalibrationArtifactMetadata {
        n_hessian,
        n_imatrix,
        json,
    })
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

    /// Admission-grade resident-oracle collection over the exact native
    /// calibration job. Family implementations override this when they can
    /// reset model state between multiple independent samples. The default is
    /// deliberately strict: legacy collectors may consume one sample only and
    /// use the historical fixed KLD top-k of 64.
    fn collect_calibration_job(
        &self,
        gpu: &mut Gpu,
        tokenizer: &crate::tokenizer::Tokenizer,
        job: &contracts::CalibrationJob,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let [sample] = job.samples.samples() else {
            return Err(format!(
                "resident calibration backend does not implement independent sample resets; job has {} samples",
                job.samples.samples().len()
            ));
        };
        if job.options.kldref && job.options.kldref_top_k != 64 {
            return Err(format!(
                "resident calibration backend uses fixed KLD top-k 64, but the job requests {}",
                job.options.kldref_top_k
            ));
        }
        let provenance = calibration_job_provenance(job, provenance)?;
        self.collect_calibration(
            gpu,
            tokenizer,
            &sample.tokens,
            job.options.kldref,
            output,
            &provenance,
        )
    }
}

/// Add the exact serialized calibration job to a resident artifact's metadata.
/// The streamed/resident comparator requires this field to prove that corpus
/// and independently reset sample maps match; a duplicate caller key is
/// rejected rather than silently overwritten by metadata assembly.
pub fn calibration_job_provenance<'a>(
    job: &contracts::CalibrationJob,
    provenance: &[(&'a str, serde_json::Value)],
) -> Result<Vec<(&'a str, serde_json::Value)>, String> {
    if provenance.iter().any(|(key, _)| *key == "job") {
        return Err("calibration provenance already contains a job field".into());
    }
    let mut result = provenance.to_vec();
    result.push((
        "job",
        serde_json::to_value(job).map_err(|error| format!("serialize calibration job: {error}"))?,
    ));
    Ok(result)
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
    sequences: &[&[u32]],
    forward: F,
) -> Result<CalibSummary, String>
where
    F: FnOnce(&mut Gpu, &[&[u32]]) -> Result<CalibForward, String>,
{
    let collector = arm(gpu, capture_names, imatrix_only);

    // The seam owns the chunking policy, so no arch can silently calibrate
    // under one unbounded context. The arch owns what "independent" means for
    // it — resetting KV / SSM / recurrent state and restarting positions at 0.
    let sequences = calib_sequences(sequences);

    // Run the arch forward, then ALWAYS disarm the capture before propagating
    // any error (a half-armed `gpu` would mis-capture a later forward).
    let forward_out = forward(gpu, &sequences);
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
    let imatrix_only = effective_imatrix_only(imatrix_only);
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
    let metadata =
        build_calibration_metadata(&descriptors, None, static_meta, &forward_out.extra_meta)?;
    let max_consistency = collector
        .write_streaming(
            gpu,
            output,
            arch_id,
            &metadata.json,
            &forward_out.extra_tensors,
        )
        .map_err(|e| format!("calib write {}: {e}", output.display()))?;
    collector.free_gpu(gpu);

    Ok(CalibSummary {
        n_hessian: metadata.n_hessian,
        n_imatrix: metadata.n_imatrix,
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
                    output_names: vec![tensor_name.to_string()],
                    diag,
                    // Weighted path stages no rows, so the shape statistics have
                    // nothing to reduce over; the imatrix is accumulated directly.
                    stats: None,
                    gcrest: None,
                    h,
                    h_f64,
                    buf,
                    buf_rows: 0,
                    buf_capacity: FLUSH_BATCH,
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
            if acc.buf_rows == acc.buf_capacity {
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
///
/// Selection, not a sort. This is called once per scored position, so on a
/// 248320-vocab model a 1175-chunk reference runs it 1.2M times; fully sorting
/// the vocab to keep 256 of it cost ~4.5 billion comparisons per chunk and put
/// the build at ~37 s/chunk, single-threaded — twelve hours for one reference.
///
/// Two things were expensive. The sort was O(n log n) to answer an O(n)
/// question, and its comparator indexed back into `logits`, so every comparison
/// took two random loads across a 1 MB array. Selecting over (logit, index)
/// pairs keeps those loads sequential and `select_nth_unstable_by` is
/// introselect, so only the k survivors get ordered.
///
/// Ties break on the index, ascending. The old `sort_unstable` left equal
/// logits in an arbitrary order, which made the reference bytes depend on the
/// sort's internal swaps; pinning the tiebreak makes the artifact reproducible.
pub fn topk_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    let mut pairs: Vec<(f32, u32)> = logits.iter().copied().zip(0u32..).collect();
    pairs.select_nth_unstable_by(k - 1, cmp);
    pairs.truncate(k);
    pairs.sort_unstable_by(cmp);
    pairs.into_iter().map(|(v, i)| (i, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection must agree with the full sort it replaced, ties included, and
    /// must not depend on how the selection algorithm happened to swap.
    #[test]
    fn topk_logits_matches_full_sort_and_is_tie_deterministic() {
        fn reference(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            // Same tiebreak the real one pins: logit desc, index asc.
            idx.sort_by(|&a, &b| {
                logits[b as usize]
                    .total_cmp(&logits[a as usize])
                    .then(a.cmp(&b))
            });
            idx.truncate(k.min(logits.len()));
            idx.into_iter().map(|i| (i, logits[i as usize])).collect()
        }

        // Deliberately tie-heavy: only 7 distinct values over 500 slots, so the
        // k-boundary lands inside a run of equal logits every time.
        let logits: Vec<f32> = (0..500).map(|i| ((i * 37) % 7) as f32).collect();
        for k in [1usize, 8, 256, 500] {
            assert_eq!(topk_logits(&logits, k), reference(&logits, k), "k={k}");
        }

        // Ordinary descending case, plus the degenerate edges.
        let plain: Vec<f32> = vec![-1.5, 9.0, 0.0, 3.25, -7.0];
        assert_eq!(topk_logits(&plain, 3), vec![(1, 9.0), (3, 3.25), (2, 0.0)]);
        assert_eq!(topk_logits(&plain, 0), vec![]);
        // k past the end clamps rather than panicking in select_nth.
        assert_eq!(topk_logits(&plain, 99), reference(&plain, 99));
    }

    #[test]
    fn compact_hessian_size_matches_diag_plus_lower_triangle() {
        assert_eq!(compact_hessian_bytes(1), 4);
        assert_eq!(compact_hessian_bytes(2), 10);
        assert_eq!(compact_hessian_bytes(3), 18);
        assert_eq!(compact_hessian_bytes(4096), 16_789_504);
    }

    #[test]
    fn canonical_metadata_matches_legacy_shape_and_dynamic_overrides() {
        let descriptors = vec![
            CalibTensorDesc {
                name: "dense.0".into(),
                has_hessian: true,
                k: 4,
                n_tokens: 12,
            },
            CalibTensorDesc {
                name: "expert.0".into(),
                has_hessian: false,
                k: 8,
                n_tokens: 7,
            },
        ];
        let metadata = build_calibration_metadata(
            &descriptors,
            Some(4),
            &[("corpus", serde_json::json!("fixture"))],
            &[(
                "artifacts".into(),
                serde_json::json!(["hessian", "imatrix", "kldref"]),
            )],
        )
        .unwrap();
        assert_eq!(metadata.n_hessian, 1);
        assert_eq!(metadata.n_imatrix, 2);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&metadata.json).unwrap(),
            serde_json::json!({
                "artifact_kind": "calibration",
                "layers_per_pass": 4,
                "n_hessian": 1,
                "n_imatrix": 2,
                "per_tensor_tokens": {"dense.0": 12, "expert.0": 7},
                "artifacts": ["hessian", "imatrix", "kldref"],
                "corpus": "fixture",
            })
        );
    }

    #[test]
    fn canonical_metadata_rejects_duplicate_tensor_descriptors() {
        let descriptors = vec![
            CalibTensorDesc {
                name: "same".into(),
                has_hessian: true,
                k: 4,
                n_tokens: 1,
            },
            CalibTensorDesc {
                name: "same".into(),
                has_hessian: false,
                k: 4,
                n_tokens: 1,
            },
        ];
        assert!(build_calibration_metadata(&descriptors, None, &[], &[]).is_err());
    }

    #[test]
    fn calibration_part_combiner_preserves_tensor_bytes_without_payload_mmaps() {
        use crate::hfq::{write_hfqm_package_mem, HfqFile};

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hipfire-calib-combine-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let part_a = root.join("part-a.hfq");
        let part_b = root.join("part-b.hfq");
        let output = root.join("combined.calib.hfq");
        write_hfqm_package_mem(
            &part_a,
            24,
            r#"{"artifact_kind":"calibration-part"}"#,
            &[HfqMemTensor {
                name: "layer.0.imatrix".into(),
                quant_type: 2,
                shape: vec![2],
                group_size: 0,
                data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
        )
        .unwrap();
        write_hfqm_package_mem(
            &part_b,
            24,
            r#"{"artifact_kind":"calibration-part"}"#,
            &[HfqMemTensor {
                name: "layer.1.imatrix".into(),
                quant_type: 2,
                shape: vec![1],
                group_size: 0,
                data: vec![9, 10, 11, 12],
            }],
        )
        .unwrap();
        combine_calib_parts(
            &output,
            24,
            r#"{"artifact_kind":"calibration"}"#,
            &[part_a.clone(), part_b.clone()],
            &[HfqMemTensor {
                name: "extra".into(),
                quant_type: 2,
                shape: vec![1],
                group_size: 0,
                data: vec![13, 14, 15, 16],
            }],
        )
        .unwrap();

        let combined = HfqFile::open_index_only(&output).unwrap();
        assert_eq!(combined.arch_id, 24);
        assert_eq!(
            combined.tensor_data_vec("layer.0.imatrix").unwrap().1,
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            combined.tensor_data_vec("layer.1.imatrix").unwrap().1,
            vec![9, 10, 11, 12]
        );
        assert_eq!(
            combined.tensor_data_vec("extra").unwrap().1,
            vec![13, 14, 15, 16]
        );
        drop(combined);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resident_job_provenance_is_exact_and_rejects_shadowing() {
        use contracts::{CalibrationJob, CalibrationOptions, CalibrationSample, SampleSet};

        let samples = SampleSet::new(
            vec![
                CalibrationSample::new("a", vec![1, 2], "fixture"),
                CalibrationSample::new("b", vec![3, 4], "fixture"),
            ],
            2,
            7,
        )
        .unwrap();
        let job = CalibrationJob::new(
            "source",
            "tokenizer",
            samples,
            CalibrationOptions::default(),
        )
        .unwrap();
        let provenance =
            calibration_job_provenance(&job, &[("oracle", serde_json::json!("resident"))]).unwrap();
        assert_eq!(provenance.len(), 2);
        assert_eq!(provenance[1].0, "job");
        assert_eq!(provenance[1].1, serde_json::to_value(&job).unwrap());
        assert!(calibration_job_provenance(&job, &[("job", serde_json::json!("shadow"))]).is_err());
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

#[cfg(test)]
mod calib_sequence_default_tests {
    use super::{calib_sequences, DEFAULT_CALIB_SEQ_LEN};

    /// The DEFAULT must split. This used to be `usize::MAX` — one unbounded
    /// sequence — so the O(n²) shape was what you got unless you knew to set
    /// `HIPFIRE_CALIB_SEQ_LEN`, and the cost grew with the very budget that
    /// makes calibration worth doing. Measured on zaya1-8b at 32768 tokens:
    /// 10746 s -> 1065 s, quality unchanged.
    ///
    /// ⚠️ The input length is a LITERAL, not derived from the constant. A first
    /// version of this test sized its input as `DEFAULT_CALIB_SEQ_LEN * 3`, so
    /// raising the constant grew the input to match and the assertion held for
    /// any value — it could not fail. Keep the literal.
    ///
    /// Guarded on the env var being unset, because these tests share a process
    /// and the variable is read at call time.
    #[test]
    fn the_default_splits_rather_than_running_one_unbounded_sequence() {
        if std::env::var("HIPFIRE_CALIB_SEQ_LEN").is_ok() {
            return; // an explicit override is being exercised elsewhere
        }
        let toks: Vec<u32> = (0..6644u32).collect(); // literal: ~3.2x of 2048
        let out = calib_sequences(&[&toks[..]]);
        assert!(
            out.len() > 1,
            "the default must chunk 6644 tokens; got {} sequence(s), which is the \
             O(n^2) shape this default exists to prevent",
            out.len()
        );
        assert!(
            out.iter().all(|s| s.len() <= 2048),
            "no sequence may exceed 2048"
        );
        // 2048 matches the n_ctx KLD references are built at; drifting from that
        // reintroduces the calibrate-long/serve-short mismatch.
        assert_eq!(DEFAULT_CALIB_SEQ_LEN, 2048);
    }

    /// A trailing 1-token remainder has no next-token target and would
    /// contribute a degenerate row, so it is dropped rather than kept.
    #[test]
    fn a_one_token_remainder_is_dropped() {
        if std::env::var("HIPFIRE_CALIB_SEQ_LEN").is_ok() {
            return;
        }
        let toks: Vec<u32> = (0..2049u32).collect(); // literal 2048 + 1
        let out = calib_sequences(&[&toks[..]]);
        assert_eq!(out.len(), 1, "the 1-token tail must not become a sequence");
        assert_eq!(out[0].len(), 2048);
    }
}
