// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! MoE kernel family: dispatching expert GEMM operations.
//!
//! Supports 3 variants:
//! - **IndexedGateUp**: gate+up projection for a single expert (indexed by token)
//! - **IndexedDown**: down projection for a single expert (indexed by token)
//! - **GroupedGemm**: batched grouped-expert GEMM (all experts in one launch)
//!
//! # Current status
//!
//! `run()` is the centralized single-token MoE decode entry — it delegates to
//! [`crate::pipeline::run_moe_decode`] (the GPU top-K fast path plus the generic
//! CPU-top-K fallback). The family owns resolution (`MoeDtypes` → `MoeResolution`);
//! the model passes only the dtype snapshot + k. One `DispatchCtx` is threaded
//! end-to-end from the call site through every inner GEMV. Scratch stays model-owned.
//! Grouped-GEMM prefill is a future arm (gated on `ShapeInfo.batch_size`).

use hipfire_rdna::DType;
use hipfire_rdna::GpuTensor;

use crate::context::DispatchCtx;
use crate::families::gemv::{GivensRef, WeightRef};
use crate::tables::moe_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── MoE eligibility lattice ────────────────────────────

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605); the routed
/// fields use experts[0] as representative (the loader builds all experts in a
/// layer with matching dtype, so [0] == all — same invariant the original
/// routed_* checks relied on).
#[derive(Clone, Copy, Debug)]
pub struct MoeDtypes {
    pub router: DType,
    pub shared_gate: DType,        // ffn.shared_expert_gate
    pub shared_expert_gate: DType, // ffn.shared_expert.gate
    pub shared_expert_up: DType,   // ffn.shared_expert.up
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType, // ffn.experts[0].gate_up
    pub routed_down: DType,    // ffn.experts[0].down
    pub has_paro_shared: bool, // ffn.paro_shared.is_some()
}

/// `HIPFIRE_QWEN35_MOE_OQ_INDEXED` — the single parse of the switch controlling
/// the indexed routed-OQ path. **ON by default** since 2026-08-12; set `0`/`off`
/// to fall back to the CPU top-K path.
///
/// The road here is worth keeping, because the default was flipped ON, reverted,
/// and flipped ON again in one day, and only the last one had evidence behind it:
///
/// 1. The per-expert AWQ rotation was wrong — routed experts do NOT share an AWQ
///    scale, so one rotation for all of them was a scale error from layer 0
///    (35B-A3B oq4.25++: KLD 5.108296, ppl 1171.67). Fixed: KLD 0.031515 / ppl
///    7.4643 against 0.030367 / 7.4622 for the fallback, layer-0 residual cosine
///    0.244 -> 0.999999.
/// 2. Flipped on those numbers, and `tests/tiny-quant-gate.sh` turned seven
///    `qwen3_5_moe` OQ cells **non-finite**. One model's KLD is not the GPU
///    correctness tier.
/// 3. Root cause: the FWHT rotates are G256 on both sides and compute
///    `n_groups = K / 256`, so the toy `moe_inter = 128` launched a ZERO-SIZED
///    grid — dispatched, no blocks, success, destination never written.
/// 4. Both halves fixed: `hipfire_rdna::dispatch::fwht_groups` errors on such a
///    `K`, and [`oq_indexed_decode_active`] refuses admission before that.
/// 5. And only then flipped: the tier now has a fixture that REACHES this path
///    (`qwen3_5_moe_indexed` — top-8, `moe_inter` 768), verified to move the KLD
///    when the switch moves, which is the thing steps 1-2 never established.
///
/// So `qwen3_5_moe` (`moe_inter = 128`) covers the fallback and
/// `qwen3_5_moe_indexed` covers this path. If you change either fixture's shape,
/// you are changing what this switch is tested by.
///
/// This parse MUST stay single. Three sites used to read it independently and
/// two spellings disagreed: the loader's MoE-block repack and this resolver
/// accepted only `"1"`, while qwen35's dispatch predicate also accepted `"on"`.
/// `=on` therefore enabled the indexed dispatch against weights that were never
/// repacked for it — guaranteed garbage, from a value that looks like it should
/// work. The same trap now points the other way: an unrecognised OFF spelling
/// leaves the path ON, so the off-values are matched explicitly and generously.
pub fn oq_indexed_decode_enabled() -> bool {
    oq_indexed_decode_enabled_from(
        std::env::var("HIPFIRE_QWEN35_MOE_OQ_INDEXED")
            .ok()
            .as_deref(),
    )
}

/// The parse itself, split out so it is testable without touching process-global
/// env (which races under a parallel test runner). The default has been inverted
/// twice, so it carries a direct assertion either way.
pub fn oq_indexed_decode_enabled_from(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false") | Some("no"))
}

/// The only top-k the indexed routed-expert kernels exist for. They are the
/// `*_k8_indexed*` family and `use_gpu_topk` admits them at this `k` and no
/// other; every other top-k decodes through the CPU-top-K fallback.
pub const INDEXED_MOE_K_TOP: usize = 8;

/// Admission for the indexed OQ path, ignoring the env switch. TWO independent
/// conditions, and both have bitten:
///
/// **Shape.** The path rotates activations with the 256-wide FWHT on BOTH sides —
/// gate_up over `K = hidden`, down over `K = mi` — and each is a G256 format, so
/// a `K` that is not a positive multiple of 256 is not a shape it has.
/// `hipfire_rdna::dispatch::fwht_groups` errors on such a `K` instead of
/// launching the zero-sized grid that used to return success without writing its
/// destination, but admission must refuse BEFORE that or the loud failure just
/// replaces a wrong answer with a dead model. The arch-6 toy MoE is `mi = 128`,
/// which is what turned seven `qwen3_5_moe` OQ cells non-finite.
///
/// **Top-k.** The loader repacks routed OQ experts into the indexed kernels'
/// BLOCK layout (132 B / 260 B), so it must repack only when those kernels can
/// actually run. At `k_top != 8` dispatch falls back to CPU top-K, which reads
/// the same tensors as canonical 130 B / 258 B — silent garbage. Measured on a
/// top-2 fixture at `mi = 768`: **NaN** with the switch on, `0.06812199` with it
/// off. Found by PR #248 while merging the shape half; the shape guard alone was
/// not enough, and flipping the default is what made it reachable.
///
/// This must agree everywhere. The loader's repack decides the weight LAYOUT, so
/// any site that disagrees runs one layout's kernels over the other's bytes.
pub fn oq_indexed_admissible(hidden: usize, mi: usize, k_top: usize) -> bool {
    hidden != 0 && mi != 0 && hidden % 256 == 0 && mi % 256 == 0 && k_top == INDEXED_MOE_K_TOP
}

/// The flag AND admission. Every site that used to read
/// [`oq_indexed_decode_enabled`] directly wants this one.
pub fn oq_indexed_decode_active(hidden: usize, mi: usize, k_top: usize) -> bool {
    oq_indexed_decode_enabled() && oq_indexed_admissible(hidden, mi, k_top)
}

/// CPU top-K expert selection + renormalisation — the routing half of
/// [`crate::pipeline::run_moe_decode_cpu_fallback`], extracted so it can be
/// tested without a GPU.
///
/// This runs for EVERY MoE whose `k != 8`, because those fail the `use_gpu_topk`
/// guard and land in the fallback — so it is the only routing code a `k = 10`
/// model like Qwen3.8-Flash-Next (`qwen4_exp`) ever executes. It was previously
/// inline, which meant the k != 8 path had no unit coverage at all, and the only
/// evidence offered for it was an end-to-end comparison in which BOTH arms ran
/// this same code, so any bug here was common-mode and cancelled exactly.
///
/// Semantics match the HF reference verbatim (`Qwen3NextTopKRouter.forward`,
/// which `Qwen4ExpTextTopKRouter` inherits unchanged): softmax is applied by the
/// caller, then top-k by probability, then divide by the selected sum when
/// `norm_topk_prob`. That flag defaults TRUE in both
/// `configuration_qwen4_exp.py` and our own config parse, and the target's
/// `config.json` omits it, so the default is load-bearing.
///
/// `probs` must already be softmaxed. Returns `(indices, weights)` with indices
/// in DESCENDING probability order — the routed loop and the telemetry histogram
/// both rely on that order.
///
/// Panics are avoided by the caller's `k ∈ [1, n_exp]` guard
/// (`cpu-topk-k-out-of-range`); `select_nth_unstable_by(k - 1)` panics otherwise.
pub fn cpu_topk_select(probs: &[f32], k: usize, norm_topk_prob: bool) -> (Vec<usize>, Vec<f32>) {
    debug_assert!(
        k >= 1 && k <= probs.len(),
        "caller must guard k in [1, n_exp]"
    );
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_indices: Vec<usize> = indices.into_iter().take(k).collect();
    topk_indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_weights: Vec<f32> = topk_indices.iter().map(|&i| probs[i]).collect();
    if norm_topk_prob {
        let sum: f32 = topk_weights.iter().sum();
        if sum > 0.0 {
            for w in topk_weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    (topk_indices, topk_weights)
}

#[cfg(test)]
mod cpu_topk_tests {
    use super::cpu_topk_select;

    /// Distinct, non-monotonic probabilities so a bug cannot pass by accident:
    /// the top-k set is not a prefix, not a suffix, and not sorted in place.
    fn probs12() -> Vec<f32> {
        vec![
            0.02, 0.13, 0.01, 0.20, 0.03, 0.11, 0.04, 0.18, 0.05, 0.09, 0.07, 0.07,
        ]
    }

    /// The qwen4_exp shape: top-10 of 12. Two experts must stay dark, and they
    /// must be the two smallest.
    #[test]
    fn selects_top_10_of_12_descending() {
        let p = probs12();
        let (idx, w) = cpu_topk_select(&p, 10, false);
        assert_eq!(idx.len(), 10);
        // the two smallest (0.01 @ 2, 0.02 @ 0) are excluded
        assert!(!idx.contains(&2), "expert 2 (p=0.01) must be dark");
        assert!(!idx.contains(&0), "expert 0 (p=0.02) must be dark");
        // strictly descending by probability
        for pair in w.windows(2) {
            assert!(pair[0] >= pair[1], "weights must be descending: {w:?}");
        }
        assert_eq!(idx[0], 3, "highest prob is expert 3 (0.20)");
    }

    #[test]
    fn renorm_sums_to_one_and_preserves_ratios() {
        let p = probs12();
        let (_, w) = cpu_topk_select(&p, 10, true);
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "renormalised weights must sum to 1, got {sum}"
        );
        let (_, raw) = cpu_topk_select(&p, 10, false);
        // ratio of the top two is unchanged by renormalisation
        assert!((w[0] / w[1] - raw[0] / raw[1]).abs() < 1e-5);
    }

    #[test]
    fn no_renorm_leaves_probabilities_untouched() {
        let p = probs12();
        let (idx, w) = cpu_topk_select(&p, 10, false);
        for (slot, &e) in idx.iter().enumerate() {
            assert_eq!(w[slot], p[e]);
        }
    }

    /// k == n_exp is degenerate (nothing is dark) and k == 1 is the other edge.
    /// Both are reachable through the same guard, so both must hold.
    #[test]
    fn k_edges_hold() {
        let p = probs12();
        let (idx, w) = cpu_topk_select(&p, p.len(), true);
        assert_eq!(idx.len(), p.len());
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let (idx1, w1) = cpu_topk_select(&p, 1, true);
        assert_eq!(idx1, vec![3]);
        assert!(
            (w1[0] - 1.0).abs() < 1e-6,
            "single expert renormalises to 1.0"
        );
    }

    /// FAULT INJECTION — proves these tests can actually fail. This is the
    /// mutant the adversarial review asked for: drop the k-th expert. If the
    /// assertions above cannot catch it, they are measuring nothing.
    #[test]
    fn tests_detect_a_dropped_kth_expert() {
        let p = probs12();
        let (good, _) = cpu_topk_select(&p, 10, false);
        let (short, _) = cpu_topk_select(&p, 9, false);
        assert_ne!(good.len(), short.len());
        let missing: Vec<_> = good.iter().filter(|e| !short.contains(e)).collect();
        assert_eq!(
            missing.len(),
            1,
            "dropping k by one must lose exactly one expert"
        );
        // and it must be the WEAKEST of the ten, not an arbitrary one
        assert_eq!(*missing[0], good[9]);
    }

    /// Ties must not lose or duplicate a slot. `select_nth_unstable_by` is
    /// unstable, so which of the tied pair wins is unspecified — but the
    /// cardinality and the absence of duplicates are not.
    #[test]
    fn ties_do_not_duplicate_or_drop() {
        let p = probs12(); // experts 10 and 11 are both 0.07
        let (idx, _) = cpu_topk_select(&p, 10, true);
        let mut seen = idx.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), idx.len(), "no duplicate expert slots");
    }
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq6: bool,
    pub routed_indexable_paro: bool,
    /// Opus Quant W4A16 routed experts (Oq4G256 gate_up + down). Indexed
    /// MoE-block kernels (132 B/group) — same FWHT-rotated activation basis as
    /// the MQ path, so it shares `needs_x_rot_local`/`rotate_x_mq`.
    pub routed_indexable_oq4: bool,
    /// Opus Quant W8A16 routed experts (Oq8G256 gate_up + down, 260 B/group).
    pub routed_indexable_oq8: bool,
    /// Routed experts are compact-resident, possibly MIXED with promoted Oq8
    /// ones in the same layer. Dispatches the `gemv_oq_compact_moe_*` kernels,
    /// which read a PER-EXPERT stride table (0 = Oq8) and so can serve both
    /// layouts in one launch.
    pub routed_indexable_oq_compact: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
}

impl MoeResolution {
    /// ⚠️ Applies the env switch with **no admission check** — no shape, no
    /// k_top. That is fine for tests, which pass shapes they control, and wrong
    /// for a dispatch site: both halves of [`oq_indexed_admissible`] exist
    /// because omitting one produced non-finite output (sub-256 `mi`) and NaN
    /// (top-k != 8). There is deliberately no production caller; the live path
    /// uses `resolve_with_oq_indexed(.., oq_indexed_decode_active(..))`. Keep it
    /// that way.
    pub fn resolve(d: &MoeDtypes, k: usize) -> Self {
        // origin/master's helper, not an inline env read: it accepts the same
        // spellings the arch layer does. The inline `== Some("1")` this replaces
        // is exactly the `=1` vs `=on` split the indexed-OQ handover called
        // "guaranteed garbage".
        let oq_indexed_decode = oq_indexed_decode_enabled();
        let resolved = Self::resolve_with_oq_indexed(d, k, oq_indexed_decode);
        // `HIPFIRE_MOE_RESOLVE_DEBUG=1` dumps the snapshot this verdict came from.
        //
        // The arch layer resolves the same question independently and traces its
        // own answer, so the two can disagree while both look correct in
        // isolation — that split has now produced two bugs: the routed-dtype
        // marshalling defaulting to F32 under paged residency, and a silent
        // drop to `run_moe_decode_cpu_fallback` (~160 s/layer on one core) while
        // the arch trace still reported `use_gpu_topk=true`. THIS is the verdict
        // that selects the path; the arch-side trace is not.
        if std::env::var("HIPFIRE_MOE_RESOLVE_DEBUG").as_deref() == Ok("1") {
            eprintln!(
                "[moe-resolve] k={k} oq_gate={oq_indexed_decode} router={:?} \
                 shared(gate/eg/eu)={:?}/{:?}/{:?} routed(gu/dn)={:?}/{:?} \
                 all_gu_mq4={} paro_shared={} => idx(mq4/mq6/paro/oq4/oq8)={}/{}/{}/{}/{} \
                 use_gpu_topk={} needs_x_rot={}",
                d.router,
                d.shared_gate,
                d.shared_expert_gate,
                d.shared_expert_up,
                d.routed_gate_up,
                d.routed_down,
                d.experts_all_gate_up_mq4,
                d.has_paro_shared,
                resolved.routed_indexable_mq4,
                resolved.routed_indexable_mq6,
                resolved.routed_indexable_paro,
                resolved.routed_indexable_oq4,
                resolved.routed_indexable_oq8,
                resolved.use_gpu_topk,
                resolved.needs_x_rot_local,
            );
        }
        resolved
    }

    pub fn resolve_with_oq_indexed(d: &MoeDtypes, k: usize, oq_indexed_decode: bool) -> Self {
        use DType::*;
        let gate_side_mq4 = d.router == MQ4G256
            && d.shared_gate == MQ4G256
            && d.shared_expert_gate == MQ4G256
            && d.shared_expert_up == MQ4G256
            && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;

        let routed_gate_up_oq4 = d.routed_gate_up == Oq4G256;
        let routed_gate_up_oq8 = d.routed_gate_up == Oq8G256;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;
        let routed_indexable_oq4 =
            oq_indexed_decode && (d.routed_down == Oq4G256) && routed_gate_up_oq4;
        let routed_indexable_oq8 =
            oq_indexed_decode && (d.routed_down == Oq8G256) && routed_gate_up_oq8;
        // Compact-resident routed experts on the indexed path. ON by default:
        // verified byte-identical to the expanded Oq8 reference on a 35B-A3B.
        //
        // It shipped OFF for one commit because a compact layer produced "!!!!".
        // The kernels were never the problem -- the cause was that the OQ branch
        // above tests only `oq4 || oq8`, so a compact layer fell PAST it into the
        // paro arm and was decoded as ParoQ4G128, while also skipping the
        // per-slot `rotate_x_mq_awq_indexed_batched` that branch performs. Two
        // faults from one missing disjunct, and neither was visible in the
        // kernels: HIPFIRE_MOE_FEED_DEBUG=1 found it by showing the probe inside
        // that branch never firing.
        //
        // HIPFIRE_MOE_COMPACT_INDEXED=0 forces the generic CPU-top-K fallback,
        // which is correct but slower and does not need the stride table.
        let routed_gate_up_oq_compact = d.routed_gate_up == OqCompactG256;
        let routed_indexable_oq_compact = oq_indexed_decode
            && (d.routed_down == OqCompactG256)
            && routed_gate_up_oq_compact
            && std::env::var("HIPFIRE_MOE_COMPACT_INDEXED").as_deref() != Ok("0");

        let routed_dtype_indexable = routed_indexable_mq4
            || routed_indexable_mq6
            || routed_indexable_paro
            || routed_indexable_oq4
            || routed_indexable_oq8
            || routed_indexable_oq_compact;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        if !use_gpu_topk {
            hipfire_rdna::kernel_trace::record_fallback(
                "moe decode: routed dtype not indexable (or k != 8) -> CPU top-K per-expert loop",
                &format!(
                    "gate_up={:?} down={:?} k={} indexable={}",
                    d.routed_gate_up, d.routed_down, k, routed_dtype_indexable
                ),
            );
        }
        // OQ routed experts are FWHT-rotated (same signs as MQ, gen_fwht_signs
        // 42/1042 uploaded by ensure_mq_signs) → they need x_rot_local too.
        let needs_x_rot_local = gate_side_mq4
            || routed_gate_up_mq4
            || routed_gate_up_mq6
            || routed_gate_up_paro
            || routed_gate_up_oq4
            || routed_gate_up_oq8
            || routed_gate_up_oq_compact;

        Self {
            gate_side_mq4,
            routed_indexable_mq4,
            routed_indexable_mq6,
            routed_indexable_paro,
            routed_indexable_oq4,
            routed_indexable_oq8,
            routed_indexable_oq_compact,
            use_gpu_topk,
            needs_x_rot_local,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4
            || self.routed_indexable_mq6
            || self.routed_indexable_paro
            || self.routed_indexable_oq4
            || self.routed_indexable_oq8
            || self.routed_indexable_oq_compact
    }
}

// ── Dispatch parameters ────────────────────────────────

/// Everything the MoE decode executor arm reads, marshaled by the model from
/// its weight/config/scratch structs. Resolution is owned by the family
/// (the model passes only the dtype snapshot + k); the executor computes
/// [`MoeResolution`] from [`MoeDtypes`] on entry.
/// Makes routed experts dereferenceable by the indexed MoE kernels under paged
/// residency.
///
/// **Why a trait and not a field.** `WeightPager` lives in `hipfire-runtime`,
/// which depends on *this* crate — so the pager cannot be held here directly
/// without a dependency cycle. The provider is implemented on the runtime/arch
/// side, where the pager already is, and passed in as a trait object.
///
/// **The contract is a safety property, not a convenience.** The indexed kernels
/// do `A = expert_ptrs[topk_indices[..]]` and dereference `A` with no validation.
/// An implementation that returns `Ok(())` while leaving any selected expert
/// non-resident causes a GPU-side null dereference, which on gfx1103 is an
/// `amdgpu` MES hang and a full device reset — it takes down every other process
/// on the GPU and is not recoverable in-process. Returning `Err` is always
/// preferable to returning `Ok` with incomplete residency.
///
/// **This does NOT patch the pointer table.** The pager maintains the table
/// itself on every residency transition (push), so an implementation only has to
/// make the experts resident and the slots follow. That split is deliberate: the
/// previous design had each dispatch site patch after ensuring, and the
/// default-ON lowered pipeline simply never did, leaving the table all-zero.
pub trait ExpertResidency {
    /// Make every expert in `selected` resident for `layer`.
    ///
    /// On `Ok(())` every expert named by `selected` MUST be resident, and hence
    /// have a live slot in the pager-maintained pointer table. Experts not in
    /// `selected` are not read by this dispatch.
    fn ensure_resident(
        &self,
        gpu: &mut hipfire_rdna::Gpu,
        layer: usize,
        selected: &[u32],
    ) -> Result<(), DispatchError>;
}

pub struct MoeParams<'a> {
    /// Index of the decoder layer this MoE block belongs to. Carried purely so
    /// the executor can attribute router-selection telemetry to a layer; the
    /// compute path never reads it. Mirrors `MoePrefillParams::layer`, which
    /// has always had it — the decode side lacked one, which is why the
    /// verbatim port of `moe_ffn_decode_impl` could not keep its histogram call.
    pub layer: usize,
    pub dtypes: MoeDtypes,
    /// Token-batch width. Decode = 1. >1 must route to grouped prefill (Step 8).
    /// Guarded at runtime matching the bias-aware decode guard.
    pub batch_size: usize,
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub smi: usize,
    pub k: usize,
    pub n_exp: usize,
    pub norm_topk_prob: bool,
    pub x_rot_prerotated: bool,
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    /// EP (expert-parallel, Ship 6 substrate-EP) routed-output redirect. When
    /// `Some`, the routed combine AND the shared-expert down accumulate into
    /// this **zeroed** partial buffer instead of `x_residual`; the EP executor
    /// then all-reduces the partial across ranks and adds it into `x_residual`
    /// once. `None` (default) = single-GPU: accumulate directly into
    /// `x_residual`, byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
    /// EP: skip the shared-expert **down** projection so the replicated shared
    /// expert is computed on rank 0 only (not summed N× by the all-reduce).
    /// `false` (default) = run it (single-GPU). Router + shared gate/up still
    /// run on every rank (they share the fused gate-side GEMV with the router).
    pub skip_shared: bool,
    // gate-side weights
    pub router: WeightRef<'a>,
    pub shared_expert_gate: WeightRef<'a>,
    pub shared_gate_w: WeightRef<'a>,
    pub shared_up_w: WeightRef<'a>,
    pub shared_down_w: WeightRef<'a>,
    // routed expert pointer tables + dims
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Per-expert compact block stride (0 = that expert is Oq8), `[n_exp]` i32.
    /// `None` where no compact routed expert is resident. Required whenever
    /// `routed_indexable_oq_compact` is set -- it is what lets one launch serve a
    /// layer that mixes compact and promoted experts.
    pub expert_gate_up_strides: Option<&'a GpuTensor>,
    pub expert_down_strides: Option<&'a GpuTensor>,
    /// Per-expert AWQ scale pointer tables, same shape and construction as the
    /// weight pointer tables above. Routed experts do NOT share one AWQ scale
    /// (each sees a different token subset, so a different imatrix), and the
    /// divide must precede the FWHT, so the rotation is per (token, krank).
    /// `None` when no expert at this layer carries a sidecar — the rotation is
    /// then plain, but still per-slot, because the indexed OQ GEMVs read `x`
    /// per slot either way.
    pub expert_gate_up_awq_ptrs: Option<&'a GpuTensor>,
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Residency provider for paged experts. `None` means either fully resident
    /// (the tables were filled at load from `MoeFfnWeights::experts`) or paged
    /// with no provider — and `check_moe_decode_supported` refuses the latter
    /// rather than dispatching against a table that is still all-zero.
    pub expert_residency: Option<&'a dyn ExpertResidency>,
    /// True when resident routed experts sit in the `oq4_arch` COMBINED layout
    /// (the default for qt=34/37), so the OQ4 grouped kernel must skip past the
    /// split nibbles and split f32 scales to reach its interleaved block stream.
    /// False when `oq_moe` repacked them, which emits that stream at offset 0.
    pub routed_oq_arch_combined: bool,
    pub routed_gate_up_k: usize,
    pub routed_down_m: usize,
    pub routed_down_k: usize,
    /// Per-expert (gate_up, down) weight refs for the generic CPU-top-K
    /// fallback (`!use_gpu_topk`: k != 8 OR routed dtype not indexable).
    /// Master's `moe_ffn_decode_impl` indexed `ffn.experts[expert_idx]` in a
    /// host loop; the indexed-kernel pointer tables above can't drive that
    /// path (they assume k=8 + an indexable routed dtype). One ref pair per
    /// expert, length `n_exp`. **Empty** when the layer is paged (the indexed
    /// GPU-top-K path is the only mode in paged residency) — the fallback
    /// asserts non-empty before use, matching master's `ffn.experts[..]`
    /// indexing (which also required resident experts).
    pub routed_experts: &'a [(WeightRef<'a>, WeightRef<'a>)],
    // paro sidecars
    pub routed_gate_up_paro: Option<GivensRef<'a>>,
    pub routed_down_paro: Option<GivensRef<'a>>,
    // scratch buffers
    pub router_logits: &'a GpuTensor,
    pub scalar_buf: &'a GpuTensor,
    pub x_rot_local: &'a GpuTensor,
    /// Fused [gate||up] scratch of length `2 * max(mi, smi)`. Used by the
    /// generic CPU-top-K fallback to receive a single routed expert's fused
    /// gate_up GEMV output (master wrote `expert.gate_up` into one buffer of
    /// width `2*mi`, then sliced gate/up halves). The GPU-top-K fast path
    /// does not read this field.
    pub gate_up_buf: &'a GpuTensor,
    pub gate_buf: &'a GpuTensor,
    pub up_buf: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
    /// `[k × hidden]` f32 — per-slot rotated gate_up input, the input-side
    /// mirror of `down_expanded`. Written by
    /// `rotate_x_mq_awq_indexed_batched`; see `expert_gate_up_awq_ptrs`.
    pub x_rot_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 bias-aware decode parameters ───────────

/// Parameters for the deepseek4 bias-aware MoE decode arm (k=6, MQ2-Lloyd routed
/// experts). Kept distinct from [`MoeParams`] because the ds4 sub-graph has no
/// fused gate-side and no shared-expert block: the shared expert is a separate
/// model-owned step (`ffn_stub`) that runs first and seeds `ffn_out`, and this
/// arm's routed-down kernel atomic-accumulates into that same buffer.
///
/// `scores` is the post-`sqrt_softplus(gate·x)` router output — the model owns
/// the router GEMV + activation. Selection adds `gate_bias` while the routing
/// weights use the *unbiased* `scores`; the bias-aware kernel handles that
/// two-score semantic and folds in `route_scale`, all in one launch. The model
/// pre-rotates the activation, so `x_rot` is consumed as-is (no re-rotation).
pub struct MoeBiasAwareParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub n_exp: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Token-batch width. Decode = 1. A value > 1 must route to the grouped
    /// prefill executor (Step 8), never this decode arm — guarded in the executor.
    pub batch_size: usize,
    // activations / residual
    /// FWHT-rotated activation (model pre-rotates; this arm does not re-rotate).
    pub x_rot: &'a GpuTensor,
    /// Residual stream the routed-down kernel atomic-accumulates into. The
    /// model's shared-expert step must have run first to seed this buffer.
    pub ffn_out: &'a GpuTensor,
    // router
    pub scores: &'a GpuTensor, // post-sqrt_softplus gate·x (weights use these)
    pub gate_bias: &'a GpuTensor, // per-expert routing bias (selection only)
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Per-expert compact block stride (0 = that expert is Oq8), `[n_exp]` i32.
    /// `None` where no compact routed expert is resident. Required whenever
    /// `routed_indexable_oq_compact` is set -- it is what lets one launch serve a
    /// layer that mixes compact and promoted experts.
    pub expert_gate_up_strides: Option<&'a GpuTensor>,
    pub expert_down_strides: Option<&'a GpuTensor>,
    // scratch buffers (model-owned)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    /// `[k_top × hidden]` per-expert down outputs for the deterministic combine.
    pub down_expanded: &'a GpuTensor,
    /// Layer index, needed only to name the layer to `ExpertResidency`.
    pub layer_idx: usize,
    /// Paged-expert residency, mirroring `MoeParams::expert_residency`.
    ///
    /// The bias-aware path carried the pointer TABLES but not this hook, so an
    /// arch decoding through it (deepseek4, top-6) could not use the pager at
    /// all and had to upload every expert resident. That is what makes an
    /// 82.8 GB artifact die at layer 19 of 43 on a 43 GB device.
    ///
    /// `None` on a fully-resident model — every caller today — and the dispatch
    /// is then byte-identical to before.
    pub expert_residency: Option<&'a dyn ExpertResidency>,
}

// ── DeepSeek-V4 batched/prefill MoE parameters ─────────

/// Router-selection mode for the batched/prefill MoE path. DeepSeek-V4 uses
/// static hash routing for the first `num_hash_layers` layers and bias-aware
/// top-k for the rest; the executor branches on this.
pub enum MoePrefillRouting<'a> {
    /// Bias-aware batched top-k (select on `scores + gate_bias`, weight on the
    /// unbiased `scores`, normalize, `*route_scale`).
    BiasAware { gate_bias: &'a GpuTensor },
    /// Static `tid2eid` hash routing (layers `0..num_hash_layers`). `tokens` is
    /// the device-side `[B]` i32 token-id buffer.
    Hash {
        tid2eid: &'a GpuTensor,
        tokens: &'a GpuTensor,
    },
}

/// Parameters for the deepseek4 batched/prefill MoE (k=6, MQ2-Lloyd). The
/// model owns RMSNorm, the shared expert, the router GEMV + `sqrt_softplus`
/// (producing `scores`); this arm runs routing → routed experts → combine,
/// accumulating into `ffn_out` (the shared expert already seeded it).
///
/// Picks the grouped-GEMM path when `batch_size >= HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE`
/// (default 128), else the scalar K4 indexed path — mirroring `ffn_batched`.
pub struct MoeBiasAwarePrefillParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub n_exp: usize,
    pub k_top: usize,
    pub batch_size: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub layer_idx: usize, // for the optional HIPFIRE_DEEPSEEK4_DUMP_TOPK header
    /// Paged-expert residency, mirroring `MoeBiasAwareParams::expert_residency`.
    ///
    /// PREFILL needs this as much as decode does. Without it the prefill pass
    /// dispatches against whatever the device pointer table happens to hold —
    /// null for every expert no earlier decode admitted, and eviction NULLS
    /// slots, so a paged model silently prefills against missing experts.
    pub expert_residency: Option<&'a dyn ExpertResidency>,
    // routing
    pub routing: MoePrefillRouting<'a>,
    pub scores: &'a GpuTensor, // post-sqrt_softplus moe_scores_batch [B, n_exp]
    pub topk_indices: &'a GpuTensor, // [B, k_top] (routing out, expert in)
    pub topk_weights: &'a GpuTensor, // [B, k_top]
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Per-expert compact block stride (0 = that expert is Oq8), `[n_exp]` i32.
    /// `None` where no compact routed expert is resident. Required whenever
    /// `routed_indexable_oq_compact` is set -- it is what lets one launch serve a
    /// layer that mixes compact and promoted experts.
    pub expert_gate_up_strides: Option<&'a GpuTensor>,
    pub expert_down_strides: Option<&'a GpuTensor>,
    // activation / residual
    pub x_rot: &'a GpuTensor,   // ffn_x_rot_batch [B, hidden]
    pub ffn_out: &'a GpuTensor, // ffn_out_batch [B, hidden] (accumulate target)
    // grouped-path scratch
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // shared scratch (grouped + scalar)
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // scalar-path scratch (expanded deterministic down)
    pub down_expert_outputs: &'a GpuTensor,
}

// ── Qwen3.5 softmax-top-k MoE prefill parameters (Ship 4.2) ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoePrefillCapturePoint {
    GateUpInput,
    DownInput,
}

/// Family-neutral view of a routed-expert activation seam. The callback sees
/// the complete teacher routing plus the already-built grouped permutation;
/// it may filter rows for calibration, but must not mutate execution routing.
pub struct MoePrefillCaptureBatch<'a> {
    pub layer: usize,
    pub point: MoePrefillCapturePoint,
    pub source: &'a GpuTensor,
    pub source_width: usize,
    pub source_row_div: usize,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub batch_size: usize,
    pub k_top: usize,
    pub num_experts: usize,
}

pub trait MoePrefillCapture: Send + Sync {
    fn capture(
        &self,
        gpu: &mut hipfire_rdna::Gpu,
        batch: &MoePrefillCaptureBatch<'_>,
    ) -> Result<(), DispatchError>;
}

/// Parameters for the qwen35 batched/prefill MoE routed-expert block.
///
/// Distinct from [`MoeBiasAwarePrefillParams`] — qwen35 uses softmax top-k
/// routing (k=8) with MQ4/MQ6/Paro routed experts, a fused gate-side, and a
/// shared expert that seeds `x_batch` before this arm runs.
///
/// The model owns RMSNorm, the router GEMV + softmax top-k (producing
/// `topk_indices` / `topk_weights`), and the shared expert (which already
/// accumulated into `x_batch`). This arm runs scatter → gate_up → unscatter →
/// SwiGLU+rotate → down → combine, accumulating into `x_batch`.
///
/// All tensor refs are `&'a GpuTensor` (shared, not `&mut` — GpuTensor is Copy).
/// Scratch tensors are model-owned; the family holds only references.
pub struct MoePrefillParams<'a> {
    pub layer: usize,
    pub capture: Option<&'a dyn MoePrefillCapture>,
    // dtype snapshot
    pub dtypes: MoeDtypes,
    // dims
    pub batch_size: usize,
    pub mi: usize,
    pub down_m: usize,
    pub down_k: usize,
    pub gate_up_k: usize,
    pub k_top: usize,
    pub n_exp: usize,
    /// m_total upper bound pre-computed by the model via
    /// `moe_grouped_m_total_bound(total_slots, n_exp)`. Used by Path 2
    /// scatter + grouped GEMM for grid sizing.
    pub m_total_max: usize,
    // routing inputs (model-produced)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    // destination = x_batch (residual; combine accumulates here)
    pub x_batch: &'a GpuTensor,
    // activation buffers
    pub x_norm_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    // routed gate_up/down pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Per-expert compact block stride (0 = that expert is Oq8), `[n_exp]` i32.
    /// `None` where no compact routed expert is resident. Required whenever
    /// `routed_indexable_oq_compact` is set -- it is what lets one launch serve a
    /// layer that mixes compact and promoted experts.
    pub expert_gate_up_strides: Option<&'a GpuTensor>,
    pub expert_down_strides: Option<&'a GpuTensor>,
    /// Per-expert AWQ scale pointer tables — see [`MoeParams`]'s fields of the
    /// same name.
    pub expert_gate_up_awq_ptrs: Option<&'a GpuTensor>,
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// True when resident routed experts sit in the `oq4_arch` COMBINED layout
    /// (the default for qt=34/37), so the OQ4 grouped kernel must skip past the
    /// split nibbles and split f32 scales to reach its interleaved block stream.
    /// False when `oq_moe` repacked them, which emits that stream at offset 0.
    pub routed_oq_arch_combined: bool,
    // intermediate buffers
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // Path 1 expanded-down scratch
    pub down_expanded: &'a GpuTensor,
    /// `[N × k_top × gate_up_k]` f32 — per-slot rotated gate_up input, the
    /// input-side mirror of `down_expanded`.
    pub x_rot_expanded: &'a GpuTensor,
    // Path 2 scatter scratch (model-owned)
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // paro sidecars (per-layer shared Givens rotation tables)
    pub paro_gate_up: Option<GivensRef<'a>>,
    pub paro_down: Option<GivensRef<'a>>,
    /// AWQ scale for the routed down weight (experts[0].down.awq_scale).
    /// Used by the AWQ-aware silu+rotate step. `None` when the routed
    /// experts are non-AWQ (the common case for A3B).
    pub down_awq_scale: Option<&'a GpuTensor>,
    /// EP (Ship 6 substrate-EP prefill): when `Some`, the **routed** combine
    /// accumulates into this **zeroed** `[batch × dim]` partial instead of
    /// `x_batch`; the EP prefill driver then all-reduce-sums the partial across
    /// ranks and adds it into each rank's `x_batch`. The **shared** expert stays
    /// in `x_batch` (replicated per rank — added once to each rank's own copy,
    /// no all-reduce). `None` (the default) accumulates routed into `x_batch`,
    /// byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
}

/// Resolved dispatch plan for the qwen35 batched MoE prefill routed block.
///
/// Distinct from [`MoeResolution`] (decode) — prefill adds the Path 0/1/2
/// grouped-vs-scalar down selection and the Paro i8/k8 levers.
/// Pure function of [`MoeDtypes`] + arch + [`FeatureFlags`].
pub struct MoePrefillResolution {
    /// Gate_up + down via grouped-GEMM scatter pipeline (Path 2).
    /// Requires WMMA-capable arch (gfx11/gfx12) + `moe_grouped_gemm` flag.
    pub use_path2: bool,
    /// Down uses atomic-accumulate GEMV (Path 0) instead of atomic-free
    /// expanded+combine (Path 1). gfx9* wave64 archs (gfx906/gfx908/gfx94x).
    pub down_path0: bool,
    /// gfx1151 Paro i8 MMQ grouped GEMM (Path 2 only).
    pub use_paro_i8: bool,
    /// gfx1151 Paro i8 MMQ k8 grouped GEMM (Path 2 only).
    pub use_paro_i8_k8: bool,
    /// Routed experts use ParoQ4G128 (determines SwiGLU+rotate kernel selection).
    pub paro_mode: bool,
}

impl MoePrefillResolution {
    /// Resolve the prefill dispatch plan from dtypes, arch, and flags.
    ///
    /// Reads MoE prefill env levers from `flags` (parsed once at `Gpu::init`),
    /// not `std::env` — mid-prefill env mutation is not honored.
    pub fn resolve(
        d: &MoeDtypes,
        arch: &hipfire_rdna::arch_caps::ArchCaps,
        flags: &hipfire_rdna::feature_flags::FeatureFlags,
    ) -> Self {
        let routed_dtype = d.routed_gate_up;
        let paro_mode = routed_dtype == DType::ParoQ4G128 && d.has_paro_shared;
        let is_gfx12 = arch.is_gfx1200() || arch.is_gfx1201();
        let grouped_supported = match routed_dtype {
            DType::MQ4G256 | DType::ParoQ4G128 => arch.has_wmma(),
            DType::MQ6G256 | DType::MQ3G256 => arch.is_gfx1151() || is_gfx12,
            DType::MQ2G256Lloyd => arch.is_gfx1151(),
            // gfx1151 uses the admitted raw WMMA kernels. Other architectures
            // retain the same grouped routing and use the portable active-route
            // fallback rather than rejecting source-model calibration.
            DType::F16 | DType::BF16 => true,
            _ => false,
        };
        // MQ3 and raw F16/BF16 have no indexed fallback, so do not let a tuning
        // opt-out route them into a nonexistent path.
        let grouped_required = matches!(routed_dtype, DType::MQ3G256 | DType::F16 | DType::BF16);
        let use_path2 = grouped_supported && (flags.moe_grouped_gemm || grouped_required);
        if !use_path2 {
            hipfire_rdna::kernel_trace::record_fallback(
                "moe prefill: no grouped MoE GEMM for this dtype/arch -> per-token indexed GEMV",
                &format!(
                    "{routed_dtype:?} arch_grouped_supported={grouped_supported} flag={}",
                    flags.moe_grouped_gemm
                ),
            );
        }
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151 && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8 && flags.moe_paro_i8_k8.unwrap_or(true);
        Self {
            use_path2,
            down_path0,
            use_paro_i8,
            use_paro_i8_k8,
            paro_mode,
        }
    }
}

// ── Family ─────────────────────────────────────────────

pub struct MoeFamily {
    registry: KernelRegistry,
}

impl MoeFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        moe_table::populate(&mut registry);
        registry
            .validate()
            .expect("moe kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given MoE variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        variant: MoeVariant,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run a single-token MoE decode step through the centralized executor.
    ///
    /// Delegates to [`crate::pipeline::run_moe_decode`], which dispatches the
    /// GPU top-K fast path (k=8 with an indexable routed dtype ∈ {MQ4G256,
    /// MQ6G256, ParoQ4G128}) or the generic CPU-top-K fallback (k != 8 or a
    /// non-indexable routed dtype). Resolution is owned here (the family
    /// resolves [`MoeDtypes`] → [`MoeResolution`]), and `ctx` is threaded
    /// through every inner GEMV so the call site builds one `DispatchCtx`
    /// per token (not 6+). Scratch stays model-owned.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut hipfire_rdna::Gpu,
        params: &MoeParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode(ctx, gpu, params)
    }

    /// Run a single-token deepseek4 bias-aware MoE decode step (k=6, MQ2-Lloyd
    /// routed experts). Delegates to [`crate::pipeline::run_moe_decode_bias_aware`].
    ///
    /// The model owns the router GEMV + `sqrt_softplus` (producing
    /// `params.scores`) and the shared expert (`ffn_stub`, which seeds
    /// `params.ffn_out`); this entry runs only the bias-aware top-k + routed
    /// MQ2-Lloyd expert sub-graph.
    ///
    /// Takes no `DispatchCtx`: the bias-aware path dispatches fixed MQ2-Lloyd
    /// kernels with no arch-gated sub-dispatch, so building a `DispatchCtx`
    /// per layer per token (an uncached `FeatureFlags::from_env` parse) would
    /// be pure waste on the decode hot path.
    pub fn run_bias_aware(
        &self,
        gpu: &mut hipfire_rdna::Gpu,
        params: &MoeBiasAwareParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_bias_aware(gpu, params)
    }

    /// Run a batched/prefill deepseek4 MoE step (k=6, MQ2-Lloyd): routing
    /// (bias-aware or hash) → routed experts (grouped GEMM when
    /// `batch_size >= gate`, else scalar K4 indexed) → combine, accumulating
    /// into `params.ffn_out`. Delegates to
    /// [`crate::pipeline::run_moe_prefill_bias_aware`]. The model owns RMSNorm,
    /// the shared expert, and the router GEMV + `sqrt_softplus`.
    pub fn run_bias_aware_prefill(
        &self,
        gpu: &mut hipfire_rdna::Gpu,
        params: &MoeBiasAwarePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill_bias_aware(gpu, params)
    }

    /// Run a batched/prefill qwen35 MoE routed-expert block (k=8, softmax
    /// top-k, MQ4/MQ6/Paro routed experts): scatter → gate_up → unscatter →
    /// SwiGLU+rotate → down → combine, accumulating into `params.x_batch`.
    ///
    /// The model owns RMSNorm, the router GEMV + softmax top-k, and the
    /// shared expert. Family owns resolution (`MoeDtypes` + arch + flags →
    /// [`MoePrefillResolution`]) and the full routed pipeline. `ctx` is
    /// decision-only (arch/env) — threaded once per chunk, not per layer.
    /// Delegates to [`crate::pipeline::run_moe_prefill`].
    pub fn run_prefill(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut hipfire_rdna::Gpu,
        params: &MoePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill(ctx, gpu, params)
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}
